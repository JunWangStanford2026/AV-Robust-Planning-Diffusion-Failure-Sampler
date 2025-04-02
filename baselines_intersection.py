import json
import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from torch.distributions import MultivariateNormal
import pomegranate.distributions as pgd
from tqdm import tqdm
from src import gmm_cross_entropy_method
from src.ast_env import ASTEnv
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from joblib import Parallel, delayed
import multiprocessing

# from src.validation_env import ValidationEnv
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')


## Intersection risk function
risk_threshold = -0.06

env = gym.make("intersection-v0",render_mode='rgb_array')
env.unwrapped.config.update({"initial_vehicle_count":1})
env.unwrapped.config.update({"spawn_probability":0.0})
env.unwrapped.configure({
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": False,
    }})
# def pendulum_step_risk(obs_traj):
#     thetas = np.arctan2(obs_traj[1], obs_traj[0])
#     thetas_degrees = np.rad2deg(thetas)
#     dist = np.max(np.abs(thetas_degrees))
#     dist = np.clip(dist, 0.0, risk_threshold)
#     return dist


# Simulate function
def sim_risk(x,perturbation_scale = 150):
    global env
    # env = copy(env_glob)


    def ttc(veh1,veh2,min_distance=0.0):
        x1 = veh1[1]
        y1 = veh1[2]
        vx1 = veh1[3]
        vy1 = veh1[4] 
    
        x2 = veh2[1]  
        y2 = veh2[2]
        vx2 = veh2[3]
        vy2 = veh2[4]   

        deltax = x2-x1
        deltay = y2-y1
        deltavx = vx2-vx1
        deltavy = vy2-vy1

        TMA = (-deltavx*deltax-deltavy*deltay)/(deltavx**2+deltavy**2)
        MIN_DISTANCE = (deltavy*deltax-deltavx*deltay)**2 / (deltavx**2+deltavy**2)

        # ttc = (min_distance**2-x1+x2-y1+y2)/(vx1**2-2*vx1*vx2+vx2**2+(vy1-vy2)**2)

        return TMA, MIN_DISTANCE

    def ttc_all_vehicles_intersection(vehicles):
        intersection = np.zeros(7)
        distance_intersection = np.array([np.linalg.norm(v[1:3]) for v in vehicles])
        ttcs = -np.array([ttc(vehicles[i],intersection) for i in range(vehicles.shape[0])])
        return ttcs, distance_intersection

    def most_critical_ttc(vehicles):
        ttcs,distance_intersection = ttc_all_vehicles_intersection(vehicles[1:2])  #exclude ego vehicle
        ttcs = np.where(np.sign(ttcs)==-1,np.nan,ttcs)
        if np.all(np.isnan(ttcs)):
            min_ttcs = 1000
        else:
            min_ttcs = np.nanmin(ttcs)
            if np.nanmin(np.where(distance_intersection<0.001,np.nan,distance_intersection)) <= 0.12:
                min_ttcs = 0.01    

        return min_ttcs

    def extract_risk_states(all_obs):
        # crashed = all_obs[-1][4]["crashed"]
        states = np.zeros((len(all_obs),2,4))
        vehicles_exist = np.ones((len(all_obs),2))
        
        for i in range(len(all_obs)):
            states[i] = all_obs[i][0][:2,1:5]
            vehicles_exist[i] = all_obs[i][0][:2,0]    
        
        mask = vehicles_exist.sum(axis=1) >= 2
        positions = states[mask,:,:2]
        distances = np.linalg.norm(np.diff(positions,axis=1).squeeze(),axis=1)
        risk = -np.min(distances)
        return risk, states

    def simulate (env,perturbations=np.zeros((24,4))):
        # perturbations = perturbations.T
            
        perturbations = perturbations.reshape((24,4)).numpy()

        #behavior constants
        md_crit = 0.05
        tma_crit = 1.5
        origin_value = -1
        alpha = -origin_value/md_crit
        beta = -origin_value/tma_crit
        gamma = origin_value
        target_speed = 4.0

        # global env    
        all_obs = []
        counter = 0    
        obs = env.reset()
        all_obs.append(copy(obs))
        # Attempt 2
        while len(all_obs)<perturbations.shape[0]:
            tma,pred_min_distance = ttc(obs[0][0],obs[0][1])
            # pred_min_distance = pred_min_distance-0.05
            # current_distance = np.linalg.norm(obs[0][0,1:3]-obs[0][1,1:3])
            # current_speed = env.road.vehicles[-1].speed #ego vehicle speed
            current_speed = env.unwrapped.road.vehicles[-1].speed
            if tma < 0:
                a = 0.0
            else:
                a = np.clip(alpha*pred_min_distance+beta*tma+gamma,-1,0)
            if a >= 0.0 and current_speed < target_speed:
                a = 0.1 #slowly acceleerate
            # print(tma,pred_min_distance,a)
            
            obs = env.step(np.array([a]))
            all_obs.append(copy(obs))
            obs[0][1,1:5] += perturbations[counter]
            counter += 1
            # im = env.render()
            # time.sleep(1.0)

        return extract_risk_states(all_obs)
    
    if x == 'random':
            x = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(96),torch.eye(96)).sample((1,))
            
    x = x/perturbation_scale    #rescale the perturbations
    risk, obs = simulate(env,x)
    # risk, obs = simulate(env)

    if np.isnan(risk):
        print("NAN ISSUES, THERE ARE VEHILCES", env.road.vehicles.__len__())

    return risk, obs

# def plot_pendulum_trajectory(obs_traj, is_failure):
#     # convert obs to angles
#     thetas = np.arctan2(obs_traj[:, 1], obs_traj[:, 0])
#     thetas_degrees = np.rad2deg(thetas)

#     # plot thetas
#     if is_failure:
#         idx_fail = np.where(np.abs(thetas_degrees) >= risk_threshold)[0][0]
#         plt.plot(thetas_degrees[:idx_fail], 'r', alpha=0.5)
#     else:
#         plt.plot(thetas_degrees, 'k', alpha=0.2)


def run_mc(mc_config):
    N = mc_config["N"]
    xdim = mc_config["xdim"]
    horizon = mc_config["horizon"]
    save_dir = mc_config["save_dir"]
    N_CORES = multiprocessing.cpu_count()
    # N_CORES = 24
    
    # px = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(xdim*horizon),torch.eye(xdim*horizon))
    # disturbances = px.sample((N,))
    # d_split = torch.split(disturbances,N_BATCH,dim=0)
    
    # processed_list = []
    # for k in range(int(N/N_BATCH)):
        
    #     print("Cycle", k, "/", int(N/N_BATCH))
    #     processed_list += Parallel(n_jobs=N_CORES)(delayed(sim_risk)(x=d_split[k][i].cpu().detach()) for i in tqdm(range(N_BATCH)))
    
    processed_list = Parallel(n_jobs=N_CORES)(delayed(sim_risk)(x='random') for i in tqdm(range(N)))
    
    risks = [p[0] for p in processed_list]
    trajectories = [p[1] for p in processed_list]
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(risks,save_dir + '/risks.pt')
    torch.save(trajectories,save_dir + '/trajectories.pt')
        
    
    


def run_cem(cem_config):
    # unpack the config
    N = cem_config['N']
    n_components = cem_config['n_components']
    n_elite = cem_config['n_elite']
    rho_target = cem_config['rho_target']
    max_iters = cem_config['max_iters']
    save_dir = cem_config['save_dir']
    Path(save_dir).mkdir(parents=True, exist_ok=True)



    # Load pendulum data
    # data_path = 'data/pendulum/output.json'
    # with open(data_path) as f:
    #     data_dict = json.load(f)

    ## Environment params
    # px_variance = data_dict['meta']['px_variance']
    # horizon = data_dict['meta']['horizon']
    # kp = data_dict['meta']['kp']
    # kd = data_dict['meta']['kd']
    xdim = 4
    px_variance = 1.0
    horizon = 24

    # # create inverted pendulum environment
    # env = gym.make('Pendulum-v1')
    # reset_options = {'x_init':0.0, 'y_init':0.0}
    # policy = lambda s: np.array([-kp*np.arctan2(s[1], s[0]) - kd* s[2]])
    # val_env = ValidationEnv(env, policy, 'action', reset_options=reset_options)

    # Create disturbance model
    px = MultivariateNormal(torch.zeros(xdim), px_variance*torch.eye(xdim))
    

    # Set up CEM
    #rho_fn = lambda x: sim_risk(x, val_env, pendulum_risk, xdim, odim, horizon)[0]
    px_cem = MultivariateNormal(torch.zeros(xdim*horizon), px_variance*torch.eye(xdim*horizon))
    components = [pgd.Normal(torch.zeros(xdim*horizon), 1.0*torch.ones(xdim*horizon), covariance_type='diag') for _ in range(n_components)]

    # Run cem
    sim_fn = lambda x: sim_risk(x)
    model = gmm_cross_entropy_method(sim_fn, rho_target, components, px_cem, N, n_elite, max_iters, save_dir, verbose=True)
    
    # Plot samples
    samples = model.sample(1000)
    evals = [sim_fn(s) for s in samples]
    risks = [e[0] for e in evals]
    obs = [e[1] for e in evals]

    rs = torch.tensor(risks)
    fs = rs >= rho_target
    obs_trajectories = torch.stack(obs)

    # plt.figure()
    # for i in range(1000):
    #     plot_pendulum_trajectory(obs[i], risks[i] >= rho_target)

    # plt.xlabel('time step')
    # plt.ylabel('theta')
    # plt.xlim(0, horizon)
    # plt.ylim(-90, 90)
    # plt.savefig(f'{save_dir}/cem_samples.png')


    # Save off samples and state trajectories
    #torch.save(samples, f'{save_dir}/samples-final.pt')

    return


# def run_ast(ast_config):
#     # unpack the config
#     n_timesteps = ast_config['n_timesteps']
#     n_samples = ast_config['n_samples']
#     save_dir = ast_config['save_dir']
#     Path(save_dir).mkdir(parents=True, exist_ok=True)

#     # Delete the old AST samples
#     if os.path.exists(f'{save_dir}/disturbances.pt'):
#         os.remove(f'{save_dir}/disturbances.pt')

#     if os.path.exists(f'{save_dir}/observations.pt'):
#         os.remove(f'{save_dir}/observations.pt')

#     # Load pendulum data
#     data_path = 'data/pendulum/output.json'
#     with open(data_path) as f:
#         data_dict = json.load(f)

#     # Environment params
#     px_variance = data_dict['meta']['px_variance']
#     horizon = data_dict['meta']['horizon']
#     kp = data_dict['meta']['kp']
#     kd = data_dict['meta']['kd']
#     xdim = 1
#     odim = 3

#     # create inverted pendulum environment
#     env = gym.make('Pendulum-v1')
#     reset_options = {'x_init':0.0, 'y_init':0.0}
#     policy = lambda s: np.array([-kp*np.arctan2(s[1], s[0]) - kd* s[2]])
#     print("action space: ", env.action_space)
#     # Create disturbance model
#     px = MultivariateNormal(torch.zeros(xdim), px_variance[0]*torch.eye(xdim))
    
#     # Create AST environment
#     policy_fn = lambda x: policy(x)
#     ast_env = ASTEnv(env,
#                      policy_fn,
#                      px,
#                      pendulum_step_risk,
#                      risk_threshold,
#                      horizon,
#                      reset_kwargs=reset_options,
#                      save_dir=save_dir,
#                      noise="actions",
#                      event_reward=10.0,
#                      save=False,
#                     )

#     # Run AST
#     model = PPO("MlpPolicy", ast_env, verbose=1)
#     model.learn(total_timesteps=n_timesteps)


#     # generate some rollouts from the policy
#     def simulate_policy(model, env, horizon):
#         obs_trjaectory = np.zeros((horizon, env.observation_space.shape[0]))
#         act_trajectory = np.zeros((horizon, env.action_space.shape[0]))
#         obs, _ = env.reset()
#         for i in range(horizon):
#             action, _states = model.predict(obs)
#             obs, rewards, _, _, info = env.step(action)
#             obs_trjaectory[i, :] = obs

#         return act_trajectory, obs_trjaectory
    

#     ast_xs = np.zeros((n_samples, horizon, xdim))
#     ast_rs = np.zeros(n_samples)
#     ast_fs = np.zeros(n_samples)

#     obs_trajs = np.zeros((n_samples, horizon, odim))

#     for i in range(n_samples):
#         act_traj, obs_traj = simulate_policy(model, ast_env, horizon)
#         risk = pendulum_risk(torch.tensor(obs_traj))
#         obs_trajs[i, :, :] = obs_traj
#         ast_xs[i, :, :] = act_traj
#         ast_rs[i] = risk
#         ast_fs[i] = risk >= risk_threshold


#     ## Plotting
#     # convert obs to angles
#     thetas = np.arctan2(obs_trajs[:, :, 1], obs_trajs[:, :, 0])
#     thetas_degrees = np.rad2deg(thetas)

#     # plot thetas
#     plt.figure()
    
#     for i in range(n_samples):
#         if ast_fs[i]:
#             idx_fail = np.where(np.abs(thetas_degrees[i, :]) >= risk_threshold)[0][0]
#             plt.plot(thetas_degrees[i, :idx_fail])
#         else:
#             plt.plot(thetas_degrees[i, :])
    
#     plt.xlabel('time step')
#     plt.ylabel('theta')
#     plt.xlim(0, horizon)
#     plt.ylim(-90, 90)
#     plt.savefig(f'{save_dir}/ast_samples.png')

#     # Save off samples from the policy
#     torch.save(ast_xs, f'{save_dir}/samples-1.pt')
#     torch.save(ast_rs, f'{save_dir}/risks-1.pt')
#     torch.save(obs_trajs, f'{save_dir}/observations-1.pt')


if __name__ == '__main__':
    Ntotal = 10000*5
    save_dir = './results_intersection/baselines_2'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    Ntrial = 5
    
    mc_config = {
        "N":2000000,
        "xdim":4,
        "horizon":24
    }

    cem_config = {
        'N': 5000,
        'n_components': 2,
        'n_elite': 500,
        'rho_target': -0.06,
        'max_iters': 50,
        #'save_dir': save_dir + '/cem2',
    }

    ast_config = {
        'n_timesteps': 10*50000,
        'n_samples': 1000,
        #'save_dir': save_dir + '/ast',
    }

    mc_configs = [mc_config]
    cem_configs = [cem_config]
    ast_configs = [ast_config]
    
    # # Run MC
    # for i, cfg in enumerate(mc_configs):
    #     save_dir_in = save_dir +'/mc' + f'/config_{i}'
    #     cfg['save_dir'] = save_dir_in
        
    #     run_mc(cfg)

    # Run CEM
    for i, cfg in enumerate(cem_configs):
        for n in tqdm(range(Ntrial)):
            save_dir_in = save_dir +'/cem3' + f'/config_{i}' + f'/{n}'
            cfg['save_dir'] = save_dir_in
            
            run_cem(cfg)
    
    # # Run AST
    # for i, cfg in enumerate(ast_configs):
    #     for n in tqdm(range(Ntrial)):
    #         save_dir_in = save_dir +'/ast' + f'/config_{i}' + f'/{n}'
    #         cfg['save_dir'] = save_dir_in    
            
    #         run_ast(cfg)
    
    # run_cem(cem_config)
    # run_ast(ast_config)