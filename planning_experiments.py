# DIFFERENCE IN SCALES:

# POSITION: OBJECT SCALE = 100 * OBSERVATION SCALE

# VELOCITY: OBJECT SCALE = 20 * OBSERVATION SCALE

import numpy as np
import torch
import math
import gurobipy as gp
from gurobipy import GRB
import gymnasium as gym
from copy import deepcopy as copy
import sys
import imageio
import matplotlib.pyplot as plt
from numpy.linalg import norm
from difs import FullyConditionedUnet, GaussianDiffusionConditional, DiFS
import concurrent.futures
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
import random
import string
from diffusion_based_planner import DiffusionBasedPlanner
from kalman_filter import KalmanFilter
import gc
import pickle

USED_SAMPLES_PER_REPLAN = 40
TOTAL_SAMPLES_PER_REPLAN = 200
FILTER_SAMPLES = 20
NUM_EXPERIMENTS = 5000
SOUTH_MODEL = "./models/run_59_seed_2_SOUTH_distilled.pt"
WEST_MODEL = "./models/run_56_seed_2_WEST_distilled.pt"
NORTH_MODEL = "./models/run_58_seed_2_NORTH_distilled.pt"
EAST_MODEL = "./models/run_57_seed_2_EAST_distilled.pt"

# Intersection risk simulation
def sim_risk(disturbances, env, perturbation_scale=0.15, horizon=24):
    # disturbances of shape (4, 24)
    # dx[0] , dx[1] , ..., dx[23]
    # dy[0] , dy[1] , ..., dy[23]
    # dvx[0], dvx[1], ..., dvx[23]
    # dvy[0], dvy[1], ..., dvy[23]

    def point_to_segment_distance(p, v, w):
        """Returns the minimum distance between point p and line segment vw."""
        l2 = np.sum((w - v)**2)  # length squared of segment vw
        if l2 == 0:
            return np.linalg.norm(p - v)  # v == w case
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)  # projection falls on segment
        return np.linalg.norm(p - projection)


    def project_polygon(polygon, axis):
        min_p, max_p = None, None
        for p in polygon:
            projected = p.dot(axis)
            if min_p is None or projected < min_p:
                min_p = projected
            if max_p is None or projected > max_p:
                max_p = projected
        return min_p, max_p


    def interval_distance(min_a: float, max_a: float, min_b: float, max_b: float):
        """
        Calculate the distance between [minA, maxA] and [minB, maxB]
        The distance will be negative if the intervals overlap
        """
        return min_b - max_a if min_a < min_b else min_a - max_b

    def minimum_separating_distance(a, b):
        """
        Computes minimum separating distance between a and b.

        See https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection

        :param a: polygon A, as a list of [x, y] points
        :param b: polygon B, as a list of [x, y] points
        :return: are intersecting, will intersect, translation vector
        """
        intersecting = True
        for polygon in [a, b]:
            for p1, p2 in zip(polygon, polygon[1:]):
                normal = np.array([-p2[1] + p1[1], p2[0] - p1[0]])
                normal /= np.linalg.norm(normal)
                min_a, max_a = project_polygon(a, normal)
                min_b, max_b = project_polygon(b, normal)

                if interval_distance(min_a, max_a, min_b, max_b) > 0:
                    intersecting = False
        
        if intersecting:
            return 0

        min_distance = float('inf')
        # Check vertex-vertex distances between the two polygons
        for vertex_a in a:
            for vertex_b in b:
                # Calculate the Euclidean distance between each pair of vertices
                distance = np.linalg.norm(vertex_a - vertex_b)
                min_distance = min(min_distance, distance)

        # Check vertex-edge distances between the two polygons
        for i in range(len(a) - 1):
            p1, p2 = a[i], a[(i + 1)]
            segment1 = np.array([p1, p2])
            
            for j in range(len(b) - 1):
                q1, q2 = b[j], b[(j + 1)]
                segment2 = np.array([q1, q2])
                
                # Check the distance between each vertex of one polygon and the edge of the other polygon
                for point in [np.array(p1), np.array(p2)]:
                    min_distance = min(min_distance, point_to_segment_distance(point, np.array(q1), np.array(q2)))
                for point in [np.array(q1), np.array(q2)]:
                    min_distance = min(min_distance, point_to_segment_distance(point, np.array(p1), np.array(p2)))

        return min_distance

    def desired_gap(ego_vehicle, intruder_vehicle, velocity_disturbance):
        # d0 = 10.0
        d0 = 25.0
        # tau = 1.5
        tau = 5.7
        ab = 15.0

        intruder_velocity = intruder_vehicle.velocity.copy()
        if velocity_disturbance is not None:
            intruder_velocity += velocity_disturbance

        dv = np.dot(ego_vehicle.velocity - intruder_velocity, ego_vehicle.direction)
        
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def not_zero(x: float, eps: float = 1e-2) -> float:
        if abs(x) > eps:
            return x
        elif x >= 0:
            return eps
        else:
            return -eps

    def simulate (env, disturbances, horizon):
        md_crit = 0.1
        tma_crit = 2.0
        origin_value = -1
        alpha = -origin_value/md_crit
        beta = -origin_value/tma_crit
        gamma = origin_value
        target_speed = 4.0
        max_acceleration = 5.0
        delta = 3.0

        road = env.unwrapped.road
        intruder_vehicle = road.vehicles[0]
        ego_vehicle = road.vehicles[1]

        robustness = float('inf')
        obs = np.empty((horizon - 1, 2, 4))
        # Attempt 2
        for i in range(1, horizon):
            robustness = min(robustness, minimum_separating_distance(ego_vehicle.polygon(), 
                                                                     intruder_vehicle.polygon()))

            # low_risk is true for the following three cases:
            # (1) ego and intruder are on opposite lanes of the same branch
            # (2) the intruder is on the same lane behind the ego
            # (3) ego has traversed the intersection, intruder on some other lane

            low_risk = False
            ego_lane_index = ego_vehicle.lane_index
            intruder_lane_index = intruder_vehicle.lane_index
            ego_position = ego_vehicle.position.copy()
            intruder_position = intruder_vehicle.position.copy()
            velocity_disturbance = None
            if disturbances is not None:
                intruder_position += disturbances[:2, i]
                velocity_disturbance = disturbances[2:, i]

            if (ego_lane_index[0] == intruder_lane_index[1] and
                ego_lane_index[1] == intruder_lane_index[0]):
                low_risk = True     # Case 1
            if (ego_lane_index[:2] == intruder_lane_index[:2]):     # Case 2, on same lane
                if (ego_lane_index[1][0] == 'o' and
                    norm(ego_position) > norm(intruder_position)):
                    low_risk = True     # both going out, with intruder behind
                if (ego_lane_index[1][0] == 'i' and
                    norm(ego_position) < norm(intruder_position)):
                    low_risk = True     # both going in, with intruder behind
            if (ego_lane_index[1][0] == 'o' and 
                ego_lane_index[:2] != intruder_lane_index[:2]):
                low_risk = True     # Case 3
            if (ego_lane_index[0] == intruder_lane_index[1]):
                low_risk = True     # Case 4

            current_speed = ego_vehicle.speed
            a = max_acceleration * (
                1
                - np.power(
                    max(current_speed, 0) / target_speed,
                    delta,
                )
            )

            if not low_risk:
                d = norm(ego_position - intruder_position)
                a -= max_acceleration * np.power(desired_gap(ego_vehicle, intruder_vehicle, velocity_disturbance) / not_zero(d), 2)
            else:
                a = 0.1 if current_speed < target_speed else 0.0

            obs[i - 1] = env.step(np.array([a]))[0][:2, 1:5]

        # Rescale to object scale
        obs[:, :, :2] *= 100
        obs[:, :, 2:] *= 20

        return robustness, obs

    disturbances /= perturbation_scale

    robustness, obs = simulate(env, disturbances, horizon)

    return robustness, obs



def RobustIDMPolicy(env, possible_intruder_states, noise, intruder_origin, validity_threshold=8):
    # env: the simulation environment at the current state
    # possible_intruder_states: list of np arrays of shape (4,)
    # noise: np array of shape (4,), noise at current time step
    def desired_gap(ego_vehicle, intruder_velocity, velocity_disturbance):
        # velocity disturbance should be None when computing on failure
        # samples, since the samples are already generated with noisy observations

        # d0 = 25.0
        d0 = 25.0
        tau = 5.7
        ab = 15.0

        if velocity_disturbance is not None:
            intruder_velocity += velocity_disturbance

        dv = np.dot(ego_vehicle.velocity - intruder_velocity, ego_vehicle.direction)
        
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def not_zero(x: float, eps: float = 1e-2) -> float:
        if abs(x) > eps:
            return x
        elif x >= 0:
            return eps
        else:
            return -eps

    target_speed = 4.0
    max_acceleration = 5.0
    delta = 3.0

    road = env.unwrapped.road
    intruder_vehicle = road.vehicles[0]
    ego_vehicle = road.vehicles[1]

    # low_risk is true for the following three cases:
    # (1) ego and intruder are on opposite lanes of the same branch
    # (2) the intruder is on the same lane behind the ego
    # (3) ego has traversed the intersection, intruder on some other lane

    low_risk = False
    ego_lane_index = ego_vehicle.lane_index
    ego_destination = int(ego_vehicle.route[-1][1][-1])
    intruder_lane_index = intruder_vehicle.lane_index
    ego_position = ego_vehicle.position.copy()
    intruder_position = intruder_vehicle.position.copy()

    intruder_position += noise[:2]
    velocity_disturbance = noise[2:]

    # Filter for realistic failure samples
    valid_intruder_states = []
    for intruder_state in possible_intruder_states:
        if norm(intruder_state[:2] - intruder_position) < validity_threshold and norm(intruder_state[:2]) >= 1e-3:
            valid_intruder_states.append(intruder_state)

    samples_in_use = len(valid_intruder_states)
    print(samples_in_use, " samples used for planning")


    if (ego_lane_index[0] == intruder_lane_index[1] and
        ego_lane_index[1] == intruder_lane_index[0]):
        low_risk = True     # Case 1
    if (ego_lane_index[:2] == intruder_lane_index[:2]):     # Case 2, on same lane
        if (ego_lane_index[1][0] == 'o' and
            norm(ego_position) > norm(intruder_position)):
            low_risk = True     # both going out, with intruder behind
        if (ego_lane_index[1][0] == 'i' and
            norm(ego_position) < norm(intruder_position)):
            low_risk = True     # both going in, with intruder behind
    if (ego_lane_index[1][0] == 'o' and 
        ego_lane_index[:2] != intruder_lane_index[:2]):
        low_risk = True     # Case 3
    if (ego_lane_index[0] == intruder_lane_index[1]):
        low_risk = True     # Case 4

    # Stopping in the middle of the intersection is in general dangerous.
    # If we are in the intersection, and if we are crossing or have already
    # crossed all possible trajectories of the intruder, we should get
    # out of the intersection as quickly as possible. So for the following
    # cases, we should always accelerate.
    #
    # 1. Ego going East and inside the crossing.
    # 2. Ego going North and inside the crossing, intruder coming from North.
    # 3. Ego going North and inside the upper lane of the crossing.
    # 4. Ego going West and inside the leftward lane.

    if ego_destination == 3 and ego_position[0] < 10 and ego_position[1] < 8:
        # Case 1
        return 0.2, None

    # if ego_destination == 2 and intruder_origin == 2 and abs(ego_position[1]) < 4:
    #     # Case 2
    #     low_risk = True

    if ego_destination == 2 and ego_position[1] < 0 and ego_position[1] > -4:
        # Case 3
        return 0.2, None

    if ego_destination == 1 and ego_position[0] < 0 and ego_position[0] > -10:
        # Case 4
        return 0.2, None

    current_speed = ego_vehicle.speed
    a = max_acceleration * (
        1
        - np.power(
            max(current_speed, 0) / target_speed,
            delta,
        )
    )

    if not low_risk:
        ds = [norm(ego_position - intruder_position),]
        for intruder_state in valid_intruder_states:
            ds.append(norm(ego_position - intruder_state[:2]))

        desired_gaps = [desired_gap(ego_vehicle, intruder_vehicle.velocity.copy(), velocity_disturbance)]
        for intruder_state in valid_intruder_states:
            desired_gaps.append(desired_gap(ego_vehicle, intruder_state[2:], None))

        deductors = [max_acceleration * np.power(desired_gaps[i] / not_zero(ds[i]), 2) for i in range(samples_in_use + 1)]

        a -= np.max(deductors)
        return a, deductors
        
    else:
        a = 0.5 if current_speed < target_speed else 0.0
        return a, None


# Number of experiments in which a collision occurs
NUM_COLLISIONS = 0
# Number of experiments in which the destination is not reached
NUM_DELAYS = 0


for experiment in range(NUM_EXPERIMENTS):
    try:
        # Environment params
        horizon = 24
        xdim = 4
        px_variance = 1.0
        
        VIDEO_SAVEPATH = "./media/" + str(experiment) + ".mp4"
        ENV_SAVEPATH = "./media/" + str(experiment) + "_env.pkl"
        LOGS_SAVEPATH = "./media/" + str(experiment) + "_logs.pkl"
        NOISE_SAVEPATH = "./media/" + str(experiment) + "_noise.pkl"
        ACTIONS_SAVEPATH = "./media/" + str(experiment) + "_act.pkl"

        # Noise for this simulation
        NOISE = torch.randn(24, 4).numpy() / 0.15
        # ACTIONS_TAKEN = []

        # Initialize Highway Environment
        env = gym.make("intersection-v0",render_mode='rgb_array')
        env.unwrapped.config.update({"initial_vehicle_count":1})
        env.unwrapped.config.update({"spawn_probability":0.0})
        env.unwrapped.configure({
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": False,
            }})

        # Reset environment and record initial states
        obs = env.reset()
        # backup_env_copy = copy(env)

        # This will later store the low-robustness state trajectories
        list_of_obs = [np.empty((24, 2, 4)) for i in range(USED_SAMPLES_PER_REPLAN)]
        for i in range(USED_SAMPLES_PER_REPLAN):
            current_obs = obs[0][:2, 1:5].copy()
            current_obs[:, :2] *= 100
            current_obs[:, 2:] *= 20
            list_of_obs[i][0] = current_obs

        # Initial states for diffusion sampling, object scale, with Gaussian Noise
        road = env.unwrapped.road
        intruder_vehicle = road.vehicles[0]
        ego_vehicle = road.vehicles[1]

        INIT_POS = ego_vehicle.position.copy()
        print("INIT_POS = ", INIT_POS)
        INIT_VEL = ego_vehicle.velocity.copy()

        EGO_DESTINATION = int(ego_vehicle.route[-1][1][-1])
        print("EGO DESTINATION = ", EGO_DESTINATION)
        relative_position = intruder_vehicle.position - ego_vehicle.position + NOISE[0, :2]
        relative_velocity = intruder_vehicle.velocity - ego_vehicle.velocity + NOISE[0, 2:4]
        initial_states = list(relative_position) + list(relative_velocity)
        initial_states.append(float(EGO_DESTINATION))
        inits = torch.Tensor(initial_states)
        inits = inits.unsqueeze(0).repeat(TOTAL_SAMPLES_PER_REPLAN, 1).to('cuda')

        # Origin of the intruder vehicle (i.e. scenario)
        INTRUDER_ORIGIN = int(intruder_vehicle.route[0][0][-1])
        print("INTRUDER ORIGIN = ", INTRUDER_ORIGIN)


        # Make copies of the simulation environment. Because the destination of the intruder
        # is not known, each copy randomly resamples the destination of the intruder.
        env_copies = []
        for i in range(TOTAL_SAMPLES_PER_REPLAN):
            # Make copy of the environment
            env_copy = copy(env)
            # Get intruder vehicle object in copy
            intruder_copy = env_copy.unwrapped.road.vehicles[0]
            # Possible destinations for the intruder
            destination_options = [0, 1, 2, 3]
            destination_options.remove(INTRUDER_ORIGIN)
            # Sample a destination for the intruder
            sampled_destination = np.random.choice(destination_options)
            # Intruder plans route to the sampled destination
            intruder_copy.plan_route_to("o" + str(sampled_destination))
            intruder_copy.randomize_behavior()
            # Append the copy to the list
            env_copies.append(env_copy)


        # LOAD THE DISTILLATION MODEL!
        model = FullyConditionedUnet(
                dim = 64,
                dim_mults = (2, 4, 8, 16),
                channels = 4,
                cond_dim = 24,
        ).to('cuda')

        diffusion = GaussianDiffusionConditional(
            model,
            seq_length=horizon,
            classifier_free_guidance=False,
            timesteps=1
        ).to('cuda')


        print("Making model...")
        sampler = DiFS(
            None,
            diffusion,
            evaluate_fn=sim_risk,
            init_disturbances=None,
            envs_list=None,
            run_serial=None,
            save_path=None,
            alpha=0.9,
            N=1,
            train_num_steps=40000,
            train_batch_size=256,
            train_lr=1e-4,
            use_wandb=False,
            save_intermediate=False
        )


        print("loading model...")
        if INTRUDER_ORIGIN == 0:
            sampler.load(SOUTH_MODEL)
        elif INTRUDER_ORIGIN == 1:
            sampler.load(WEST_MODEL)
        elif INTRUDER_ORIGIN == 2:
            sampler.load(NORTH_MODEL)
        elif INTRUDER_ORIGIN == 3:
            sampler.load(EAST_MODEL)
        else:
            assert False

        # Draw samples from the model
        print("Setting conditions...")
        cond = torch.full((TOTAL_SAMPLES_PER_REPLAN, ), 0.0).to('cuda')

        print("Drawing samples...")
        samples = sampler.sample(cond, inits, no_grad=True).cpu().numpy()

        print("Simulating on samples drawn...")
        elite_idxs = None
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit tasks with their indices
            futures = {executor.submit(sim_risk, samples[i], env_copies[i]): i for i in range(samples.shape[0])}

            # Collect results in order of completion, storing the index
            results = []
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))

            # Sort results by index and extract the values
            results.sort(key=lambda x: x[0])
            simulations = [result for _, result in results]
            robustnesses = np.array([robustness for robustness, _ in simulations])
            elite_idxs = np.argsort(robustnesses)[:USED_SAMPLES_PER_REPLAN]
            for i in range(USED_SAMPLES_PER_REPLAN):
                list_of_obs[i][1:] = simulations[elite_idxs[i]][1]

        # intruder state trajectories in low robustness samples
        intruder_trajectories = [trajectory[:, 1, :].transpose() for trajectory in list_of_obs]

        # SET UP GUROBI OPTIMIZER
        planner = DiffusionBasedPlanner(EGO_DESTINATION, USED_SAMPLES_PER_REPLAN)
        OPTIMAL_CONTROLS = planner.fit(INIT_POS, INIT_VEL, intruder_trajectories, 24)

        # frames = [env.render(),]
        # logs = []

        # Note: horizon = (number of remaining action steps) + 1

        # Diffusion-Based Planning Loop
        while True:
            # Perform next action
            obs = env.step(np.array([OPTIMAL_CONTROLS[0]]))
            # ACTIONS_TAKEN.append(OPTIMAL_CONTROLS[0])
            horizon -= 1
            # frames.append(env.render())

            road = env.unwrapped.road
            intruder_vehicle = road.vehicles[0]
            ego_vehicle = road.vehicles[1]
            # New ego vehicle states
            INIT_POS = ego_vehicle.position.copy()
            print("INIT_POS = ", INIT_POS)
            INIT_VEL = ego_vehicle.velocity.copy()

            if abs(INIT_POS[1]) < 35 or abs(intruder_vehicle.position[1]) < 25:
                # Ego/intruder vehicle is too close to the intersection; diffusion model not applicable
                break

            # Get new states (with noise) for diffusion sampling
            relative_position = intruder_vehicle.position - ego_vehicle.position + NOISE[24 - horizon, :2]
            relative_velocity = intruder_vehicle.velocity - ego_vehicle.velocity + NOISE[24 - horizon, 2:4]
            initial_states = list(relative_position) + list(relative_velocity)
            initial_states.append(float(EGO_DESTINATION))
            inits = torch.Tensor(initial_states)
            inits = inits.unsqueeze(0).repeat(TOTAL_SAMPLES_PER_REPLAN, 1).to('cuda')

            # This will later store the low-robustness state trajectories
            list_of_obs = [np.empty((horizon, 2, 4)) for i in range(USED_SAMPLES_PER_REPLAN)]
            for i in range(USED_SAMPLES_PER_REPLAN):
                current_obs = obs[0][:2, 1:5].copy()
                current_obs[:, :2] *= 100
                current_obs[:, 2:] *= 20
                list_of_obs[i][0] = current_obs

            # Make copies of the simulation environment. Because the destination of the intruder
            # is not known, each copy randomly resamples the destination of the intruder.
            env_copies = []
            for i in range(TOTAL_SAMPLES_PER_REPLAN):
                # Make copy of the environment
                env_copy = copy(env)
                # Get intruder vehicle object in copy
                intruder_copy = env_copy.unwrapped.road.vehicles[0]
                # Possible destinations for the intruder
                destination_options = [0, 1, 2, 3]
                destination_options.remove(INTRUDER_ORIGIN)
                # Sample a destination for the intruder
                sampled_destination = np.random.choice(destination_options)
                # Intruder plans route to the sampled destination
                intruder_copy.plan_route_to("o" + str(sampled_destination))
                intruder_copy.randomize_behavior()
                # Append the copy to the list
                env_copies.append(env_copy)

            # Draw samples from the model
            print("Drawing samples...")
            samples = sampler.sample(cond, inits, no_grad=True).cpu().numpy()

            print("Simulating on samples drawn...")
            elite_idxs = None
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Submit tasks with their indices
                futures = {executor.submit(sim_risk, samples[i], env_copies[i], 0.15, horizon): i for i in range(samples.shape[0])}

                # Collect results in order of completion, storing the index
                results = []
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    results.append((idx, future.result()))

                # Sort results by index and extract the values
                results.sort(key=lambda x: x[0])
                simulations = [result for _, result in results]
                robustnesses = np.array([robustness for robustness, _ in simulations])
                print("MEAN ROBUSTNESS OF DIFFUSION SAMPLES = ", np.mean(robustnesses))
                elite_idxs = np.argsort(robustnesses)[:USED_SAMPLES_PER_REPLAN]
                for i in range(USED_SAMPLES_PER_REPLAN):
                    list_of_obs[i][1:] = simulations[elite_idxs[i]][1]

            intruder_trajectories = [trajectory[:, 1, :].transpose() for trajectory in list_of_obs]
            
            # OPTIMIZATION-BASED PLANNER
            OPTIMAL_CONTROLS = planner.fit(INIT_POS, INIT_VEL, intruder_trajectories, horizon)

            for env_cpy in env_copies:
                env_cpy.close()

        # IDM LOOP
        # Since our previous replan, we have performed 1 action.
        # This puts us at index-1 on the failure trajectories

        idm_timestep = 1
        total_actions_remaining = horizon - 1

        intruder_position = env.unwrapped.road.vehicles[0].position.copy()
        intruder_velocity = env.unwrapped.road.vehicles[0].velocity.copy()
        o = np.concatenate((intruder_position, intruder_velocity))
        o += NOISE[24 - horizon]

        # We maintain a kalman filter for the actual observation
        # as well as each failure trajectory
        kalman_filters = [KalmanFilter(o),]
        for trajectory in intruder_trajectories:
            kalman_filters.append(KalmanFilter(trajectory[:, 1]))


        for i in range(total_actions_remaining):
            if i > 0:
                # update filter for real observation
                intruder_position = env.unwrapped.road.vehicles[0].position.copy()
                intruder_velocity = env.unwrapped.road.vehicles[0].velocity.copy()
                o = np.concatenate((intruder_position, intruder_velocity))
                o += NOISE[24 - horizon]
                kalman_filters[0].update(o)

                # update filters for failure trajectories
                for i in range(len(intruder_trajectories)):
                    trajectory = intruder_trajectories[i]
                    kalman_filters[i+1].update(trajectory[:, idm_timestep])


            possible_intruder_states = [trajectory[:, idm_timestep] for trajectory in intruder_trajectories]
            for kalman_filter in kalman_filters:
                possible_intruder_states += kalman_filter.sample(FILTER_SAMPLES)

            noise = NOISE[24 - horizon]
            action, deductors = RobustIDMPolicy(env, possible_intruder_states, noise, intruder_origin=INTRUDER_ORIGIN)
            # logs.append(deductors)
            idm_timestep += 1
            horizon -= 1

            env.step(np.array([action]))
            # ACTIONS_TAKEN.append(action)
            # frames.append(env.render())

        road = env.unwrapped.road
        intruder_vehicle = road.vehicles[0]
        ego_vehicle = road.vehicles[1]

        if env.unwrapped.vehicle.crashed:
            NUM_COLLISIONS += 1
            # imageio.mimsave(VIDEO_SAVEPATH, frames, fps=5)
            # with open(ENV_SAVEPATH, 'wb') as file:
            #     pickle.dump(backup_env_copy, file)
            # with open(LOGS_SAVEPATH, 'wb') as file:
            #     pickle.dump(logs, file)
            # with open(NOISE_SAVEPATH, 'wb') as file:
            #     pickle.dump(NOISE, file)
            # with open(ACTIONS_SAVEPATH, 'wb') as file:
            #     pickle.dump(ACTIONS_TAKEN, file)

        if EGO_DESTINATION == 1 and ego_vehicle.position[0] > -10:
            NUM_DELAYS += 1
        elif EGO_DESTINATION == 2 and ego_vehicle.position[1] > -10:
            NUM_DELAYS += 1
        elif EGO_DESTINATION == 3 and ego_vehicle.position[0] < 10:
            NUM_DELAYS += 1

        env.close()
        for env_cpy in env_copies:
            env_cpy.close()

        print("-----------------------------")
        print("Experiments performed: ", experiment + 1)
        print("Delays Found: ", NUM_DELAYS)
        print("Collisions Found: ", NUM_COLLISIONS)
        print("-----------------------------")
        gc.collect()

    except:
        continue

print("Total number of experiments: ", NUM_EXPERIMENTS)
print("Total delays: ", NUM_DELAYS)
print("Total collisions: ", NUM_COLLISIONS)

