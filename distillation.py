# Serial Number of this Run
RUN_SERIAL = 59

import torch
from torch import nn
from torch.optim import Adam, AdamW
from difs import FullyConditionedResnet, FullyConditionedUnet, GaussianDiffusionConditional, GaussianDiffusionConditionalTrainer, DiFS
import random
import numpy as np
from numpy.linalg import norm
import gymnasium as gym
from copy import deepcopy as copy
from multiprocessing import cpu_count
from accelerate import Accelerator, DataLoaderConfiguration
from ema_pytorch import EMA
import concurrent.futures
from tqdm import tqdm
import sys
from prdc import compute_prdc
import wandb

# Setting
SCENARIO = "SOUTH"
RANDOM_SEED = 2

# HYPERPARAMETERS
TIMESTEPS = 1
LAMB = 1.0
DIS_BATCH_SIZE = 2048
GEN_BATCH_SIZE = 2048
# intersection robustness threshold
RHO_THRESHOLD = 0.0
GENERATOR_TRAIN_LR = 3e-4
DISCRIMINATOR_TRAIN_LR = 3e-4
GENERATOR_REG = 6e-1
DISCRIMINATOR_REG = 1e-1
# DISCRIMINATOR_DROPOUT_RATE = 0.5
DISCRIMINATOR_DIM = 64

# Training logistics
NUM_ITERS = 10000
CHECK_EVERY = 1

# CONSTANTS
XDIM = 4
HORIZON = 24

# Initial models (if any)

# Model obtained after difs pretraining
INIT = None
# For supervised pretraining and distillation
TEACHER_SAMPLES = None
TEACHER_CONDS = None
TEACHER_INITS = None

# Model obtained after supervised pretraining
PRETRAINED_MODEL = None

# Teacher model
TEACHER_MODEL = None

# Enter the save path of the output model!!!!
# !!!!!!!!!!!!!!!!!!
# For difs pretraining, enter the save path in difs/trainer.py: DiFS.train()
# !!!!!!!!!!!!!!!!!!
SAVE_PATH = "models/run_59_seed_2_SOUTH_difs_pretrained.pt"
assert SAVE_PATH != None


config = {
    "teacher": TEACHER_MODEL,
    "scenario": SCENARIO,
    "random_seed": RANDOM_SEED,
    "lambda": LAMB,
    "timesteps": TIMESTEPS,
    "rho_threshold": RHO_THRESHOLD,
    "gen_batch_size": GEN_BATCH_SIZE,
    "dis_batch_size": DIS_BATCH_SIZE,
    "generator_lr": GENERATOR_TRAIN_LR,
    "discriminator_lr": DISCRIMINATOR_TRAIN_LR,
    # "discriminator_dropout": DISCRIMINATOR_DROPOUT_RATE,
    "generator L2 regularization": GENERATOR_REG,
    "discriminator L2 regularization": DISCRIMINATOR_REG,
    "discriminator_dim": DISCRIMINATOR_DIM
}

wandb.init(entity="distillation_difs", 
           project="GAN",
           config=config,
           )


def sim_risk(disturbances, env, perturbation_scale=0.15):
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

    def simulate (env, disturbances):
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
        # Attempt 2
        for i in range(1, 24):
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

            env.step(np.array([a]))

        return robustness

    disturbances /= perturbation_scale

    robustness = simulate(env, disturbances)

    if robustness == float('inf'):
        print("NAN ISSUES, THERE ARE VEHILCES", env.road.vehicles.__len__())

    return robustness







def evaluate(x, env):
    """
    Compute robustness and state trajectory. Wrapper for the simulation function
    """
    return sim_risk(x, env)



# class Unsqueeze(nn.Module):
#     def forward(self, x):
#         return x.unsqueeze(1)


# class Discriminator(nn.Module):
#     def __init__(self, dropout_prob=DISCRIMINATOR_DROPOUT_RATE):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3)
#         self.dropout1 = nn.Dropout(p=dropout_prob)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
#         self.dropout2 = nn.Dropout(p=dropout_prob)
#         self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
#         self.batchnorm = nn.BatchNorm1d(64)
#         self.fc1 = nn.Linear(64, 32)
#         self.fc2 = nn.Linear(32, 1)  # Output layer, predicting single value
    
#     def forward(self, x):
#         # x shape: (N, 4, 24)
#         x = F.relu(self.conv1(x))  # Apply 1D conv with ReLU activation
#         x = self.dropout1(x)
#         x = F.relu(self.conv2(x))  # Apply another 1D conv with ReLU activation
#         x = self.dropout2(x)
#         x = x.permute(0, 2, 1)  # Reshape for LSTM: (N, 24, 32)
#         _, (hn, _) = self.lstm(x)  # Get final hidden state of LSTM
#         x = F.relu(self.fc1(self.batchnorm(hn[-1])))  # Feed LSTM hidden state to FC layer with ReLU
#         x = self.fc2(x)  # Output layer, no activation for regression
#         return x.squeeze(1)  # Squeeze to make output shape (N,)

# import torch.nn.functional as F



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Discriminator(nn.Module):
    def __init__(self, discriminator_dim = DISCRIMINATOR_DIM, horizon = HORIZON):
        super(Discriminator, self).__init__()

        # self.resnet = FullyConditionedResnet(XDIM, discriminator_dim, 
        #                                      time_emb_dim=None, cond_emb_dim=None, inits_emb_dim=5)
        self.resnet = FullyConditionedResnet(XDIM, discriminator_dim, 
                                             time_emb_dim=None, cond_emb_dim=None, inits_emb_dim=4)
        self.flattener = Flatten()
        self.fc = nn.Linear(discriminator_dim * horizon, 1)
    
    def forward(self, x, i):
        # x shape: (N, 4, 24)
        # i shape: (N, 5)
        return self.fc(self.flattener(self.resnet(x, time_emb=None, cond_emb=None, inits_emb=i))).squeeze(1)



def discriminator():
    """
    Build and return a PyTorch model implementing the discriminator architecture.
    """

    model = Discriminator().to('cuda')
    return model


def generator(mode: str, envs_list,
              channels=XDIM, horizon=HORIZON, 
              train_lr=GENERATOR_TRAIN_LR, px_variance=1.0,
              evaluate_fn=evaluate, N=40000):
    model = FullyConditionedUnet(
        dim = 64,
        dim_mults = (2, 4, 8, 16),
        channels = 4,
        cond_dim = 24,
    ).to('cuda')

    if "pretraining" in mode:
        diffusion = GaussianDiffusionConditional(
            model,
            seq_length=horizon,
            classifier_free_guidance=False,
            timesteps=TIMESTEPS
        ).float().to('cuda')
    else:
        diffusion = GaussianDiffusionConditionalTrainer(
            model,
            seq_length=horizon,
            classifier_free_guidance=False,
            timesteps=TIMESTEPS
        ).float().to('cuda')

    # Initial disturbances
    data = px_variance * torch.randn(N,channels,horizon).to('cuda')

    difs = DiFS(
        RANDOM_SEED,
        diffusion,
        evaluate_fn=evaluate,
        init_disturbances=data,
        envs_list=envs_list,
        run_serial=RUN_SERIAL,
        save_path=SAVE_PATH,
        alpha=0.9,
        N=N,
        train_num_steps=40000,
        train_batch_size=256,
        train_lr=GENERATOR_TRAIN_LR,
        use_wandb=True,
        save_intermediate=True
    )

    if mode == "supervised_pretraining":
        print("Loading model")
        difs.load(INIT)

    if "gan" in mode:
        print("Loading Model")
        difs.load(PRETRAINED_MODEL)

    return difs

def teacher(envs_list, channels=XDIM, horizon=HORIZON, px_variance=1.0,
            evaluate_fn=evaluate, N=1000):
    model = FullyConditionedUnet(
        dim = 64,
        dim_mults = (2, 4, 8, 16),
        channels = 4,
        cond_dim = 24,
    ).to('cuda')

    diffusion = GaussianDiffusionConditionalTrainer(
        model,
        seq_length=horizon,
        classifier_free_guidance=False,
        timesteps=1000
    ).float().to('cuda')

    for param in diffusion.parameters():
        # Teacher model must be fixed
        param.requires_grad = False

    # Initial disturbances
    data = px_variance * torch.randn(N,channels,horizon).to('cuda')

    difs = DiFS(
        RANDOM_SEED,
        diffusion,
        evaluate_fn=evaluate,
        init_disturbances=data,
        envs_list=envs_list,
        run_serial=RUN_SERIAL,
        save_path=None,
        alpha=0.9,
        N=N,
        train_num_steps=40000,
        train_batch_size=256,
        train_lr=GENERATOR_TRAIN_LR,
        use_wandb=True,
        save_intermediate=True
    )

    print("Loading model")
    difs.load(TEACHER_MODEL)

    return difs


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function in PyTorch.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    bce = nn.BCEWithLogitsLoss()
    return bce(input.squeeze(), target)

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    true_labels = torch.ones(logits_real.shape[0]).to('cuda')
    fake_labels = torch.zeros(logits_fake.shape[0]).to('cuda')
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, fake_labels)

    return loss

def generator_gan_loss(logits_fake):
    tgt_logits = torch.ones(logits_fake.shape[0]).to('cuda')
    # GAN loss
    return bce_loss(logits_fake, tgt_logits)

def generator_loss(logits_fake, generator_outputs, teacher_reconstructions, reconstruction_weights, lamb=LAMB):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    - generator_outputs: Pytorch Tensor of shape (N, 4, 24), outputs of generator
    - teacher_reconstructions: Pytorch Tensor of shape (N, Nt, 4, 24) containing 
      teacher reconstructions of generator outputs at Nt different timesteps
    - reconstruction_weights: Pytorch Tensor of shape (Nt,) giving the weights of all timesteps

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    tgt_logits = torch.ones(logits_fake.shape[0]).to('cuda')
    # GAN loss
    loss = bce_loss(logits_fake, tgt_logits)

    if abs(lamb) < 0.01:
        # Pure GAN without distillation loss
        return loss

    batch_size, num_timesteps, _, _ = teacher_reconstructions.shape

    generator_outputs = generator_outputs.unsqueeze(1).repeat(1, num_timesteps, 1, 1)
    # difference[i, j] is difference b/w i-th generator output and j-th teacher reconstruction of the i-th generator output
    differences = torch.norm(torch.abs(generator_outputs - teacher_reconstructions), dim=(2, 3))
    reconstruction_weights = reconstruction_weights.unsqueeze(0).repeat(batch_size, 1)
    differences = reconstruction_weights * differences
    # Add distillation loss
    loss += (lamb * torch.mean(differences))

    return loss


def get_optimizer(model, gen_train_lr=GENERATOR_TRAIN_LR, dis_train_lr=DISCRIMINATOR_TRAIN_LR):
    """
    Construct and return an optimizer for the model.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    if type(model) == DiFS:
        return AdamW(model.model.parameters(), lr=gen_train_lr, weight_decay=GENERATOR_REG)
    
    return AdamW(model.parameters(), lr=dis_train_lr, weight_decay=DISCRIMINATOR_REG)
    
def supervised_pretraining(G, data, conds, inits, envs_list, rho_threshold=RHO_THRESHOLD):
    """
    Use samples from the teacher model to pretrain the diffusion
    model of the generator.
    """
    print("Pretraining the Generator...")
    # cond1 = torch.load("./diffusion_samples/model-14-milestone-final-50000-cond-1.pt")
    # cond2 = torch.load("./diffusion_samples/model-14-milestone-final-50000-cond-2.pt")
    # cond3 = torch.load("./diffusion_samples/model-14-milestone-final-50000-cond-3.pt")
    # cond4 = torch.load("./diffusion_samples/model-14-milestone-final-50000-cond-4.pt")
    # cond5 = torch.load("./diffusion_samples/model-14-milestone-final-50000-cond-5.pt")
    # cond = torch.cat((cond1, cond2, cond3, cond4, cond5), 0).to(float)
    G.pretrain(data.float().to('cuda'),
               conds.float().to('cuda'),
               inits.float().to('cuda'))

    print("------EVALUATE CURRENT GENERATOR------")
    print("Taking 1000 samples from the pretrained generator...")
    rho_sample = torch.full((1000, ), rho_threshold).to('cuda')

    # inits = torch.empty((1000, 5), device='cuda')
    inits = torch.empty((1000, 4), device='cuda')
    for i in range(1000):
        env = envs_list[i]
        road = env.unwrapped.road
        intruder_vehicle = road.vehicles[0]
        ego_vehicle = road.vehicles[1]
        inits[i, :2] = torch.from_numpy(intruder_vehicle.position - ego_vehicle.position).to(inits.device)
        inits[i, 2 : 4] = torch.from_numpy(intruder_vehicle.velocity - ego_vehicle.velocity).to(inits.device)
        # inits[i, 4] = float(ego_vehicle.route[2][1][-1])

    samples = G.sample(rho_sample, inits, no_grad=True).clone().detach().cpu()
    print("DISTURBANCE SCALE OF CURRENT BATCH: ", torch.mean(torch.abs(samples).to(torch.float)))
    print("DISTURBANCE MEAN: ", torch.mean(samples.to(torch.float)))
    print("DISTURBANCE STD: ", torch.std(samples.to(torch.float)))

    # Evaluate robustness of each sample
    print("Evaluating robustness of samples drawn...")
    sample_trajectories = [samples[i] for i in range(samples.shape[0])]
    simulations = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks with their indices
        futures = {executor.submit(evaluate, sample_trajectories[i].clone().detach().numpy(), envs_list[i]): i for i in range(len(sample_trajectories))}

        # Collect results in order of completion, storing the index
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx = futures[future]
            results.append((idx, future.result()))

        # Sort results by index and extract the values
        results.sort(key=lambda x: x[0])
        simulations = [result for _, result in results]

    mean_robustness = np.mean(simulations)

    print("MEAN ROBUSTNESS OF CURRENT BATCH: ", mean_robustness)

    print("Saving Distillation Model...")
    G.save(SAVE_PATH)
    print("Model saved.")
        

def difs_pretraining(G):
    """
    Use samples from the teacher model to pretrain the diffusion
    model of the generator.
    """
    print("Pretraining the Generator...")
    G.train()

def compute_weight_change(old_state_dict, new_state_dict):
    result = 0.0
    for i in range(len(old_state_dict)):
        old_param = old_state_dict[i].clone().cpu()
        new_param = new_state_dict[i].clone().cpu()
        result += torch.sum(torch.abs(new_param - old_param)).item()
    return result

def distill(envs_list, D, G, T, D_solver, G_solver, discriminator_loss, generator_loss, 
            real_data, real_inits, mc_risks, mc_disturbances, mc_trajectories,
            dis_iters=40, gen_iters=40, dis_warmup=40, show_every=CHECK_EVERY, 
            dis_batch_size=DIS_BATCH_SIZE, gen_batch_size=GEN_BATCH_SIZE,
            num_iters=NUM_ITERS, rho_threshold=RHO_THRESHOLD,
            lamb=LAMB, teacher_reconstructions=1):
    """
    Train a GAN to distill the diffusion model!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - num_iters: Number of training iterations.
    - rho_threshold: Robustness threshold for the distillation model
    """
    dis_rho_sample = torch.full((dis_batch_size, ), rho_threshold).unsqueeze(1).repeat(1, 24).to('cuda')
    gen_rho_sample = torch.full((gen_batch_size, ), rho_threshold).unsqueeze(1).repeat(1, 24).to('cuda')
    teacher_betas = T.model.betas
    teacher_alphas = 1 - teacher_betas

    best_mean_robustness = float('inf')
    
    for iter in range(num_iters):
        d_total_error = None
        discriminator_iterations = dis_warmup if iter == 0 else dis_iters
        print("Training discriminator...")
        for dis_iter in range(discriminator_iterations):
            # Compute logits for a batch of real data
            D_solver.zero_grad()
            current_batch = random.sample(range(len(real_data)), dis_batch_size)
            real_data_sample = real_data[current_batch].float().to('cuda')
            real_inits_sample = real_inits[current_batch].float().to('cuda')
            logits_real = D(real_data_sample, real_inits_sample)

            # Use student model to diffuse real data
            real_data_diffused = G.model.diffuse(real_data_sample, TIMESTEPS, no_grad=True)

            # Use student model to reconstruct and compute logits
            fake_data = G.model.inference(cond=dis_rho_sample,
                                          inits=real_inits_sample,
                                          starting_timestep=TIMESTEPS,
                                          starting_data=real_data_diffused,
                                          no_grad=True)
            logits_fake = D(fake_data, real_inits_sample)

            # Train discriminator
            d_total_error = discriminator_loss(logits_real=logits_real, logits_fake=logits_fake)
            d_total_error.backward()
            D_solver.step()

            true_positive_rate = np.mean((torch.sigmoid(logits_real) > 0.5).cpu().numpy().astype(int))
            false_positive_rate = np.mean((torch.sigmoid(logits_fake) > 0.5).cpu().numpy().astype(int))

            print("\n")
            print(true_positive_rate, " of real samples pass the discriminator.")
            print(false_positive_rate, " of generated samples fool the discriminator.")
            print("\n")

            wandb.log({'discriminator true positive rate': true_positive_rate,
                       'discriminator false positive rate': false_positive_rate,
                       'discriminator loss': d_total_error})


        # Generate another batch of fake data and train generator
        g_error = None
        print("Training generator...")
        for gen_iter in range(gen_iters):
            D_solver.zero_grad()
            G_solver.zero_grad()

            # Draw some samples from the dataset
            current_batch = random.sample(range(len(real_data)), gen_batch_size)
            real_data_sample = real_data[current_batch].float().to('cuda')
            real_inits_sample = real_inits[current_batch].float().to('cuda')
            # Let the generator reconstruct
            real_data_diffused = G.model.diffuse(real_data_sample, TIMESTEPS, no_grad=False)

            fake_data = G.model.inference(cond=gen_rho_sample,
                                          inits=real_inits_sample, 
                                          starting_timestep=TIMESTEPS,
                                          starting_data=real_data_diffused,
                                          no_grad=False)
            # Calculate discriminator logits for reconstructed data
            gen_logits_fake = D(fake_data, real_inits_sample)

            if lamb >= 0.01:
                # Use teacher model to reconstruct generator outputs
                reconstruction_timesteps = random.sample(range(1, 1001), teacher_reconstructions)
                B, C, L = fake_data.shape
                fake_data_reconstructed = torch.empty(B, teacher_reconstructions, C, L).to('cuda')  # (N, Nt, 4, 24)

                for i in range(teacher_reconstructions):
                    fake_data_diffused = T.model.diffuse(fake_data, reconstruction_timesteps[i], no_grad=True)
                    current_reconstruction = T.model.inference(cond=gen_rho_sample, inits=real_inits_sample,
                                                            starting_timestep=reconstruction_timesteps[i],
                                                            starting_data=fake_data_diffused, no_grad=True)
                    fake_data_reconstructed[:, i, :, :] = current_reconstruction

                reconstruction_weights = torch.tensor([teacher_alphas[timestep - 1] for timestep in reconstruction_timesteps]).to('cuda')
                
                # Calculate GAN-distillation hybrid loss
                g_error = generator_loss(gen_logits_fake, fake_data, fake_data_reconstructed, reconstruction_weights, lamb=lamb)
            else:
                g_error = generator_gan_loss(gen_logits_fake)


            g_error.backward()
            G_solver.step()

            disturbance_scale = torch.mean(torch.abs(fake_data))
            disturbance_mean = torch.mean(fake_data)
            disturbance_std = torch.std(fake_data)
            print("DISTURBANCE SCALE: ", disturbance_scale)
            print("DISTURBANCE MEAN: ", disturbance_mean)
            print("DISTURBANCE STD: ", disturbance_std)

            success_rate = np.mean((torch.sigmoid(gen_logits_fake) > 0.5).cpu().numpy().astype(int))
            wandb.log({'disturbance scale': disturbance_scale,
                       'disturbance mean': disturbance_mean,
                       'disturbance std': disturbance_std,
                       'discriminator false positive rate': success_rate,
                       'generator loss': g_error})
            print("\n")
            print(success_rate, " of generated samples fool the discriminator.")
            print("\n")

        
        if (iter % show_every == 0):
            print("------EVALUATE CURRENT GENERATOR------")
            print("Taking samples...")
            rhos_for_eval = torch.full((1000,), rho_threshold)
            print("Reinitializing environments & Getting initial states...")
            # inits = torch.empty((1000, 5)).to('cuda')
            inits = torch.empty((1000, 4)).to('cuda')
            for i in range(1000):
                env = envs_list[i]
                env.reset(seed=(RANDOM_SEED + i))
                road = env.unwrapped.road
                intruder_vehicle = road.vehicles[0]
                ego_vehicle = road.vehicles[1]
                inits[i, :2] = torch.from_numpy(intruder_vehicle.position - ego_vehicle.position).to(inits.device)
                inits[i, 2 : 4] = torch.from_numpy(intruder_vehicle.velocity - ego_vehicle.velocity).to(inits.device)
                # inits[i, 4] = float(ego_vehicle.route[2][1][-1])

            fake_data = G.sample(rhos_for_eval, inits, no_grad=True).cpu()
            fake_data = fake_data.detach()
            
            print("Simulating on samples...")

            sample_trajectories = [fake_data[i] for i in range(1000)]
            simulations = []

            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Submit tasks with their indices
                futures = {executor.submit(evaluate, sample_trajectories[i].clone().detach().numpy(), envs_list[i]): i for i in range(1000)}

                # Collect results in order of completion, storing the index
                results = []
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    idx = futures[future]
                    results.append((idx, future.result()))

                # Sort results by index and extract the values
                results.sort(key=lambda x: x[0])
                simulations = np.array([result for _, result in results])

            mean_robustness = np.mean(simulations)
            fraction_equal_zero = np.mean((simulations == 0.0).astype(int))
            fraction_less_than_one = np.mean((simulations <= 1.0).astype(int))
            fraction_less_than_five = np.mean((simulations <= 5.0).astype(int))

            wandb.log({"mean robustness": mean_robustness,
                       "fraction with zero rho": fraction_equal_zero,
                       "fraction with <= 1 rho": fraction_less_than_one,
                       "fraction with <= 5 rho": fraction_less_than_five})

            if mean_robustness <= best_mean_robustness:
                best_mean_robustness = mean_robustness
                print("Saving Distillation Model...")
                G.save(SAVE_PATH)
                print("Model saved.")

            # sample_trajectories = [fake_data[i] for i in range(1000)]
            # risks = []
            # observations = []
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     futures = [executor.submit(sim_risk, sample_traj) for sample_traj in sample_trajectories]
            #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            #         current_risk, current_obs = future.result()
            #         assert current_risk <= 0
            #         risks.append(current_risk)
            #         observations.append(torch.from_numpy(current_obs).unsqueeze(0))

            # risks = torch.tensor(risks)
            # observations = torch.cat(observations, dim=0)

            # failure_idx = torch.where(risks>=-0.06)[0]
            # difs_trajectories = observations[failure_idx].reshape(-1,96*2)
            # no_mc = mc_trajectories.shape[0]
            # no_difs = difs_trajectories.shape[0]
            # idx = np.arange(no_difs)
            # np.random.shuffle(idx)
            # failure_rate = failure_idx.shape[0]/risks.shape[0]


            # try:
            #     metrics = compute_prdc(real_features=mc_trajectories,fake_features=difs_trajectories[idx[:no_mc]],nearest_k=5)
            #     print("failure rate: ", failure_rate)
            #     print("density: ", metrics['density'])
            #     print("coverage: ", metrics['coverage'])
            #     wandb.log({"failure rate": failure_rate,
            #                "density": metrics['density'],
            #                "coverage": metrics['coverage']})
            #     if (metrics['coverage'] >= 0.65) or (metrics['coverage'] >= 0.61 and failure_rate >= 0.8):
            #         save_path = "./Distillation_Models/F-model-14-distilled-" + str(TIMESTEPS) + "-steps-iter-" + str(iter) + ".pt"
            #         print("Saving Distillation Model...")
            #         G.save(save_path)
            #         print("Model saved.")
            # except:
            #     wandb.log({'failure rate': failure_rate})



def main(argv):
    mode = argv[0]
    # create traffic intersection environments
    # these are purely for evaluation purposes
    envs_list = []
    print("Initializing simulation environments...")
    num_environments = 40000 if mode == "difs_pretraining" else 1000
    for i in tqdm(range(num_environments)):
        env = gym.make("intersection-v0",render_mode='rgb_array')
        env.unwrapped.config.update({"initial_vehicle_count":1})
        env.unwrapped.config.update({"spawn_probability":0.0})
        env.unwrapped.configure({
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": False,
            }})
        env.reset(seed=(RANDOM_SEED + i))
        envs_list.append(env)


    # Shape of samples is: (N, 4, 24)
    G = generator(mode=mode, envs_list=envs_list)


    if mode == "difs_pretraining":
        difs_pretraining(G)
        return

    samples = torch.from_numpy(torch.load(TEACHER_SAMPLES))
    conds = torch.from_numpy(torch.load(TEACHER_CONDS)).unsqueeze(1).repeat(1, 24)
    inits = torch.load(TEACHER_INITS)
    if mode == "supervised_pretraining":
        supervised_pretraining(G, samples, conds, inits, envs_list=envs_list)
        return


    D = discriminator()
    T = teacher(envs_list=envs_list)
    D_solver = get_optimizer(D)
    G_solver = get_optimizer(G)
    # mc_risks = torch.Tensor(torch.load("./monte_carlo_failure_samples/config_1/risks.pt"))
    # mc_disturbances = torch.stack(torch.load("./monte_carlo_failure_samples/config_1/failure_disturbances.pt")).reshape(-1,96)
    # mc_trajectories = torch.stack([torch.Tensor(a) for a in torch.load("./monte_carlo_failure_samples/config_1/failure_trajectories.pt")]).reshape(-1,96*2)


    if len(argv) == 1:
        distill(envs_list=envs_list, D=D, G=G, T=T, D_solver=D_solver, G_solver=G_solver, 
                discriminator_loss=discriminator_loss, 
                generator_loss=generator_loss, real_data=samples, real_inits=inits,
                mc_risks=None, mc_disturbances=None, mc_trajectories=None)
    else:
        lamb = float(argv[1])
        distill(envs_list=envs_list, D=D, G=G, T=T, D_solver=D_solver, G_solver=G_solver, 
                discriminator_loss=discriminator_loss, 
                generator_loss=generator_loss, real_data=samples, real_inits=inits,
                mc_risks=None, mc_disturbances=None, mc_trajectories=None, lamb = lamb)



if __name__ == "__main__":
    main(sys.argv[1:])