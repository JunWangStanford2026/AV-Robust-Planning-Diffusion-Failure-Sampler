RUN_SERIAL = 41
RANDOM_SEED = 1

print("importing packages...")
import torch
from torch import nn
from torch.optim import Adam, AdamW
import random
import numpy as np

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
random.seed(RANDOM_SEED)                  
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from difs import FullyConditionedUnet, GaussianDiffusionConditional, DiFS
import gymnasium as gym
from copy import deepcopy as copy
from multiprocessing import cpu_count
from accelerate import Accelerator, DataLoaderConfiguration
from ema_pytorch import EMA
import concurrent.futures
from tqdm import tqdm
import sys
import wandb
from numpy.linalg import norm
import wandb

wandb.login(key="0008e2d125e8e34c1d04105171081d4e8584ad93")

TIMESTEPS = 1000    # diffusion sampling timesteps
N = 40000
horizon = 24
xdim = 4
px_variance = 1.0
ALPHA = 0.9
TRAIN_NUM_STEPS = 40000  # number of diffusion training steps per iteration
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 3e-4
SCENARIO = "EAST"
DIM = 64
DIM_MULTS = (2, 4, 8, 16)
USE_CFG = False

print("Generating initial disturbances...")
data = px_variance * torch.randn(N,xdim,horizon)

envs_list = []
print("Initializing simulation environments...")
for i in tqdm(range(N)):
    env = gym.make("intersection-v0",render_mode='rgb_array')
    env.unwrapped.config.update({"initial_vehicle_count":1})
    env.unwrapped.config.update({"spawn_probability":0.0})
    env.unwrapped.configure({
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": False,
        }})

    env.reset(seed=RANDOM_SEED + i)
    envs_list.append(env)



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

print("Making model...")
model = FullyConditionedUnet(
        dim = DIM,
        # dim_mults = (1, 2, 4),
        dim_mults = DIM_MULTS,
        channels = 4,
        cond_dim = 24,
).to('cuda')

diffusion = GaussianDiffusionConditional(
    model,
    seq_length=horizon,
    classifier_free_guidance=USE_CFG,
    timesteps=TIMESTEPS
).to('cuda')

sampler = DiFS(
    RANDOM_SEED,
    diffusion,
    evaluate_fn=sim_risk,
    init_disturbances=data,
    envs_list=envs_list,
    run_serial=RUN_SERIAL,
    alpha=ALPHA,
    N=N,
    train_num_steps=TRAIN_NUM_STEPS,
    train_batch_size=TRAIN_BATCH_SIZE,
    train_lr=TRAIN_LR,
    use_wandb=True,
    save_intermediate=True
)

print("Logging hyperparameter choice to wandb...")
config = {
    "alpha": ALPHA,
    "dim": DIM,
    "dim_mults": DIM_MULTS,
    "timesteps": TIMESTEPS,
    "N": N,
    "train_num_steps": TRAIN_NUM_STEPS,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "train_lr": TRAIN_LR,
    "scenario": SCENARIO,
    "classifier_free_guidance": USE_CFG,
    "random_seed": RANDOM_SEED
}

wandb.init(entity="distillation_difs", 
           project="GAN",
           config=config,
           )

print("TRAINING STARTS!")

sampler.train()
