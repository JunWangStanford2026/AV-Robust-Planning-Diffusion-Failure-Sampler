import numpy as np
from copy import deepcopy as copy
import gymnasium as gym
import torch
import concurrent.futures
from tqdm import tqdm

# REMEMBER TO EDIT intersection.py !!!!!
SCENARIO = "SOUTH(AWS8)"
NUM_SIMULATIONS = 10000000
SAVE_PATH = "failure_trajectories_" + SCENARIO + ".pt"

def simulate(disturbances, perturbation_scale=0.15):
    # disturbances of shape (4, 24)
    # dx[0] , dx[1] , ..., dx[23]
    # dy[0] , dy[1] , ..., dy[23]
    # dvx[0], dvx[1], ..., dvx[23]
    # dvy[0], dvy[1], ..., dvy[23]

    # create simulation environment
    env = gym.make("intersection-v0",render_mode='rgb_array')
    env.unwrapped.config.update({"initial_vehicle_count":1})
    env.unwrapped.config.update({"spawn_probability":0.0})
    env.unwrapped.configure({
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": False,
        }})

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

    def simulate (disturbances):
        md_crit = 0.1
        tma_crit = 2.0
        origin_value = -1
        alpha = -origin_value/md_crit
        beta = -origin_value/tma_crit
        gamma = origin_value
        target_speed = 4.0
        max_acceleration = 5.0
        delta = 3.0

        obs = np.empty((24, 2, 4))
        obs[0] = env.reset()[0][:2, 1:5]

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
                    np.linalg.norm(ego_position) > np.linalg.norm(intruder_position)):
                    low_risk = True     # both going out, with intruder behind
                if (ego_lane_index[1][0] == 'i' and
                    np.linalg.norm(ego_position) < np.linalg.norm(intruder_position)):
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
                d = np.linalg.norm(ego_position - intruder_position)
                a -= max_acceleration * np.power(desired_gap(ego_vehicle, intruder_vehicle, velocity_disturbance) / not_zero(d), 2)
            else:
                a = 0.1 if current_speed < target_speed else 0.0

            obs[i] = env.step(np.array([a]))[0][:2, 1:5]

        return robustness, obs

    disturbances /= perturbation_scale

    robustness, obs = simulate(disturbances.cpu().numpy())

    if robustness == float('inf'):
        print("NAN ISSUES, THERE ARE VEHILCES", env.road.vehicles.__len__())

    return robustness, obs

failures = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(simulate, torch.randn(4, 24)) for i in range(NUM_SIMULATIONS)]

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        result = future.result()
        if result[0] == 0:
            # a failure is found!
            failures.append(result[1])

print("Number of failures found: ", len(failures))
torch.save(failures, SAVE_PATH)