import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys

class DiffusionBasedPlanner(object):
    TARGET_POSITION_TABLE = {1: (-0.1, -0.02),  2: (0.02, -0.1),  3: (0.1, 0.02)}
    VX_LOWER_TABLE =        {1: -0.501,         2: -0.0,          3: -0.001}
    VX_UPPER_TABLE =        {1: 0.001,          2: 0.0,           3: 0.501}
    AX_LOWER_TABLE =        {1: -0.1,           2: 0.0,           3: -0.2}
    AX_UPPER_TABLE =        {1: 0.2,            2: 0.0,           3: 0.1}
    MEAN_VX_LIMIT_TABLE =   {1: (-0.4, 0.0),    2: (0.0, 0.0),    3: (0.0, 0.4)}

    def __init__(self, ego_destination, used_samples_per_replan):
        self.USED_SAMPLES_PER_REPLAN = used_samples_per_replan
        self.EGO_DESTINATION = ego_destination
        self.TARGET_POSITION = [100 * coord for coord in self.TARGET_POSITION_TABLE[ego_destination]]
        self.VX_LOWER = self.VX_LOWER_TABLE[ego_destination] * 20
        self.VX_UPPER = self.VX_UPPER_TABLE[ego_destination] * 20
        self.VY_LOWER = -0.501 * 20
        self.VY_UPPER = 0.001 * 20
        self.AX_LOWER = self.AX_LOWER_TABLE[ego_destination] * 20
        self.AX_UPPER = self.AX_UPPER_TABLE[ego_destination] * 20
        self.AY_LOWER = -0.2 * 20
        self.AY_UPPER = 1 * 20
        self.MEAN_VX_LIMIT = [20 * component for component in self.MEAN_VX_LIMIT_TABLE[ego_destination]]
        self.MEAN_VY_LIMIT = [-0.3 * 20, 0.0 * 20]
        self.MANHATTAN_DISTANCE_THRESHOLD = 0.085 * 100

    def fit(self, init_pos, init_vel, intruder_trajectories, horizon):
        # print("TARGET_POSITION = ", self.TARGET_POSITION)
        # print("init_pos = ", init_pos)
        # print("init_vel = ", init_vel)
        # print("Sample intruder trajectory = ", intruder_trajectories[0])


        opt_model = gp.Model()

        # Variables for x positions and velocities
        x = np.array([[opt_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"x_{t}") for t in range(horizon)],
                    [opt_model.addVar(vtype=GRB.CONTINUOUS, lb=self.VX_LOWER, ub=self.VX_UPPER, name=f"vx_{t}") for t in range(horizon)]])  # x[0, j] = x_ego_j; x[1, j] = vx_ego_j
        # Variables for y positions and velocities
        y = np.array([[opt_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"y_{t}") for t in range(horizon)],
                    [opt_model.addVar(vtype=GRB.CONTINUOUS, lb=self.VY_LOWER, ub=self.VY_UPPER, name=f"vy_{t}") for t in range(horizon)]])  # y[0, j] = y_ego_j; y[1, j] = vy_ego_j
        # Variable for Manhattan distance between ego and intruder
        manhattan_distances = np.array([[opt_model.addVar(vtype=GRB.CONTINUOUS, name=f"manhattan_dis_traj_{traj}_step_{t}") for t in range(horizon)] for traj in range(self.USED_SAMPLES_PER_REPLAN)])

        # Constraints for initial states
        opt_model.addConstr(x[0, 0] == init_pos[0], name="init_x_pos_constraint")
        opt_model.addConstr(y[0, 0] == init_pos[1], name="init_y_pos_constraint")
        opt_model.addConstr(x[1, 0] == init_vel[0], name="init_x_vel_constraint")  
        opt_model.addConstr(y[1, 0] == init_vel[1], name="init_y_vel_constraint")


        # Constraints for reaching destination
        if self.EGO_DESTINATION == 1:
            opt_model.addConstr(x[0, horizon - 1] <= self.TARGET_POSITION[0], name="target_x_pos_constraint")
            opt_model.addConstr(y[0, horizon - 1] <= self.TARGET_POSITION[1] + 1, name="target_y_pos_constraint1")
            opt_model.addConstr(y[0, horizon - 1] >= self.TARGET_POSITION[1] - 1, name="target_y_pos_constraint2")
        elif self.EGO_DESTINATION == 2:
            opt_model.addConstr(x[0, horizon - 1] == self.TARGET_POSITION[0], name="target_x_pos_constraint")
            opt_model.addConstr(y[0, horizon - 1] <= self.TARGET_POSITION[1], name="target_y_pos_constraint")
        elif self.EGO_DESTINATION == 3:
            opt_model.addConstr(x[0, horizon - 1] >= self.TARGET_POSITION[0], name="target_x_pos_constraint")
            opt_model.addConstr(y[0, horizon - 1] <= self.TARGET_POSITION[1] + 1, name="target_y_pos_constraint1")
            opt_model.addConstr(y[0, horizon - 1] >= self.TARGET_POSITION[1] - 1, name="target_y_pos_constraint2")
        else:
            assert False

        M = 1000        # Big-M used to set conditional lanekeeping constraints

        for timestep in range(horizon):
            # Lanekeeping Constraints
            if self.EGO_DESTINATION == 1:
                y_is_positive = opt_model.addVar(vtype=GRB.BINARY, name=f"1[y>=0], step {timestep}")
                opt_model.addConstr(y[0, timestep] >= -M * (1 - y_is_positive), name=f"south branch indicator, step {timestep}")
                opt_model.addConstr(y[0, timestep] <= M * y_is_positive, name=f"south branch indicator, step {timestep}")
                opt_model.addGenConstrIndicator(y_is_positive, 1, x[0, timestep] >= 1, name=f"x lanekeeping1, step {timestep}")
                opt_model.addGenConstrIndicator(y_is_positive, 1, x[0, timestep] <= 3, name=f"x lanekeeping2, step {timestep}")

                x_is_negative = opt_model.addVar(vtype=GRB.BINARY, name=f"1[x<=0], step {timestep}")
                opt_model.addConstr(x[0, timestep] <= M * (1 - x_is_negative), name=f"west branch indicator, step {timestep}")
                opt_model.addConstr(x[0, timestep] >= -M * x_is_negative, name=f"west branch indicator, step {timestep}")
                opt_model.addGenConstrIndicator(x_is_negative, 1, y[0, timestep] >= -3, name=f"y lanekeeping1, step {timestep}")
                opt_model.addGenConstrIndicator(x_is_negative, 1, y[0, timestep] <= -1, name=f"y lanekeeping2, step {timestep}")
            
            elif self.EGO_DESTINATION == 2:
                opt_model.addConstr(x[0, timestep] >= 1, name=f"x lanekeeping1, step {timestep}")
                opt_model.addConstr(x[0, timestep] <= 3, name=f"x lanekeeping2, step {timestep}")
            
            elif self.EGO_DESTINATION == 3:
                in_south_branch = opt_model.addVar(vtype=GRB.BINARY, name=f"1[y>=4], step {timestep}")
                opt_model.addConstr(y[0, timestep] >= 4 - (M * (1 - in_south_branch)), name=f"south branch indicator, step {timestep}")
                opt_model.addConstr(y[0, timestep] <= 4 + M * in_south_branch, name=f"south branch indicator, step {timestep}")
                opt_model.addGenConstrIndicator(in_south_branch, 1, x[0, timestep] >= 1, name=f"x lanekeeping1, step {timestep}")
                opt_model.addGenConstrIndicator(in_south_branch, 1, x[0, timestep] <= 3, name=f"x lanekeeping2, step {timestep}")

                in_east_branch = opt_model.addVar(vtype=GRB.BINARY, name=f"1[x>=4], step {timestep}")
                opt_model.addConstr(x[0, timestep] >= 4 - (M * (1 - in_east_branch)), name=f"east branch indicator, step {timestep}")
                opt_model.addConstr(x[0, timestep] <= 4 + M * in_east_branch, name=f"east branch indicator, step {timestep}")
                opt_model.addGenConstrIndicator(in_east_branch, 1, y[0, timestep] >= 1, name=f"y lanekeeping1, step {timestep}")
                opt_model.addGenConstrIndicator(in_east_branch, 1, y[0, timestep] <= 3, name=f"y lanekeeping2, step {timestep}")

            else:
                assert False

            # Kinematics Constraints
            if timestep < horizon - 1:
                # Kinematics constraints for x direction
                opt_model.addConstr(x[0, timestep + 1] == x[0, timestep] + 0.5 * (x[1, timestep + 1] + x[1, timestep]), name=f"step_{timestep}_x_update_constraint")
                opt_model.addConstr(x[1, timestep + 1] >= x[1, timestep] + 0.25 * self.AY_LOWER, name=f"step_{timestep}_x_deceleration_constraint")
                opt_model.addConstr(x[1, timestep + 1] <= x[1, timestep] + 0.25 * self.AY_UPPER, name=f"step_{timestep}_x_acceleration_constraint")
                    
                # Kinematics constraints for y direction
                opt_model.addConstr(y[0, timestep + 1] == y[0, timestep] + 0.5 * (y[1, timestep + 1] + y[1, timestep]), name=f"step_{timestep}_y_update_constraint")
                opt_model.addConstr(y[1, timestep + 1] >= y[1, timestep] + 0.25 * self.AY_LOWER, name=f"step_{timestep}_y_deceleration_constraint")
                opt_model.addConstr(y[1, timestep + 1] <= y[1, timestep] + 0.25 * self.AY_UPPER, name=f"step_{timestep}_y_acceleration_constraint")
                
                
                # # This is a convenience constraint: we do not want zero acceleration values cause that causes numerical instability in the simulator
                # x_velocity_change = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="dvx_step_{timestep}")
                # y_velocity_change = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="dvy_step_{timestep}")
                # model.addConstr(x_velocity_change == x[1, timestep + 1] - x[1, timestep], name="calculate_dvx_step_{timestep}")
                # model.addConstr(y_velocity_change == y[1, timestep + 1] - y[1, timestep], name="calculate_dvy_step_{timestep}")
                # x_velocity_change_abs = model.addVar(vtype=GRB.CONTINUOUS, name="|dvx|_step_{timestep}")
                # y_velocity_change_abs = model.addVar(vtype=GRB.CONTINUOUS, name="|dvy|_step_{timestep}")
                # model.addConstr(x_velocity_change_abs == gp.abs_(x_velocity_change), name="calculate_|dvx|_step_{timestep}")
                # model.addConstr(y_velocity_change_abs == gp.abs_(y_velocity_change), name="calculate_|dvy|_step_{timestep}")
                # model.addConstr(x_velocity_change_abs >= 1e-8, name="avoid_zero_ax_step_{timestep}")
                # model.addConstr(x_velocity_change_abs >= 1e-8, name="avoid_zero_ay_step_{timestep}")

                
            for trajectory in range(self.USED_SAMPLES_PER_REPLAN):
                # Filter out trash data
                if abs(intruder_trajectories[trajectory][0, timestep]) + abs(intruder_trajectories[trajectory][1, timestep]) >= 1e-5:
                    # Horizontal distance to possible intruder positions
                    horizontal_displacement = opt_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"step_{timestep}_traj_{trajectory}_dx")
                    horizontal_distance = opt_model.addVar(vtype=GRB.CONTINUOUS, name=f"step_{timestep}_traj_{trajectory}_|dx|")
                    opt_model.addConstr(horizontal_displacement == intruder_trajectories[trajectory][0, timestep] - x[0, timestep], name=f"step_{timestep}_traj_{trajectory}_set_dx")
                    opt_model.addConstr(horizontal_distance == gp.abs_(horizontal_displacement), name=f"step_{timestep}_traj_{trajectory}_set_|dx|")

                    # Vertical distance to possible intruder positions
                    vertical_displacement = opt_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"step_{timestep}_traj_{trajectory}_dy")
                    vertical_distance = opt_model.addVar(vtype=GRB.CONTINUOUS, name=f"step_{timestep}_traj_{trajectory}_|dy|")
                    opt_model.addConstr(vertical_displacement == intruder_trajectories[trajectory][1, timestep] - y[0, timestep], name=f"step_{timestep}_traj_{trajectory}_set_dy")
                    opt_model.addConstr(vertical_distance == gp.abs_(vertical_displacement), name=f"step_{timestep}_traj_{trajectory}_set_|dy|")
                    
                    opt_model.addConstr(manhattan_distances[trajectory, timestep] == vertical_distance + horizontal_distance, name=f"step_{timestep}_traj_{trajectory}_compute_manhattan_distance")

        opt_model.addConstr(np.sum(x[1, :]) <= horizon * self.MEAN_VX_LIMIT[1], name="Mean X-Velocity Upper Limit")
        opt_model.addConstr(np.sum(x[1, :]) >= horizon * self.MEAN_VX_LIMIT[0], name="Mean X-Velocity Lower Limit")

        opt_model.addConstr(np.sum(y[1, :]) <= horizon * self.MEAN_VY_LIMIT[1], name="Mean Y-Velocity Upper Limit")
        opt_model.addConstr(np.sum(y[1, :]) >= horizon * self.MEAN_VY_LIMIT[0], name="Mean Y-Velocity Lower Limit")

        minimum_distance = opt_model.addVar(vtype=GRB.CONTINUOUS, name="minimum manhattan distance")
        opt_model.addConstr(minimum_distance == gp.min_(manhattan_distances.flatten().tolist()))

        opt_model.setObjective(minimum_distance, GRB.MAXIMIZE)

        try:
            # Optimize the model
            opt_model.setParam(GRB.Param.InfUnbdInfo, 1)
            opt_model.optimize()

            # If the model is infeasible or unbounded
            if opt_model.status == GRB.INFEASIBLE:
                print("INFEASIBLE MODEL!!!")
            if opt_model.status == GRB.UNBOUNDED:
                print("UNBOUNDED MODEL!!!")
            if opt_model.status == GRB.INFEASIBLE or opt_model.status == GRB.UNBOUNDED or opt_model.status == GRB.INF_OR_UNBD:
                try:
                    # Compute the IIS
                    opt_model.computeIIS()
                    
                    # Output the IIS to a file
                    opt_model.write("model.ilp")
                    
                    print("IIS written to 'model.ilp'.")
                    print("The following constraints and variables are in the IIS:")
                    
                    # Print the names of the constraints and variables that are in the IIS
                    for constr in opt_model.getConstrs():
                        if constr.IISConstr:
                            print(f"Constraint: {constr.constrName}")
                    
                    for var in opt_model.getVars():
                        if var.IISLB > 0 or var.IISUB > 0:
                            print(f"Variable: {var.varName}")
                except:
                    print("Cannot compute IIS")

                # Iterate through all variables in the model
                free_variables = []
                for v in opt_model.getVars():
                    # Check if the variable is unbounded (free)
                    if v.lb == -GRB.INFINITY and v.ub == GRB.INFINITY:
                        free_variables.append(v.varName)

                # Print out the free variables
                if free_variables:
                    print("Free Variables:")
                    for var in free_variables:
                        print(var)
                else:
                    print("No free variables found.")
                        
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        # if opt_model.status == GRB.OPTIMAL:
        #     # Extract the solution values
        #     x = np.array([[x[i, j].X for j in range(x.shape[1])] for i in range(x.shape[0])])
        #     y = np.array([[y[i, j].X for j in range(y.shape[1])] for i in range(y.shape[0])])
        #     print("Optimal x trajectory planned: ", x)
        #     print("Optimal y trajectory planned: ", y)
        # else:
        #     sys.exit("No optimal solution found.")

        # Retrieve control inputs from optimized state trajectories
        x_velocities = np.array([x[1, t].X for t in range(horizon)]) / 20
        y_velocities = np.array([y[1, t].X for t in range(horizon)]) / 20
        speeds = np.sqrt((x_velocities ** 2) + (y_velocities ** 2))

        opt_model.dispose()  # Explicitly free the model resources
        opt_model = None  # Remove reference to the model

        return 4 * (speeds[1:] - speeds[:-1])