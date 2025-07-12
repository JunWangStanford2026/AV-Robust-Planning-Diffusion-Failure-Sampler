import numpy as np

class KalmanFilter():
    def __init__(self, init_obs, perturbation_scale=0.15):
        # init_obs must be of format [x, y, v_x, v_y]
        self.perturbation_scale = perturbation_scale
        # belief covariance
        self.cov = 3 * ((np.identity(4) / perturbation_scale) ** 2)
        # Transition matrix
        self.Ts = np.array([[1, 0, 0.5, 0],
                            [0, 1, 0, 0.5]])
        # the only source of transitional uncertainty
        # is the uncertainty in the observation of the
        # current velocity
        self.Es = ((np.identity(2) / perturbation_scale) ** 2) / 4
        # observation covariance
        self.Eo = (np.identity(4) / perturbation_scale) ** 2

        # belief mean
        self.mean = init_obs.copy()
            

    def update(self, obs):
        intruder_position = obs[:2].copy()
        intruder_velocity = obs[2:].copy()
        o = obs.copy()

        # predict step
        predicted_mean = np.empty((4,))
        predicted_mean[2:] = intruder_velocity
        predicted_mean[:2] = self.mean[:2] + 0.5 * (self.mean[2:] + intruder_velocity)
        predicted_cov = (np.identity(4) / self.perturbation_scale) ** 2
        predicted_cov[:2, :2] = (self.Ts @ self.cov @ (self.Ts.T)) + self.Es

        # update step
        kalman_gain = predicted_cov @ np.linalg.inv(predicted_cov + self.Eo)
        self.mean = predicted_mean + kalman_gain @ (o - predicted_mean)
        self.cov = (np.identity(4) - kalman_gain) @ predicted_cov

            

    def sample(self, num_samples):
        samples = np.random.multivariate_normal(self.mean, self.cov, num_samples)
        return [samples[i] for i in range(samples.shape[0])]
            


    