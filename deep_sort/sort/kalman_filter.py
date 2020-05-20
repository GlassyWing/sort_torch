import tensorflow as tf
import numpy as np

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter:

    def __init__(self):

        ndim, dt = 4, 1
        #  Create Kalman filter model matrices (8, 8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._motion_mat = tf.constant(self._motion_mat, dtype=tf.float32)

        self._motion_mat = tf.transpose(self._motion_mat)

        # (8, 4)
        self._update_mat = tf.eye(2 * ndim, ndim, dtype=tf.float32)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h. shape of (4, 1)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement.numpy()
        mean_vel = tf.zeros_like(measurement)
        mean = tf.reshape(tf.concat([mean_pos, mean_vel], -1), (1, -1))  # (1, 8)

        std = tf.constant([
            [
                2 * self._std_weight_position * mean_pos[3],
                2 * self._std_weight_position * mean_pos[3],
                1e-2,
                2 * self._std_weight_position * mean_pos[3],
                10 * self._std_weight_velocity * mean_pos[3],
                10 * self._std_weight_velocity * mean_pos[3],
                1e-5,
                10 * self._std_weight_velocity * mean_pos[3]
            ]
        ], dtype=tf.float32)

        covariance = tf.linalg.diag(tf.pow(std, 2))  # (1, 8, 8)
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of shape (*, 8) of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step. which shape is (*, 8, 8)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """

        std_pos = tf.constant([[
            self._std_weight_position,
            self._std_weight_position,
            0.,
            self._std_weight_position]], dtype=tf.float32)
        std_pos_back = tf.constant([[
            0.,
            0.,
            1e-2,
            0.
        ]], dtype=tf.float32)
        std_vel = tf.constant([[
            self._std_weight_velocity,
            self._std_weight_velocity,
            0.,
            self._std_weight_velocity]], dtype=tf.float32)
        std_vel_back = tf.constant([[
            0.,
            0.,
            1e-5,
            0.
        ]], dtype=tf.float32)

        std_pos = mean[:, 3:4] * std_pos + std_pos_back  # (*, 4)
        std_vel = mean[:, 3:4] * std_vel + std_vel_back  # (*, 4)

        # (*, 8, 8)
        motion_cov = tf.linalg.diag(tf.pow(tf.concat([std_pos, std_vel], -1), 2))

        mean = tf.matmul(mean, self._motion_mat)  # (*, 8)

        # (*, 8, 8)
        covariance = tf.matmul(
            tf.transpose(tf.matmul(tf.transpose(covariance, (0, 2, 1)), self._motion_mat), (0, 2, 1)),
            self._motion_mat)
        return mean, covariance + motion_cov

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (*, 8).
        covariance : ndarray
            The state's covariance matrix (*, 8, 8).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """

        std = tf.constant([[
            self._std_weight_position,
            self._std_weight_position,
            0.,
            self._std_weight_position]], dtype=tf.float32)
        std_back = tf.constant([[0., 0., 1e-1, 0.]], dtype=tf.float32)

        std = mean[:, 3:4] * std + std_back  # (*, 4)

        # (*, 4, 4)
        innovation_cov = tf.linalg.diag(tf.pow(std, 2))

        # (*, 8) dot (8, 4)
        mean = tf.matmul(mean, self._update_mat)  # (*, 4)

        # (*, 4, 4)
        covariance = tf.matmul(
            tf.transpose(tf.matmul(tf.transpose(covariance, (0, 2, 1)), self._update_mat), (0, 2, 1)),
            self._update_mat)
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (*, 8).
        covariance : ndarray
            The state's covariance matrix (*,8,8).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """

        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor = tf.linalg.cholesky(projected_cov)

        # (*, 8, 4)
        kalman_gain = tf.transpose(
            tf.linalg.cholesky_solve(
                chol_factor,
                tf.transpose(tf.matmul(covariance, self._update_mat), (0, 2, 1))
            ),
            (0, 2, 1)
        )

        # (*, 4)
        innovation = tf.reshape(measurement, (-1, 4)) - projected_mean

        kalman_gain_t = tf.transpose(kalman_gain, (0, 2, 1))
        new_mean = mean + tf.reshape(tf.matmul(tf.expand_dims(innovation, 1), kalman_gain_t), (-1, 8))  # (*, 8)
        new_covariance = covariance - tf.matmul(
            tf.transpose(tf.matmul(tf.transpose(projected_cov, (0, 2, 1)), kalman_gain_t), (0, 2, 1)),
            kalman_gain_t)

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (n, 8).
        covariance : ndarray
            Covariance of the state distribution (n, 8, 8).
        measurements : ndarray
            An Mx4 dimensional matrix of M measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an matrix of shape N X M, where the i-th row contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[j]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:, None, :2], covariance[:, :2, :2]
            measurements = measurements[None, :, :2]
        else:
            mean = tf.expand_dims(mean, 1)
            measurements = tf.expand_dims(measurements, 0)

        # (n, 4, 4)
        cholesky_factor = tf.linalg.cholesky(covariance)
        d = - mean + measurements  # (n, m, 4)
        z = tf.linalg.triangular_solve(cholesky_factor, tf.transpose(d, (0, 2, 1)))
        # z = torch.cholesky_solve(d.permute(0, 2, 1),
        #                          cholesky_factor,
        #                          upper=False)  # (n, 4, m)
        squared_maha = tf.reduce_sum(z ** 2, axis=1)  # (n, m)
        return squared_maha


if __name__ == '__main__':
    kf = KalmanFilter()
    mean, covariance = kf.initiate(tf.constant([10, 15, 0.5, 10]))

    mean, covariance = kf.predict(mean, covariance)
    mean, covariance = kf.update(mean, covariance, tf.constant([12, 20, 0.6, 11]))

    mean_2, covariance_2 = kf.initiate(tf.constant([12, 13, 0.7, 5]))
    mean_2, covariance_2 = kf.predict(mean_2, covariance_2)
    mean_2, covariance_2 = kf.update(mean_2, covariance_2, tf.constant([13, 14, 0.7, 8]))

    squared_maha = kf.gating_distance(tf.concat((mean, mean_2), 0),
                                      tf.concat((covariance, covariance_2), 0),
                                      tf.constant([[12, 20, 0.6, 11],
                                                   [20, 16, 0.4, 18]]))
    print(squared_maha)
