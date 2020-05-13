import torch

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

    def __init__(self, use_cuda=False):
        self._device = "cpu" if not use_cuda else "cuda:0"

        ndim, dt = 4, 1
        #  Create Kalman filter model matrices (8, 8)
        self._motion_mat = torch.eye(2 * ndim, 2 * ndim, device=self._device)
        for i in range(ndim):
            self._motion_mat[i, ndim + 1] = dt

        # (4, 8)
        self._update_mat = torch.eye(ndim, 2 * ndim, device=self._device)

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
        mean_pos = measurement.to(self._device)
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel], dim=-1).view(-1, 1)  # (8, 1)

        std = torch.tensor([
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]],
            device=mean_pos.device)

        covariance = torch.diag(torch.pow(std, 2))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of shape (8, 1) of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = torch.tensor([
            self._std_weight_position * mean[3, 0],
            self._std_weight_position * mean[3, 0],
            1e-2,
            self._std_weight_position * mean[3, 0]], device=mean.device)
        std_vel = torch.tensor([
            self._std_weight_velocity * mean[3, 0],
            self._std_weight_velocity * mean[3, 0],
            1e-5,
            self._std_weight_velocity * mean[3, 0]], device=mean.device)

        # (8, 8)
        motion_cov = torch.diag(torch.pow(torch.cat([std_pos, std_vel], dim=-1), 2))

        mean = torch.mm(self._motion_mat, mean)  # (8, 1)
        covariance = torch.chain_matmul(self._motion_mat, covariance, self._motion_mat.t()) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = torch.tensor([
            self._std_weight_position * mean[3, 0],
            self._std_weight_position * mean[3, 0],
            1e-1,
            self._std_weight_position * mean[3, 0]], device=mean.device)

        # (4, 4)
        innovation_cov = torch.diag(torch.pow(std, 2))

        mean = torch.mm(self._update_mat, mean)  # (4, 1)
        covariance = torch.chain_matmul(self._update_mat, covariance, self._update_mat.t())  # (8, 8)
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
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
        chol_factor = torch.cholesky(projected_cov, upper=False)
        kalman_gain = torch.cholesky_solve(torch.mm(covariance, self._update_mat.t()).t(),
                                           chol_factor,
                                           upper=False).t()

        innovation = measurement.view(1, -1) - projected_mean.view(1, -1)

        new_mean = mean + torch.mm(innovation, kalman_gain.t()).t()  # (8, 1)
        new_covariance = covariance - torch.chain_matmul(kalman_gain, projected_cov, kalman_gain.t())  # (8, 8)

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2, :], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = torch.cholesky(covariance)
        d = measurements - mean.t()
        z = torch.triangular_solve(d.t(), cholesky_factor, upper=False)[0]
        squared_maha = torch.sum(z * z, dim=0)
        return squared_maha
