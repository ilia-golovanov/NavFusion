import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import simulator

RAD2DEG = 180.0 / math.pi


def kalman_predict(x, P, F, Q, B=None, u=None):
    x = F.dot(x)

    if (B is not None) and (u is not None):
        x += B.dot(u)
    P = F.dot(P).dot(F.T) + Q


def kalman_update(x, P, inn, H, R):
    # z_p = H.dot(x)
    # y = z - H.dot(x)
    y = inn
    S = H.dot(P).dot(H.T) + R

    # L = scipy.linalg.cholesky(S, lower=True)
    Sinv = np.linalg.inv(S)

    K = P.dot(H.T).dot(Sinv)
    x_new = x + K.dot(y)

    IKH = np.identity(x.shape[1]) - K.dot(H)
    #P_new = IKH.dot(P).dot(IKH.T) + K.dot(R).dot(K.T)
    P_new2 = IKH.dot(P)

    return x_new, P_new2


# model: x - y - vel_x, vel_x, sin/cos bearing (3DOF)
# meas: acc_x, acc_y, gyro_z
# meas: x, y - noisy


class Filter:
    def __init__(self, data):
        self.R_gps = np.diag([1, 1])
        self.H_gps = np.zeros((2, 5))
        self.H_gps[0, 0] = 1.0
        self.H_gps[1, 1] = 1.0

        self.x = None
        self.P = None

        self.data = data

    def init(self):
        self.x = np.zeros((6, 1))
        self.x[0] = self.data['gps'][0, 0]
        self.x[1] = self.data['gps'][0, 1]
        self.x[2] = 10.0
        self.x[3] = 0.0
        self.x[4] = 0.0
        self.x[5] = 1.0

        self.P = np.diag([1, 1, 0.1, 0.1, 0.01])
        self.P = self.P ** 2

    @staticmethod
    def rot_bw(yaw):
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([[c, -s],
                         [s,  c]])

    def apply_error_state(self, dx):
        self.x.flat[0:4] = self.x[0:4].T + dx[0:4].T
        d_yaw = dx.flat[4]

        yaw = math.atan2(self.x[4], self.x[5])
        yaw += d_yaw
        self.x[4] = math.sin(yaw)
        self.x[5] = math.cos(yaw)

    def imu_apply(self, acc_b, gyr_z):
        gyr_noise = 1e-8
        acc_noise = 1e-6

        yaw = math.atan2(self.x[4], self.x[5])
        dt = 0.1
        rot_bw = self.rot_bw(yaw)

        # Update error state covariance
        Qi = np.diag(np.array([0., 0., acc_noise, acc_noise, gyr_noise]))
        Qi = (Qi * dt) ** 2

        dvdt = rot_bw.dot(np.array([[0., -1.],
                                    [1., 0.]])).dot(acc_b)
        F = np.identity(5)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2:4, 4:5] = dvdt

        self.P = F.dot(self.P).dot(F.T) + Qi

        # predict nominal state
        acc_w = rot_bw.dot(acc_b)

        dx = np.zeros((5, 1))
        dx[0:2] = self.x[2:4] * dt + acc_w * dt * dt / 2.0
        dx[2:4] = acc_w * dt
        dx[4] = gyr_z * dt
        self.apply_error_state(dx)

    def gps_apply(self, x, y):
        inn = np.array([x, y]) - self.x.flat[0:2]
        dx = np.zeros((1, 5))
        dx, self.P = kalman_update(dx, self.P, inn, self.H_gps, self.R_gps)
        self.apply_error_state(dx)

    def run(self):
        self.init()

        length = self.data['gps'].shape[0]
        history_x = np.ndarray(shape=(length, 6))
        history_P = np.ndarray(shape=(length, 5))

        for i in range(0, length):
            acc_b = self.data['acc'][i, :].reshape(-1, 1)
            gyr = self.data['gyr'][i]
            gps = self.data['gps'][i, :]
            if i > 0:
                self.imu_apply(acc_b, gyr)

            if i % 10 == 0:
                self.gps_apply(gps[0], gps[1])
            history_x[i, :] = self.x.T
            history_P[i, :] = self.P.diagonal() ** 0.5

        return history_x, history_P, self.data


def plot_results(history_x, history_P, data):
    plt.plot(history_x[:, 0], history_x[:, 1], label="output")
    plt.plot(data['truth'].x, data['truth'].y, label="gt")
    plt.legend()

    fig, axis = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    axis[0].plot(history_x[:, 2], label="vx")
    axis[1].plot(history_x[:, 3], label="vy")
    yaw = np.arctan2(history_x[:, 4], history_x[:, 5])
    axis[2].plot(yaw * RAD2DEG, label='yaw')
    axis[2].plot(data['truth'].yaw * RAD2DEG, label="gt")
    for i in axis:
        i.legend()
    fig.tight_layout()

    fig, axis = plt.subplots(nrows=5, ncols=1, figsize=(12, 9))
    axis[0].plot(history_P[:, 0], label="x")
    axis[1].plot(history_P[:, 1], label="y")
    axis[2].plot(history_P[:, 2], label="vx")
    axis[3].plot(history_P[:, 3], label="vy")
    axis[4].plot(history_P[:, 4], label="dyaw")
    for i in axis:
        i.legend()

    fig.canvas.set_window_title("sd")
    fig.tight_layout()


if __name__ == '__main__':
    data = simulator.circle()

    f = Filter(data)
    res = f.run()

    plot_results(res[0], res[1], res[2])
    plt.show()
