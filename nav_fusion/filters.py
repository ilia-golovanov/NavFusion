import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import simulator


def kalman_predict(x, P, F, Q, B=None, u=None):
    x = F.dot(x)

    if (B is not None) and (u is not None):
        x += B.dot(u)
    P = F.dot(P).dot(F.T) + Q


def kalman_update(x, P, z, H, R):
    z_p = H.dot(x)
    y = z - H.dot(x)
    S = H.dot(P).dot(H.T)

    # L = scipy.linalg.cholesky(S, lower=True)
    Sinv = np.linalg.inv(S)

    K = P.dot(H.T).dot(Sinv)
    x += K.dot(y)

    IKH = np.identity(x.shape[0])
    P = IKH.dot(P).dot(IKH.T) + K.dot(R).dot(K.T)


# model: x - y - vel_x, vel_x, sin/cos bearing (3DOF)
# meas: acc_x, acc_y, gyro_z
# meas: x, y - noisy


class Filter:
    def __init__(self, data):
        self.R_gps = np.diag([1, 1])
        self.H_gps = np.zeros((2, 6))
        self.H_gps[0, 0] = 1.0
        self.H_gps[1, 1] = 1.0

        self.x = np.zeros((6, 1))
        self.P = np.diag([3, 3, 20, 20, 0.2, 0.2])

        self.data = data

    @staticmethod
    def rot_bw(yaw):
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([[c, -s],
                         [s,  c]])

    def imu_apply(self, acc_b, gyr_z):
        yaw = math.atan2(self.x[4], self.x[5])
        dt = 0.1 ## ?
        rot_bw = self.rot_bw(yaw)
        yaw += gyr_z * dt
        self.x[4] = math.sin(yaw)
        self.x[5] = math.cos(yaw)

        acc_w = rot_bw.dot(acc_b)
        self.x[2:4] += acc_w * dt
        self.x[0:2] += self.x[2:4] * dt + acc_w * (dt ** 2) / 2.0

        F = np.identity(6)
        F[0, 2] = dt
        F[1, 3] = dt

        B = np.zeros((6, 2))
        dx = (acc_w * dt ** 2) / 2.0
        dv = acc_w * dt
        B[0, 0] = dx[0]
        B[1, 1] = dx[1]
        B[2, 0] = dv[0]
        B[3, 1] = dv[1]

        acc_noise = 0.0001
        Q = B.dot(B.T) * acc_noise

        self.P = F.dot(self.P).dot(F.T) + Q

    def gps_apply(self, x, y):
        kalman_update(self.x, self.P, np.array([[x], [y]]), self.H_gps, self.R_gps)

    def init(self):
        self.x[0] = self.data['gps'][0, 0]
        self.x[1] = self.data['gps'][0, 1]
        self.x[2] = 1.0
        self.x[3] = 0.0

        self.P = np.diag([3, 3, 20, 20, 2.0, 2.0])

    def run(self):
        self.init()

        length = self.data['gps'].shape[0]
        history_x = np.ndarray(shape=(length, 6))

        for i in range(length):
            acc_b = self.data['acc'][i, :].reshape(-1, 1)
            gyr = self.data['gyr'][i]
            gps = self.data['gps'][i, :]
            self.imu_apply(acc_b, gyr)
            self.gps_apply(gps[0], gps[1])
            history_x[i, :] = self.x.T

        plt.plot(history_x[:, 0], history_x[:, 1])


if __name__ == '__main__':
    data = simulator.circle()

    f = Filter(data)
    f.run()
    plt.show()
