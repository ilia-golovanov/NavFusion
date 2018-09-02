import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

RAD2DEG = 180.0 / math.pi


def circle():
    radius = 100.0
    velocity = 10.0
    duration = 40.0
    dt = 0.1

    period = 2 * math.pi * radius / velocity
    length = int(duration / dt)
    time = np.arange(0.0, duration, dt)
    x = radius * np.sin(2 * math.pi * time / period)
    y = radius * np.cos(2 * math.pi * time / period)

    gps = np.stack([x, y], axis=1) # pd.DataFrame({'x': x, 'y': y})
    acc_x = np.ndarray((length,))
    acc_x[:] = 0.0
    acc_y = np.ndarray((length,))
    acc_y[:] = velocity ** 2 / radius
    acc = np.stack([acc_x, acc_y], axis=1)

    gyr_z = np.ndarray((length,))
    w = velocity / radius
    gyr_z[:] = w

    print(w)
    yaw = time * w
    truth = pd.DataFrame({'x': x, 'y': y, 'yaw': yaw})

    data = dict()
    data['gps'] = gps  # [::10]
    data['acc'] = acc
    data['gyr'] = gyr_z
    data['truth'] = truth
    return data


def plot():
    data = circle()
    # plt.figure()
    # plt.plot(data['acc'][:, 0])
    # plt.plot(data['acc'][:, 1])
    #
    # plt.figure()
    # plt.plot(data['truth'].yaw * RAD2DEG)
    # plt.show()


if __name__ == '__main__':
    plot()
