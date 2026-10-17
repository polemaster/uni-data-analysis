"""
 2-D function drawing with points

"""

import matplotlib.pyplot as plt
import numpy as np


def fun4(x1, x2):
    y = (
        np.sin(x1 * 3) / (abs(x1) + 1)
        + np.sin(x2 * 5 - 1) / (abs(x2 / 2 - 1) + 1)
        + ((x1 - 5) ** 2 + (x2 - 5) ** 2) / 50
    )
    A = np.array(
        [
            [1, 1, 2, -0.07],
            [4, 4, 2.3, 0.5],
            [-7, 6, 1.5, 1],
            [5, 8, 3, 0.8],
            [-2, -5, 0.7, -3],
            [9, -8, 2.4, 2.5],
            [-8, -8, 2.5, 6],
            [-7, 0, 4, 0.4],
        ]
    )
    for i in range(len(A)):
        y -= (
            (x1 - A[i, 0]) * (x1 - A[i, 0]) + (x2 - A[i, 1]) * (x2 - A[i, 1])
            < A[i, 2] * A[i, 2]
        ) * A[i, 3]
    return y


def draw_map():
    X1, X2 = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
    Z = fun4(X1, X2)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    cs = plt.contour(
        X1, X2, Z, levels=[-2.0, -1.8, -1.5, -1, -0.5, 0, 1, 2, 3, 4, 5, 7, 10]
    )


def show_fun():

    draw_map()
    plt.title("2D map: highlands - yellow, valleys - blue")
    plt.show()


def show_point_sequence(Points, title):
    draw_map()
    plt.plot(Points[:, 0], Points[:, 1], "", color="green")
    plt.plot(Points[:, 0], Points[:, 1], "o", color="green", markersize=12, linewidth=2)
    plt.plot(Points[-1, 0], Points[-1, 1], "+", color="red", markersize=20, linewidth=3)
    plt.title(title)
    plt.show()


def show_the_point(x, title):
    draw_map()
    plt.plot(x[0], x[1], "+", color="red", markersize=12, linewidth=3)
    plt.title(title)
    plt.show()


draw_map()
plt.show()
