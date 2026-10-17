"""
   Simulated Annealing  - the minimum of 2D function searching

"""

import map_min_search as mm
import matplotlib.pyplot as plt
import numpy as np


def plot_temperatures(iterations, initial_temp):
    # array of iterations from 0 to iterations - 1
    iterations = [i for i in range(iterations)]
    # temperatures for each iterations
    temperatures = [initial_temp / float(i + 1) for i in iterations]
    # plot iterations vs temperatures
    plt.plot(iterations, temperatures)
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.show()


# from wikipedia, e_prim - new point
def accept_prob_fun(e, e_prim, T):
    if e_prim < e:
        return 1
    return np.exp(-(e_prim - e) / T)


def main():
    num_of_steps = 2000  # number of steps: do not change

    num_of_parameters = 2  # number of solution parameters
    N = num_of_parameters

    Temp = 2000
    Temp_min = 0.01  # minimal temperature
    delta_T = 0.995  # change of temperature
    c = 0.1  # constant due to the influence of T for acceptance probability

    # plot_temperatures(num_of_steps, Temp)

    # my parameters
    bounds = np.array([[-10, 10], [-10, 10]])
    step_size = 0.1
    x_min = (bounds[:, 1] - bounds[:, 0]) / 2 + bounds[:, 0]
    Temp_0 = Temp

    Solution = np.random.rand(N) * 20 - 10  # initial solution - random point

    E_min = 10e30  # minimal function value
    E_prev = 0  # previous value of the function
    Records = np.empty((0, N))  # array of record solutions

    mm.show_the_point(Solution, "initial solution")
    # P(delta E) = 1 / (1 + exp(delta E / cT))

    for ep in range(num_of_steps):
        # new solution (should be near previous one and dependent on Temp )
        NewSolution = Solution + np.random.uniform(
            -1, 1, size=2
        )  # Generate new solution
        # NewSolution = (
        #     Solution
        #     + (np.random.rand(len(x_min)) * 2 - 1)
        #     * (bounds[:, 1] - bounds[:, 0])
        #     * step_size
        # )

        E = mm.fun4(
            NewSolution[0], NewSolution[1]
        )  # function value for point coordinates

        dE = (
            E - E_prev
        )  # change of function value (dE < 0 means than new solution is better)

        # accept_prob = accept_prob_fun(E_prev, E, Temp)  # acceptance probability

        accept_prob = 1 / (1 + np.exp(dE / (c * Temp)))
        if np.random.rand() < accept_prob or dE < 0:
            Solution = NewSolution
            E_prev = E

        if E < E_min:
            print(
                "new minimum = "
                + str(E)
                + " for point x1 = "
                + str(NewSolution[0])
                + " x2 = "
                + str(NewSolution[1])
                + "\n"
            )
            E_min = E
            Solution_min = NewSolution
            Records = np.append(Records, [NewSolution], axis=0)

        # Temp = Temp_0 / float(ep + 1)
        Temp *= delta_T
        # Temp = Temp_0 * (2 ** (2.62 - 1) - 1) / ((2 + ep) ** (2.62 - 1) - 1)

        if Temp < Temp_min:
            Temp = Temp_min
    # end of steps loop
    text = (
        "best solution, value = "
        + str(E_min)
        + " for point x1 = "
        + str(Solution_min[0])
        + " x2 = "
        + str(Solution_min[1])
        + " temp: "
        + str(Temp)
    )
    print(text + "\n")
    mm.show_point_sequence(Records, "record sequence, " + text)


if __name__ == "__main__":
    main()
