import matplotlib.pyplot as plt
import numpy as np


class Environment:
    def __init__(self, n):
        self.mus = np.random.normal(size=[n])

    def action(self, a):
        mu = self.mus[a]
        return np.random.normal(loc=mu)


def create_violinplot(env, reps=500):
    all_data = []
    n = len(env.mus)
    for a in range(n):
        data = [env.action(a) for _ in range(reps)]
        all_data.append(data)
    plt.violinplot(all_data, range(n), showmeans=True)
    plt.plot(env.mus)
    plt.show()


def main():
    n = 10
    env = Environment(n)
    create_violinplot(env)


main()
