import matplotlib.pyplot as plt
import numpy as np


class Environment:
    def __init__(self, k):
        self.mus = np.random.normal(size=[k])

    def action(self, a):
        mu = self.mus[a]
        return np.random.normal(loc=mu)

    def num_arms(self):
        return len(self.mus)


def create_violinplot(env, reps=500):
    all_data = []
    n = env.num_arms()
    for a in range(n):
        data = [env.action(a) for _ in range(reps)]
        all_data.append(data)
    plt.violinplot(all_data, range(n), showmeans=True)
    plt.plot(env.mus)
    plt.show()


def main():
    env = Environment(10)
    # create_violinplot(env)

    k = env.num_arms()
    estimations = np.array([-np.inf for _ in range(k)])
    counts = np.zeros(k)

    steps = 1000
    epsilon = 0.1
    for _ in range(steps):
        if np.random.uniform() < epsilon:
            a = np.random.random_integers(0, 9)
        else:
            a = np.argmax(estimations / counts)
        r = env.action(a)
        counts[a] += 1
        estimations[a] += r


main()
