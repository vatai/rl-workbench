import matplotlib as mpl

mpl.use("Qt5Agg")  # <- important to call before any other mpl imports/inits
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


def egreedy(env, epsilon, steps):
    k = env.num_arms()
    estimations = np.zeros(k)
    # estimations = np.array([-np.inf for _ in range(k)])
    counts = np.zeros(k)
    rewards = np.zeros(steps)
    for i in range(steps):
        if np.random.uniform() < epsilon:
            a = np.random.randint(k)
        else:
            a = np.argmax(estimations / counts)
        r = env.action(a)
        counts[a] += 1
        estimations[a] += r
        rewards[i] = (estimations / counts).mean()
    return rewards


def experiment(env, epsilon, num_runs, steps):
    rewards = np.zeros([num_runs, steps])
    for r in range(num_runs):
        rewards[r] = egreedy(env, epsilon, steps)
    rewards = rewards.mean(axis=0)
    plt.plot(rewards)


def main():
    # np.random.seed(4)
    env = Environment(10)
    print(f"{env.mus=}")
    for epsilon in [0, 0.1, 0.01]:
        experiment(env, epsilon=epsilon, num_runs=2000, steps=1000)
    plt.legend([0, 0.1, 0.01])
    # plt.ylim(0, 2)
    plt.show()


main()
