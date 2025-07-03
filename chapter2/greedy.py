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


def egreedy(env, epsilon=0.1, steps=1000):
    k = env.num_arms()
    estimations = np.zeros(k)
    # estimations = np.array([-np.inf for _ in range(k)])
    counts = np.zeros(k)
    rewards = []
    for _ in range(steps):
        if np.random.uniform() < epsilon:
            a = np.random.randint(k)
        else:
            a = np.argmax(estimations / counts)
        r = env.action(a)
        rewards.append(r)
        counts[a] += 1
        estimations[a] += r
    return np.array(rewards)


def experiment(env, epsilon, num_runs=2000):
    rewards = []
    for _ in range(num_runs):
        r = egreedy(env, epsilon)
        rewards.append(r)
    rewards = np.vstack(rewards)
    rewards = rewards.mean(axis=0)
    plt.plot(rewards)


def main():
    env = Environment(10)
    experiment(env, 0)
    experiment(env, 0.1)
    experiment(env, 0.01)
    plt.legend([0, 0.1, 0.01])
    plt.show()


main()
