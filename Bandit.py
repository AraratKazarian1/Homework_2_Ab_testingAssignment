"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # log average reward (use f strings to make it informative)
        # log average regret (use f strings to make it informative)
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon Greedy algorithm for multi-armed bandit problem.
    """

    def __init__(self, p, epsilon):
        self.p = p
        self.epsilon = epsilon
        self.counts = [0] * len(p)
        self.values = [0] * len(p)

    def __repr__(self):
        return f"EpsilonGreedy(p={self.p}, epsilon={self.epsilon})"

    def pull(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.p))
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            arm = self.pull()
            reward = np.random.choice(self.p[arm])
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        pass

#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling algorithm for multi-armed bandit problem.
    """

    def __init__(self, p, precision):
        self.p = p
        self.precision = precision
        self.alpha = [1] * len(p)
        self.beta = [1] * len(p)

    def __repr__(self):
        return f"ThompsonSampling(p={self.p}, precision={self.precision})"

    def pull(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(len(self.p))]
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            arm = self.pull()
            reward = np.random.choice(self.p[arm])
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        pass

#--------------------------------------#

class Visualization():
    """
    Class for visualizing results of bandit algorithms.
    """

    @staticmethod
    def plot_learning_process(rewards_egreedy, rewards_thompson, window_size=100):
        avg_rewards_egreedy = [sum(rewards_egreedy[i:i+window_size])/window_size for i in range(0, len(rewards_egreedy), window_size)]
        avg_rewards_thompson = [sum(rewards_thompson[i:i+window_size])/window_size for i in range(0, len(rewards_thompson), window_size)]

        plt.plot(avg_rewards_egreedy, label="Epsilon Greedy")
        plt.plot(avg_rewards_thompson, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Average Reward")
        plt.title("Learning Process (Average Reward)")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_cumulative_rewards(rewards_egreedy, rewards_thompson):
        cum_rewards_egreedy = np.cumsum(rewards_egreedy)
        cum_rewards_thompson = np.cumsum(rewards_thompson)
        plt.plot(cum_rewards_egreedy, label="Epsilon Greedy")
        plt.plot(cum_rewards_thompson, label="Thompson Sampling")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards Comparison")
        plt.legend()
        plt.show()

#--------------------------------------#

def store_rewards_csv_combined(rewards_egreedy, rewards_thompson):
    num_bandits_egreedy = len(rewards_egreedy)
    num_bandits_thompson = len(rewards_thompson)

    bandit_numbers_egreedy = list(range(1, num_bandits_egreedy + 1))
    bandit_numbers_thompson = list(range(num_bandits_egreedy + 1, num_bandits_egreedy + num_bandits_thompson + 1))

    bandit_numbers = bandit_numbers_egreedy + bandit_numbers_thompson
    algorithms = ['Epsilon Greedy'] * num_bandits_egreedy + ['Thompson Sampling'] * num_bandits_thompson

    rewards = rewards_egreedy + rewards_thompson

    data = {'Bandit': bandit_numbers,
            'Reward': rewards,
            'Algorithm': algorithms}

    df = pd.DataFrame(data)
    df.to_csv('rewards_combined.csv', index=False)

def main():
    Bandit_Reward = [[1], [2], [3], [4]]
    num_trials = 20000
    epsilon = 0.1  # Choose epsilon value
    precision = 0.01  # Choose precision for Thompson Sampling

    # Initialize bandit algorithms
    epsilon_greedy = EpsilonGreedy(Bandit_Reward, epsilon)
    thompson_sampling = ThompsonSampling(Bandit_Reward, precision)

    # Run experiments
    rewards_egreedy = epsilon_greedy.experiment(num_trials)
    rewards_thompson = thompson_sampling.experiment(num_trials)

    # Visualize learning process
    Visualization.plot_learning_process(rewards_egreedy, rewards_thompson)

    # Visualize cumulative rewards
    Visualization.plot_cumulative_rewards(rewards_egreedy, rewards_thompson)

    # Print cumulative rewards
    print(f"Cumulative Reward - Epsilon Greedy: {sum(rewards_egreedy)}")
    print(f"Cumulative Reward - Thompson Sampling: {sum(rewards_thompson)}")

    # Calculate cumulative regret
    best_arm_reward = max([sum(arm) for arm in Bandit_Reward])
    regret_egreedy = [best_arm_reward - reward for reward in rewards_egreedy]
    regret_thompson = [best_arm_reward - reward for reward in rewards_thompson]

    # Print cumulative regret
    print(f"Cumulative Regret - Epsilon Greedy: {sum(regret_egreedy)}")
    print(f"Cumulative Regret - Thompson Sampling: {sum(regret_thompson)}")

    # Store rewards in CSV file
    store_rewards_csv_combined(rewards_egreedy, rewards_thompson)

if __name__ == '__main__':
    main()