"""
  Run this file at first, in order to see what it is printing.
  Instead of print(), we use the respective log level from loguru.
"""

from abc import ABC, abstractmethod
from loguru import logger
import random
import math
import matplotlib.pyplot as plt


class Bandit(ABC):
    """Abstract base class for a multi-armed bandit."""

    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """
        Args:
            p (list[float]): List of true means for each arm.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Returns a string representation of the bandit."""
        pass

    @abstractmethod
    def pull(self):
        """Simulate pulling a chosen arm, returning a reward."""
        pass

    @abstractmethod
    def update(self):
        """Update internal parameters (posterior or estimates) after receiving a reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Run the multi-armed bandit experiment for a specified number of trials."""
        pass

    @abstractmethod
    def report(self):
        """
        Store data in CSV, print average reward (use f-strings),
        and print average regret (use f-strings).
        """
        pass


#--------------------------------------#
class Visualization:
    """
    Visualization class with two methods:
      - plot1(algorithm_name, rewards_history):
          Plot rolling-average reward over time.
      - plot2(egreedy_rewards, thompson_rewards):
          Compare cumulative rewards of both algorithms.
    """

    def plot1(self, algorithm_name, rewards_history):
        """Visualize the rolling-average reward for a given algorithm.

        Args:
            algorithm_name (str): Name of the bandit algorithm.
            rewards_history (list[float]): List of reward values from each trial.
        """
        if not rewards_history:
            logger.warning(f"No rewards to plot for {algorithm_name}.")
            return

        # Compute rolling average
        rolling_avg = []
        cumsum = 0.0
        for i, reward in enumerate(rewards_history, start=1):
            cumsum += reward
            rolling_avg.append(cumsum / i)

        # Plot
        plt.figure()
        plt.plot(rolling_avg)
        plt.title(f"{algorithm_name}: Rolling Average Reward Over Time")
        plt.xlabel("Trial")
        plt.ylabel("Average Reward")
        plt.show()

    def plot2(self, egreedy_rewards, thompson_rewards):
        """Compare cumulative rewards for E-greedy and Thompson Sampling.

        Args:
            egreedy_rewards (list[float]): Rewards from EpsilonGreedy per trial.
            thompson_rewards (list[float]): Rewards from ThompsonSampling per trial.
        """
        # Compute cumulative sums
        cum_eg = []
        cum_ts = []
        eg_sum = 0.0
        ts_sum = 0.0

        max_len = max(len(egreedy_rewards), len(thompson_rewards))
        for i in range(max_len):
            if i < len(egreedy_rewards):
                eg_sum += egreedy_rewards[i]
            if i < len(thompson_rewards):
                ts_sum += thompson_rewards[i]
            cum_eg.append(eg_sum)
            cum_ts.append(ts_sum)

        # Plot
        plt.figure()
        plt.plot(cum_eg, label="Epsilon-Greedy")
        plt.plot(cum_ts, label="Thompson Sampling")
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.show()


#--------------------------------------#
class EpsilonGreedy(Bandit):
    def __init__(self, p):
        """
        p: list of true means for each arm (e.g., [1, 2, 3, 4]).

        Args:
            p (list[float]): The true means for each arm.
        """
        self.p = p                      
        self.n_arms = len(p)          
        self.estimates = [0.0] * self.n_arms   
        self.counts = [0] * self.n_arms        

        self.rewards_history = []
        self.regret_history = []

        self.best_mean = max(p)

    def __repr__(self):
        """
        Returns:
            str: The name of this bandit algorithm, "EpsilonGreedy".
        """
        return "EpsilonGreedy"

    def pull(self, arm):
        """Simulate pulling the specified arm.

        We'll assume each arm's reward is drawn
        from a Normal distribution with mean = p[arm] and std dev = 1.

        Args:
            arm (int): Index of the arm to pull.

        Returns:
            float: The simulated reward.
        """
        reward = random.gauss(self.p[arm], 1.0)
        return reward

    def update(self, arm, reward):
        """Update estimated mean for the given arm based on the received reward.

        Args:
            arm (int): Which arm was pulled.
            reward (float): The observed reward from that pull.
        """
        self.counts[arm] += 1
        n = self.counts[arm]

        current_est = self.estimates[arm]

        new_est = current_est + (reward - current_est) / n
        self.estimates[arm] = new_est

    def experiment(self, n_trials=10000):
        """Run an epsilon-greedy strategy for `n_trials`.

        We'll use a simple decaying epsilon = 1/t approach.

        Args:
            n_trials (int, optional): Number of trials to run. Defaults to 10000.
        """
        for t in range(1, n_trials + 1):

            epsilon = 1.0 / t

            # Exploration vs. exploitation
            if random.random() < epsilon:
                arm = random.randint(0, self.n_arms - 1)
            else:
                arm = max(range(self.n_arms), key=lambda x: self.estimates[x])

            reward = self.pull(arm)
            self.update(arm, reward)
            regret = self.best_mean - self.p[arm]

            self.rewards_history.append(reward)
            self.regret_history.append(regret)

    def report(self):
        """Compute and print the average reward and regret.
        Also store the trial-by-trial data in a CSV file.
        """
        import csv

        total_trials = len(self.rewards_history)
        avg_reward = sum(self.rewards_history) / total_trials if total_trials > 0 else 0.0
        avg_regret = sum(self.regret_history) / total_trials if total_trials > 0 else 0.0

        logger.info(f"[EpsilonGreedy] Average Reward: {avg_reward:.3f}")
        logger.info(f"[EpsilonGreedy] Average Regret: {avg_regret:.3f}")

        csv_filename = "egreedy_results.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "reward", "regret"])
            for i, (r, rg) in enumerate(zip(self.rewards_history, self.regret_history), start=1):
                writer.writerow([i, r, rg])

        logger.info(f"[EpsilonGreedy] Results saved to {csv_filename}")


#--------------------------------------#
class ThompsonSampling(Bandit):
    def __init__(self, p):
        """
        p: true means for each arm (e.g., [1, 2, 3, 4]).
        We'll treat these means as the underlying distribution's means
        for a Normal(mean=p[arm], stddev=1) reward, just like EpsilonGreedy.

        Args:
            p (list[float]): The true means for each arm.
        """
        self.p = p
        self.n_arms = len(p)

        self.means = [0.0] * self.n_arms
        self.precisions = [1.0] * self.n_arms 
        self.counts = [0] * self.n_arms

        self.rewards_history = []
        self.regret_history = []
        self.best_mean = max(p)

    def __repr__(self):
        """
        Returns:
            str: The name of this bandit algorithm, "ThompsonSampling".
        """
        return "ThompsonSampling"

    def pull(self, arm):
        """Simulate pulling the specified arm.

        We'll assume each arm's reward is drawn from Normal(p[arm], 1.0).

        Args:
            arm (int): Index of the arm to pull.

        Returns:
            float: The simulated reward.
        """
        reward = random.gauss(self.p[arm], 1.0)
        return reward

    def update(self, arm, reward):
        """Update the posterior for the chosen arm given the observed reward.

        For Normal with variance=1, the posterior is Normal with:
          precision_new = precision_old + 1
          mean_new = (mean_old * precision_old + reward) / precision_new

        Args:
            arm (int): Which arm was pulled.
            reward (float): The observed reward from that pull.
        """
        old_mean = self.means[arm]
        old_precision = self.precisions[arm]

        new_precision = old_precision + 1.0
        new_mean = (old_mean * old_precision + reward) / new_precision

        self.means[arm] = new_mean
        self.precisions[arm] = new_precision
        self.counts[arm] += 1

    def experiment(self, n_trials=10000):
        """Run Thompson Sampling for n_trials.

        Args:
            n_trials (int, optional): Number of trials to run. Defaults to 10000.
        """
        for t in range(1, n_trials + 1):
            samples = []
            for arm in range(self.n_arms):
                var = 1.0 / self.precisions[arm]
                sample_mean = random.gauss(self.means[arm], math.sqrt(var))
                samples.append(sample_mean)

            chosen_arm = max(range(self.n_arms), key=lambda i: samples[i])
            reward = self.pull(chosen_arm)
            self.update(chosen_arm, reward)

            regret = self.best_mean - self.p[chosen_arm]
            self.rewards_history.append(reward)
            self.regret_history.append(regret)

    def report(self):
        """Compute and print the average reward and regret.
        Also store the trial-by-trial data in a CSV file.
        """
        import csv

        total_trials = len(self.rewards_history)
        if total_trials > 0:
            avg_reward = sum(self.rewards_history) / total_trials
            avg_regret = sum(self.regret_history) / total_trials
        else:
            avg_reward = 0.0
            avg_regret = 0.0

        logger.info(f"[ThompsonSampling] Average Reward: {avg_reward:.3f}")
        logger.info(f"[ThompsonSampling] Average Regret: {avg_regret:.3f}")

        csv_filename = "thompson_results.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "reward", "regret"])
            for i, (r, rg) in enumerate(zip(self.rewards_history, self.regret_history), start=1):
                writer.writerow([i, r, rg])

        logger.info(f"[ThompsonSampling] Results saved to {csv_filename}")


def comparison():
    """Run a comparison of EpsilonGreedy vs ThompsonSampling on 4 arms, 20k trials."""

    bandit_reward = [1, 2, 3, 4]
    n_trials = 20000

    egreedy_bandit = EpsilonGreedy(bandit_reward)
    egreedy_bandit.experiment(n_trials=n_trials)
    egreedy_bandit.report()

    ts_bandit = ThompsonSampling(bandit_reward)
    ts_bandit.experiment(n_trials=n_trials)
    ts_bandit.report()

    viz = Visualization()

    viz.plot1("EpsilonGreedy", egreedy_bandit.rewards_history)
    viz.plot1("ThompsonSampling", ts_bandit.rewards_history)

    viz.plot2(egreedy_bandit.rewards_history, ts_bandit.rewards_history)

    logger.info("Comparison complete. Check CSV files and plots for details.")


if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison()
