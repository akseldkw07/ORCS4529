import numpy as np
import wandb
import tqdm
from value_nn import ValueFunction
from policy_nn import Policy
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

# parameter initializations (you can change any of these)
alpha = 1e-2  # learning rate for PG
beta = 1e-3  # learning rate for baseline
numtrajs = 5  # num of trajecories from the current policy to collect in each iteration
iterations = 300  # total num of iterations
envname = "CartPole-v0"  # environment name
gamma = 0.99  # discount


def discounted_rewards(r: list[float], gamma: float):
    """take 1D float array of rewards and compute discounted reward"""
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0, len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


def train_policy_gradient(
    policy: Policy, baseline: ValueFunction, env: CartPoleEnv, rrecord: list, max_traj_steps: int
):
    for iter in tqdm.tqdm(range(iterations)):

        # To record trajectories generated from current policy
        OBS = []  # observations
        ACTS = []  # actions
        ADS = []  # advantages (to compute policy gradient)
        VAL = []  # Monte carlo value predictions (to compute baseline, and policy gradient)

        for num in range(numtrajs):
            # To keep a record of states actions and reward for each episode
            obss = []  # states
            acts = []  # actions
            rews = []  # instant rewards

            obs, info = env.reset()

            # TODO: run one episode using the current policy "actor"
            # TODO: record all observations (states, actions, rewards) from the epsiode in  obss, acts, rews
            done = False
            while not done:
                obss.append(obs.copy())
                prob = policy.compute_prob(np.expand_dims(obs, 0))
                prob: np.ndarray = prob / np.sum(prob)  # normalizing again to account for numerical errors
                action = np.random.choice(policy.actsize, p=prob.flatten(), size=1).item()
                acts.append(action)
                obs, reward, done, term, info = env.step(action)
                rews.append(reward)
                done = done or len(rews) > max_traj_steps

            # Below is for logging training performance
            rrecord.append(np.sum(rews))

            # TODO:  Use discounted_rewards function to compute \hat{V}s/\hat{Q}s  from instant rewards in rews
            discounted_r = discounted_rewards(rews, gamma)
            # TODO: record the computed \hat{V}s in VAL, states obss in OBS, and actions acts in ACTS, for batch update
            VAL.extend(discounted_r)
            OBS.extend(obss)
            ACTS.extend(acts)

        # AFTER collecting numtrajs trajectories:

        # 1. TODO: train baseline
        """
            Use the batch (OBS, VAL) of states and value predictions as targets to train baseline.
            Use baseline.train : note that this takes as input numpy array, so you may have to convert
            lists into numpy array using np.array()
        """
        baseline.train(np.array(OBS), np.array(VAL))

        # 2.TODO: Update the policy
        """
            Compute baselines: use baseline.compute_values for states in the batch OBS
            Compute advantages ADS using VAL and computed baselines
            Update policy using actor.train using OBS, ACTS and ADS
        """
        baselines = baseline.compute_values(np.array(OBS))
        ADS = np.array(VAL) - baselines
        policy.train(np.array(OBS), np.array(ACTS), np.array(ADS))

        # printing moving averages for smoothed visualization.
        # Do not change below: this assume you recorded the sum of rewards in each episide in the list rrecord
        fixedWindow = 100
        movingAverage = 0
        if len(rrecord) >= fixedWindow:
            movingAverage = np.mean(rrecord[len(rrecord) - fixedWindow : len(rrecord) - 1])

        # wandb logging
        wandb.log({"training reward": rrecord[-1], "training reward moving average": movingAverage})
