import numpy as np
import torch
import typing as t
from gymnasium.wrappers.common import TimeLimit
from gymnasium.spaces import Discrete, Box
import q_model as q_model
from q_model import Qfunction
from replay_buffer import ReplayBuffer

# from .q_model import Qfunction
# from .replay_buffer import ReplayBuffer

# remove above line if you do not want to see inline plots from wandb

# hyper-parameters
lr = 1e-3  # learning rate for gradient update
batchsize = 64  # batchsize for buffer sampling
maxlength = 1000  # max number of tuples held by buffer
envname = "CartPole-v0"  # environment name
tau = 100  # time steps for target update
episodes = 300  # number of episodes to run
initialsize = 500  # initial time steps before start training
trainfreq = 20  # frequency of training steps
epsilon = 0.2  # constant for exploration
gamma = 0.99  # discount

# initialize environment
# env: TimeLimit = gym.make(envname)
# env.observation_space = t.cast(Box, env.observation_space)
# env.action_space = t.cast(Discrete, env.action_space)
# obssize = t.cast(Box, env.observation_space).low.size
# actsize = t.cast(Discrete, env.action_space).n

# # initialize Q-function networks (princpal and target)
# Qprincipal = Qfunction(obssize, actsize, lr)
# Qtarget = Qfunction(obssize, actsize, lr)

# # initialization of graph and buffer
# buffer = ReplayBuffer(maxlength)


def run_target_update(Qprincipal: Qfunction, Qtarget: Qfunction):
    for v, v_ in zip(Qprincipal.model.parameters(), Qtarget.model.parameters()):
        v_.data.copy_(v.data)


def training_loop(
    Qprincipal: Qfunction, Qtarget: Qfunction, buffer: ReplayBuffer, env: TimeLimit
):
    # main iteration
    rrecord = []
    totalstep = 0
    for episode in range(episodes):

        obs, info = env.reset()
        done = False
        rsum = 0

        while not done:

            # greedy choice below. Use epsilon greedy for exploration
            action = Qprincipal.compute_argmaxQ(np.expand_dims(obs, 0), epsilon=epsilon)

            newobs, r, done, _, info = env.step(action)
            done_ = 1 if done else 0
            e = (obs, action, r, done_, newobs)

            # IF NOT USING BUFFER:
            # use single sample (obs, action, r, done_, newobs) with Qtarget to compute target and train Qprincipal

            # ELSE IF USING REPLAY BUFFER
            # append experiences e to buffer
            buffer.append(e)

            # every few episodes (decide the frequency) sample a minibatch from buffer
            # compute targets in batch using Qtarget and train Qprincipal
            if totalstep > initialsize and totalstep % trainfreq == 0:
                # sample a minibatch
                states, actions, rewards, newstates, dones = buffer.sample_batch(
                    batchsize, decay_denom=maxlength
                )

                # convert ALL batch arrays to torch on the same device
                device = next(Qtarget.model.parameters()).device
                states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                newstates_t = torch.as_tensor(
                    newstates, dtype=torch.float32, device=device
                )
                dones_t = torch.as_tensor(
                    dones, dtype=torch.float32, device=device
                )  # {0,1} → float

                # compute max_a Q(s', a) with the TARGET net; ensure it's 1-D (N,)
                qnews = Qtarget.compute_maxQvalues(newstates_t).detach().view(-1)

                # sanity check (optional, great for catching mismatches fast)
                assert (
                    qnews.shape == rewards_t.shape
                ), f"qnews {qnews.shape} vs rewards {rewards_t.shape}"

                # Bellman targets: d = r + γ * (1 - done) * max_a Q_target(s', a)
                targets_t = rewards_t + gamma * (1.0 - dones_t) * qnews

                # train principal net (expects NumPy or Torch? yours accepts either; let’s pass Torch)
                loss = Qprincipal.train(states_t, actions_t, targets_t)

            # UPDATE target network
            # every tau steps update copy the principal network to the target network
            if totalstep % tau == 0:
                run_target_update(Qprincipal, Qtarget)

            # update
            totalstep += 1
            rsum += r  # type: ignore
            obs = newobs

        # The code below is for printing and debugging at the end of episode

        rrecord.append(rsum)

        # printing functions for debugging purposes. Feel free to add more
        if episode % 10 == 0:
            print("buffersize {}".format(len(buffer)))
            print(
                "episode {} ave training returns {}".format(
                    episode, np.mean(rrecord[-10:])
                )
            )

        # printing moving averages for smoothed visualization.
        fixedWindow = 100
        movingAverage = 0
        # if len(rrecord) >= fixedWindow:
        #     movingAverage = np.mean(rrecord[len(rrecord) - fixedWindow : len(rrecord) - 1])
        # wandb.log({"training reward": rsum, "train reward moving average": movingAverage})
