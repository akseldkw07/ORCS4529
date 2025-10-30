import numpy as np
import wandb
import tqdm
from q_model import Qfunction
from replay_buffer import ReplayBuffer
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from kret_studies import kret_torch as uks_torch

# from .q_model import Qfunction
# from .replay_buffer import ReplayBuffer

# remove above line if you do not want to see inline plots from wandb

# hyper-parameters
envname = "CartPole-v0"  # environment name
lr = 1e-3  # learning rate for gradient update
batchsize = 64  # batchsize for buffer sampling
maxlength = 1000  # max number of tuples held by buffer
tau = 100  # time steps for target update
episodes = 300  # number of episodes to run
initialsize = 500  # initial time steps before start training
trainfreq = 10  # frequency of training steps
epsilon = 0.2  # constant for exploration
gamma = 0.99  # discount
print_debug = False  # whether to print debug info
max_episode_steps = 200  # max number of steps per episode
epsilon_init = 0.5  # initial epsilon for exploration
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
    Qprincipal: Qfunction,
    Qtarget: Qfunction,
    buffer: ReplayBuffer,
    env: CartPoleEnv,
    rrecord: list,
    totalstep: int,
):
    # main iteration
    for episode in tqdm.tqdm(range(episodes)):

        obs, info = env.reset()
        done = False
        rsum = 0

        while not done:
            # greedy choice below. Use epsilon greedy for exploration
            epsilon = uks_torch.exp_decay(episode, epsilon_init, half_life=500)
            action = Qprincipal.compute_argmaxQ(np.expand_dims(obs, 0), epsilon=epsilon)

            newobs, r, done, trunc, info = env.step(action)
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
                states, actions, rewards, newstates, dones = buffer.sample_batch(batchsize, decay_denom=maxlength)

                qnews = Qtarget.compute_maxQvalues(newstates).detach().view(-1)

                # Bellman targets: d = r + γ * (1 - done) * max_a Q_target(s', a)
                targets_t = rewards + gamma * (1.0 - dones) * qnews.numpy()

                Qprincipal.train(states, actions, targets_t)

            # UPDATE target network
            # every tau steps update copy the principal network to the target network
            if totalstep % tau == 0:
                run_target_update(Qprincipal, Qtarget)

            # update
            totalstep += 1
            rsum += r  # type: ignore
            obs = newobs
            done = done or rsum > max_episode_steps  # to avoid truncation issues

        # The code below is for printing and debugging at the end of episode

        rrecord.append(rsum)

        # printing functions for debugging purposes. Feel free to add more
        if print_debug and episode % 10 == 0:
            print(f"buffersize {len(buffer)}")
            print(f"episode {episode} ave training returns {np.mean(rrecord[-10:])}")

        # printing moving averages for smoothed visualization.
        fixedWindow = 100
        movingAverage = 0
        if len(rrecord) >= fixedWindow:
            movingAverage = np.mean(rrecord[len(rrecord) - fixedWindow : len(rrecord) - 1])
        wandb.log({"training reward": rsum, "train reward moving average": movingAverage})
        return rrecord, totalstep
