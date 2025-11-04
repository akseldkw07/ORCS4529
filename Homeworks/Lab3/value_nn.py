# TODO: define value function as a class. You need to define the model and set the loss.
import torch
import numpy as np


class ValueFunction:

    def __init__(self, obssize, lr):
        """
        obssize: size of states
        """
        # TODO DEFINE THE MODEL
        self.model = torch.nn.Sequential(
            # TODO
            # input layer of input size obssize
            # intermediate layers
            # output layer of output size 1
        )

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # RECORD HYPER-PARAMS
        self.obssize = obssize
        self.actsize = actsize

        # TEST
        self.compute_values(np.random.randn(obssize).reshape(1, -1))

    def compute_values(self, states):
        """
        compute value function for given states
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples]
        """
        states = torch.FloatTensor(states)
        return self.model(states).cpu().data.numpy()

    def train(self, states, targets):
        """
        states: numpy array
        targets: numpy array
        """
        states = torch.FloatTensor(states)
        targets = torch.FloatTensor(targets)

        # COMPUTE Value PREDICTIONS for states
        self.model(states)

        # LOSS
        # TODO: set LOSS as square error of predicted values compared to targets
        # loss =

        # BACKWARD PASS
        self.optimizer.zero_grad()
        loss.backward()

        # UPDATE
        self.optimizer.step()

        return loss.detach().cpu().data.numpy()
