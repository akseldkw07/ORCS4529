# TODO: define value function as a class. You need to define the model and set the loss.
from collections.abc import Callable
import torch
import numpy as np


class ValueFunction:
    obssize: int
    h1: int
    h2: int
    model: Callable[[torch.Tensor], torch.Tensor]
    optimizer: torch.optim.Optimizer

    def __init__(self, obssize: int, lr: float):
        """
        obssize: size of states
        """
        # RECORD HYPER-PARAMS
        self.obssize = obssize

        self.h1 = int(np.clip(2 * obssize, 64, 512))
        self.h2 = int(np.clip(obssize, 64, 512))

        # TODO DEFINE THE MODEL
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obssize, self.h1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h1, self.h2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h2, 1),
        )

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # TEST
        self.compute_values(np.random.randn(obssize).reshape(1, -1))

    def compute_values(self, states: np.ndarray | torch.Tensor):
        """
        compute value function for given states
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples]
        """
        states = torch.FloatTensor(states)
        return self.model(states).cpu().data.numpy()

    def train(self, states: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor):
        """
        states: numpy array
        targets: numpy array
        """
        states = torch.FloatTensor(states)
        targets = torch.FloatTensor(targets)

        # COMPUTE Value PREDICTIONS for states
        predictions: torch.Tensor = self.model(states)

        # LOSS
        # TODO: set LOSS as square error of predicted values compared to targets
        loss = torch.nn.functional.mse_loss(predictions.squeeze(), targets)

        # BACKWARD PASS
        self.optimizer.zero_grad()
        loss.backward()

        # UPDATE
        self.optimizer.step()

        return loss.detach().cpu().data.numpy()
