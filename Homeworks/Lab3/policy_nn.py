# define neural net \pi_\phi(s) as a class
import torch
import numpy as np
from collections.abc import Callable


class Policy:
    obssize: int
    actsize: int
    h1: int
    h2: int
    model: Callable[[torch.Tensor], torch.Tensor]
    optimizer: torch.optim.Optimizer

    def __init__(self, obssize: int, actsize: int | np.signedinteger, lr: float):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # RECORD HYPER-PARAMS
        self.obssize = obssize
        self.actsize = int(actsize)

        self.h1 = int(np.clip(2 * obssize, 64, 512))
        self.h2 = int(np.clip(obssize, 64, 512))

        # TODO DEFINE THE MODEL
        self.model = torch.nn.Sequential(
            # input layer of input size obssize
            torch.nn.Linear(obssize, self.h1),
            torch.nn.ReLU(),
            # intermediate layers
            torch.nn.Linear(self.h1, self.h2),
            torch.nn.ReLU(),
            # output layer of output size actsize
            torch.nn.Linear(self.h2, self.actsize),
        )

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # TEST
        self.compute_prob(np.random.randn(obssize).reshape(1, -1))

    def compute_prob(self, states: np.ndarray | torch.Tensor):
        """
        compute prob distribution over all actions given state: pi(s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, actsize]
        """
        states = torch.FloatTensor(states)
        prob = torch.nn.functional.softmax(self.model(states), dim=-1)
        return prob.cpu().data.numpy()

    def _to_one_hot(self, y: torch.Tensor, num_classes: int):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def train(
        self, states: np.ndarray | torch.Tensor, actions: np.ndarray | torch.Tensor, Qs: np.ndarray | torch.Tensor
    ):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        Qs = torch.FloatTensor(Qs)

        # COMPUTE probability vector pi(s) for all s in states
        logits = self.model(states)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        # Compute probaility pi(s,a) for all s,a
        action_onehot = self._to_one_hot(actions, self.actsize)
        prob_selected: torch.Tensor = torch.sum(prob * action_onehot, axis=-1)  # type: ignore

        # FOR ROBUSTNESS
        prob_selected += 1e-8

        # TODO define loss function as described in the text above
        loss = torch.mean(Qs * torch.log(prob_selected)) * -1.0

        # BACKWARD PASS
        self.optimizer.zero_grad()
        loss.backward()

        # UPDATE
        self.optimizer.step()

        return loss.detach().cpu().data.numpy()
