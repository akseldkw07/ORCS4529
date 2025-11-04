# define neural net \pi_\phi(s) as a class
import torch
import numpy as np


class Policy:

    def __init__(self, obssize, actsize, lr):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # TODO DEFINE THE MODEL
        self.model = torch.nn.Sequential(
            # input layer of input size obssize
            # intermediate layers
            # output layer of output size actsize
        )

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # RECORD HYPER-PARAMS
        self.obssize = obssize
        self.actsize = actsize

        # TEST
        self.compute_prob(np.random.randn(obssize).reshape(1, -1))

    def compute_prob(self, states):
        """
        compute prob distribution over all actions given state: pi(s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, actsize]
        """
        states = torch.FloatTensor(states)
        prob = torch.nn.functional.softmax(self.model(states), dim=-1)
        return prob.cpu().data.numpy()

    def _to_one_hot(self, y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def train(self, states, actions, Qs):
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
        action_onehot = self._to_one_hot(actions, actsize)
        prob_selected = torch.sum(prob * action_onehot, axis=-1)

        # FOR ROBUSTNESS
        prob_selected += 1e-8

        # TODO define loss function as described in the text above
        # loss = ....

        # BACKWARD PASS
        self.optimizer.zero_grad()
        loss.backward()

        # UPDATE
        self.optimizer.step()

        return loss.detach().cpu().data.numpy()
