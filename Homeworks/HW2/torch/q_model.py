# define neural net Q_\theta(s,a) as a class
import torch
import numpy as np


class Qfunction(object):
    obssize: int
    actsize: int
    h1: int
    h2: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, obssize: int, actsize: int | np.signedinteger, lr: float):
        """
        obssize: dimension of state space
        actsize: dimension of action space
        sess: sess to execute this Qfunction
        optimizer:
        """
        # RECORD HYPER-PARAMS
        self.obssize = obssize
        self.actsize = int(actsize)

        self.h1 = min(512, max(64, 2 * obssize))
        self.h2 = min(512, max(64, int(obssize + actsize)))

        # DEFINE THE MODEL
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.obssize, self.h1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h1, self.h2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h2, self.actsize),  # outputs Q(s,a) for all a
        )

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _to_one_hot(self, y: torch.Tensor, num_classes: int):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

        y = y.view(-1, 1).long()
        out = torch.zeros(y.size(0), num_classes, dtype=torch.float32, device=y.device)
        out.scatter_(1, y, 1.0)
        return out

    def compute_Qvalues(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        input: list of numsamples state-action pairs
        output: List of Q values for each input (s,a). The output will have size [numsamples, 1]
        """
        # Below is example code when neural network is set to take as input state and output Q-value for all actions.
        # This will be different for neural network that takes as input a state-action pair

        states = states.float().view(-1, self.obssize)  # (N, 4)
        actions = actions.view(-1, 1).long()  # (N, 1)
        q_all = self.model(states)  # (N, 2)
        return q_all.gather(1, actions).squeeze(1)

    def compute_maxQvalues(self, states: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        input: a list of numsamples states
        output: max_a Q(s,a) values for every input state s in states. The output will have size numsamples
        """
        # Below is example code when neural network is set to take as input state and output Q-value for all actions.
        # if the neural takes as input a state-action pair, then the code will need to loop over all actions to compute all values
        states = torch.from_numpy(states) if isinstance(states, np.ndarray) else states
        states = states.float().view(-1, self.obssize)  # <— force (N, 4)
        q_all = self.model(states)  # (N, actsize)
        return torch.max(q_all, dim=1).values

    def compute_argmaxQ(self, state: np.ndarray | torch.Tensor, epsilon: float = 0):
        """
        input:
            state: (obssize,) or (1, obssize)
        output:
            greedy action index as np.int64
        """
        if isinstance(state, np.ndarray):
            st = torch.from_numpy(state).float()
        else:
            st = state.float()

        if st.dim() == 1:
            st = st.unsqueeze(0)  # (1, S)

        if np.random.rand() < epsilon:
            a = np.random.randint(self.actsize)
        else:
            q: torch.Tensor = self.model(st)  # (1, A)
            a: int = int(torch.argmax(q, dim=1).item())

        return a

    def train(
        self,
        states: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
    ):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)

        # COMPUTE Q PREDICTIONS for all state-action pairs
        q_preds_selected = self.compute_Qvalues(states, actions)

        # LOSS
        # print(q_preds_selected.shape, targets.shape)
        loss = torch.mean((q_preds_selected - targets) ** 2)

        # BACKWARD PASS
        self.optimizer.zero_grad()
        loss.backward()

        # UPDATE
        self.optimizer.step()

        return loss.detach().cpu().data.numpy()
