from functools import cache
from typing import SupportsFloat
import numpy as np
import typing as t
from collections import deque
from numpy.typing import NDArray


@cache
def get_weights(size: int, decay_denom: int):
    x = np.arange(1, size + 1)
    unweighted = np.exp((x - size) / decay_denom)
    return unweighted


Transition = tuple[
    np.ndarray,
    int | np.signedinteger,
    float | SupportsFloat,
    t.Literal[0, 1],
    np.ndarray,
]


class ReplayBuffer(deque[Transition]):
    x: NDArray[np.int_]
    idxs_last: NDArray[np.int_]  # indexes of last sampled batch

    def __init__(self, maxlen: int):
        super().__init__(maxlen=maxlen)

    def _sample_idxs(self, batch_size: int, decay_denom: int | None = None):
        unweighted = get_weights(self.maxlen, decay_denom or self.maxlen)[0 : len(self)]
        weights = unweighted / np.sum(unweighted)

        batch_size = min(batch_size, len(self))
        self.idxs_last = np.random.choice(
            len(self), batch_size, replace=False, p=weights
        )

    def sample_batch(self, batch_size: int, decay_denom: int | None = None):
        self._sample_idxs(batch_size, decay_denom)

        # in ReplayBuffer.sample_batch, just before returning
        s, a, r, s2, d = zip(*(self[i] for i in self.idxs_last))

        fix = lambda x: np.asarray(x, dtype=np.float32).reshape(-1)  # force (4,)
        states = np.stack([fix(si) for si in s])
        newst = np.stack([fix(s2i) for s2i in s2])

        return (
            states,
            np.asarray(a),
            np.asarray(r, dtype=np.float32),
            newst,
            np.asarray(d, dtype=np.bool_),
        )

    def sample_batch_deprecated(self, batch_size: int, decay_denom: int | None = None):
        self._sample_idxs(batch_size, decay_denom)

        # return [self[i] for i in idxs]
        s, a, r, s2, d = zip(*(self[i] for i in self.idxs_last))
        return (
            np.stack(s),
            np.asarray(a),
            np.asarray(r, dtype=np.float32),
            np.stack(s2),
            np.asarray(d, dtype=np.bool_),
        )
