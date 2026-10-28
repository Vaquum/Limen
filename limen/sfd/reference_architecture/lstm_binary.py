from collections.abc import Callable, Iterator
import re
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override

from limen._optional import require_optional
from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ['LSTMBinary', 'lstm_binary']
_FloatArray = npt.NDArray[np.float64]
_THRESHOLD = 0.5
_MATRIX_DIMENSIONS = 2
_BIDIRECTIONAL = 2


class _Network(Protocol):
    def __call__(self, inputs: 'Tensor') -> 'Tensor': ...
    def parameters(self) -> Iterator['Tensor']: ...
    def eval(self) -> object: ...


def _network(params: dict[str, object]) -> _Network:
    _ = require_optional('torch', 'PyTorch', 'neural')
    import torch
    from torch import Tensor, nn

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sequence = cast(Callable[..., nn.LSTM], nn.LSTM)(
                input_size=1, batch_first=True, device='cpu', dtype=torch.float32, **({'hidden_size': 16} | params))
            self.directions = _BIDIRECTIONAL if self.sequence.bidirectional else 1
            width = self.sequence.proj_size or self.sequence.hidden_size
            self.head = nn.Linear(width * self.directions, 1, device='cpu', dtype=torch.float32)

        @override
        def forward(self, inputs: Tensor) -> Tensor:
            _, (hidden, _) = cast(tuple[Tensor, tuple[Tensor, Tensor]], self.sequence(inputs))
            final = hidden[-self.directions:].transpose(0, 1).flatten(1)
            return cast(Tensor, self.head(final)).reshape(-1)

    return cast(_Network, Network())


class LSTMBinary(ReferenceModel):
    '''CPU LSTM over completed return windows, with train-only scalar normalization.'''
    deterministic, inference_output = False, ('binary', True)

    def __init__(self) -> None:
        super().__init__()
        self.model: _Network | None = None
        self._columns: list[str] = []
        self._indices: list[int] = []
        self._width = 0
        self._mean, self._scale = 0.0, 1.0

    def _windows(self, inputs: object) -> _FloatArray:
        if isinstance(inputs, pl.DataFrame):
            values = inputs.select(self._columns).to_numpy()
        else:
            matrix = np.asarray(inputs, dtype=np.float64)
            if matrix.ndim != _MATRIX_DIMENSIONS or matrix.shape[1] != self._width:
                raise ValueError('LSTMBinary inference requires the fitted feature order and width')
            values = matrix[:, self._indices]
        windows = np.asarray(values, dtype=np.float64)
        if not np.isfinite(windows).all():
            raise ValueError('LSTMBinary requires finite return windows')
        return windows

    @override
    def train(self, data: dict[str, object], *, epochs: int = 10, batch_size: int = 64,
              learning_rate: float = 0.01, weight_decay: float = 0.0,
              betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
              random_state: int = 42, **params: object) -> 'LSTMBinary':
        _ = require_optional('torch', 'PyTorch', 'neural')
        import torch
        if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError('LSTMBinary epochs and batch_size must be positive integers')
        inputs = data['x_train']
        if not isinstance(inputs, pl.DataFrame):
            raise TypeError('LSTMBinary training requires named ret_1_lag_N columns')
        matches = sorted((int(match[1]), column) for column in inputs.columns
                         if (match := re.fullmatch(r'ret_1_lag_(\d+)', column)))
        if not matches or [lag for lag, _ in matches] != list(range(len(matches))):
            raise ValueError('LSTMBinary requires contiguous ret_1_lag_0..N columns')
        self._columns = [column for _, column in reversed(matches)]
        self._indices = [inputs.columns.index(column) for column in self._columns]
        self._width = inputs.width
        windows = self._windows(inputs)
        target = np.asarray(data['y_train'])
        if target.ndim != 1 or len(target) != len(windows) or not np.array_equal(np.unique(target), [0, 1]):
            raise ValueError('LSTMBinary requires aligned numeric binary training classes 0 and 1')
        self._mean, self._scale = float(windows.mean()), float(windows.std())
        if self._scale == 0:
            raise ValueError('LSTMBinary training returns have zero variance')
        features = torch.from_numpy(np.asarray((windows - self._mean) / self._scale, dtype=np.float32)[:, :, None])
        labels = torch.from_numpy(np.asarray(target, dtype=np.float32))
        generator = np.random.default_rng(random_state)
        with torch.random.fork_rng(devices=[]):
            seeded = torch.Generator(device='cpu')
            _ = seeded.manual_seed(random_state)
            torch.set_rng_state(seeded.get_state())
            self.model = _network(params)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                         weight_decay=weight_decay, betas=betas, eps=eps)
            for _epoch in range(epochs):
                order = generator.permutation(len(target))
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    optimizer.zero_grad()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(self.model(features[indices]), labels[indices])
                    loss.backward()
                    _ = optimizer.step()
        _ = self.model.eval()
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        if self.model is None:
            raise ValueError('LSTMBinary is not fitted')
        import torch
        windows = (self._windows(data['x_test']) - self._mean) / self._scale
        with torch.no_grad():
            logits = self.model(torch.from_numpy(np.asarray(windows, dtype=np.float32)[:, :, None]))
            probability = np.asarray(torch.sigmoid(logits).numpy(), dtype=np.float64)
        return {'_preds': (probability > _THRESHOLD).astype(np.int8), '_probs': probability}

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        prediction = self.predict(data)
        preds, probs = cast(npt.NDArray[np.int8], prediction['_preds']), cast(_FloatArray, prediction['_probs'])
        self._record_probabilities(data, prediction)
        result = binary_metrics(data, preds, probs)
        result['_preds'] = preds
        if inline_metrics:
            result.update(self._compute_confusion(preds, np.asarray(data['y_test']), data.get('price_data_for_backtest')))
            result.update(self._compute_backtest(preds, data))
        return result


def lstm_binary(data: dict[str, object], hidden_size: int = 16, num_layers: int = 1,
                bias: bool = True, dropout: float = 0.0, bidirectional: bool = False,
                proj_size: int = 0, epochs: int = 10, batch_size: int = 64,
                learning_rate: float = 0.01, weight_decay: float = 0.0,
                betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                random_state: int = 42) -> dict[str, object]:
    '''Fit a bounded CPU sequence classifier; return observed-label test metrics.'''
    model = LSTMBinary().train(data, hidden_size=hidden_size, num_layers=num_layers,
        bias=bias, dropout=dropout, bidirectional=bidirectional, proj_size=proj_size,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        weight_decay=weight_decay, betas=betas, eps=eps, random_state=random_state)
    result = model.evaluate(data)
    result['_model'] = model
    return result
