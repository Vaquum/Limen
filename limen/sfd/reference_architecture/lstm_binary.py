from collections.abc import Callable
from typing import cast

from typing_extensions import override

from limen._optional import require_optional
from limen.sfd.reference_architecture._sequence_binary import _Network, _WindowBinary

__all__ = ['LSTMBinary', 'lstm_binary']
_BIDIRECTIONAL = 2


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


class LSTMBinary(_WindowBinary):
    '''CPU LSTM over completed return windows, with train-only scalar normalization.'''

    def __init__(self) -> None:
        super().__init__()

    @override
    def _build_network(self, params: dict[str, object]) -> _Network:
        return _network(params)

    @override
    def train(self, data: dict[str, object], *, epochs: int = 10, batch_size: int = 64,
              learning_rate: float = 0.01, weight_decay: float = 0.0,
              betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
              random_state: int = 42, **params: object) -> 'LSTMBinary':
        _ = super().train(data, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas,
            eps=eps, random_state=random_state, **params)
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        return super().predict(data)

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        return super().evaluate(data, inline_metrics)


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
