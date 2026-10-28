from collections.abc import Callable
from typing import cast

from typing_extensions import override

from limen._optional import require_optional
from limen.sfd.reference_architecture._sequence_binary import Network as NetworkProtocol, WindowBinary

__all__ = ['CNN1DBinary', 'cnn1d_binary']


def _network(out_channels: int = 8, conv_layers: int = 1, kernel_size: int = 3,
             stride: int = 1, dilation: int = 1, bias: bool = True,
             activation: str = 'relu', pooling: str = 'mean', dropout: float = 0.0) -> NetworkProtocol:
    _ = require_optional('torch', 'PyTorch', 'neural')
    import torch
    from torch import Tensor, nn

    if type(conv_layers) is not int or conv_layers < 1:
        raise ValueError('CNN1DBinary conv_layers must be a positive integer')
    activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
    if activation not in activations or pooling not in ('mean', 'max'):
        raise ValueError('CNN1DBinary requires relu/tanh/gelu activation and mean/max pooling')

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            channels = 1
            for _layer in range(conv_layers):
                layers.extend((nn.Conv1d(channels, out_channels, kernel_size, stride=stride,
                    dilation=dilation, bias=bias, device='cpu', dtype=torch.float32), activations[activation]()))
                channels = out_channels
            self.blocks = nn.Sequential(*layers)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(out_channels, 1, device='cpu', dtype=torch.float32)

        @override
        def forward(self, inputs: Tensor) -> Tensor:
            temporal = cast(Tensor, self.blocks(inputs.transpose(1, 2)))
            pooled = temporal.mean(dim=2) if pooling == 'mean' else temporal.amax(dim=2)
            return cast(Tensor, self.head(self.dropout(pooled))).reshape(-1)

    return cast(NetworkProtocol, Network())


class CNN1DBinary(WindowBinary):
    '''CPU Conv1d over the time axis of completed raw return windows.'''

    def __init__(self) -> None:
        super().__init__()

    @override
    def _build_network(self, params: dict[str, object]) -> NetworkProtocol:
        return cast(Callable[..., NetworkProtocol], _network)(**params)

    @override
    def train(self, data: dict[str, object], *, epochs: int = 10, batch_size: int = 64,
              learning_rate: float = 0.01, weight_decay: float = 0.0,
              betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
              random_state: int = 42, **params: object) -> 'CNN1DBinary':
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


def cnn1d_binary(data: dict[str, object], out_channels: int = 8, conv_layers: int = 1,
                 kernel_size: int = 3, stride: int = 1, dilation: int = 1,
                 bias: bool = True, activation: str = 'relu', pooling: str = 'mean',
                 dropout: float = 0.0, epochs: int = 10, batch_size: int = 64,
                 learning_rate: float = 0.01, weight_decay: float = 0.0,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 random_state: int = 42) -> dict[str, object]:
    '''Fit a bounded temporal convolution classifier; score observed binary outcomes.'''
    model = CNN1DBinary().train(data, out_channels=out_channels, conv_layers=conv_layers,
        kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias,
        activation=activation, pooling=pooling, dropout=dropout, epochs=epochs,
        batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
        betas=betas, eps=eps, random_state=random_state)
    return model.evaluate(data) | {'_model': model}
