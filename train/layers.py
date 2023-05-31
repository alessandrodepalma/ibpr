import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax
from typing import Sequence, Any


class TorchInit(hk.initializers.Initializer):
    """
    Reproduce PyTorch's initialization (adapts UniformScaling).

    Initializes by sampling from a uniform distribution, but with the variance
    scaled by the inverse square root of the number of input units, multiplied by
    the scale.
    """
    def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        if len(shape) >= 2:
            # Save the weight's shape for use in the bias' initialization, as it's the case for PyTorch.
            self.weight_shape = shape
        input_size = np.product(self.weight_shape[:-1])
        max_val = 1./np.sqrt(input_size)
        return hk.initializers.RandomUniform(-max_val, max_val)(shape, dtype)


class Conv2d(hk.Module):

    def __init__(self, out_channels, kernel_size, stride=1, padding='VALID', bias=True):
        super().__init__()
        torch_initializer = TorchInit()
        self.conv = hk.Conv2D(
            output_channels=out_channels, kernel_shape=kernel_size, stride=stride, padding=padding, with_bias=bias,
            w_init=torch_initializer,
            b_init=torch_initializer,
        )

    def __call__(self, x):
        return self.conv(x)


class Sequential(hk.Module):

    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.size = 0
        # in order to avoid problems in haiku, register the layers as separate attributes like "layer_0"
        for clayer in layers:
            self.__setattr__(f"layer_{self.size}", clayer)
            self.size += 1

    def forward_until(self, i, x):
        for i in range(i + 1):
            x = self.__getattribute__(f"layer_{i}")(x)
        return x

    def forward_from(self, i, x):
        for i in range(i + 1, self.size):
            x = self.__getattribute__(f"layer_{i}")(x)
        return x

    def total_abs_l1(self, x):
        ret = 0
        for i in range(self.size):
            x = self.__getattribute__(f"layer_{i}")(x)
            ret += x.l1()
        return ret

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.__getattribute__(f"layer_{i}")

    def __call__(self, x):
        for i in range(self.size):
            layer = self.__getattribute__(f"layer_{i}")
            x = layer(x)
        return x

    def forward(self, x, start=None, stop=None):
        if start is not None:
            return self.forward_from(start, x)
        elif stop is not None:
            return self.forward_until(stop, x)
        else:
            return self(x)


class ReLU(hk.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return jax.nn.relu(x)


class Flatten(hk.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x.reshape((x.shape[0], -1))


class Linear(hk.Module):

    def __init__(self, out_features, bias=True):
        super().__init__()
        torch_initializer = TorchInit()
        self.linear = hk.Linear(out_features, with_bias=bias, w_init=torch_initializer, b_init=torch_initializer)

    def __call__(self, x):
        return self.linear(x)


class Normalization(hk.Module):

    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = mean
        self.sigma = sigma

    def __call__(self, x):
        return (x - self.mean) / self.sigma


class Dropout(hk.Module):

    def __init__(self, p=0.5, train=False):
        # dropout is activated only if train=True
        super().__init__()
        self.p = p
        self.train = train

    def __call__(self, x):
        return hk.dropout(hk.next_rng_key(), self.p, x) if self.train else x
