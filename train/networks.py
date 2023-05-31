import haiku as hk
from train.layers import Conv2d, Normalization, ReLU, Flatten, Linear, Sequential, Dropout


class SeqNet(hk.Module):

    def __init__(self, layers):
        super().__init__()
        self.blocks = Sequential(layers)

    def __call__(self, x):
        x = self.blocks(x)
        return x

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x

    def forward(self, x, start=None, stop=None):
        return self.blocks.forward(x, start=start, stop=stop)


class FFNN(SeqNet):

    def __init__(self, mean, sigma, sizes, n_class=10):
        hk.Module.__init__(self)
        layers = [Normalization(mean, sigma), Flatten(), Linear(sizes[0]), ReLU()]
        for i in range(1, len(sizes)):
            layers += [
                Linear(sizes[i]),
                ReLU(),
            ]
        layers += [Linear(n_class)]
        super().__init__(layers)


class ConvMed(SeqNet):

    def __init__(self, mean, sigma, n_class=10, width1=1, width2=1, linear_size=100):
        hk.Module.__init__(self)
        layers = [
            Normalization(mean, sigma),
            Conv2d(16*width1, 5, stride=2, padding=(2, 2)),
            ReLU(),
            Conv2d(32*width2, 4, stride=2, padding=(1, 1)),
            ReLU(),
            Flatten(),
            Linear(linear_size),
            ReLU(),
            Linear(n_class),
        ]
        super().__init__(layers)

        
class ConvMedBig(SeqNet):

    def __init__(self, mean, sigma, n_class=10, width1=1, width2=1, width3=1, linear_size=100):
        hk.Module.__init__(self)
        layers = [
            Normalization(mean, sigma),
            Conv2d(16*width1, 3, stride=1, padding=(1, 1)),
            ReLU(),
            Conv2d(16*width2, 4, stride=2, padding=(1, 1)),
            ReLU(),
            Conv2d(32*width3, 4, stride=2, padding=(1, 1)),
            ReLU(),
            Flatten(),
            Linear(linear_size),
            ReLU(),
            Linear(n_class),
        ]
        super().__init__(layers)


# Deeper than ConvMedBig, yet fewer parameters due to the smaller last conv layer output
class ConvMedBigger(SeqNet):

    def __init__(self, mean, sigma, n_class=10, width1=1, width2=1, width3=1, linear_size=100, train=False,
                 dropout=False):
        hk.Module.__init__(self)
        layers = [
            Normalization(mean, sigma),
            Conv2d(16*width1, 3, stride=1, padding=(1, 1)),
            ReLU()
        ]

        if train and dropout:
            layers += [Dropout(train=train and dropout, p=0.5)]

        layers += [
            Conv2d(16*width2, 4, stride=2, padding=(1, 1)),
            ReLU(),
            Conv2d(32*width3, 4, stride=2, padding=(1, 1)),
            ReLU()
        ]

        if train and dropout:
            layers += [Dropout(train=train and dropout, p=0.5)]

        layers += [
            Conv2d(64*width3, 4, stride=2, padding=(1, 1)),
            ReLU(),
            Flatten(),
            Linear(linear_size),
            ReLU(),
            Linear(n_class)
        ]
        super().__init__(layers)


# DM-Large from the CROWN-IBP paper. 17190602 parameters, 229888 activations (significantly larger than ConvMedBigger)
class DMLarge(SeqNet):
    def __init__(self, mean, sigma, n_class=10):
        hk.Module.__init__(self)
        layers = [
            Normalization(mean, sigma),
            Conv2d(64, 3, stride=1, padding=(1, 1)),
            ReLU(),
            Conv2d(64, 3, stride=1, padding=(1, 1)),
            ReLU(),
            Conv2d(128, 3, stride=2, padding=(1, 1)),
            ReLU(),
            Conv2d(128, 3, stride=1, padding=(1, 1)),
            ReLU(),
            Conv2d(128, 3, stride=1, padding=(1, 1)),
            ReLU(),
            Flatten(),
            Linear(512),
            ReLU(),
            Linear(n_class),
        ]
        super().__init__(layers)


# Shallow (yet possibly wide) fully connected network. 70588010 parameters, 14009 activations
class FCShallow(SeqNet):
    def __init__(self, mean, sigma, lin_size, n_class=10):
        hk.Module.__init__(self)
        layers = [
            Normalization(mean, sigma),
            Flatten(),
            Linear(lin_size),
            ReLU(),
            Linear(lin_size),
            ReLU(),
            Linear(n_class),
        ]
        super().__init__(layers)
