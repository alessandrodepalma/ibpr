import jax.numpy as jnp
import numpy as np
from torch.utils import data
import torchvision
import torchvision.transforms as transforms


def get_mean_sigma(dataset):
    if dataset == 'cifar10':
        mean = jnp.reshape(jnp.array([0.4914, 0.4822, 0.4465]), (1, 1, 1, 3))
        sigma = jnp.reshape(jnp.array([0.2023, 0.1994, 0.2010]), (1, 1, 1, 3))
    else:
        mean = jnp.reshape(jnp.array([0.1307]), (1, 1, 1, 1))
        sigma = jnp.reshape(jnp.array([0.3081]), (1, 1, 1, 1))
    return mean, sigma


def get_mnist():
    transform_train = Cast()
    transform_test = Cast()
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_fashion():
    transform_train = Cast()
    transform_test = Cast()
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_svhn():
    train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    return train_set, test_set, 32, 3, 10


def get_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        Cast(),
    ])

    transform_test = transforms.Compose([
        Cast(),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 32, 3, 10


# Taken from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# Taken from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)


# Taken from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
class Cast(object):
    # From PIC image to numpy.
    def __init__(self, convert_to_0_1=True):
        super().__init__()
        self.convert_to_0_1 = convert_to_0_1

    def __call__(self, pic):
        img = np.array(pic, dtype=np.float32)
        img = img.reshape((pic.size[1], pic.size[0], len(pic.getbands())))
        if self.convert_to_0_1:
            # Convert from [0, 255] into [0, 1], as implicitly done by transforms.ToTensor() in the torch version
            img = img / 255.
        return img


def get_loaders(args, test_only=False):
    if args.dataset == 'cifar10':
        train_set, test_set, input_size, input_channels, n_class = get_cifar10()
    elif args.dataset == 'mnist':
        train_set, test_set, input_size, input_channels, n_class = get_mnist()
    elif args.dataset == 'fashion':
        train_set, test_set, input_size, input_channels, n_class = get_fashion()
    elif args.dataset == 'svhn':
        train_set, test_set, input_size, input_channels, n_class = get_svhn()
    else:
        raise NotImplementedError('Unknown dataset')

    if args.n_valid is not None:
        print('Using validation set of size %d!' % args.n_valid)
        # create a validation set (returned instead of the test set) with the last <args.n_valid> datapoints.
        test_set = train_set[-args.n_valid:]
        train_set = train_set[:-args.n_valid]

    if not test_only:
        train_loader = NumpyLoader(train_set, batch_size=args.train_batch, shuffle=True, num_workers=0, drop_last=True)
    else:
        train_loader = None
    test_loader = NumpyLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=0, drop_last=True)
    return len(train_set), train_loader, test_loader, input_size, input_channels, n_class


if __name__ == "__main__":
    # Example usage of the above.
    train_set, test_set, input_size, input_channels, n_class = get_mnist()
    test_loader = NumpyLoader(test_set, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

    for inputs, targets in test_loader:
        # the only solution with the PyTorch loader is to convert to Jax at this stage.
        inputs, targets = jnp.array(inputs), jnp.array(targets)
        print(type(inputs), inputs.shape)
