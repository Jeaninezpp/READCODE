import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision import datasets
from PIL import Image
import pickle
import os
import torch
from torchvision.datasets import VisionDataset


class MyRotation4times(object):
    def __call__(self, img):
        img = [F.rotate(img, angle) for angle in range(0, 271, 90)]
        return img


class MyCompose(transforms.Compose):
    def __call__(self, img):
        for t in self.transforms:
            if isinstance(img, list):
                img = [t(im) for im in img]
            else:
                img = t(img)
        return tuple(img) if isinstance(img, list) else img


class UspsDataset(datasets.VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, return_params=False):
        super(UspsDataset, self).__init__(root)
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.return_params = return_params

        self.create()
        filename = 'training.pt' if self.train else 'test.pt'

        self.data, self.targets = torch.load(os.path.join(self.root, filename))

    def create(self):
        if not os.path.exists(os.path.join(self.root, 'training.pt')):
            if not os.path.exists(self.root):
                os.makedirs(self.root)

            def load_usps(data_path='./data/usps'):
                with open(data_path + '/usps_train.jf') as f:
                    data = f.readlines()
                data = data[1:-1]
                data = [list(map(float, line.split())) for line in data]
                data = np.array(data)
                data_train, labels_train = data[:, 1:], data[:, 0]

                with open(data_path + '/usps_test.jf') as f:
                    data = f.readlines()
                data = data[1:-1]
                data = [list(map(float, line.split())) for line in data]
                data = np.array(data)
                data_test, labels_test = data[:, 1:], data[:, 0]

                x = np.concatenate((data_train, data_test)).astype('float32') * 255.0
                x = x.round().astype(np.uint8)
                y = np.concatenate((labels_train, labels_test)).astype(np.int64)
                x = x.reshape([-1, 16, 16])
                print('USPS samples', x.shape)
                return x, y

            x, y = load_usps(self.root)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            idx = torch.randperm(y.shape[0])
            x, y = x[idx], y[idx]
            with open(os.path.join(self.root, 'training.pt'), 'wb') as f:
                torch.save((x, y), f)
            with open(os.path.join(self.root, 'test.pt'), 'wb') as f:
                torch.save((x, y), f)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def loaddata(db='cifar10'):
    data_dir = 'data/' + db
    if db == 'cifar10':
        mean = (0.4914, 0.48216, 0.44653)
        std = (0.24703, 0.24349, 0.26159)
        db_train = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=MyCompose([
                                        MyRotation4times(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                    ]))
        db_test = datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    elif db == 'mnist':
        db_train = datasets.MNIST(data_dir, True, download=True,
                                  transform=MyCompose([
                                      transforms.Pad(2),
                                      MyRotation4times(),
                                      transforms.ToTensor()
                                  ]))
        db_test = datasets.MNIST(data_dir, False, download=True,
                                 transform=transforms.Compose([
                                     transforms.Pad(2),
                                     transforms.ToTensor()
                                 ]))
    elif db == 'usps':
        db_train = UspsDataset(data_dir, train=True, return_params=True,
                               transform=MyCompose([
                                   transforms.Pad(2),
                                   MyRotation4times(),
                                   transforms.ToTensor()]))
        db_test = UspsDataset(data_dir, train=False,
                              transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]))

    return db_train, db_test


if __name__ == "__main__":
    print('utils')
    db_train, db_test = loaddata('cifar10')
    print(len(db_train))
    from torch.utils.data import DataLoader

    loader = DataLoader(db_train, 5)
    for x, y in loader:
        print(len(x))
        print(x[0].size())
        print(y)
        break
