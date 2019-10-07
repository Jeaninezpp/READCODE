import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import numpy as np
import csv
from time import time
from sklearn.metrics import pairwise_distances_argmin
from traditional_clustering import basic_clustering, acc, nmi, ari


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class ApproximatePCA(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ApproximatePCA, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        if x.dim() != 2:
            x = x.view(x.size(0), -1)  # flatten to vector
        f_cluster = x - x.detach().mean(0)  # zero mean
        f_cluster = self.linear(f_cluster)  # transform
        return f_cluster


class SSDC(nn.Module):
    def __init__(self, in_channel=3, n_clusters=10, im_size=32, dropRate=0.0):
        super(SSDC, self).__init__()
        self.cluster_dim = 20
        self.n_clusters = n_clusters
        self.channels = in_channel
        self.pretrained = False

        self.feature = nn.Sequential()
        self.feature.add_module('conv1', nn.Conv2d(self.channels, 32, 3, padding=1))
        self.feature.add_module('bn1', nn.BatchNorm2d(32))
        self.feature.add_module('relu1', nn.ReLU(True))
        self.feature.add_module('drop1', nn.Dropout2d(dropRate, False))
        self.feature.add_module('conv2', nn.Conv2d(32, 32, 3, padding=1))
        self.feature.add_module('bn2', nn.BatchNorm2d(32))
        self.feature.add_module('relu2', nn.ReLU(True))
        self.feature.add_module('drop2', nn.Dropout2d(dropRate, False))
        self.feature.add_module('pool1', nn.MaxPool2d(2, 2))

        self.feature.add_module('conv3', nn.Conv2d(32, 64, 3, padding=1))
        self.feature.add_module('bn3', nn.BatchNorm2d(64))
        self.feature.add_module('relu3', nn.ReLU(True))
        self.feature.add_module('drop3', nn.Dropout2d(dropRate, False))
        self.feature.add_module('conv4', nn.Conv2d(64, 64, 3, padding=1))
        self.feature.add_module('bn4', nn.BatchNorm2d(64))
        self.feature.add_module('relu4', nn.ReLU(True))
        self.feature.add_module('drop4', nn.Dropout2d(dropRate, False))
        self.feature.add_module('pool2', nn.MaxPool2d(2, 2))
        self.feature.add_module('flatten', Flatten())

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc5', nn.Linear(64 * (im_size // 4) ** 2, 512))
        self.classifier.add_module('bn5', nn.BatchNorm1d(512))
        self.classifier.add_module('relu5', nn.ReLU(True))
        self.classifier.add_module('drop5', nn.Dropout2d(dropRate, False))
        self.classifier.add_module('fc6', nn.Linear(512, 4))

        self.cluster = ApproximatePCA(64 * (im_size // 4) ** 2, self.cluster_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = 1. / np.sqrt(m.weight.data.size(1))
                m.weight.data.uniform_(-n, n)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, path=('feature', 'cluster', 'classifier')):
        return_tuple = True
        if isinstance(path, str):
            path = (path,)
            return_tuple = False
        outputs = []
        h = self.feature(x)
        if 'feature' in path:
            outputs.append(h)
        if 'cluster' in path:
            outputs.append(self.cluster(h))
        if 'classifier' in path:
            out = self.classifier(h)
            outputs.append(out)

        return outputs if return_tuple else outputs[0]

    def predict(self, dataset, path=('feature', 'cluster', 'classifier')):
        device = next(self.parameters()).device
        loader = dataset if isinstance(dataset, DataLoader) else DataLoader(dataset, batch_size=64, shuffle=False,
                                                                            pin_memory=True, num_workers=1)
        if isinstance(path, str):
            path = (path,)
        self.eval()
        outputs = [[] for _ in range(len(path))]
        with torch.no_grad():
            for x, _ in loader:
                x = x[0] if isinstance(x, tuple) or isinstance(x, list) else x
                output = self(x.to(device), path)
                for i in range(len(path)):
                    outputs[i].append(output[i].to('cpu').numpy())
        for i in range(len(path)):
            outputs[i] = np.concatenate(outputs[i])
        return outputs if len(outputs) > 1 else outputs[0]

    def train_classifier(self, loader, args):
        print('Begin Training' + '-' * 70)
        device = next(self.parameters()).device

        logfile = open(args.save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss'])
        logwriter.writeheader()

        t0 = time()
        self.optimizer = optim.Adam(self.parameters(), args.lr)
        for epoch in range(args.pretrain_epochs):
            self.train()  # set to training mode
            ti = time()
            training_loss = 0.0
            for x, _ in loader:  # batch training
                bs = x[0].size(0)
                x = torch.cat(x, 0).to(device)
                y = torch.LongTensor([0] * bs + [1] * bs + [2] * bs + [3] * bs).to(device)

                self.optimizer.zero_grad()  # set gradients of optimizer to zero
                out = self(x, 'classifier')  # forward
                loss = F.cross_entropy(out, y)
                loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
                training_loss += loss.item() * bs  # record the batch loss
                self.optimizer.step()  # update the trainable parameters with computed gradients

            training_loss = training_loss / len(loader.dataset)
            print("==> Epoch %02d: loss=%.5f, time=%ds" % (epoch, training_loss, time() - ti))

            logwriter.writerow(dict(epoch=epoch, loss=training_loss))
            logfile.flush()
            if epoch > 0 and epoch % args.save_steps == 0:
                torch.save(self.state_dict(), args.save_dir + '/model_epoch%02d.pkl' % epoch)
        torch.save(self.state_dict(), args.save_dir + '/model_final.pkl')
        logwriter.writerow(dict(epoch='total time %ds' % (time() - t0)))
        logfile.close()
        self.pretrained = True
        print('Trained model saved to \'%s/model_final.pkl\'' % args.save_dir)
        print("Total time = %ds" % (time() - t0))
        print('End Classifier Training' + '-' * 70)

    def fit(self, loader, args, method='kmeans'):
        epochs, batch_size, save_dir, lr, gamma = args.epochs, args.batch_size, args.save_dir, args.lr, args.gamma
        device = next(self.parameters()).device

        # pretrain classifier
        if not self.pretrained:
            if args.model_file is None:
                self.train_classifier(loader, args)
            else:
                self.load_state_dict(torch.load(args.model_file))

        print('Initializing cluster centers...')
        features = self.predict(loader, 'feature')
        features = np.reshape(features, [features.shape[0], -1])

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.cluster_dim)
        features_pca = pca.fit_transform(features)
        self.cluster.linear.__setattr__('weight',
                                        nn.Parameter(torch.from_numpy(pca.components_).to(torch.float).to(device)))

        self.y_pred, self.centers = basic_clustering(features_pca, self.n_clusters, method)
        # self.centers = get_centers(features_pca, self.y_pred)

        # Step 2: deep clustering
        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dc_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        self.optimizer = optim.Adam(self.parameters(), lr)
        ti = time()
        training_loss = 0
        for epoch in range(epochs + 1):

            # shuffle data
            # idx = shuffle_data(loader, loader_trans)
            # self.y_pred = self.y_pred[idx]
            y = loader.dataset.targets if hasattr(loader.dataset, 'targets') else loader.dataset.labels
            y = np.array(y) if isinstance(y, list) or isinstance(y, np.ndarray) else y.numpy()

            # evaluate
            acc_val = np.round(acc(y, self.y_pred), 5)
            nmi_val = np.round(nmi(y, self.y_pred), 5)
            ari_val = np.round(ari(y, self.y_pred), 5)

            logdict = dict(epoch=epoch, acc=acc_val, nmi=nmi_val, ari=ari_val, loss=training_loss)
            logwriter.writerow(logdict)
            logfile.flush()
            print('Epoch %d: acc=%.5f, nmi=%.5f, ari=%.5f; loss=%.5f; time=%.1f' %
                  (epoch, acc_val, nmi_val, ari_val, training_loss, time() - ti))

            ti = time()
            target_centers = torch.FloatTensor(self.centers[self.y_pred])
            self.train()  # set to training mode
            training_loss = 0.0
            for iter, (x, _) in enumerate(loader):
                bs = x[0].size(0)
                x0 = x[0].to(device)
                x123 = torch.cat(x[1:], 0).to(device)
                y_class = torch.LongTensor([0] * bs + [1] * bs + [2] * bs + [3] * bs).to(device)
                y_cluster = target_centers[iter * batch_size: min(len(loader.dataset), (iter + 1) * batch_size)].to(
                    device)

                self.optimizer.zero_grad()  # set gradients of optimizer to zero
                [out_cluster, out_class0] = self(x0, ['cluster', 'classifier'])  # forward
                out_class123 = self(x123, 'classifier')
                out_class = torch.cat((out_class0, out_class123), 0)
                loss = gamma * F.mse_loss(out_cluster, y_cluster) + \
                       F.cross_entropy(out_class, y_class)

                loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
                training_loss += loss.item() * bs  # record the batch loss
                self.optimizer.step()  # update the trainable parameters with computed gradients

            training_loss = training_loss / len(loader.dataset)

            # update clustering result
            cluster_features = self.predict(loader.dataset, 'cluster')
            self.y_pred = pairwise_distances_argmin(cluster_features, self.centers)

        torch.save(self.state_dict(), save_dir + '/model_cluster.pkl')
        if not logfile.closed:
            logfile.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='usps', type=str,
                        choices=['usps', 'mnist', 'cifar10'])
    parser.add_argument('--ncluster', default=10, type=int)
    parser.add_argument('--model-file', default=None, type=str)
    parser.add_argument('--save-dir', default='result/temp', type=str)
    parser.add_argument('--pretrain-epochs', default=20, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--save-steps', default=50, type=int)
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # args.model_file = 'result/temp/model_final.pkl'
    device = torch.device("cuda")
    from utils import loaddata

    db_train, db_test = loaddata(args.db)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_train = DataLoader(db_train, batch_size=args.batch_size, shuffle=False, **kwargs)
    channels = db_test[0][0].size(0)
    im_size = db_test[0][0].size(1)
    model = SSDC(in_channel=channels, n_clusters=args.ncluster, im_size=im_size).to(device)
    model.fit(loader_train, args)
