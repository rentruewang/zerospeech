from argparse import ArgumentParser
from os import listdir, path

import numpy as np
import torch
from numpy import random
from numpy.fft import fft, rfft
from scipy.io import wavfile
from sklearn.cluster import KMeans
from torch import cuda, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# utilities


def xor(a, b):
    a = True if a else False
    b = True if b else False
    return not a == b


def pad(arr, length, pad_val):
    arr = list(arr)
    while len(arr) < length:
        arr.append(pad_val)
    return np.array(arr)


def squared_dist(arr1, arr2):
    return ((arr1-arr2)**2).sum()


def k_means(use_sklearn=True):
    class KM:
        def compute_distance(self, a, b):
            return (a-b)**2

        def __init__(self, n_clusters=8, max_iter=300):
            self.n_clusters = n_clusters
            self.max_iter = max_iter

        def fit(self, X):
            x_p = X.copy()
            random.shuffle(x_p)
            self.centre = [x_p[i] for i in range(int(self.n_clusters))]
            for _ in range(self.max_iter):
                args = self.predict(X)
                for i in range(len(self.centre)):
                    lst = [X[a] for a in args if a == i]
                    if len(lst) == 0:
                        random.shuffle(x_p)
                        self.centre[i] = x_p[0]
                    else:
                        self.centre[i] = sum(lst)/len(lst)

        def predict(self, X):
            p = np.zeros(len(X))
            for i, entry in enumerate(X):
                p[i] = np.argmin(np.array([self.compute_distance(
                    entry, point) for point in self.centre]))
            return p.astype('int')

        def fit_predict(self, X):
            self.fit(X)
            return self.predict(X)
    return KMeans if use_sklearn else KM


def history(dataset, module, device='cpu', rnn=True):
    outputs = []
    if rnn:
        states = [torch.zeros([1, 1, h], device=device)
                  for h in module.hidden_sizes]
        for data in dataset:
            for d in data:
                d = d.view(1, 1, -1)
                out, states = module(d, states)
                outputs.append(out.detach().numpy().squeeze())
    else:
        for data in dataset:
            out = module(data)
            outputs.extend([o for o in out.detach().numpy()])
    return outputs


def one_hot(index, size):
    tensor = np.zeros(size)
    tensor[index] = 1
    return tensor.astype('float32')


class WavDataset(Dataset):
    def __init__(self, dirname, fft_s, device, dtype='float32'):
        super().__init__()
        self.device = device

        files = listdir(dirname)
        rate, data = [], []
        for f in files:
            r, d = wavfile.read(path.join(dirname, f))
            rate.append(r)
            data.append(d)

        self.rate = sum(rate)//len(rate)
        assert not any([r-self.rate for r in rate])

        fft_algorithm, timesteps = fft_s
        self.data = []
        for entry in data:
            new_entry = []
            for i in range(0, len(entry)-timesteps+1, timesteps):
                new_entry.append(fft_algorithm(entry[i:i+timesteps]))
            self.data.append(np.array(new_entry).astype(dtype).real)
        self.data = [torch.tensor(d, device=device) for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class FetchData:
    TRAIN = random.randn()
    TEST = random.randn()

    def __call__(self, dirname, mode, fft_s, device):
        if mode == self.TEST:
            dataset = WavDataset(
                path.join(dirname, 'test'), fft_s, device)
        elif mode[0] == self.TRAIN:
            dataset = WavDataset(
                path.join(dirname, 'train', mode[1]), fft_s, device
            )
        return dataset


# modules
# `class Module` can be `Encoder`, `Decoder`, or even `Critic`
class Module(nn.Module):
    def __init__(self, input_len, output_len, num_layers=3, rnn=False, discrete=False):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_layers = num_layers
        self.rnn = rnn
        self.discrete = discrete
        module_list = []
        if rnn:
            self.hidden_sizes = []
        for i in range(num_layers):
            if rnn:
                input_size = (input_len*(num_layers-i) +
                              output_len*i)//num_layers
                hidden_size = (input_len*(num_layers-i-1) +
                               output_len*(i+1))//num_layers
                self.hidden_sizes.append(hidden_size)
                module_list.append(nn.GRU(input_size=input_size,
                                          hidden_size=hidden_size,
                                          num_layers=1))
            else:
                in_features = (input_len*(num_layers-i) +
                               output_len*i)//num_layers
                out_features = (input_len*(num_layers-i-1) +
                                output_len*(i+1))//num_layers
                module_list.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(module_list)

    def forward(self, inputs, states=None):
        # inputs: if rnn: timestep=1, N, out else: N,out
        if self.rnn:
            for i, layer in enumerate(self.layers[:-1]):
                inputs, states[i] = layer(inputs, states[i].detach())
                inputs = F.relu(inputs)
            output, states[-1] = self.layers[-1](inputs, states[-1])
            outputs = output, states
        else:
            for layer in self.layers[:-1]:
                inputs = F.relu(layer(inputs))
            outputs = self.layers[-1](inputs)
        return outputs


# training loops

def train_vae(data, from_speaker, enc, dec, cls1,
              rnn=True, device='cpu', noclass=True):
    enc, enc_optim = enc
    dec, dec_optim = dec
    cls1, cls1_optim = cls1
    if rnn:
        enc_states = [torch.zeros([1, 1, h], device=device)
                      for h in enc.hidden_sizes]
        enc_out, _ = enc(data, enc_states)
        dec_states = [torch.zeros([1, 1, h], device=device)
                      for h in dec.hidden_sizes]
        dec_out, _ = dec(enc_out, dec_states)
        cls_states = [torch.zeros([1, 1, h], device=device)
                      for h in cls1.hidden_sizes]
        cls_out, _ = cls1(enc_out, cls_states)

    else:
        enc_out = enc(data)
        dec_out = dec(enc_out)
        cls_states = [torch.zeros([1, 1, h]) for h in cls1.hidden_sizes]
        cls_out, _ = cls1(enc_out.unsqueeze(
            1), [s.clone() for s in cls_states])

    if not noclass:
        closs = F.cross_entropy(F.softmax(cls_out[-1], -1), from_speaker)

        cls1_optim.zero_grad()
        closs.backward(retain_graph=True)
        cls1_optim.step()

        loss = F.mse_loss(dec_out, data) - closs
    else:
        loss = F.mse_loss(dec_out, data)

    enc_optim.zero_grad()
    dec_optim.zero_grad()
    loss.backward()
    enc_optim.step()
    dec_optim.step()


def train_gan(data, from_speaker, enc, dec, gen, cls2, dis,
              rnn=True, device='cpu', noclass=True):
    enc, _ = enc
    dec, _ = dec
    gen, gen_optim = gen
    cls2, cls2_optim = cls2
    dis, dis_optim = dis
    if rnn:
        enc_states = [torch.zeros([1, 1, h], device=device)
                      for h in enc.hidden_sizes]
        enc_out, _ = enc(data, enc_states)
        dec_states = [torch.zeros([1, 1, h], device=device)
                      for h in dec.hidden_sizes]
        dec_out, _ = dec(enc_out, dec_states)
        gen_states = [torch.zeros([1, 1, h], device=device)
                      for h in gen.hidden_sizes]
        gen_out, _ = gen(enc_out, gen_states)
        output = gen_out + dec_out
        cls_states = [torch.zeros([1, 1, h], device=device)
                      for h in cls2.hidden_sizes]
        cls_out, _ = cls2(output, cls_states)
        dis_states = [torch.zeros([1, 1, h], device=device)
                      for h in dis.hidden_sizes]
        dis_real, _ = dis(data, [d.clone() for d in dis_states])
        dis_fake, _ = dis(output, [d.clone() for d in dis_states])

    else:
        enc_out = enc(data)
        dec_out = dec(enc_out)
        gen_out = gen(enc_out)
        output = gen_out + dec_out
        cls_states = [torch.zeros([1, 1, h], device=device)
                      for h in cls2.hidden_sizes]
        cls_out, _ = cls2(output.unsqueeze(1), cls_states)
        dis_states = [torch.zeros([1, 1, h], device=device)
                      for h in dis.hidden_sizes]
        dis_real, _ = dis(data.unsqueeze(1), [d.clone() for d in dis_states])
        dis_fake, _ = dis(output.unsqueeze(1), [d.clone() for d in dis_states])

    if not noclass:
        closs = F.cross_entropy(F.softmax(cls_out[-1], -1), from_speaker)

        cls2_optim.zero_grad()
        closs.backward(retain_graph=True)
        cls2_optim.step()

    real_loss = F.binary_cross_entropy(
        F.sigmoid(dis_real[-1]), torch.tensor([[1.]], device=device))
    fake_loss = F.binary_cross_entropy(
        F.sigmoid(dis_fake[-1]), torch.tensor([[0.]], device=device))
    loss = real_loss+fake_loss

    dis_optim.zero_grad()
    loss.backward(retain_graph=True)
    dis_optim.step()

    if not noclass:
        loss = -closs-fake_loss
    else:
        loss = fake_loss
    gen_optim.zero_grad()
    loss.backward()
    gen_optim.step()


def train_ddec(inputs, target, dec, rnn=True):
    dec, dec_optim = dec
    if rnn:
        states = [torch.zeros([1, 1, h]) for h in dec.hidden_sizes]
        output, _ = dec(inputs, states)

    else:
        output = dec(inputs)

    loss = F.mse_loss(output, target)

    dec_optim.zero_grad()
    loss.backward()
    dec_optim.step()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-t', '--timesteps', type=int, default=None)
    parser.add_argument('-s', '--stepspersec', type=int, default=None)
    parser.add_argument('-dr', '--dir', type=str, required=True)
    parser.add_argument('-l', '--latent', type=int, default=200)
    parser.add_argument('-n', '--num_layers', type=int, default=3)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-dv', '--device', type=str, default='cpu')
    parser.add_argument('-ln', '--linear', action='store_true')
    parser.add_argument('-rf', '--rfft', action='store_true')
    parser.add_argument('-ns', '--n_speakers', type=int, default=2)
    parser.add_argument('-K', '--n_clusters', type=int, default=500)
    parser.add_argument('-ts', '--test', action='store_true')
    args = parser.parse_args()

    epochs = args.epochs
    timesteps = args.timesteps
    stepspersec = args.stepspersec
    latent = args.latent
    device = args.device if cuda.is_available() else 'cpu'
    dirname = args.dir
    lr = args.learning_rate
    rnn = not args.linear
    num_layers = args.num_layers
    n_speakers = args.n_speakers
    n_clusters = args.n_clusters
    is_test = args.test

    fetch = FetchData()
    # train_data = fetch(
    #     dirname, [FetchData.TRAIN, 'unit'], (fft, timesteps), device)
    if not xor(timesteps, stepspersec):
        raise ValueError('not timesteps xor stepspersec')
    if is_test:
        train_data = fetch(dirname, FetchData.TEST,
                           (fft, timesteps), device)
    else:
        train_data = fetch(dirname, [FetchData.TRAIN, 'unit'],
                           (fft, timesteps), device)
    if timesteps:
        stepspersec = train_data.rate//timesteps
    elif stepspersec:
        timesteps = train_data.rate//stepspersec

    enc = Module(input_len=timesteps, output_len=latent,
                 rnn=rnn, num_layers=num_layers)
    dec = Module(input_len=latent, output_len=timesteps,
                 rnn=rnn, num_layers=num_layers)
    dis = Module(input_len=timesteps, output_len=1,
                 rnn=True, num_layers=num_layers)
    gen = Module(input_len=latent, output_len=timesteps,
                 rnn=rnn, num_layers=num_layers)
    cls1 = Module(input_len=latent, output_len=n_speakers,
                  rnn=True, num_layers=num_layers)
    cls2 = Module(input_len=timesteps, output_len=n_speakers,
                  rnn=True, num_layers=num_layers)
    toslice = Module(input_len=latent, output_len=n_speakers,
                     rnn=True, num_layers=num_layers)

    enc_optim = optim.RMSprop(params=enc.parameters(), lr=lr)
    dec_optim = optim.RMSprop(params=dec.parameters(), lr=lr)
    dis_optim = optim.RMSprop(params=dis.parameters(), lr=lr)
    gen_optim = optim.RMSprop(params=gen.parameters(), lr=lr)
    cls1_optim = optim.RMSprop(params=cls1.parameters(), lr=lr)
    cls2_optim = optim.RMSprop(params=cls2.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        print('vae epoch: {}/{}'.format(epoch, epochs))

        for index in range(len(train_data)):
            input_data = train_data[index].unsqueeze(
                1) if rnn else train_data[index]
            train_vae(input_data, None, [enc, enc_optim], [
                      dec, dec_optim], [cls1, cls1_optim], rnn, device, True)

    for epoch in range(1, epochs+1):
        print('gan epoch: {}/{}'.format(epoch, epochs))

        for index in range(len(train_data)):
            input_data = train_data[index].unsqueeze(
                1) if rnn else train_data[index]
            train_gan(input_data, None, [enc, enc_optim], [dec, dec_optim], [
                      gen, gen_optim], [cls2, cls2_optim], [dis, dis_optim],  rnn, device, True)

    ht = history(train_data, enc, device=device, rnn=rnn)
    kmeans = k_means(True)(n_clusters=n_clusters)
    kmeans.fit(ht)

    target_dec = Module(input_len=n_clusters, output_len=dec.output_len,
                        num_layers=num_layers, rnn=rnn)
    td_optim = optim.RMSprop(target_dec.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        print('target epoch: {}/{}'.format(epoch, epochs))

        for data in train_data:
            cls_list = []
            real_output = []
            output = []

            if rnn:
                e_states = [torch.zeros([1, 1, h], device=device)
                            for h in enc.hidden_sizes]
                d_states = [torch.zeros([1, 1, h], device=device)
                            for h in dec.hidden_sizes]
                td_states = [torch.zeros([1, 1, h], device=device)
                             for h in target_dec.hidden_sizes]
                for d in data:
                    d = d.view(1, 1, -1)
                    encoded, e_states = enc(d, e_states)
                    cls = kmeans.predict(encoded.detach().numpy().squeeze(1))
                    cls = torch.from_numpy(
                        one_hot(cls, n_clusters).reshape([1, 1, -1])).to(device)
                    d_out, d_states = dec(encoded, d_states)
                    real_output.append(d_out)
                    td_out, td_states = target_dec(cls, td_states)
                    output.append(td_out)

                real_output = torch.cat(real_output, dim=0)
                output = torch.cat(output, dim=0)

            else:
                encoded = enc(data)
                cls = kmeans.predict(encoded.detach().numpy().squeeze())
                cls_list.extend([torch.from_numpy(
                    one_hot(c, size=n_clusters)).to(device) for c in cls])
                output = target_dec(torch.stack(cls_list, dim=0))
                real_output = dec(encoded)

            loss = F.mse_loss(real_output, output)

            td_optim.zero_grad()
            loss.backward()
            td_optim.step()
