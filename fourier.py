from argparse import ArgumentParser
from os import listdir, makedirs, path

import numpy as np
import torch
from numpy import random
from numpy.fft import fft
from scipy.io import wavfile
from sklearn.cluster import KMeans
from torch import cuda, nn, optim
from torch.distributions import Categorical
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
    return ((arr1 - arr2) ** 2).sum()


def k_means(use_sklearn=True):
    class KM:
        def compute_distance(self, a, b):
            return (a - b) ** 2

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
                        self.centre[i] = sum(lst) / len(lst)

        def predict(self, X):
            p = np.zeros(len(X))
            for i, entry in enumerate(X):
                p[i] = np.argmin(
                    np.array(
                        [self.compute_distance(entry, point) for point in self.centre]
                    )
                )
            return p.astype("int")

        def fit_predict(self, X):
            self.fit(X)
            return self.predict(X)

    return KMeans if use_sklearn else KM


def history(dataset, module, device="cpu", rnn=True):
    outputs = []
    if rnn:
        states = [torch.zeros([1, 1, h], device=device) for h in module.hidden_sizes]
        for data, source, speaker in dataset:
            for d in data:
                d = d.view(1, 1, -1)
                out, states = module(d, states)
                outputs.append(out.detach().numpy().squeeze())
    else:
        for data, source, speaker in dataset:
            out = module(data)
            outputs.extend([o for o in out.detach().numpy()])
    return outputs


def one_hot(index, size):
    tensor = np.zeros(size)
    tensor[index] = 1
    return tensor.astype("float32")


class WavDataset(Dataset):
    def __init__(self, dirname, fft_s, device, dtype="float32"):
        super().__init__()
        self.device = device

        files = listdir(dirname)
        rate, data = [], []
        source, speaker = [], []
        for f in files:
            src = f[0]
            spk = int(f[1:4])
            r, d = wavfile.read(path.join(dirname, f))
            rate.append(r)
            data.append(d)
            source.append(src)
            speaker.append(spk)

        self.rate = sum(rate) // len(rate)
        assert not any([r - self.rate for r in rate])

        fft_algorithm, timesteps = fft_s
        self.data = []
        for entry in data:
            new_entry = []
            for i in range(0, len(entry) - timesteps + 1, timesteps):
                new_entry.append(fft_algorithm(entry[i : i + timesteps]))
            self.data.append(np.absolute(np.array(new_entry).astype(dtype)))
        self.data = [torch.tensor(d, device=device) for d in self.data]
        self.source = source
        speaker = self.reduce_speaker(speaker)
        self.speaker = [torch.tensor([s], device=device) for s in speaker]

    def __len__(self):
        assert len(self.data) == len(self.source) == len(self.speaker)
        return len(self.speaker)

    def __getitem__(self, index):
        return self.data[index], self.source[index], self.speaker[index]

    def reduce_speaker(self, speaker):
        speaker_list = []
        for i in speaker:
            if i in speaker_list:
                pass
            else:
                speaker_list.append(i)
        reduced = dict([(elem, i) for i, elem in enumerate(speaker_list)])
        for i in range(len(speaker)):
            speaker[i] = reduced[speaker[i]]
        return speaker


class FetchData:
    TRAIN = random.randn()
    TEST = random.randn()

    def __call__(self, dirname, mode, fft_s, device):
        if mode == self.TEST:
            dataset = WavDataset(path.join(dirname, "test"), fft_s, device)
        elif mode[0] == self.TRAIN:
            dataset = WavDataset(path.join(dirname, "train", mode[1]), fft_s, device)
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
                input_size = (
                    input_len * (num_layers - i) + output_len * i
                ) // num_layers
                hidden_size = (
                    input_len * (num_layers - i - 1) + output_len * (i + 1)
                ) // num_layers
                self.hidden_sizes.append(hidden_size)
                module_list.append(
                    nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1)
                )
            else:
                in_features = (
                    input_len * (num_layers - i) + output_len * i
                ) // num_layers
                out_features = (
                    input_len * (num_layers - i - 1) + output_len * (i + 1)
                ) // num_layers
                module_list.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(module_list)

    def forward(self, inputs, states=None):
        # inputs: if rnn: timesteps, N, out else: N,out
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


class Decoder(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = nn.Module(*args, **kwargs)
        self.linear = nn.Linear(
            in_features=module.input_len + 1, out_features=module.output_len
        )

    def forward(self, inputs, cls, states=None):
        outputs = self.module(inputs, states)
        net = torch.cat([outputs, cls], -1)
        return self.linear(net)


# training loops
def train_vae(data, from_speaker, enc, dec, cls1, rnn=True, device="cpu", noclass=True):
    enc, enc_optim = enc
    dec, dec_optim = dec
    cls1, cls1_optim = cls1
    if rnn:
        enc_states = [torch.zeros([1, 1, h], device=device) for h in enc.hidden_sizes]
        enc_out, _ = enc(data, enc_states)
        dec_states = [torch.zeros([1, 1, h], device=device) for h in dec.hidden_sizes]
        dec_out, _ = dec(enc_out, from_speaker, dec_states)
        cls_states = [torch.zeros([1, 1, h], device=device) for h in cls1.hidden_sizes]
        cls_out, _ = cls1(enc_out, cls_states)

    else:
        enc_out = enc(data)
        dec_out = dec(enc_out)
        cls_states = [torch.zeros([1, 1, h]) for h in cls1.hidden_sizes]
        cls_out, _ = cls1(enc_out.unsqueeze(1), [s.clone() for s in cls_states])

    if not noclass:
        closs = F.cross_entropy(cls_out[-1], from_speaker)

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


def train_gan(
    data, from_speaker, enc, dec, gen, cls2, dis, rnn=True, device="cpu", noclass=True
):
    enc, _ = enc
    dec, _ = dec
    gen, gen_optim = gen
    cls2, cls2_optim = cls2
    dis, dis_optim = dis
    if rnn:
        enc_states = [torch.zeros([1, 1, h], device=device) for h in enc.hidden_sizes]
        enc_out, _ = enc(data, enc_states)
        dec_states = [torch.zeros([1, 1, h], device=device) for h in dec.hidden_sizes]
        dec_out, _ = dec(enc_out, from_speaker, dec_states)
        gen_states = [torch.zeros([1, 1, h], device=device) for h in gen.hidden_sizes]
        gen_out, _ = gen(enc_out, gen_states)
        output = gen_out + dec_out
        cls_states = [torch.zeros([1, 1, h], device=device) for h in cls2.hidden_sizes]
        cls_out, _ = cls2(output, cls_states)
        dis_states = [torch.zeros([1, 1, h], device=device) for h in dis.hidden_sizes]
        dis_real, _ = dis(data, [d.clone() for d in dis_states])
        dis_fake, _ = dis(output, [d.clone() for d in dis_states])

    else:
        enc_out = enc(data)
        dec_out = dec(enc_out)
        gen_out = gen(enc_out)
        output = gen_out + dec_out
        cls_states = [torch.zeros([1, 1, h], device=device) for h in cls2.hidden_sizes]
        cls_out, _ = cls2(output.unsqueeze(1), cls_states)
        dis_states = [torch.zeros([1, 1, h], device=device) for h in dis.hidden_sizes]
        dis_real, _ = dis(data.unsqueeze(1), [d.clone() for d in dis_states])
        dis_fake, _ = dis(output.unsqueeze(1), [d.clone() for d in dis_states])

    if not noclass:
        closs = F.cross_entropy(cls_out[-1], from_speaker)

        cls2_optim.zero_grad()
        closs.backward(retain_graph=True)
        cls2_optim.step()

    real_loss = F.binary_cross_entropy(
        torch.sigmoid(dis_real[-1]), torch.tensor([[1.0]], device=device)
    )
    fake_loss = F.binary_cross_entropy(
        torch.sigmoid(dis_fake[-1]), torch.tensor([[0.0]], device=device)
    )
    loss = real_loss + fake_loss

    dis_optim.zero_grad()
    loss.backward(retain_graph=True)
    dis_optim.step()

    if not noclass:
        loss = -closs - fake_loss
    else:
        loss = fake_loss
    gen_optim.zero_grad()
    loss.backward()
    gen_optim.step()


def train_ddec(train_data, enc, dec, target_dec, kmeans, rnn=True, device="cpu"):
    enc, _ = enc
    dec, _ = dec
    target_dec, td_optim = target_dec

    for data, source, speaker in train_data:
        if rnn:
            e_states = [torch.zeros([1, 1, h], device=device) for h in enc.hidden_sizes]
            d_states = [torch.zeros([1, 1, h], device=device) for h in dec.hidden_sizes]
            td_states = [
                torch.zeros([1, 1, h], device=device) for h in target_dec.hidden_sizes
            ]

            encoded, _ = enc(data.unsqueeze(1), e_states)
            cls = kmeans.predict(encoded.detach().numpy().squeeze(1))
            cls = torch.from_numpy(
                np.expand_dims(np.array([one_hot(c, n_clusters) for c in cls]), axis=1)
            ).to(device)
            output, _ = target_dec(cls, td_states)
            real_output, _ = dec(encoded, d_states)

        else:
            cls_list = []
            encoded = enc(data)
            cls = kmeans.predict(encoded.detach().numpy())
            cls = torch.from_numpy(np.array([one_hot(c, n_clusters) for c in cls])).to(
                device
            )
            output = target_dec(cls)
            real_output = dec(encoded)

        loss = F.mse_loss(real_output, output)

        td_optim.zero_grad()
        loss.backward()
        td_optim.step()


# trained using reinforcement learning
def shuffle(action_list, input_sentence):
    assert len(action_list) == len(input_sentence)
    res = []
    index = 0
    for i in range(len(action_list) - 1):
        if action_list[i].item() == 1:
            res.append(input_sentence[index : i + 1])
            index = i + 1
    if index != len(action_list):
        res.append(input_sentence[index:])
    random.shuffle(res)
    result = sum(res, [])
    return torch.stack(result, dim=0)


def train_eos(train_data, enc, eos, ldis, rnn=True, device="cpu"):
    enc, _ = enc
    eos, eos_optim = eos
    ldis, dis_optim = ldis

    for data, _, _ in train_data:
        temporal_output = []
        log_prob = []
        past_actions = []
        eos_states = [torch.zeros([1, 1, h], device=device) for h in eos.hidden_sizes]
        if rnn:
            enc_states = [
                torch.zeros([1, 1, h], device=device) for h in enc.hidden_sizes
            ]
            data = data.unsqueeze(1)
            for d in data:
                d = d.view(1, 1, -1)
                encoded, enc_states = enc(d, enc_states)
                h, eos_states = eos(encoded, eos_states)

                softmax_output = F.softmax(h, -1)
                dist = Categorical(softmax_output)
                action_taken = dist.sample()
                log_prob.append(dist.log_prob(action_taken).squeeze())
                temporal_output.append(encoded.squeeze())
                past_actions.append(action_taken)

        else:
            encoded = enc(data)
            for e in encoded:
                e = e.view(1, 1, -1)
                h, enc_states = eos(e, eos_states)
                softmax_output = F.softmax(h, -1)
                dist = Categorical(softmax_output)
                action_taken = dist.sample()
                log_prob.append(dist.log_prob(action_taken).squeeze())
                temporal_output.append(e.squeeze())
                past_actions.append(action_taken)
        shuffled_output = shuffle(past_actions, temporal_output).unsqueeze_(1)

        if rnn:
            enc_states = [
                torch.zeros([1, 1, h], device=device) for h in enc.hidden_sizes
            ]
            encoded, _ = enc(data, enc_states)
        else:
            encoded = enc(data)
            encoded = encoded.unsqueeze_(1)
        dis_states = [torch.zeros([1, 1, h], device=device) for h in ldis.hidden_sizes]
        F_output, _ = ldis(shuffled_output, dis_states.copy())
        T_output, _ = ldis(encoded, dis_states.copy())

        loss = F.binary_cross_entropy(
            torch.sigmoid(F_output[-1]).sum(), torch.tensor(0.0, device=device)
        ) + F.binary_cross_entropy(
            torch.sigmoid(T_output[-1]).sum(), torch.tensor(0.0, device=device)
        )
        dis_optim.zero_grad()
        loss.backward(retain_graph=True)
        dis_optim.step()

        loss = torch.tensor(0.0, device=device)
        for log_p in log_prob:
            loss -= log_p * torch.sigmoid(F_output[-1]).sum().item()
        eos_optim.zero_grad()
        loss.backward()
        eos_optim.step()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-E", "--episodes", type=int, required=True)
    parser.add_argument("-t", "--timesteps", type=int, default=None)
    parser.add_argument("-s", "--stepspersec", type=int, default=None)
    parser.add_argument("-dr", "--dir", type=str, required=True)
    parser.add_argument("-l", "--latent", type=int, default=200)
    parser.add_argument("-n", "--num_layers", type=int, default=3)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
    parser.add_argument("-dv", "--device", type=str, default="cuda")
    parser.add_argument("-ln", "--linear", action="store_true")
    parser.add_argument("-K", "--n_clusters", type=int, default=500)
    parser.add_argument("-ts", "--test", action="store_true")
    parser.add_argument("-nc", "--noclass", action="store_true")
    parser.add_argument("-wd", "--weight_dir", type=str, default="_weight_dir")
    parser.add_argument("-ds", "--dontsave", action="store_true")
    parser.add_argument("-rs", "--restart", action="store_true")
    args = parser.parse_args()

    epochs = args.epochs
    episodes = args.episodes
    timesteps = args.timesteps
    stepspersec = args.stepspersec
    latent = args.latent
    device = args.device if cuda.is_available() else "cpu"
    dirname = args.dir
    lr = args.learning_rate
    rnn = not args.linear
    num_layers = args.num_layers
    n_clusters = args.n_clusters
    is_test = args.test
    noclass = args.noclass
    weight_dir = args.weight_dir
    save = not args.dontsave
    restart = args.restart

    if save:
        makedirs(weight_dir, exist_ok=True)

    fetch = FetchData()
    if not xor(timesteps, stepspersec):
        raise ValueError("not timesteps xor stepspersec")
    if is_test:
        train_data = fetch(dirname, FetchData.TEST, (fft, timesteps), device)
    else:
        train_data = fetch(dirname, [FetchData.TRAIN, "unit"], (fft, timesteps), device)
    if timesteps:
        stepspersec = train_data.rate // timesteps
    elif stepspersec:
        timesteps = train_data.rate // stepspersec

    speaker_dict = {}
    for speaker_num in train_data.speaker:
        try:
            speaker_dict[speaker_num] += 1
        except KeyError:
            speaker_dict[speaker_num] = 1
    n_speakers = len([n for n in speaker_dict.keys()])

    enc_path = path.join(weight_dir, "enc.pt")
    enc = Module(
        input_len=timesteps, output_len=latent, rnn=rnn, num_layers=num_layers
    ).to(device)

    dec_path = path.join(weight_dir, "dec.pt")
    dec = Decoder(
        input_len=latent, output_len=timesteps, rnn=rnn, num_layers=num_layers
    )

    dis_path = path.join(weight_dir, "dis.pt")
    dis = Module(input_len=timesteps, output_len=1, rnn=True, num_layers=num_layers)

    gen_path = path.join(weight_dir, "gen.pt")
    gen = Module(input_len=latent, output_len=timesteps, rnn=rnn, num_layers=num_layers)

    cls1_path = path.join(weight_dir, "cls1.pt")
    cls1 = Module(
        input_len=latent, output_len=n_speakers, rnn=True, num_layers=num_layers
    )

    cls2_path = path.join(weight_dir, "cls2.pt")
    cls2 = Module(
        input_len=timesteps, output_len=n_speakers, rnn=True, num_layers=num_layers
    )
    eos_path = path.join(weight_dir, "eos.pt")
    eos = Module(input_len=latent, output_len=2, rnn=True, num_layers=num_layers)

    ldis_path = path.join(weight_dir, "ldis.pt")
    ldis = Module(input_len=latent, output_len=1, rnn=True, num_layers=num_layers)

    td_path = path.join(weight_dir, "td.pt")
    target_dec = Module(
        input_len=n_clusters, output_len=dec.output_len, num_layers=num_layers, rnn=rnn
    )

    enc_optim = optim.RMSprop(params=enc.parameters(), lr=lr)
    dec_optim = optim.RMSprop(params=dec.parameters(), lr=lr)
    dis_optim = optim.RMSprop(params=dis.parameters(), lr=lr)
    gen_optim = optim.RMSprop(params=gen.parameters(), lr=lr)
    cls1_optim = optim.RMSprop(params=cls1.parameters(), lr=lr)
    cls2_optim = optim.RMSprop(params=cls2.parameters(), lr=lr)
    eos_optim = optim.RMSprop(params=eos.parameters(), lr=lr)
    ldis_optim = optim.RMSprop(params=ldis.parameters(), lr=lr)
    td_optim = optim.RMSprop(params=target_dec.parameters(), lr=lr)

    if not restart:
        try:
            enc_dict = torch.load(enc_path, map_location=device)
            enc.load_state_dict(enc_dict["module"])
            enc_optim.load_state_dict(enc_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain enc")

        try:
            dec_dict = torch.load(dec_path, map_location=device)
            dec.load_state_dict(dec_dict["module"])
            dec_optim.load_state_dict(dec_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain dec")

        try:
            dis_dict = torch.load(dis_path, map_location=device)
            dis.load_state_dict(dis_dict["module"])
            dis_optim.load_state_dict(dis_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain dis")

        try:
            gen_dict = torch.load(gen_path, map_location=device)
            gen.load_state_dict(gen_dict["module"])
            gen_optim.load_state_dict(gen_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain gen")

        try:
            cls1_dict = torch.load(cls1_path, map_location=device)
            cls1.load_state_dict(cls1_dict["module"])
            cls1_optim.load_state_dict(cls1_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain cls1")

        try:
            cls2_dict = torch.load(cls2_path, map_location=device)
            cls2.load_state_dict(cls2_dict["module"])
            cls2_optim.load_state_dict(cls2_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain cls2")

        try:
            eos_dict = torch.load(eos_path, map_location=device)
            eos.load_state_dict(eos_dict["module"])
            eos_optim.load_state_dict(eos_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain eos")

        try:
            ldis_dict = torch.load(ldis_path, map_location=device)
            ldis.load_state_dict(ldis_dict["module"])
            ldis_optim.load_state_dict(ldis_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain ldis")

        try:
            td_dict = torch.load(td_path, map_location=device)
            target_dec.load_state_dict(td_dict["module"])
            td_optim.load_state_dict(td_dict["optim"])
        except (FileNotFoundError, RuntimeError):
            print("retrain target_dec")

    for epoch in range(1, epochs + 1):
        print("vae epoch: {}/{}".format(epoch, epochs))

        for index in range(len(train_data)):
            input_data = (
                train_data[index][0].unsqueeze(1) if rnn else train_data[index][0]
            )
            input_speaker = train_data[index][2]
            train_vae(
                input_data,
                input_speaker,
                [enc, enc_optim],
                [dec, dec_optim],
                [cls1, cls1_optim],
                rnn,
                device,
                noclass,
            )

        if save:
            torch.save(
                {"module": enc.state_dict(), "optim": enc_optim.state_dict()},
                f=enc_path,
            )
            torch.save(
                {"module": dec.state_dict(), "optim": dec_optim.state_dict()},
                f=dec_path,
            )
            torch.save(
                {"module": cls1.state_dict(), "optim": cls1_optim.state_dict()},
                f=cls1_path,
            )

    for epoch in range(1, epochs + 1):
        print("gan epoch: {}/{}".format(epoch, epochs))

        for index in range(len(train_data)):
            input_data = (
                train_data[index][0].unsqueeze(1) if rnn else train_data[index][0]
            )
            input_speaker = train_data[index][2]
            train_gan(
                input_data,
                input_speaker,
                [enc, enc_optim],
                [dec, dec_optim],
                [gen, gen_optim],
                [cls2, cls2_optim],
                [dis, dis_optim],
                rnn,
                device,
                noclass,
            )

        if save:
            torch.save(
                {"module": gen.state_dict(), "optim": gen_optim.state_dict()},
                f=gen_path,
            )
            torch.save(
                {"module": dis.state_dict(), "optim": dis_optim.state_dict()},
                f=dis_path,
            )
            torch.save(
                {"module": cls2.state_dict(), "optim": cls2_optim.state_dict()},
                f=cls2_path,
            )

    ht = history(train_data, enc, device=device, rnn=rnn)
    kmeans = k_means(True)(n_clusters=n_clusters)
    kmeans.fit(ht)

    for epoch in range(1, epochs + 1):
        print("target epoch: {}/{}".format(epoch, epochs))
        train_ddec(
            train_data,
            [enc, enc_optim],
            [dec, dec_optim],
            [target_dec, td_optim],
            kmeans,
            rnn,
            device,
        )

        if save:
            torch.save(
                {"module": target_dec.state_dict(), "optim": td_optim.state_dict()},
                f=td_path,
            )

    for epoch in range(1, episodes + 1):
        print("End of sentence episode: {}/{}".format(epoch, epochs))
        train_eos(
            train_data,
            [enc, enc_optim],
            [eos, eos_optim],
            [ldis, ldis_optim],
            rnn,
            device,
        )

        if save:
            torch.save(
                {"module": eos.state_dict(), "optim": eos_optim.state_dict()},
                f=eos_path,
            )
            torch.save(
                {"module": ldis.state_dict(), "optim": ldis_optim.state_dict()},
                f=ldis_path,
            )
