'''
This module generates random utterances.
'''
from argparse import ArgumentParser

import numpy as np
import torch
from numpy import random
from torch import cuda, nn
from torch.nn import functional as F


def halt_signals(indices, from_len, to_len, var):
    def map_end_of_syllable(index, from_len, to_len, var):
        co = 1 if random.uniform() < .5 else -1
        ratio = to_len / from_len
        lam = var * ratio
        return min(
            [max([int(index * ratio + co * random.poisson(lam)), 0]), to_len - 1])
    hs = []
    if isinstance(indices, int):
        return map_end_of_syllable(indices, from_len, to_len, var)
    for i in indices:
        hs.append(map_end_of_syllable(i, from_len, to_len, var))
    return np.array(hs)


class VoiceToLatent(nn.Module):
    def __init__(self, kernel_size=5, stride=2,
                 hidden_size=175, num_layers=2, bidirectional=False,
                 state_decay=.9, device='cpu'):
        super().__init__()
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=16,
                      kernel_size=kernel_size, stride=stride),
            nn.Conv1d(in_channels=16, out_channels=64,
                      kernel_size=kernel_size, stride=stride),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=kernel_size, stride=stride),
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=kernel_size, stride=stride),
            nn.Conv1d(in_channels=256, out_channels=512,
                      kernel_size=kernel_size, stride=stride),
            nn.Conv1d(in_channels=512, out_channels=512,
                      kernel_size=kernel_size, stride=stride),
            nn.Conv1d(in_channels=512, out_channels=512,
                      kernel_size=kernel_size, stride=stride)
        ])

        self.rnn_enc = nn.ModuleDict({
            'lin': nn.Linear(in_features=512, out_features=hidden_size),
            'rnn': nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional)
        })
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decay_factor = state_decay
        self.device = device

    def forward(self, inputs, states=None):
        # inputs: a batch of sequences -> shape: N, timesteps
        batch = inputs.shape[0]
        outputs = inputs.unsqueeze(1)
        for conv1d in self.conv1d_list:
            outputs = F.relu(conv1d(outputs), inplace=True)
        outputs = outputs.permute(2, 0, 1)
        states = states.detach() if states is not None else \
            torch.zeros(self.num_layers, batch, self.hidden_size)
        state_history = []
        output_history = []
        for i in range(len(outputs)):
            out = F.relu(self.rnn_enc['lin'](outputs[i:i + 1]))
            timestep_output, states = self.rnn_enc['rnn'](
                out, states * self.decay_factor)
            output_history.append(timestep_output)
            state_history.append(states)
        return outputs, torch.cat(output_history, dim=0), state_history


class LatentToVoice(nn.Module):
    def __init__(self, kernel_size=5, stride=2,
                 input_hidden_size=175, num_layers=2, bidirectional=False,
                 state_decay=.9, device='cpu'):
        super().__init__()
        self.conv1d_T = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=512, out_channels=512,
                               kernel_size=kernel_size, stride=stride),
            nn.ConvTranspose1d(in_channels=512, out_channels=512,
                               kernel_size=kernel_size, stride=stride),
            nn.ConvTranspose1d(in_channels=512, out_channels=256,
                               kernel_size=kernel_size, stride=stride),
            nn.ConvTranspose1d(in_channels=256, out_channels=128,
                               kernel_size=kernel_size, stride=stride),
            nn.ConvTranspose1d(in_channels=128, out_channels=64,
                               kernel_size=kernel_size, stride=stride),
            nn.ConvTranspose1d(in_channels=64, out_channels=16,
                               kernel_size=kernel_size, stride=stride),
            nn.ConvTranspose1d(in_channels=16, out_channels=1,
                               kernel_size=kernel_size, stride=stride)
        ])

        self.rnn_enc = nn.ModuleDict({
            'rnn': nn.GRU(input_size=input_hidden_size, hidden_size=input_hidden_size,
                          num_layers=num_layers, bidirectional=bidirectional),
            'lin': nn.Linear(in_features=input_hidden_size, out_features=512)
        })
        self.hidden_size = input_hidden_size
        self.num_layers = num_layers
        self.decay_factor = state_decay
        self.device = device

    def forward(self, inputs, rnn_inputs, states_list):
        # inputs: previous outputs -> shape: timesteps, N, channels
        assert len(inputs) == len(rnn_inputs) == len(states_list)
        l = len(inputs)
        states_history = []
        rnn_outputs = []
        for i in range(l):
            output, state = self.rnn_enc['rnn'](
                rnn_inputs[i:i + 1], states_list[i])
            output = self.rnn_enc['lin'](F.relu(output, inplace=True))
            rnn_outputs.append(output)
            states_history.append(state)
        outputs = (inputs + torch.cat(rnn_outputs, dim=0)).permute(1, 2, 0)
        for layer in self.conv1d_T[:-1]:
            outputs = F.relu(layer(outputs), inplace=True)
        outputs = self.conv1d_T[-1](outputs)
        return outputs, torch.cat(rnn_outputs, dim=0), states_history


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, categories=1, ns=.95):
        super().__init__()
        self.trans = nn.Linear(in_features=input_size,
                               out_features=hidden_size)
        module_list = [nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
                       for _ in range(num_layers)]
        self.rnn = nn.ModuleList(module_list)
        self.score = nn.Linear(in_features=hidden_size, out_features=1)
        self.hidden_size = hidden_size
        self.ns = ns

    def forward(self, inputs, states=None):
        # inputs: encoded content -> shape timesteps, N, channels
        states = states.detach() if states is not None else torch.zeros(
            1, inputs.shape[1], self.hidden_size)
        outputs = F.relu(self.trans(inputs))

        for layer in self.rnn:
            outputs, states = layer(outputs, states)
            outputs = F.leaky_relu(outputs, negative_slope=self.ns)
        final_score = F.sigmoid(self.score(states)).squeeze_()
        return final_score


class SyllableEnds(nn.Module):
    def __init__(self, state_num_layers, state_hidden_size):
        super().__init__()
        self.score = nn.Linear(
            in_features=state_num_layers * state_hidden_size, out_features=1)

    def forward(self, state_inputs):
        state_list = []
        for state in state_inputs:
            state_list.append(state.view(state.shape[1], -1))
        B = [int(s.shape[0]) for s in state_list]
        max_b = max(B)
        assert not any([max_b - b for b in B])
        del B
        return [F.sigmoid(self.score(s).squeeze_(-1)) for s in state_list]


class NextSyllable(nn.Module):
    def __init__(self, state_num_layers, state_hidden_size, hidden_size=50):
        super().__init__()
        self.map = nn.ModuleDict({
            'enc': nn.Linear(in_features=state_num_layers *
                             state_hidden_size, out_features=hidden_size),
            'dec': nn.Linear(out_features=state_num_layers *
                             state_hidden_size, in_features=hidden_size)
        })

    def forward(self, inputs, batch=False):
        # inputs: encoded states -> shape: num_layers, N, hidden if batch else
        # num_layers, hidden
        shapes = inputs.shape
        if batch:
            inputs = inputs.view(shapes[1], -1)
        else:
            inputs = inputs.view(-1)
        output = F.relu(self.map['enc'](inputs))

        return self.map['dec'](output).view(*shapes)


class FromStatesToVoice(nn.Module):
    def __init__(self, state_num_layers, state_hidden_size, hidden_size=50):
        super().__init__()
        self.compact = nn.Linear(in_features=state_hidden_size,
                                 out_features=hidden_size)
        self.scale = nn.ModuleList([
            nn.Linear(in_features=1, out_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=1)
        ])
        self.rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=state_num_layers)
        self.hidden_size = hidden_size
        self.num_layers = state_num_layers

    def forward(self, previous_state, next_state, timesteps):
        shapes = list(previous_state.shape)
        assert list(next_state.shape) == shapes
        compact_p = F.tanh(self.compact(previous_state))
        compact_n = F.tanh(self.compact(next_state))
        diff = compact_n - compact_p
        shapes[-1] = self.hidden_size
        states = diff.view(*shapes)
        voice = torch.zeros((1, shapes[1], 1))
        voices = []
        for _ in range(timesteps):
            voice = self.scale[0](voice)
            voice, states = self.rnn(voice, states)
            voice = self.scale[1](voice)
            voices.append(voice)
        return torch.cat(voices, dim=0)


def slice_from_tensor(tensor, timesteps):
    tensor_slices = []
    for i in range(0, tensor.shape[1] - timesteps + 1, timesteps):
        tensor_slices.append(tensor[:, i:i + timesteps])
    return tensor_slices


def extract(list_states, slices):
    return [list_states[i] for i in slices]


def debug():
    # debug
    debug = LatentToVoice
    device = 'cuda' if cuda.is_available() else 'cpu'

    batch = 47
    timesteps = 131
    kernel_size = 7
    num_layers = 2
    bidirectional = False
    hidden_size = 11
    stride = 1

    from_len = 137
    to_len = 355

    if debug == halt_signals:
        for i in range(100):
            r = random.randint(low=0, high=from_len)
            print(r)
            print(halt_signals(r, from_len, to_len, 1))
            print()
        r = random.randint(0, from_len, size=[100])
        print(r.shape)
        h = halt_signals(r, from_len, to_len, 1)
        print(h)
        print(h.shape)

    if debug == VoiceToLatent:
        vtl = VoiceToLatent(kernel_size=kernel_size, stride=stride,
                            hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, device=device)
        inputs = torch.randn(batch, timesteps)
        out, rout, rstate = vtl(inputs)
        print(out.shape)
        print(rout.shape)
        print(torch.stack(rstate, dim=0).shape)
        ostate = torch.randn(num_layers, batch, hidden_size)
        out, rout, rstate = vtl(inputs, ostate)
        print(out.shape)
        print(rout.shape)
        print(torch.stack(rstate, dim=0).shape)

    if debug == LatentToVoice:
        inputs = torch.randn(batch, timesteps)
        vtl = VoiceToLatent(kernel_size=kernel_size, stride=stride,
                            hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, device=device)
        out, rout, rstate = vtl(inputs)
        print(out.shape)
        print(rout.shape)
        print(torch.stack(rstate, dim=0).shape)
        ltv = LatentToVoice(kernel_size=kernel_size, stride=stride,
                            input_hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, device=device)
        out, rout, rstate = ltv(out, rout, rstate)
        print(out.shape)
        print(rout.shape)
        print(torch.stack(rstate, dim=0).shape)

    if debug == Critic:
        vtl = VoiceToLatent(kernel_size=kernel_size, stride=stride,
                            hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, device=device)
        inputs = torch.randn(batch, timesteps)
        out, rout, rstate = vtl(inputs)
        print(out.shape)
        print(rout.shape)
        print(torch.stack(rstate, dim=0).shape)
        cri = Critic(input_size=out.shape[-1], hidden_size=hidden_size,
                     num_layers=num_layers, categories=1)
        score = cri(out)
        print(score.shape)

    if debug == SyllableEnds:
        vtl = VoiceToLatent(kernel_size=kernel_size, stride=stride,
                            hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, device=device)
        inputs = torch.randn(batch, timesteps)
        out, rout, rstate = vtl(inputs)
        print(torch.stack(rstate, dim=0).shape)
        se = SyllableEnds(state_num_layers=num_layers,
                          state_hidden_size=hidden_size)
        sout = se(rstate)
        print(torch.stack(sout, dim=0).shape)

    if debug == NextSyllable:
        vtl = VoiceToLatent(kernel_size=kernel_size, stride=stride,
                            hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, device=device)
        inputs = torch.randn(batch, timesteps)
        out, rout, rstate = vtl(inputs)
        ns = NextSyllable(state_num_layers=num_layers,
                          state_hidden_size=hidden_size)
        nout = ns(rstate[0])
        print(nout.shape)

    if debug == FromStatesToVoice:
        vtl = VoiceToLatent(kernel_size=kernel_size, stride=stride,
                            hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, device=device)
        inputs = torch.randn(batch, timesteps)
        out, rout, rstate = vtl(inputs)
        fstv = FromStatesToVoice(state_num_layers=num_layers,
                                 state_hidden_size=hidden_size)
        fout = fstv(rstate[0], rstate[1], timesteps)
        print(fout.shape)


def test():
    def test_on_batch(batch, timesteps, sample_var,
                      enc, dec, cls_1, gen, dis, cls_2, agent, nsyl, fstv):
        # batch: a batch of sequence data -> shape: N, total_timesteps
        batch_size, timesteps = batch.shape

        # enc
        batch = slice_from_tensor(batch, timesteps)
        states = torch.zeros(
            (enc.num_layers, batch_size, enc.hidden_size), device=enc.device)
        enc_out, enc_rnnout, enc_states = [], [], []
        for B in batch:
            out, rnnout, states = enc(B, states=states)
            enc_out.append(out)
            enc_rnnout.append(rnnout)
            enc_states.extend(states)
        enc_out = torch.cat(enc_out, dim=0)
        enc_rnnout = torch.cat(enc_rnnout, dim=0)

        # dec
        dec_out, dec_rnnout, dec_states = dec(enc_out, enc_rnnout, enc_states)

        # cls_1
        cls_1_out = cls_1(enc_out)

        # gen
        gen_out, gen_rnnout, gen_states = gen(enc_out, enc_rnnout, enc_states)

        # dis
        dis_out = dis(enc_out)

        # cls_2
        cls_2_out = cls_2(enc_out)

        # agent
        agent_out = agent(enc_states)
        # agent_out: a list (len timesteps) of scores in batch

        states_tensor = torch.stack(
            enc_states, dim=0).permute(2, 0, 1, 3).contiguous()
        agent_out_tensor = torch.stack(
            agent_out, dim=0).round().int().permute(1, 0).numpy()
        agent_out_tensor = random.uniform(0, 1, agent_out_tensor.shape).round()

        end_states = []
        for s, a in zip(states_tensor, agent_out_tensor):
            end_states.append([s[i] for i in np.argwhere(a == 1).squeeze(-1)])

        interval = []
        for aot in agent_out_tensor:
            args1 = np.argwhere(aot == 1).squeeze(-1)
            interval.append([args1[i + 1] - args1[i]
                             for i in range(len(args1) - 1)])
        print(interval[-1][-1])

        # nsyl
        nsyl_out = []
        for es in end_states:
            nsyl_out.append([nsyl(es[i]) for i in range(len(es) - 1)])

        # fstv
        fstv_out = []
        for es, itv in zip(end_states, interval):
            lst = []
            for t in range(len(es) - 1):
                a, b = es[t], es[t + 1]
                # shape: num_layers, hidden_size
                a, b = a.unsqueeze(1), b.unsqueeze(1)
                lst.append(fstv(a, b, itv[t]))
            fstv_out.append(lst)

    device = 'cuda' if cuda.is_available() else 'cpu'
    batch = 47
    timesteps = 131
    kernel_size = 7
    num_layers = 2
    bidirectional = False
    hidden_size = 11
    stride = 1

    vtl = VoiceToLatent(kernel_size=kernel_size, stride=stride,
                        hidden_size=hidden_size, num_layers=num_layers,
                        bidirectional=bidirectional, device=device).to(device)
    ltv = LatentToVoice(kernel_size=kernel_size, stride=stride,
                        input_hidden_size=hidden_size, num_layers=num_layers,
                        bidirectional=bidirectional, device=device).to(device)
    inputs = torch.randn(batch, timesteps)
    out, rout, rstate = vtl(inputs)
    cri = Critic(input_size=out.shape[-1], hidden_size=hidden_size,
                 num_layers=num_layers, categories=1).to(device)
    se = SyllableEnds(state_num_layers=num_layers,
                      state_hidden_size=hidden_size).to(device)
    ns = NextSyllable(state_num_layers=num_layers,
                      state_hidden_size=hidden_size).to(device)
    fstv = FromStatesToVoice(state_num_layers=num_layers,
                             state_hidden_size=hidden_size).to(device)
    test_on_batch(torch.randn(batch, timesteps), timesteps, 3,
                  vtl, ltv, cri, ltv, cri, cri, se, ns, fstv)
    print('test passed')


debug()
test()
