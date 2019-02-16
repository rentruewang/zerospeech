from argparse import ArgumentParser

import torch
from torch import cuda

from models import *
from utils import *


def slice_from_tensor(tensor, timesteps):
    tensor_slices = []
    for i in range(0, tensor.shape[1]-timesteps+1, timesteps):
        tensor_slices.append(tensor[:, i:i+timesteps])
    return tensor_slices


def extract(list_states, slices):
    return [list_states[i] for i in slices]


def train_on_batch(batch, timesteps, sample_var,
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
        interval.append([args1[i+1]-args1[i] for i in range(len(args1)-1)])
    print(interval[-1][-1])

    # nsyl
    nsyl_out = []
    for es in end_states:
        nsyl_out.append([nsyl(es[i]) for i in range(len(es)-1)])

    # fstv
    fstv_out = []
    for es, itv in zip(end_states, interval):
        lst = []
        for t in range(len(es)-1):
            a, b = es[t], es[t+1]
            # shape: num_layers, hidden_size
            a, b = a.unsqueeze(1), b.unsqueeze(1)
            lst.append(fstv(a, b, itv[t]))
        fstv_out.append(lst)


def test():

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
    train_on_batch(torch.randn(batch, timesteps), timesteps, 3,
                   vtl, ltv, cri, ltv, cri, cri, se, ns, fstv)
    print('test passed')


def main():
    test()
    parser = ArgumentParser()
    parser.add_argument('--test', action='store_true')


test()
# main()
