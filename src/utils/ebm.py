import torch
import random
import numpy as np 
from numpy import linalg as LA
from torch.nn import functional as F

class SampleBuffer:
    def __init__(self, max_samples=10000, train_feat=None, n_classes=5, p=0.95):
        self.max_samples = max_samples
        self.buffer = []
        self.n_classes = n_classes
        self.p = p
        self.train_feat = train_feat

    def __len__(self):
        return len(self.buffer)

    def push(self, samples, class_ids=None):
        samples = samples.detach().to('cpu')
        class_ids = class_ids.detach().to('cpu')

        for sample, class_id in zip(samples, class_ids):
            self.buffer.append((sample.detach(), class_id))

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples, device='cpu:0'):
        items = random.choices(self.buffer, k=n_samples)
        samples, class_ids = zip(*items)
        samples = torch.stack(samples, 0)
        class_ids = torch.tensor(class_ids)
        samples = samples.to(device)
        class_ids = class_ids.to(device)

        return samples, class_ids

    def init_sample(self, data_shape, data_device):
        """
            Initialize negative sample from either uniform distribution or feature of data in train set
        """
        if self.train_feat is None:
            return torch.rand(data_shape, device=data_device)
        else:
            n_samples = data_shape[0]
            idx = np.random.choice(self.train_feat.shape[0], n_samples)
            return torch.tensor(self.train_feat[idx], device=data_device)


    def sample_buffer(self, pos_data, norm_kwargs=None):
        batch_size = pos_data.size(0)
        device = pos_data.device

        if len(self.buffer) < 1:
            return (
                data_normalize(self.init_sample(pos_data.shape, device), norm_kwargs),
                torch.randint(0, self.n_classes, (batch_size,), device=device),
            )

        n_replay = (np.random.rand(batch_size) < self.p).sum()

        replay_sample, replay_id = self.get(n_replay)
        replay_sample, replay_id = replay_sample.to(device), replay_id.to(device)
        random_sample = data_normalize(self.init_sample((batch_size - n_replay, pos_data.size(1)), data_device=device), norm_kwargs)
        random_id = torch.randint(0, self.n_classes, (batch_size - n_replay,), device=device)

        return (
            torch.cat([replay_sample, random_sample], 0),
            torch.cat([replay_id, random_id], 0),
        )


# def sample_buffer(buffer, pos_data, n_classes=5, p=0.95, norm_kwargs=None):
#     batch_size = pos_data.size(0)
#     device = pos_data.device

#     if len(buffer) < 1:
#         return (
#             data_normalize(torch.rand(pos_data.shape, device=device), norm_kwargs),
#             torch.randint(0, n_classes, (batch_size,), device=device),
#         )

#     n_replay = (np.random.rand(batch_size) < p).sum()

#     replay_sample, replay_id = buffer.get(n_replay)
#     replay_sample, replay_id = replay_sample.to(device), replay_id.to(device)
#     random_sample = data_normalize(torch.rand(batch_size - n_replay, pos_data.size(1), device=device), norm_kwargs)
#     random_id = torch.randint(0, n_classes, (batch_size - n_replay,), device=device)

#     return (
#         torch.cat([replay_sample, random_sample], 0),
#         torch.cat([replay_id, random_id], 0),
#     )

def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)

def data_normalize(inp, kwargs):
    if kwargs is None:
        return inp 
        
    norm_type = kwargs['norm_type']
    if norm_type == 'L2N':
        return F.normalize(inp, p=2, dim=1)
    elif norm_type == 'CL2N':
        mean = kwargs['mean']
        return F.normalize(inp-mean, p=2, dim=1)
    else: #unnormalize
        return inp

def sample_ebm(model, replay_buffer, pos_data, step_size=0.001, sample_step=10, norm_kwargs=None):
    data_shape = pos_data.shape
    batch_size = data_shape[0]

    neg_data, neg_id = replay_buffer.sample_buffer(pos_data, norm_kwargs=norm_kwargs)
    neg_data.requires_grad = True 

    requires_grad(model.parameters(), False)
    model.eval() 

    for k in range(sample_step):
        noise = torch.randn(pos_data.shape, device=pos_data.device)
        noise.normal_(0, 0.0005)
        neg_data.data.add_(noise.data)

        neg_energy = -1.0 * torch.logsumexp(model(neg_data), 1)
        neg_energy.sum().backward()
        
        # TODO: update hyperparam
        neg_data.grad.data.clamp_(-1, 1)
        neg_data.data.add_(-step_size, neg_data.grad.data)

        neg_data.grad.detach_()
        neg_data.grad.zero_()

        # neg_data.data = data_normalize(neg_data.data, norm_kwargs)

    neg_data = neg_data.detach()

    requires_grad(model.parameters(), True)
    model.train()
    model.zero_grad()

    return neg_data, neg_id

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))
