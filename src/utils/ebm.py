import torch
import numpy as np 


class SampleBuffer:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.buffer = []

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


def sample_buffer(buffer, pos_data, n_classes=5, p=0.95):
    batch_size = pos_data.size(0)
    device = pos_data.device

    if len(buffer) < 1:
        return (
            torch.rand(pos_data.shape, device=device),
            torch.randint(0, n_classes, (batch_size,), device=device),
        )

    n_replay = (np.random.rand(batch_size) < p).sum()

    replay_sample, replay_id = buffer.get(n_replay)
    random_sample = torch.rand(batch_size - n_replay, 3, 32, 32, device=device)
    random_id = torch.randint(0, n_classes, (batch_size - n_replay,), device=device)

    return (
        torch.cat([replay_sample, random_sample], 0),
        torch.cat([replay_id, random_id], 0),
    )


def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)

def sample_ebm(model, replay_buffer, pos_data, step_size=0.001, sample_step=20):
    data_shape = pos_data.shape
    batch_size = data_shape[0]

    neg_data, neg_id = sample_buffer(replay_buffer, batch_size)
    neg_data.requires_grad = True 

    requires_grad(model.parameters(), False)
    model.eval() 

    for k in range(sample_step):
        noise = torch.randn(pos_data.shape, device=pos_data.device)
        noise.normal_(0, 0.0005)
        neg_data.add_(noise)

        neg_energy = -1.0 * torch.logsumexp(model(neg_data))
        neg_energy.sum().backward()
        
        # TODO: update hyperparam
        neg_data.grad.data.clamp_(-0.01, 0.01)
        neg_data.data.add_(-step_size, neg_data.grad.data)

        neg_data.grad.detach_()
        neg_data.grad.zero_()

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