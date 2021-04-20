import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import linalg as LA
from .model import VERAGenerator, MLPNetwork


def train_vera(model, e_optimizer, generator, g_optimizer, x_d, y_d, args):
    batch_size = args['batch_size']
    cd_training = args['cd_training']
    gen_training = args['gen_training']
    p_control = args['p_control']
    n_control = args['n_control']
    pg_control = args['pg_control']
    energy_weight = args['energy_weight']
    ent_weight = args['ent_weight']
    max_sigma = args['max_sigma']
    min_sigma = args['min_sigma']

    x_g, h_g = generator.sample(batch_size, requires_grad=True)
    x_d.requires_grad_()

    if cd_training:  # use contrastive divergence objective
        x_g_detach = x_g.detach().requires_grad_()

        # energy of generative data
        lg_detach = model(x_g_detach).squeeze()

        # energy and logits of true data
        ld, ld_logits = model(x_d, return_logits=True)

        grad_ld = torch.autograd.grad(ld.sum(), x_d, create_graph=True)[
            0].flatten(start_dim=1).norm(2, 1)

        logp_obj = (ld - lg_detach).mean()

        e_loss = - logp_obj + \
            (ld ** 2).mean() * p_control + \
            (lg_detach ** 2).mean() * n_control + \
            (grad_ld ** 2. / 2).mean() * pg_control

        ce_loss = torch.nn.CrossEntropyLoss()(ld_logits, y_d)

        loss = ce_loss + e_loss * energy_weight

        e_optimizer.zero_grad()
        loss.backward()
        e_optimizer.step()

    # Train Generator
    if gen_training:
        lg = model(x_g).squeeeze()
        grad = torch.autograd.grad(lg.sum(), x_g, retain_graph=True)[0]
        # ebm_gn = grad.norm(2, 1).mean()

        if ent_weight != 0:
            entropy_obj, _ = generator.entropy_obj(x_g, h_g)

        logq_obj = lg.mean() + ent_weight * entropy_obj

        g_loss = - logq_obj

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        generator.clamp_sigma(max_sigma, min_sigma)
