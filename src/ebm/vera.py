import torch
import torch.nn as nn
import torch.nn.functional as F 

from .model import VERAGenerator, MLPNetwork

def train_vera(model, generator, x_d, y_d):
    batch_size = 256
    cd_training = True 
    gen_training = True
    no_g_batch_norm = False
    p_control = 0
    n_control = 0
    pg_control = 0.1
    energy_weight = 1
    ent_weight = 0.0001
    lr = 0.001
    max_sigma = 0.3 
    min_sigma = 0.01
    noise_dim = 128
    g_inp_dim = 128
    g_out_dim = 128
    
    mlp = MLPNetwork(g_inp_dim, g_out_dim)
    generator = VERAGenerator(mlp, noise_dim)
    e_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(model.parameter(), lr=lr)
    
    x_g, h_g = generator.sample(batch_size, requires_grad=True)
    x_d.requires_grad_()  #TODO: Verify this
    
    if cd_training: # use contrastive divergence objective
        x_g_detach = x_g.detach().requires_grad_()
        if no_g_batch_norm:
            model.eval() 
        
        # energy of generative data
        lg_detach = model(x_g_detach).squeeze()

        if no_g_batch_norm:
            model.train()
        
        # energy and logits of true data     
        ld, ld_logits = model(x_d, return_logits=True)
        
        # FIXME: Check all the parameters
        grad_ld = torch.autograd.grad(ld.sum(), x_d, create_graph=True)[0].flatten(start_dim=1).norm(2, 1)

        # energy objective i.e. E(pos_data) - E(neg_data)
        logp_obj = (ld - lg_detach).mean()
        
        #TODO: Verify minimize energy 
        e_loss = logp_obj + \
                 (ld ** 2).mean() * p_control + \
                 (lg_detach ** 2).mean() * n_control + \
                 (grad_ld ** 2. / 2).mean() * pg_control # \
                 # unsup_ent.mean() * clf_ent_weight # wtf
                 
        # classify loss:
        ce_loss = F.cross_entropy(ld_logits, y_d)
        
        loss = ce_loss + e_loss * energy_weight
        
        e_optimizer.zero_grad()
        loss.backward()
        e_optimizer.step()
        
    # Train Generator
    if gen_training:
        lg = model(x_g).squeeeze()
        grad = torch.autograd.grad(lg.sum(), x_g, retain_graph=True)[0]
        ebm_gn = grad.norm(2, 1).mean()
        
        if ent_weight != 0:
            entropy_obj, ent_gn = generator.entropy_obj(x_g, h_g)
        
        logq_obj = lg.mean() + ent_weight * entropy_obj
        
        g_loss = - logq_obj
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        generator.clamp_sigma(max_sigma, min_sigma)
        