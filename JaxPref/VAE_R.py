from ml_collections import ConfigDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from  JaxPref.flow import Flow

class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset['observations'])

    def __getitem__(self, idx):
        observations = self.pref_dataset['observations'][idx]
        observations_2 = self.pref_dataset['observations_2'][idx]
        labels = self.pref_dataset['labels'][idx]
        return dict(observations=observations, observations_2=observations_2, labels=labels)
    
    def get_mode_data(self, batch_size):
        idxs = np.random.choice(range(len(self)), size=batch_size, replace=False)
        return dict(observations=self.pref_dataset['observations'][idxs], 
                    observations_2=self.pref_dataset['observations_2'][idxs])
        

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(input_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = self.FC_output(h)
        return x_hat

class VAEModel(nn.Module):
    def __init__(self, encoder_input, decoder_input, 
                 latent_dim, hidden_dim, annotation_size, 
                 size_segment, learned_prior=False, flow_prior=False):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder(encoder_input, hidden_dim, latent_dim)
        self.Decoder = Decoder(decoder_input, hidden_dim, 1)
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=learned_prior)
        self.log_var = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=learned_prior)
        self.annotation_size = annotation_size
        self.size_segment = size_segment
        self.learned_prior = learned_prior

        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, 'planar', 4)
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def encode(self, s1, s2, y):
        s1_ = s1.view(s1.shape[0], s1.shape[1], -1) # Batch x Ann x (T*State)
        s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
        y = y.reshape(s1.shape[0], s1.shape[1], -1) # Batch x Ann x 1

        encoder_input = torch.cat([s1_, s2_, y], dim=-1).view(s1.shape[0], -1) # Batch x Ann x (2*T*State + 1)
        mean, log_var = self.Encoder(encoder_input)
        return mean, log_var
    
    def decode(self, obs, z):
        r = torch.cat([obs, z], dim=-1) #Batch x Ann x T x (State + Z)        
        r = self.Decoder(r) # Batch x Ann x T x 1
        return r

    def get_reward(self, r):
        r = self.Decoder(r) # Batch x Ann x T x 1
        return r

    def transform(self, mean, log_var):
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)
    
    def reconstruction_loss(self, x, x_hat):
        return nn.CrossEntropyLoss(reduction='sum')(x_hat, x)

    def accuracy(self, x, x_hat):
        predicted_class = torch.argmax(x_hat, dim=1)
        target_class = torch.argmax(x, dim=1)
        return torch.mean((predicted_class == target_class).float())

    def latent_loss(self, mean, log_var):
        if self.learned_prior:
            kl = - torch.sum(1+ (log_var-self.log_var) - (log_var-self.log_var).exp() - (mean.pow(2)-self.mean.pow(2))/(self.log_var.exp()))
        else:
            kl = - torch.sum(1. + log_var - mean.pow(2) - log_var.exp())
        return kl
    
    def forward(self, s1, s2, y, kl_weight): # Batch x Ann x T x State, Batch x Ann x 1
        mean, log_var = self.encode(s1, s2, y)
        # s1_ = s1.view(s1.shape[0], s1.shape[1], -1) # Batch x Ann x (T*State)
        # s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
        # y = y.reshape(s1.shape[0], s1.shape[1], -1) # Batch x Ann x 1

        # encoder_input = torch.cat([s1_, s2_, y], dim=-1).view(s1.shape[0], -1) # Batch x Ann x (2*T*State + 1)
        # mean, log_var = self.Encoder(encoder_input)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # Batch x Z
            log_det = None
        z = z.repeat((1, self.annotation_size*self.size_segment)).view(-1, self.annotation_size, self.size_segment, z.shape[1]) #Batch x Ann x T x Z

        r0 = self.decode(s1, z) # Batch x Ann x T x 1
        r1 = self.decode(s2, z) # Batch x Ann x T x 1
        
        r_hat1 = r0.sum(axis=2)
        r_hat2 = r1.sum(axis=2)

        r_hat = torch.cat([r_hat1, r_hat2], dim=-1).view(-1, 2)
        labels = y.view(-1, 2)

        reconstruction_loss = self.reconstruction_loss(labels, r_hat)
        accuracy = self.accuracy(labels, r_hat)
        latent_loss = self.latent_loss(mean, log_var)
        
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            'loss': loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'latent_loss': latent_loss.item(),
            'accuracy': accuracy.item(),
            'kl_weight': kl_weight
        }

        return loss, r0, r1, metrics

    def sample(self, size):
        """Generates samples from the prior.

        Args:
            size: number of samples to generate.
        Returns:
            generated samples.
        """
        z = torch.randn(size, self.latent_dim).cuda()
        if self.dataset == 'mnist':	
            return torch.sigmoid(self.decode(z))
        else:
            return self.decode(z)
        
class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return

def get_latent(env, reward_model, dataset, label_type, mode, n=1, return_mean=True):
    batch = dataset.get_mode_data(n)
    means = []
    for i in range(len(batch['observations'])):
        seg_reward_1, seg_reward_2 = env.get_preference_rewards(batch['observations'][i], batch['observations_2'][i], mode=mode)
        # sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        # sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        # binary_label = 1*(sum_r_t_1 > sum_r_t_2)
        # if label_type == 0: # perfectly rational
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1*(sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        # elif label_type == 1:
        #     sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        #     sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        #     binary_label = 1*(sum_r_t_1 < sum_r_t_2)
        #     rational_labels = np.zeros((len(binary_label), 2))
        #     rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        #     margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= 0).reshape(-1)
        #     rational_labels[margin_index] = 0.5
        
        device = next(reward_model.parameters()).device
        obs1 = torch.from_numpy(batch['observations'][i]).float().to(device)[None]
        obs2 = torch.from_numpy(batch['observations_2'][i]).float().to(device)[None]
        labels = torch.from_numpy(rational_labels).float().to(device)[None]
        with torch.no_grad():
            mean, _ = reward_model.encode(obs1, obs2, labels)
            means.append(mean.cpu().numpy())
    
    if return_mean:
        return np.array(means)
    else:
        return np.concatenate(means, axis=0)

def get_latent_from_env(env, reward_model, mode, n=8):
    means = []
    for i in range(n):
        obs1, obs2, seg_reward_1, seg_reward_2 = env.get_data_for_z(n, mode) 
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1*(sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        
        device = next(reward_model.parameters()).device
        obs1 = torch.from_numpy(obs1).float().to(device)[None]
        obs2 = torch.from_numpy(obs2).float().to(device)[None]
        labels = torch.from_numpy(rational_labels).float().to(device)[None]
        with torch.no_grad():
            mean, _ = reward_model.encode(obs1, obs2, labels)
            means.append(mean.cpu().numpy())
    return np.concatenate(means, axis=0)

def sample_latent(reward_model, n=1):
    return np.random.normal(
        loc=reward_model.mean.detach().cpu().numpy(), 
        scale=np.exp(0.5*reward_model.log_var.detach().cpu().numpy()), 
        size=(n, reward_model.latent_dim)
    )
