from ml_collections import ConfigDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
    def __init__(self, encoder_input, decoder_input, latent_dim, annotation_size, size_segment, learned_prior=True):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder(encoder_input, 512, latent_dim)
        self.Decoder = Decoder(decoder_input, 512, 1)
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=learned_prior)
        self.log_var = torch.nn.Parameter(torch.zeros(latent_dim), requires_grad=learned_prior)
        self.annotation_size = annotation_size
        self.size_segment = size_segment
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
    def forward(self, s1, s2, y): # Batch x Ann x T x State, Batch x Ann x 1
        s1_ = s1.view(s1.shape[0], s1.shape[1], -1) # Batch x Ann x (T*State)
        s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
        y = y.reshape(s1.shape[0], s1.shape[1], -1) # Batch x Ann x 1

        encoder_input = torch.cat([s1_, s2_, y], dim=-1).view(s1.shape[0], -1) # Batch x Ann x (2*T*State + 1)
        mean, log_var = self.Encoder(encoder_input)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # Batch x Z
        z = z.repeat((1, self.annotation_size*self.size_segment)).view(-1, self.annotation_size, self.size_segment, z.shape[1]) #Batch x Ann x T x Z
        
        x0 = torch.cat([s1, z], dim=-1) #Batch x Ann x T x (State + Z)
        x1 = torch.cat([s2, z], dim=-1) #Batch x Ann x T x (State + Z)

        r0 = self.Decoder(x0) # Batch x Ann x T x 1
        r1 = self.Decoder(x1) # Batch x Ann x T x 1
        return r0, r1, mean, log_var

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
    
def get_latent(env, reward_model, dataset, label_type, mode, n=1):
    batch = dataset.get_mode_data(n)
    means = []
    for i in range(len(batch['observations'])):
        seg_reward_1, seg_reward_2 = env.get_preference_rewards(batch['observations'][i], batch['observations_2'][i], mode=mode)
        if label_type == 0: # perfectly rational
            sum_r_t_1 = np.sum(seg_reward_1, axis=1)
            sum_r_t_2 = np.sum(seg_reward_2, axis=1)
            binary_label = 1*(sum_r_t_1 < sum_r_t_2)
            rational_labels = np.zeros((len(binary_label), 2))
            rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        elif label_type == 1:
            sum_r_t_1 = np.sum(seg_reward_1, axis=1)
            sum_r_t_2 = np.sum(seg_reward_2, axis=1)
            binary_label = 1*(sum_r_t_1 < sum_r_t_2)
            rational_labels = np.zeros((len(binary_label), 2))
            rational_labels[np.arange(binary_label.size), binary_label] = 1.0
            margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= 0).reshape(-1)
            rational_labels[margin_index] = 0.5
        
        device = next(reward_model.parameters()).device
        obs1 = torch.from_numpy(batch['observations']).float().to(device)[None]
        obs2 = torch.from_numpy(batch['observations_2']).float().to(device)[None]
        labels = torch.from_numpy(rational_labels).float().to(device)[None]
        with torch.no_grad():
            mean, _ = reward_model.encode(obs1, obs2, labels)
            means.append(mean.cpu().numpy())
    return np.array(means)

def sample_latent(reward_model, n=1):
        return np.random.normal(
            loc=reward_model.mean.detach().cpu().numpy(), 
            scale=reward_model.log_var.detach().cpu().numpy(), 
            size=(n, reward_model.latent_dim)
        )