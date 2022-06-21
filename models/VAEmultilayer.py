"""
Dawen Liang et al., Variational Autoencoders for Collaborative Filtering. WWW 2018.
https://arxiv.org/pdf/1802.05814
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.BaseModel import BaseModel
from utils.Tools import activation_map, gaussian_nll

class VAEmultilayer(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device,
                 observation_std=0.01):
        super(VAEmultilayer, self).__init__()

        self.hidden_dim = model_conf.hidden_dim
        self.num_users = num_users
        self.num_items = num_items
        self.act = model_conf.act
        self.weighted_recon = model_conf.weighted_recon
        self.weight_decay = model_conf.weight_decay
        self.sparse_normalization = model_conf.sparse_normalization
        self.dropout_ratio = model_conf.dropout_ratio
        self.observation_std = observation_std
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(self.num_items, self.hidden_dim*4))
        self.encoder.append(activation_map(self.act))
        self.encoder.append(nn.Linear(self.hidden_dim*4, self.hidden_dim*2))
        for layer in self.encoder:
            if 'weight' in dir(layer):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                
        self.decoder = nn.Linear(self.hidden_dim, self.num_items, bias=False)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        # torch.nn.init.zeros_(self.decoder.bias)
        
        self.total_anneal_steps = model_conf.total_anneal_steps
        self.anneal_cap = model_conf.anneal_cap

        self.anneal = 0.
        self.update_count = 0
        self.device = device
        self.to(self.device)

    def forward(self, rating_matrix):
        # encoder
        mu_q, logvar_q = self.get_mu_logvar(rating_matrix)
        std_q = self.logvar2std(logvar_q)
        eps = torch.randn_like(std_q)   # reparametrization trick
        sampled_z = mu_q + self.training * eps * std_q  # apply reparameterization if in training mode?

        output = self.decoder(sampled_z)  # pass through the decoder

        if self.training:
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # not averaged yet
            # kl_loss = -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())
            kl_loss = -0.5 * torch.mean(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())
            return output, kl_loss
        else:  # evaluation mode
            return output

    def get_mu_logvar(self, rating_matrix):

        if self.training and self.dropout_ratio >0 :
            rating_matrix = F.dropout(rating_matrix, p=self.dropout_ratio) * (1 - self.dropout_ratio)

        if self.sparse_normalization:
            deno = torch.sum(rating_matrix>0, axis=1, keepdim=True) + 1e-5
            rating_matrix = rating_matrix / deno

        # un-embedded
        h = rating_matrix
        for layer in self.encoder:  # pass through encoder layer
            h = layer(h)
        mu_q = h[:, :self.hidden_dim]
        logvar_q = h[:, self.hidden_dim:]  # log sigmod^2  
        return mu_q, logvar_q

    def logvar2std(self, logvar):
        return torch.exp(0.5 * logvar)  # sigmod 

    def train_one_epoch(self, train_matrix, optimizer, batch_size, verbose, **kwargs):
        self.train()

        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))

        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)

            # used for assignment of beta value
            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap

            pred_matrix, kl_loss = self.forward(batch_matrix)

            '''Gaussian log-likelihood loss'''
            mask = batch_matrix != 0
            sigma = self.observation_std * torch.ones([], device=pred_matrix.device)
            # recon_loss = torch.sum(gaussian_nll(pred_matrix, sigma, batch_matrix) * mask) / torch.sum(mask)
            recon_loss = gaussian_nll(pred_matrix * mask, sigma, batch_matrix * mask)

            # for the unobserved entries
            mask0 = batch_matrix == 0
            sigma0 = self.observation_std * torch.ones([], device=pred_matrix.device)
            # recon_loss0 = torch.sum(gaussian_nll(pred_matrix, sigma0, batch_matrix) * mask0) / torch.sum(mask0)
            recon_loss0 = gaussian_nll(pred_matrix * mask0, sigma0, batch_matrix * mask0)

            # recon_loss = torch.sum(gaussian_nll(pred_matrix, sigma, batch_matrix) * mask)

            # l2 norm regularization, also regularizing the keyphrases' stdev embeddings
            l2_reg = torch.tensor(0., requires_grad=True)
            for layer in self.encoder:
                if 'weight' in dir(layer):
                    l2_reg = l2_reg + torch.norm(layer.weight)

            l2_reg = l2_reg + torch.norm(self.decoder.weight)

            # vae loss with annealing
            batch_loss = recon_loss + self.weighted_recon * recon_loss0\
                         + kl_loss * self.anneal\
                         + self.weight_decay * l2_reg

            batch_loss.backward()
            optimizer.step()

            self.update_count += 1

            loss += batch_loss
            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss.detach().cpu()

    # make predictions for recommendation
    def predict(self, input_matrix):
        '''
        Args:
            input_matrix: a input UI matrix
        Returns:
            pred_matrix: a predicted UI matrix
        '''
        with torch.no_grad():
            input_batch_matrix = torch.FloatTensor(input_matrix.toarray()).to(self.device)
            pred_batch_matrix = self.forward(input_batch_matrix).cpu().numpy()

        return pred_batch_matrix

    # def predict(self, input_matrix, test_matrix, test_batch_size):
    #     total_preds = []
    #     total_ys = []
    #     with torch.no_grad():
    #         num_data = input_matrix.shape[0]
    #         num_batches = int(np.ceil(num_data / test_batch_size))
    #         perm = list(range(num_data))
    #         for b in range(num_batches):
    #             if (b + 1) * test_batch_size >= num_data:
    #                 batch_idx = perm[b * test_batch_size:]
    #             else:
    #                 batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
    #
    #             input_batch_matrix = torch.FloatTensor(input_matrix[batch_idx].toarray()).to(self.device)
    #             test_batch_matrix = torch.FloatTensor(test_matrix[batch_idx].toarray())
    #
    #             pred_batch_matrix = self.forward(input_batch_matrix).cpu().numpy()
    #
    #             preds = pred_batch_matrix[test_batch_matrix != 0]
    #             ys = test_batch_matrix[test_batch_matrix != 0]
    #             if len(ys) > 0:
    #                 total_preds.append(preds)
    #                 total_ys.append(ys)
    #
    #     total_preds = np.concatenate(total_preds)
    #     total_ys = np.concatenate(total_ys)
    #
    #     return total_preds, total_ys