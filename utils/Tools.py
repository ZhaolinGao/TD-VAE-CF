import math
# import torch.nn.functional as F
from random import sample
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
ELK code referring to: https://github.com/schelotto/Gaussian_Word_Embedding/blob/master/main.py
'''

# pass in embeddings of one target keyphrase with the embeddings of all the keyphrases and compute the similarity score
# with all the keyphrases in the corpus
def compute_cosSimilarity(target_vector, all_vectors):
    return cosine_similarity(target_vector, all_vectors).flatten()


def gaussian_nll(mu, sigma, x):
    # return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
    se = 0.5 * torch.sum(torch.pow((x - mu), 2)) / (len(torch.nonzero(mu)) * (2 * torch.pow(sigma, 2))) + torch.log(sigma)
    return se / (len(mu))


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)  # plus |min|, relu activated

    return result_tensor


def apply_activation(act_name, x):
    if act_name == 'sigmoid':
        return torch.sigmoid(x)
    elif act_name == 'tanh':
        return torch.tanh(x)
    elif act_name == 'relu':
        return torch.relu(x)
    elif act_name == 'elu':
        return torch.elu(x)
    elif act_name == 'linear':
        return x
    else:
        raise NotImplementedError('Choose appropriate activation function. (current input: %s)' % act_name)


def activation_map(act_name):
    if act_name == 'sigmoid':
        return nn.Sigmoid()
    elif act_name == 'tanh':
        return nn.Tanh()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'elu':
        return nn.ELU()
    else:
        raise NotImplementedError('Choose appropriate activation function. (current input: %s)' % act_name)


def kernel_selection(method, mu_i, mu_j, var_i, var_j, c=1e-5):
    # so far only implemented kernels for Guassian distributions
    # could experiment with the ones for point vectors
    if method == 'ELK':
        return elk_metric(mu_i, mu_j, var_i, var_j, c)
    elif method == 'BK':
        return bk_metric(mu_i, mu_j, var_i, var_j, c)
    elif method == 'W2':
        return wasserstein2_distance(mu_i, mu_j, var_i, var_j)
    elif method == 'MB':
        return mahalanobis_distance(mu_i, mu_j, var_i, var_j, c)
    elif method == 'Euclidean':
        return euclidean_distance(mu_i, mu_j)
    else:
        raise NotImplementedError('Choose appropriate kernel function. (current input: %s)' % method)


# Expected Likelihood Kernel (ELK)
def elk_metric(mu_i, mu_j, var_i, var_j, c):
    """
    param mu_i: mu of word i: [batch, embed]
    param mu_j: mu of word j: [batch, embed]
    param var_i: variance of word i: [batch, embed]
    param var_j: variance of word j: [batch, embed]
    param exp: if apply exponentiation to the returned value
    param c: constant term added to denominator (since positive samples don't need exponentials)
    return: the energy function between the two batchs of  data: [batch]
    """
    embedding_shape = mu_i.shape[1]

    # print(var_i.device, var_j.device)
    # assertion of batch size equality
    assert mu_i.size()[0] == mu_j.size()[0]

    # log volume of ellipse
    det_fac = torch.sum(torch.log(var_i + var_j + c), 1)
    # det_fac = torch.sum(torch.log(var_i + var_j), 1)

    # mahalanobis distance between the means
    diff_mu = torch.sum((mu_i - mu_j) ** 2 / (var_i + var_j + c), 1)
    # diff_mu = torch.sum((mu_i - mu_j) ** 2 / (var_i + var_j), 1)

    # return torch.exp(0.5 * (det_fac + diff_mu))
    # returning the original value
    return -0.5 * (det_fac + diff_mu + embedding_shape * math.log(2 * math.pi))


# Bhattacharyya kernel
def bk_metric(mu_i, mu_j, var_i, var_j, c):
    # Computing the sigma values
    # sigma_i = var_i ** 0.5
    # sigma_j = var_j ** 0.5

    # sigma sum
    sigma_sum = 2 * torch.sum(torch.log(var_i ** 0.5 / (var_j ** 0.5 + c) + (var_j ** 0.5) / ((var_i ** 0.5) + c)), 1)

    # mahalanobis distance between the means
    diff_mu = torch.sum((mu_i - mu_j) ** 2 / (var_i + var_j + c), 1)

    return -0.25 * (sigma_sum + diff_mu)

# Mahalanobis distance
def mahalanobis_distance(mu_i, mu_j, var_i, var_j, c):
    # mahalanobis distance between the means
    ma_distance = torch.sum((mu_i - mu_j) ** 2 / (var_i + var_j + c), 1)

    return -1 * ma_distance


# 2-Wasserstein distance between Gaussian distributions
def wasserstein2_distance(mu_i, mu_j, var_i, var_j):
    # Computing the sigma values
    sigma_i = var_i ** 0.5
    sigma_j = var_j ** 0.5

    diff_mu = torch.sum((mu_i - mu_j) ** 2, 1)
    diff_sigma = torch.sum((sigma_i - sigma_j) ** 2, 1)

    return -(diff_mu + diff_sigma)


'''
    Similarity measures for mean embeddings only 
'''


def euclidean_distance(mu_i, mu_j):
    diff_mu = torch.sum((mu_i - mu_j) ** 2, 1)
    # return torch.cdist(mu_i, mu_j, p=2)
    return -diff_mu


class RunningAverage:
    def __init__(self):
        self.sum = 0
        self.history = []
        self.total = 0

    def update(self, value):
        self.sum += value
        self.history.append(value)
        self.total += 1

    @property
    def mean(self):
        return self.sum / self.total


# Need to chagne the sample number parameter
# Delete avoid self sampling parameter
def sampling(idx, matrix, uk_pos_num, uk_neg_num, kk_pos_num, kk_neg_num):
    # matrix_test
    # sample_method
    '''
    Args:
        idx: index used to locate user or keyphrase for sampling
        matrix: matrix uk or kk used to assist sampling for contrastive learning
        matrix_test: the test matrix used for evaluating performance, test positive samples should not be negative samples
        sample_number: number of positive and negative samples, equal number
        avoid_self_sample: needed when sampling for KK, not for UK

    Returns:
        pos_samples: an array that stores the positive sample idx [idx.shape[0] * sample_number]
        neg_samples: an array that stores the negative sample idx [idx.shape[0] * sample_number]
    '''

    # for the yelp_SIGIR dataset
    def random_sample():
        # for each anchoring user point in the batch
        for i in range(pos_entries.shape[0]):
            # negative entries would be the ones that are not positive.
            # if test_pos_entries is not None:
            #     neg_entry = list(set(range(matrix.shape[1])) - set(pos_entries[i]) - set(test_pos_entries[i]))
            # else:
            pos = np.random.choice(pos_entries[i], uk_pos_num).tolist()
            neg_entry = list(set(range(matrix.shape[1])) - set(pos))

            neg = np.random.choice(neg_entry, uk_neg_num).tolist()
            pos_samples.append(pos)
            neg_samples.append(neg)

            # for each anchoring keyphrase from the positive keyphrase
            # Need to handle the case where there's only 1 positive keyphrase
            for j in range(len(pos)):
                # sample from the previously defined lists.
                current_pos_candidates = pos_entries[i].copy()
                if pos[j] in current_pos_candidates:
                    current_pos_candidates.remove(pos[j])  # avoid self sampling
                pos_k = np.random.choice(current_pos_candidates, kk_pos_num).tolist()

                current_neg_candidates = list(set(range(matrix.shape[1])) - set(pos_k))
                if pos[j] in current_neg_candidates:
                    current_neg_candidates.remove(pos[j])  # avoid self sampling

                neg_k = np.random.choice(current_neg_candidates, kk_neg_num).tolist()
                pos_samples_kk.append(pos_k)
                neg_samples_kk.append(neg_k)

    # for the yelp_SIGIR dataset
    def experiment_random_sample():
        # for each anchoring user point in the batch
        for i in range(pos_entries.shape[0]):
            pos = np.random.choice(pos_entries[i], uk_pos_num).tolist()
            neg_entry = list(set(range(matrix.shape[1])) - set(pos))

            neg = np.random.choice(neg_entry, uk_neg_num).tolist()
            pos_samples.append(pos)
            neg_samples.append(neg)

            # for each anchoring keyphrase from the positive keyphrase
            # Need to handle the case where there's only 1 positive keyphrase
            for j in range(len(pos)):
                # sample from the previously defined lists.
                current_pos_candidates = pos.copy()
                if pos[j] in current_pos_candidates:
                    current_pos_candidates.remove(pos[j])  # avoid self sampling
                pos_k = np.random.choice(current_pos_candidates, kk_pos_num).tolist()

                current_neg_candidates = list(set(range(matrix.shape[1])) - set(pos_k))
                if pos[j] in current_neg_candidates:
                    current_neg_candidates.remove(pos[j])  # avoid self sampling

                neg_k = np.random.choice(current_neg_candidates, kk_neg_num).tolist()
                pos_samples_kk.append(pos_k)
                neg_samples_kk.append(neg_k)

    # for uk
    pos_samples = []  # m positive keyphrases
    neg_samples = []  # n negative keyphrasess

    # for kk
    pos_samples_kk = []  # m' positive keyphrasess
    neg_samples_kk = []  # n' negative keyphrases

    # gets an array of lists, which contains the positive column indices [idx.shape[0] * sample_numer]
    pos_entries = matrix[idx].tolil().rows

    # if matrix_test is not None:
    #     test_pos_entries = matrix_test[idx].tolil().rows
    # else:
    #     test_pos_entries = None

    # perform random sampling
    random_sample()
    # experiment_random_sample()

    # convert the samples to arrays
    pos_samples = np.array(pos_samples)
    neg_samples = np.array(neg_samples)

    pos_samples_kk = np.array(pos_samples_kk)
    neg_samples_kk = np.array(neg_samples_kk)

    # assertions
    assert pos_samples.shape[0] == pos_entries.shape[0]
    assert neg_samples.shape[0] == pos_samples.shape[0]
    try:
        assert pos_samples.shape[1] == uk_pos_num
    except:
        print('debugging')
    assert neg_samples.shape[1] == uk_neg_num
    assert pos_samples_kk.shape[0] == pos_entries.shape[0] * uk_pos_num
    assert neg_samples_kk.shape[0] == pos_entries.shape[0] * uk_pos_num
    assert pos_samples_kk.shape[1] == kk_pos_num
    assert neg_samples_kk.shape[1] == kk_neg_num

    # for arr_idx in range(len(pos_samples)):
    #     if test_pos_entries is not None and sample_method != 'simple_distribution':
    #         assert len(set(pos_samples[arr_idx]).intersection(test_pos_entries[arr_idx])) == 0
    #         assert len(set(neg_samples[arr_idx]).intersection(test_pos_entries[arr_idx])) == 0
    #     if avoid_self_sample:
    #         assert idx[arr_idx] not in pos_samples[arr_idx]
    #         assert idx[arr_idx] not in neg_samples[arr_idx]
    return pos_samples, neg_samples, pos_samples_kk, neg_samples_kk


# generate samples based on Gaussian embeddings
def generate_sample_embeddings(keyphrase_idx_check, sample_num, mean_embed, stdev_embed):
    # generate sample embeddings for all
    for index, kidx in enumerate(keyphrase_idx_check):
        # sample embedding for current keyphrase
        sample_embed = np.array([np.random.normal(mean_embed[kidx], stdev_embed[kidx])
                                 for _ in range(sample_num)])
        if index == 0:
            sampled_embedding = sample_embed.copy()
        else:
            sampled_embedding = np.vstack((sampled_embedding, sample_embed))

    # loop through the rest of 70 random samples
    for kidx in sample(range(mean_embed.shape[0]), 70):
        if kidx in keyphrase_idx_check:
            continue
        else:
            # sample embedding for current keyphrase
            sample_embed = np.array([np.random.normal(mean_embed[kidx], stdev_embed[kidx])
                                     for _ in range(sample_num)])
            sampled_embedding = np.vstack((sampled_embedding, sample_embed))

    return sampled_embedding


def logsumexp(pos_x, neg_x, num_pos_samples, num_neg_samples):
    # for the same anchoring point, concatenating smaples horizontally
    input_shape = pos_x.shape
    pos_x = pos_x.reshape(-1, num_pos_samples)
    neg_x = neg_x.reshape(-1, num_neg_samples)
    assert pos_x.shape[0] == neg_x.shape[0]
    concat = torch.cat((pos_x, neg_x), 1).detach()
    # find the maximum sampling point for each anchor [batch_size * 1]
    c, _ = torch.max(concat, dim=1, keepdim=True)

    pos_x_exp = torch.exp(pos_x - c)
    neg_x_exp = torch.exp(neg_x - c)

    sum_kernels = c + torch.log(torch.sum(pos_x_exp, axis=1, keepdim=True) + torch.sum(neg_x_exp, axis=1, keepdim=True))
    sum_kernels = torch.repeat_interleave(sum_kernels, repeats=num_pos_samples, dim=0)

    return sum_kernels.reshape(input_shape)  # summing of the denominator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count
