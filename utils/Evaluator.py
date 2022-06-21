import numpy as np
import torch
from utils.Tools import kernel_selection
from utils.KAVgenerator import KAVgenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time

class Evaluator:
    def __init__(self, rec_atK, explain_atK, lamb, std):
        self.rec_atK = rec_atK  # a list of the topK indecies
        self.rec_maxK = max(self.rec_atK)
        self.explain_atK = explain_atK
        self.explain_maxK = max(self.explain_atK)

        self.global_metrics = {
            "R-Precision": r_precision,
            "NDCG": ndcg
        }

        self.local_metrics = {
            "Precision": precisionk,
            "Recall": recallk,
            "MAP": average_precisionk,
            "NDCG": ndcg
        }

        self.global_metrics_embeddings = {
            "UK_R-Precision": r_precision,
            "UK_NDCG": ndcg
        }

        self.local_metrics_embeddings = {
            "UK_NDCG": ndcg,
            "UK_Precision": precisionk,
            "UK_Recall": recallk,
            "UK_MAP": average_precisionk
        }

        self.lamb = lamb
        self.std = std

    # evaluate Gaussian embeddings, explanations and keyphraes relationships
    def evaluate_embeddings(self, model, train_matrix_uk, test_matrix_uk,
                            mu_user, var_user, ndcg_only, data_name, analytical=False):
        """
        Args:
            model: passed in model, e.g., VAE, VAE-contrast
            train_matrix_uk: test matrix of UK
            test_matrix_uk: test matrix of UK
            mu_user: mean embedding for users with all historical item entires known
            var_user: sigma embedding for users with all historical item entires known
            analytical: False if getting the confidence interval value

        Returns: a dictionary of metric scores
        """
        # switch to evaluation mode
        model.eval()
        model.before_evaluate()

        mu_user = torch.from_numpy(mu_user).to(model.device)
        var_user = torch.from_numpy(var_user).to(model.device)

        assert mu_user.shape[0] == train_matrix_uk.shape[0]
        assert torch.all(torch.gt(var_user, torch.zeros(size=var_user.size()).to(var_user.device)))

        with torch.no_grad():
            keyphrase_mean_embeddings = model.keyphrase_mu.weight.data
            keyphrase_var_embeddings = torch.exp(model.keyphrase_log_var.weight.data)

        # Get corresponding keyphrases predictions
        predicted_uk = self.kernel_predict(train_matrix_uk, test_matrix_uk, mu_user, var_user,
                                           keyphrase_mean_embeddings, keyphrase_var_embeddings,
                                           model.kernel_method, model.temperature_tau_u, data_name)

        uk_results = self.evaluation(predicted_uk, test_matrix_uk, eval_type='embeddings', ndcg_only=ndcg_only,
                                     analytical=analytical)

        return uk_results

    def kernel_predict(self, train_matrix, test_matrix, mu_anchor, var_anchor, mu_samples, var_samples,
                       kernel_method, temperature, data_name):

        # maxK = self.explain_maxK
        pos_entries = train_matrix.tolil().rows  # array of lists, to not consider
        # ground_entries = test_matrix.tolil().rows

        prediction = []
        for i in range(pos_entries.shape[0]):  # for each user

            # skipping the negative keyphrase cases
            if len(pos_entries[i]) == 0:
                prediction.append(np.zeros(self.explain_maxK, dtype=np.float32))

            else:
                # topK = max(len(ground_entries[i]), maxK)  # max of number of ground truth entries and topK
                # only care about those unk entries
                # if 'yelp_SIGIR' not in data_name:  # for the non-yelp_SIGIR datasets
                #     unk_entries = list(set(range(train_matrix.shape[1])))
                # else:
                #     unk_entries = list(set(range(train_matrix.shape[1])) - set(pos_entries[i]))
                unk_entries = list(set(range(train_matrix.shape[1])))
                mu_anchor_i = torch.repeat_interleave(mu_anchor[i].reshape(1, -1), repeats=len(unk_entries), dim=0).to(mu_anchor.device)
                var_anchor_i = torch.repeat_interleave(var_anchor[i].reshape(1, -1), repeats=len(unk_entries), dim=0).to(mu_anchor.device)

                # corresponding unknown keyphrases' embeddings
                mu_sample_i = mu_samples[unk_entries]
                var_sample_i = var_samples[unk_entries]

                assert mu_anchor_i.shape == mu_sample_i.shape
                assert var_anchor_i.shape == var_sample_i.shape

                # Becomes the predictions
                kernel = torch.divide(kernel_selection(kernel_method, mu_anchor_i,
                                                       mu_sample_i, var_anchor_i,
                                                       var_sample_i), temperature)

                # check kernel shape correspondence
                assert kernel.shape[0] == len(unk_entries)

                # select argmax
                top_index = (torch.argsort(kernel, dim=-1, descending=True)[:self.explain_maxK]).cpu().data.numpy()
                top_predict = np.array(unk_entries)[top_index]

                prediction.append(top_predict)

        # predicted item indecies
        # predicted_items = np.vstack(prediction)
        predicted_items = prediction.copy()
        assert len(predicted_items) == train_matrix.shape[0]
        return predicted_items

    def evaluate_recommendations(self, dataname, item_tag1, item_tag2, tagged_items, epoch, strong_tag1, strong_tag2, model, input_matrix, 
                                test_matrix, mse_only, ndcg_only, test_batch_size, analytical=False):
        # switch to evaluation mode
        model.eval()
        # operations before evaluation, does not perform for VAE models
        model.before_evaluate()

        # get prediction data, in matrix form
        # get prediction data, in matrix form, not masking, for recommendation results
        pred_matrix = model.predict(input_matrix)
        pred_matrix = np.array(pred_matrix)
        assert pred_matrix.shape == input_matrix.shape
        RMSE = round(np.sqrt(np.mean((input_matrix.toarray() - pred_matrix) ** 2)), 4)
        # preds, ys = model.predict(input_matrix, test_matrix, test_batch_size=test_batch_size)
        # RMSE = round(np.sqrt((np.sum((preds - ys) ** 2)) / len(ys)),4)

        if mse_only:
            recommendation_results = {"RMSE": (RMSE,0)}
        else:
            # get predicted item index
            prediction = []
            # get prediction data, in matrix form, not masking, for recommendation results
            # pred_matrix = model.simple_predict(input_matrix)
            # assert pred_matrix.shape == input_matrix.shape
            num_users = pred_matrix.shape[0]

            # Prediction section
            for user_index in range(num_users):
                vector_prediction = pred_matrix[user_index]
                vector_train = input_matrix[user_index]

                if len(vector_train.nonzero()[0]) > 0:
                    vector_predict = sub_routine(vector_prediction, vector_train, topK=self.rec_maxK)
                else:
                    vector_predict = np.zeros(self.rec_maxK, dtype=np.int64)

                prediction.append(vector_predict)

            # predicted item indecies
            predicted_items = prediction.copy()
            recommendation_results = self.evaluation(predicted_items, test_matrix, eval_type='recommendations',
                                                     ndcg_only=ndcg_only, analytical=analytical)
            recommendation_results["RMSE"] = (RMSE,0)

        # CAVs
        user_embeddings, user_logvar = model.get_mu_logvar(torch.FloatTensor(input_matrix.toarray()).to(model.device))
        user_embeddings = np.array(user_embeddings.detach().cpu().numpy())
        item_embeddings = model.decoder.weight.detach().cpu().numpy()
        if dataname == 'yelp_SIGIR':
            generator = KAVgenerator(item_embeddings[item_tag1], item_embeddings[item_tag2])
        elif dataname == 'reddit':
            generator = KAVgenerator(user_embeddings[strong_tag1], user_embeddings[strong_tag2])
        cavs = np.squeeze(generator.get_all_mean_cav(20, 10))
        metrics = ['Recall@5', 'Recall@10', 'Recall@20', 'Recall@50', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']

        # MMR
        num_users = pred_matrix.shape[0]
        temp = input_matrix.toarray()
        temp[temp > 0] = np.inf

        # MMR using CAV
        mmr_predictions = np.zeros((num_users, self.rec_maxK), dtype=np.int32)
        item_item_sim = np.dot(item_embeddings, cavs)
        user_item_sim = np.dot(user_embeddings, cavs)
        user_item_sim = -np.absolute(np.expand_dims(item_item_sim, 0) - np.expand_dims(user_item_sim, 1))-temp
        item_item_sim = -np.absolute(np.expand_dims(item_item_sim, 0) - np.expand_dims(item_item_sim, 1))
        for i in tqdm(range(mmr_predictions.shape[1])):
            if i == 0:
                mmr_predictions[:, 0] = np.argmax(user_item_sim, axis=1)
                user_item_sim[np.arange(num_users), mmr_predictions[:, 0]] = np.NINF
                continue
            for u in range(num_users):
                mmr_predictions[u, i] = np.argmax(self.lamb*user_item_sim[u]-(1-self.lamb)*np.max(item_item_sim[mmr_predictions[u, :i]], axis=0))
                user_item_sim[u, mmr_predictions[u, i]] = np.NINF
        mmr_results = self.evaluation(mmr_predictions, test_matrix, eval_type='recommendations',
                                             ndcg_only=ndcg_only, analytical=analytical)
        # get diversity for MMR
        _, s_precision_2, ks_test, prob1, prob2, prob_product = diversity_metric(mmr_predictions, tagged_items, input_matrix)
        print('T-MMR', self.lamb, ':', s_precision_2, ks_test, prob_product)
        for m in metrics:
            print(mmr_results[m][0])
        print('\n')

        # MMR using cosine sim
        mmr_predictions = np.zeros((num_users, self.rec_maxK), dtype=np.int32)
        user_item_sim = np.dot(user_embeddings, item_embeddings.T)-temp
        item_item_sim = np.dot(item_embeddings, item_embeddings.T)
        for i in tqdm(range(mmr_predictions.shape[1])):
            if i == 0:
                mmr_predictions[:, 0] = np.argmax(user_item_sim, axis=1)
                user_item_sim[np.arange(num_users), mmr_predictions[:, 0]] = np.NINF
                continue
            for u in range(num_users):
                mmr_predictions[u, i] = np.argmax(self.lamb*user_item_sim[u]-(1-self.lamb)*np.max(item_item_sim[mmr_predictions[u, :i]], axis=0))
                user_item_sim[u, mmr_predictions[u, i]] = np.NINF
        mmr_results = self.evaluation(mmr_predictions, test_matrix, eval_type='recommendations',
                                             ndcg_only=ndcg_only, analytical=analytical)
        # get diversity for MMR
        _, s_precision_2, ks_test, prob1, prob2, prob_product = diversity_metric(mmr_predictions, tagged_items, input_matrix)
        print('U-MMR', self.lamb, ':', s_precision_2, ks_test, prob_product)
        for m in metrics:
            print(mmr_results[m][0])
        print('\n')
        del temp
        del user_item_sim
        del item_item_sim

        # get diversity for VAE-CF
        _, s_precision_2, ks_test, prob1, prob2, prob_product = diversity_metric(prediction, tagged_items, input_matrix)
        print('VAE-CF:', s_precision_2, ks_test, prob_product)
        for m in metrics:
            print(recommendation_results[m][0])
        print('\n')

        # VAE-CF + CAV appendix for flatten the filter bubble
        cav_norm = np.sqrt(sum(cavs**2))
        new_user_embeddings = user_embeddings - (np.expand_dims(np.dot(user_embeddings, cavs), 1)/(cav_norm**2))*cavs

        new_predictions = None
        for i in range(10):
            div_new_user_embeddings = new_user_embeddings+np.random.normal(scale=self.std, size=(user_embeddings.shape[0], 1))
            if i == 0:
                new_predictions = np.dot(div_new_user_embeddings, item_embeddings.T)
            else:
                new_predictions += np.dot(div_new_user_embeddings, item_embeddings.T)
        new_predictions /= 10

        # get predicted item index
        new_prediction = []

        # Prediction section
        for user_index in range(num_users):
            vector_prediction = new_predictions[user_index]
            vector_train = input_matrix[user_index]

            if len(vector_train.nonzero()[0]) > 0:
                vector_predict = sub_routine(vector_prediction, vector_train, topK=self.rec_maxK)
            else:
                vector_predict = np.zeros(self.rec_maxK, dtype=np.int64)

            new_prediction.append(vector_predict)

        # predicted item indecies
        predicted_items = new_prediction.copy()
        new_results = self.evaluation(predicted_items, test_matrix, eval_type='recommendations',
                                                 ndcg_only=ndcg_only, analytical=analytical)

        # get diversity for CAV
        _, s_precision_2, ks_test, prob1, prob2, prob_product = diversity_metric(new_prediction, tagged_items, input_matrix)
        print('TD-VAE-CF Flatten', self.std, ':', s_precision_2, ks_test, prob_product)
        for m in metrics:
            print(new_results[m][0])
        print('\n')

        # VAE-CF + TCAV
        cav_norm = np.sqrt(sum(cavs**2))
        new_user_embeddings = user_embeddings - (1-self.lamb)*(np.expand_dims(np.dot(user_embeddings, cavs), 1)/(cav_norm**2))*cavs
        new_predictions = np.dot(new_user_embeddings, item_embeddings.T)

        # get predicted item index
        new_prediction = []

        # Prediction section
        for user_index in range(num_users):
            vector_prediction = new_predictions[user_index]
            vector_train = input_matrix[user_index]

            if len(vector_train.nonzero()[0]) > 0:
                vector_predict = sub_routine(vector_prediction, vector_train, topK=self.rec_maxK)
            else:
                vector_predict = np.zeros(self.rec_maxK, dtype=np.int64)

            new_prediction.append(vector_predict)

        # predicted item indecies
        predicted_items = new_prediction.copy()
        new_results = self.evaluation(predicted_items, test_matrix, eval_type='recommendations',
                                                 ndcg_only=ndcg_only, analytical=analytical)

        # get diversity for CAV
        _, s_precision_2, ks_test, prob1, prob2, prob_product = diversity_metric(new_prediction, tagged_items, input_matrix)
        print('TD-VAE-CF', self.lamb, ':', s_precision_2, ks_test, prob_product)
        for m in metrics:
            print(new_results[m][0])
        print('\n')

        return recommendation_results

    # function to perform evaluation on metrics
    def evaluation(self, predicted_items, test_matrix, eval_type, ndcg_only, analytical=False):
        if eval_type == 'recommendations' and ndcg_only:
            local_metrics = None
            global_metrics = {"NDCG": ndcg}
            atK = self.rec_atK
        elif eval_type == 'recommendations' and not ndcg_only:
            local_metrics = self.local_metrics
            global_metrics = self.global_metrics
            atK = self.rec_atK
        elif eval_type == 'embeddings'and ndcg_only:
            local_metrics = None
            global_metrics = {"UK_NDCG": ndcg}
            atK = self.explain_atK
        elif eval_type == 'embeddings' and not ndcg_only:
            local_metrics = self.local_metrics_embeddings
            global_metrics = self.global_metrics_embeddings
            atK = self.explain_atK
        else:
            raise NotImplementedError("Please select proper evaluation type, current choice: %s" % eval_type)

        num_users = test_matrix.shape[0]

        # evaluation section
        output = dict()

        # The @K metrics
        if local_metrics:
            for k in atK:
                results = {name: [] for name in local_metrics.keys()}

                # topK_Predict = predicted_items[:, :k]
                for user_index in range(num_users):
                    # vector_predict = topK_Predict[user_index]
                    vector_predict = predicted_items[user_index][:k]
                    if (len(vector_predict.nonzero()[0]) > 0):
                        vector_true_dense = test_matrix[user_index].nonzero()[1]

                        if vector_true_dense.size > 0:  # only if length of validation set is not 0
                            hits = np.isin(vector_predict, vector_true_dense)
                            for name in local_metrics.keys():
                                results[name].append(local_metrics[name](vector_true_dense=vector_true_dense,
                                                                         vector_predict=vector_predict,
                                                                         hits=hits))

                results_summary = dict()
                if analytical:
                    for name in local_metrics.keys():
                        results_summary['{0}@{1}'.format(name, k)] = results[name]
                else:
                    for name in local_metrics.keys():
                        results_summary['{0}@{1}'.format(name, k)] = (np.average(results[name]),
                                                                      1.96 * np.std(results[name]) / np.sqrt(
                                                                          len(results[name])))
                output.update(results_summary)

        # The global metrics
        results = {name: [] for name in global_metrics.keys()}
        for user_index in range(num_users):
            vector_predict = predicted_items[user_index]

            if len(vector_predict.nonzero()[0]) > 0:
                vector_true_dense = test_matrix[user_index].nonzero()[1]
                hits = np.isin(vector_predict, vector_true_dense)

                if vector_true_dense.size > 0:
                    for name in global_metrics.keys():
                        results[name].append(global_metrics[name](vector_true_dense=vector_true_dense,
                                                                  vector_predict=vector_predict, hits=hits))
        results_summary = dict()
        if analytical:
            for name in global_metrics.keys():
                results_summary[name] = results[name]
        else:
            for name in global_metrics.keys():
                results_summary[name] = (
                    np.average(results[name]), 1.96 * np.std(results[name]) / np.sqrt(len(results[name])))
        output.update(results_summary)

        return output


def diversity_metric(prediction, tagged_items, input_matrix):
    ks_test = []
    s_precision = []
    s_precision_2 = []
    prob1 = []
    prob2 = []
    for u in tqdm(range(len(prediction)), total=len(prediction)):
        tag1_in = (np.isin(prediction[u], tagged_items[0]).nonzero()[0]).tolist()
        tag2_in = (np.isin(prediction[u], tagged_items[1]).nonzero()[0]).tolist()
        if not tag1_in and not tag2_in:
            continue
        if not tag1_in and tag2_in:
            ks_test.append(1)
            s_precision.append(0)
            prob1.append(0)
            prob2.append(1)
            continue
        elif tag1_in and not tag2_in:
            ks_test.append(1)
            s_precision.append(0)
            prob1.append(1)
            prob2.append(0)
            continue

        # calculate s-precision
        s_precision.append((min(tag1_in[0], tag2_in[0])+1)/(max(tag1_in[0], tag2_in[0])+1))
        s_precision_2.append(2/(max(tag1_in[0], tag2_in[0])+1))

        # calculate probabilities
        tag1_in_training = np.isin(input_matrix[u].nonzero()[1], tagged_items[0]).sum()
        tag2_in_training = np.isin(input_matrix[u].nonzero()[1], tagged_items[1]).sum()
        ratio_tag1 = len(tag1_in) / (len(tagged_items[0]) - tag1_in_training)
        ratio_tag2 = len(tag2_in) / (len(tagged_items[1]) - tag2_in_training)
        prob1.append(ratio_tag1/(ratio_tag1+ratio_tag2))
        prob2.append(ratio_tag2/(ratio_tag1+ratio_tag2))

        # calculate ks-test
        tag1_step = 1/len(tag1_in)
        tag2_step = 1/len(tag2_in)
        i_1, i_2, diff, r_1, r_2 = 0, 0, 0, 0, 0
        while True:
            if i_1 == len(tag1_in)-1:
                diff = max(diff, 1 - r_2)
                break
            elif i_2 == len(tag2_in)-1:
                diff = max(diff, 1 - r_1)
                break
            if tag1_in[i_1] < tag2_in[i_2]:
                i_1 += 1
                r_1 += tag1_step
                diff = max(diff, abs(r_1 - r_2))
            elif tag1_in[i_1] > tag2_in[i_2]:
                i_2 += 1
                r_2 += tag2_step
                diff = max(diff, abs(r_1 - r_2))
            else:
                i_1 += 1
                r_1 += tag1_step
                i_2 += 1
                r_2 += tag2_step
                diff = max(diff, abs(r_1 - r_2))
        ks_test.append(diff)

    ks_test = np.mean(np.array(ks_test))
    s_precision = np.mean(np.array(s_precision))
    s_precision_2 = np.mean(np.array(s_precision_2))
    prob1 = np.array(prob1)
    prob2 = np.array(prob2)
    prob_product = np.mean(prob1*prob2)

    return s_precision, s_precision_2, 1-ks_test, prob1, prob2, prob_product


def sub_routine(vector_predict, vector_train, topK):
    train_index = vector_train.nonzero()[1]

    # take the top recommended items
    candidate_index = np.argpartition(-vector_predict, topK + len(train_index))[:topK + len(train_index)]
    vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]

    # vector_predict = np.argsort(-vector_predict)[:topK + len(train_index)]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]


def recallk(vector_true_dense, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits) / len(vector_true_dense)


def precisionk(vector_predict, hits, **unused):
    hits = len(hits.nonzero()[0])
    return float(hits) / len(vector_predict)


def average_precisionk(vector_predict, hits, **unused):
    precisions = np.cumsum(hits, dtype=np.float32) / range(1, len(vector_predict) + 1)
    return np.mean(precisions)


def r_precision(vector_true_dense, vector_predict, **unused):
    vector_predict_short = vector_predict[:len(vector_true_dense)]
    hits = len(np.isin(vector_predict_short, vector_true_dense).nonzero()[0])
    return float(hits) / len(vector_true_dense)


def _dcg_support(size):
    arr = np.arange(1, size + 1) + 1
    return 1. / np.log2(arr)


def ndcg(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg / idcg
