import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from utils.Table import Table
from utils.Tools import generate_sample_embeddings
from utils.io import pickle_dump


class Trainer:
    def __init__(self, dataname, target, dataset, model, evaluator, logger, conf, experiment=None,
                 plot_graph=False, run_samples=False):
        self.dataname = dataname
        self.dataset = dataset
        self.train_matrix, self.test_matrix = self.dataset.eval_data()
        self.uk_train = self.dataset.all_uk()

        # yelp
        if dataname == 'yelp_SIGIR':
            self.strong_tag1 = []
            self.strong_tag2 = []
            temp = self.uk_train.toarray()
            top_tags = np.argsort(-temp, axis=1)[:, :10]
            if target == 'veg_bbq':
                tag1 = 170
                tag2 = 203
                for u in range(len(top_tags)):
                    tag1_in = tag1 in top_tags[u]
                    tag2_in = tag2 in top_tags[u]
                    if tag1_in and not tag2_in and temp[u, tag1]-temp[u, tag2]>1:
                        self.strong_tag1.append(u)
                    elif not tag1_in and tag2_in and temp[u, tag2]-temp[u, tag1]>2:
                        self.strong_tag2.append(u)
            elif target == 'fried_salad':
                tag1 = 47
                tag2 = 37
                for u in range(len(top_tags)):
                    tag1_in = tag1 in top_tags[u]
                    tag2_in = tag2 in top_tags[u]
                    if tag1_in and not tag2_in:
                        self.strong_tag1.append(u)
                    elif not tag1_in and tag2_in:
                        self.strong_tag2.append(u)
            ik_label_matrix = dataset.ik_label_matrix.toarray()
            self.item_tag1 = ik_label_matrix[:, tag1]-ik_label_matrix[:, tag2]
            self.item_tag1[self.item_tag1<0] = 0
            self.item_tag1 = self.item_tag1.nonzero()[0]
            self.item_tag2 = ik_label_matrix[:, tag2]-ik_label_matrix[:, tag1]
            self.item_tag2[self.item_tag2<0] = 0
            self.item_tag2 = self.item_tag2.nonzero()[0]
            self.tagged_items = [self.item_tag1, self.item_tag2]
            print(len(self.strong_tag1), len(self.strong_tag2))
            print(len(self.item_tag1), len(self.item_tag2))

        elif dataname == 'reddit':

            # reddit
            item_idtoi = self.dataset.item_idtoi
            if target == 'rep_dem':
                tag1 = [item_idtoi['democrats'], item_idtoi['DemocratsforDiversity'], \
                        item_idtoi['DemocraticSocialism'], item_idtoi['Forum_Democratie'], \
                        item_idtoi['Impeach_Trump'], item_idtoi['neoliberal'], item_idtoi['AskALiberal'], \
                        item_idtoi['Liberal'], item_idtoi['Classical_Liberals'], item_idtoi['JoeBiden']]
                tag2 = [item_idtoi['Republican'], item_idtoi['Conservative'], item_idtoi['ConservativesOnly'], \
                        item_idtoi['askaconservative'], item_idtoi['conservatives'], \
                        item_idtoi['republicans'], item_idtoi['Trumpgret'], \
                        item_idtoi['IronFrontUSA'], item_idtoi['AskThe_Donald'], item_idtoi['AskTrumpSupporters'], \
                        item_idtoi['TheBidenshitshow']]
                thresh_1, thresh_2 = 2, 4
            elif target == 'men_women':
                tag1 = [item_idtoi['women'], item_idtoi['WomenWhoDontSell'], item_idtoi['AskWomen'], \
                        item_idtoi['AskWomenOver30'], item_idtoi['askwomenadvice'], item_idtoi['WomensHealth']]
                tag2 = [item_idtoi['Divorce_Men'], item_idtoi['AskMen'], item_idtoi['MensRights'], \
                        item_idtoi['AskMenAdvice'], item_idtoi['AskMenOver30']]
                thresh_1, thresh_2 = 2, 1
            self.item_tag1 = tag1
            self.item_tag2 = tag2
            self.tagged_items = [tag1, tag2]
            temp = self.uk_train.toarray()
            top_tags = np.argsort(-temp, axis=1)[:, :10]
            self.strong_tag1 = []
            self.strong_tag2 = []
            for i in range(len(temp)):
                tag1_in = np.sum(np.isin(top_tags[i], tag1))
                tag2_in = np.sum(np.isin(top_tags[i], tag2))
                if tag1_in and not tag2_in and tag1_in > thresh_1:
                    self.strong_tag1.append(i)
                elif not tag1_in and tag2_in and tag2_in > thresh_2:
                    self.strong_tag2.append(i)
            print(len(self.strong_tag1), len(self.strong_tag2))
            print(len(self.item_tag1), len(self.item_tag2))


        self.model = model
        self.evaluator = evaluator
        self.logger = logger
        self.conf = conf
        self.experiment = experiment
        self.plot_graphs = plot_graph
        self.run_samples = run_samples

        self.num_epochs = conf.num_epochs
        self.lr = conf.learning_rate
        self.batch_size = conf.batch_size
        self.test_batch_size = conf.test_batch_size

        self.early_stop = conf.early_stop
        self.patience = conf.patience
        self.endure = 0
        self.skip_eval = conf.skip_eval

        self.best_epoch = -1
        self.best_score = None
        self.best_params = None
        self.best_rec_score = None
        self.best_uk_score = None

        # save the best keyphrase embeddings during best epochs
        self.mean_embeddings = None
        self.stdev_embeddings = None

        # for use case, save selected user and keyphrase embeddings during training
        self.sampled_user_idx = 799  # a single number
        self.sample_user_embeddings = None
        self.sample_user_embeddings_std = None
        self.sampled_keyphrase_idx = [225, 429, 674]  # a list of keyphrase idx
        self.sampled_keyphrase_embeddings = None
        self.sampled_keyphrase_embeddings_std = None
        self.sampled_epoch = [10, 50, 100, 200, 250, 300]  # a list of training epochs to sample
        self.score_comparison_df = pd.DataFrame(columns=['MSE', 'uk_rprec', 'epoch'])

    def train(self):
        self.logger.info(self.conf)

        # pass module parameters to the optimizer
        if len(list(self.model.parameters())) > 0:
            optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr)
            # optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr)
        else:
            optimizer = None

        # create table for logging
        score_table = Table(table_name='Scores')

        for epoch in range(1, self.num_epochs + 1):
            # train for an epoch
            epoch_start = time.time()
            loss = self.model.train_one_epoch(train_matrix=self.train_matrix,
                                              # uk_matrix=self.uk_train,
                                              # uk_test=self.uk_valid,
                                              optimizer=optimizer,
                                              batch_size=self.batch_size,
                                              verbose=False)
                                              # experiment=self.experiment)  # verbose/printing false

            # log epoch loss
            if self.experiment: self.experiment.log_metric(name='epoch_loss', value=loss, epoch=epoch)
            print("Epoch:", epoch, "Loss:", loss)

            train_elapsed = time.time() - epoch_start

            #and epoch >= 50 
            if (not self.skip_eval and epoch % 20 == 0) or (self.skip_eval and epoch == self.num_epochs):
                if not self.skip_eval and self.early_stop:  # get scores during training only
                    # recommendation performance
                    rec_score = self.evaluator.evaluate_recommendations(self.dataname, self.item_tag1, self.item_tag2, 
                                                                        self.tagged_items, epoch, self.strong_tag1, 
                                                                        self.strong_tag2, self.model, self.train_matrix,
                                                                        self.test_matrix, mse_only=False,
                                                                        ndcg_only=True, analytical=False,
                                                                        test_batch_size=self.test_batch_size)

                else:  # At the end of training epochs, during evaluation
                    # recommendation performance
                    rec_score = self.evaluator.evaluate_recommendations(self.dataname, self.item_tag1, self.item_tag2, 
                                                                        self.tagged_items, epoch, self.strong_tag1, 
                                                                        self.strong_tag2, self.model, self.train_matrix,
                                                                        self.test_matrix, mse_only=False,
                                                                        ndcg_only=False, analytical=False,
                                                                        test_batch_size=self.test_batch_size)

                # score we want to check during training
                score = {"Loss": float(loss),
                         "RMSE": rec_score['RMSE'][0]}
                if "NDCG" in rec_score.keys():
                    score['NDCG'] = rec_score['NDCG'][0]

                score_str = ' '.join(['%s=%.4f' % (m, score[m]) for m in score])
                epoch_elapsed = time.time() - epoch_start

                self.logger.info('[Epoch %3d/%3d, epoch time: %.2f, train_time: %.2f] %s' % (
                    epoch, self.num_epochs, epoch_elapsed, train_elapsed, score_str))

                # log for comet ml, per 10 epochs
                if self.experiment:
                    self.experiment.log_metric(name='RMSE', value=score['RMSE'], \
                                               epoch=epoch)
                    if "NDCG" in rec_score.keys():
                        self.experiment.log_metric(name='NDCG', value=score['NDCG'],
                                                   epoch=epoch)
                # update if ...
                standard = 'NDCG'
                if self.best_score is None or score[standard] > self.best_score[standard]:
                    self.best_epoch = epoch
                    self.best_score = score
                    self.best_rec_score = rec_score
                    self.best_params = self.model.parameters()

                    self.endure = 0

                    # log stats plot, every 50 epoch is enough
                    if self.plot_graphs and epoch >= 50 and epoch % 50 == 0:
                        self.log_stats_plot(epoch)
                else:
                    self.endure += 10
                    if self.early_stop and self.endure >= self.patience:
                        print('Early Stop Triggered...')
                        break

        # log plot at the end of training, and log last epoch embeddings
        if self.plot_graphs:
            self.log_stats_plot(epoch)
            # close plt records
            plt.clf()
            plt.cla()
            plt.close()

        print('Training Finished.')
        score_table.add_row('Best at epoch %d' % self.best_epoch, self.best_score)
        self.logger.info(score_table.to_string())

    # create scatter plot for user embedding values
    def create_scatter(self, embedding, axis_value):
        log_freq = self.dataset.log_freq_array
        log_freq_keyphrase = self.dataset.log_freq_array_keyphrase
        avg_stdev = np.mean(embedding, axis=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
        ax1.scatter(log_freq, avg_stdev)
        ax1.set_xlabel('log total rating frequency')
        ax1.set_ylabel('avg_{}'.format(axis_value))

        ax2.scatter(log_freq_keyphrase, avg_stdev)
        ax2.set_xlabel('log total keyphrase frequency')
        ax2.set_ylabel('avg_{}'.format(axis_value))

        fig.suptitle('Rating frequencies & keyphrase mentioning frequencies vs. averaged {}'.format(axis_value))

        return fig

    def create_scatter_keyhrase(self, axis_value):
        if axis_value == 'stdev':
            embedding = np.array(torch.exp(0.5 * self.model.keyphrase_log_var.weight.data))
        elif axis_value == 'mean':
            embedding = np.array(self.model.keyphrase_mu.weight.data)
        else:
            raise NotImplementedError('Choose appropriate embedding parameter. (current input: %s)' % axis_value)

        log_freq_ku = self.dataset.log_freq_array_ku
        log_freq_ki = self.dataset.log_freq_array_ki

        avg_embedding = np.mean(embedding, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
        ax1.scatter(log_freq_ku, avg_embedding)
        ax1.set_xlabel('log user mentioning frequency')
        ax1.set_ylabel('avg_{}'.format(axis_value))

        ax2.scatter(log_freq_ki, avg_embedding)
        ax2.set_xlabel('log item labeled frequency')
        ax2.set_ylabel('avg_{}'.format(axis_value))

        fig.suptitle('User mention frequencies & item labeled frequencies vs. averaged {}'.format(axis_value))
        # file_dir = log_dir + conf_name + '.png'
        # plt.savefig(file_dir)

        return fig

    def create_TSNE(self):
        # manual modifications
        keyphrase_list = ['kbbq', 'bbq', 'kebab', 'korean bbq', 'korean', 'pizza', 'pizzeria libretto',
                          'pistachio', 'pita bread']
        sample_num = 150
        keyphrase_idx_check = [self.dataset.keyphrase_2_idx[word] for word in keyphrase_list]

        sampled_embedding = generate_sample_embeddings(keyphrase_idx_check=keyphrase_idx_check,
                                                       sample_num=sample_num,
                                                       mean_embed=self.mean_embeddings,
                                                       stdev_embed=self.stdev_embeddings)
        tsne = TSNE(n_components=2, perplexity=5, early_exaggeration=4)  # , verbose=1 , n_iter=500, , perplexity=50
        tsne_results = tsne.fit_transform(sampled_embedding)

        # feed sample embedding to tsne & create plots
        df_plot = {
            'keyphrase': np.array(
                [[self.dataset.idx_2_keyphrase[idx]] * sample_num for idx in keyphrase_idx_check]).flatten()}
        df_plot = pd.DataFrame(data=df_plot)

        samples_to_plot = len(keyphrase_idx_check) * sample_num
        df_plot['tsne-2d-one'] = tsne_results[:samples_to_plot, 0]
        df_plot['tsne-2d-two'] = tsne_results[:samples_to_plot, 1]

        tsne = TSNE(n_components=2, perplexity=5, early_exaggeration=4)  # , verbose=1 , n_iter=500, , perplexity=50
        tsne_results = tsne.fit_transform(sampled_embedding)

        # feed sample embedding to tsne & create plots
        df_plot = {
            'keyphrase': np.array(
                [[self.dataset.idx_2_keyphrase[idx]] * sample_num for idx in keyphrase_idx_check]).flatten()}
        df_plot = pd.DataFrame(data=df_plot)

        samples_to_plot = len(keyphrase_idx_check) * sample_num
        df_plot['tsne-2d-one'] = tsne_results[:samples_to_plot, 0]
        df_plot['tsne-2d-two'] = tsne_results[:samples_to_plot, 1]

        plt.figure(figsize=(10, 8))
        sns_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="keyphrase",
            palette=sns.color_palette("hls", len(df_plot.keyphrase.unique())),
            data=df_plot,
            legend="full",
            alpha=0.3
        )

        return sns_plot.get_figure()

    def plot_comparison_plot(self):
        plt.clf()
        plt.figure()
        sns_plot = sns.scatterplot(data=self.score_comparison_df,
                                   x="MSE",
                                   y="uk_rprec",
                                   hue="epoch")

        return sns_plot.get_figure()

    def log_stats_plot(self, epoch_num):
        mean_embedding, stdev_embedding = self.get_mu_S()

        stats_figure = self.create_scatter(stdev_embedding, 'stdev')
        # stats_figure_mean = self.create_scatter(mean_embedding, 'mean')

        stats_figure_k = self.create_scatter_keyhrase('stdev')
        stats_figure_mean_k = self.create_scatter_keyhrase('mean')

        self.experiment.log_figure(figure_name='stats_fig_' + str(epoch_num), figure=stats_figure, overwrite=True)
        # self.experiment.log_figure(figure_name='stats_fig_mean_' + str(epoch_num), figure=stats_figure_mean,
        #                            overwrite=True)
        self.experiment.log_figure(figure_name='stats_fig_keyphrase_' + str(epoch_num), figure=stats_figure_k,
                                   overwrite=True)
        # self.experiment.log_figure(figure_name='stats_fig_mean_keyphrase_' + str(epoch_num),
        #                            figure=stats_figure_mean_k, overwrite=True)

        # close plt records
        plt.clf()
        plt.cla()
        plt.close()

    def save_best_embeddings(self, mean_embeddigns, stdev_embeddings):
        self.mean_embeddings = mean_embeddigns
        self.stdev_embeddings = stdev_embeddings

    def log_embeddings_asset(self, mean_embeddings, stdev_embeddings, epoch):
        embedded_mean_df = pd.DataFrame(mean_embeddings, columns=list(range(1, mean_embeddings.shape[1] + 1)),
                                        index=list(self.dataset.keyphrase_2_idx.keys()))

        embedded_std_df = pd.DataFrame(stdev_embeddings, columns=list(range(1, stdev_embeddings.shape[1] + 1)),
                                       index=list(self.dataset.keyphrase_2_idx.keys()))

        # self.experiment.log_dataframe_profile(dataframe=embedded_std_df, name='embedded_stdev')
        # self.experiment.log_dataframe_profile(dataframe=embedded_mean_df, name='embedded_mean')
        self.experiment.log_table('embedded_means{}.csv'.format(epoch), embedded_mean_df)
        self.experiment.log_table('embedded_stdev{}.csv'.format(epoch), embedded_std_df)

    def get_mu_S(self):
        input_matrix = self.dataset.all_data()
        i = torch.FloatTensor(input_matrix.toarray()).to(self.model.device)
        with torch.no_grad():
            mu, logvar = self.model.get_mu_logvar(i)
            std = self.model.logvar2std(logvar)
        mu, std = mu.cpu().data.numpy(), std.cpu().data.numpy()

        return mu, std
