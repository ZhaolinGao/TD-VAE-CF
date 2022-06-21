import argparse
import json
from comet_ml import Experiment
import numpy as np
import pandas as pd
import torch
import models
from utils.Dataset import Dataset, Reddit_Dataset
from utils.Evaluator import Evaluator
from utils.HPShelper import conf_dict_generator
from utils.Logger import Logger
from utils.Params import Params
from utils.Trainer import Trainer
from utils.io import load_dataframe_csv, save_dataframe_csv

def fit(experiment_, model_name, data_name_, target_, lamb_, std_, dataset_, log_directory, device_, skip_eval, plot_graph, run_samples):
    # dictionary generate from experiment
    d = conf_dict_generator[model_name](experiment_)
    d['skip_eval'] = skip_eval
    conf_dict = Params()
    conf_dict.update_dict(d)

    model_base = getattr(models, model_name)
    if 'contrast' in model_name:
        model_ = model_base(conf_dict, dataset_.num_users, dataset_.num_items, dataset_.num_keyphrases, device_)
    else:
        model_ = model_base(conf_dict, dataset_.num_users, dataset_.num_items, device_)

    evaluator = Evaluator(rec_atK=[5, 10, 15, 20, 50], explain_atK=[5, 10, 15, 20, 50], lamb=lamb_, std=std_)
    logger = Logger(log_directory)
    logger.info(conf_dict)
    logger.info(dataset_)

    trainer = Trainer(
        dataname=data_name_,
        target=target_,
        dataset=dataset_,
        model=model_,
        evaluator=evaluator,
        logger=logger,
        conf=conf_dict,
        experiment=experiment_,
        plot_graph=plot_graph,  # plot the stats for embeddings
        run_samples=run_samples  # run a 2D use case
    )

    trainer.train()
    return (trainer.best_rec_score, trainer.best_uk_score,
            trainer.best_epoch, model_)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='VAEmultilayer')
    parser.add_argument('--data_name', type=str, default='yelp_SIGIR')
    parser.add_argument('--target', type=str, default='veg_bbq', help='[veg_bbq, fried_salad, men_women, rep_dem]')
    parser.add_argument('--fold_name', type=str, default='fold0')
    parser.add_argument('--top_items', type=int, default=10, help='used to indicate top labels for each item')
    parser.add_argument('--top_users', type=int, help='if cuting the matrix with top user numbers')
    parser.add_argument('--rating_threshold', type=float, default=1,
                        help='used to indicate user liked items for generating uk matrices')
    parser.add_argument('--lamb', type=float, default=0.6)
    parser.add_argument('--std', type=float, default=10)

    parser.add_argument('--plot_graph', action='store_true', help='Whether plotting the statistical graphs')
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--run_samples', action='store_true')
    parser.add_argument('--conf', type=str, default='VAEmultilayer.config')
    parser.add_argument('--seed', type=int, default=201231)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    # where the training data files are stored
    log_dir = "{}/{}/{}/".format("./saves", p.data_name, p.model_name)
    config_dir = "{}/{}/{}".format("./conf", p.data_name, p.conf)
    table_dir = "{}/{}/{}/".format("./tables", p.data_name, p.model_name)
    print('config_dir:', config_dir, 'table_dir:', table_dir)

    with open(config_dir) as f:
        conf = json.load(f)

    if p.data_name in ['yelp_SIGIR']:
        data_dir = "{}/{}/{}/".format("./data", p.data_name, p.fold_name)
        dataset = Dataset(data_dir=data_dir, top_keyphrases=p.top_items, rating_threshold=p.rating_threshold,
                          top_users=p.top_users)
    elif p.data_name in ['reddit']:
        data_dir = "{}/{}/".format("./data", p.data_name)
        dataset = Reddit_Dataset(data_dir=data_dir, top_keyphrases=p.top_items, target=p.target)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    project_name = p.data_name + '-' + 'main'
    experiment = Experiment(api_key='8VXZFCmgpRu8EyvczH3qA5gUs', project_name=project_name)
    experiment.log_parameters(conf)

    # training
    try:
        rec_score, uk_score, epoch, model = fit(experiment, p.model_name, p.data_name, p.target, p.lamb, p.std,
                                                dataset, log_dir, device, skip_eval=p.skip_eval, 
                                                plot_graph=p.plot_graph, run_samples=p.run_samples)

        experiment.log_metric("best_epoch", epoch)
        experiment.log_metrics({k: v[0] for k, v in rec_score.items()})
        if uk_score is not None:
            experiment.log_metrics({k: v[0] for k, v in uk_score.items()})

        experiment.log_others({
            "model_desc": p.model_name
        })

        # save results table
        result_dict = conf_dict_generator[p.model_name](experiment)
        result_dict['best_epoch'] = epoch
        try:
            df = load_dataframe_csv(table_dir, p.conf.split('.')[0]+'.csv')
        except:
            df = pd.DataFrame(columns=result_dict.keys())

        for name in rec_score.keys():
            result_dict[name] = [round(rec_score[name][0], 4), round(rec_score[name][1], 4)]
        if uk_score is not None:
            for name in uk_score.keys():
                result_dict[name] = [round(uk_score[name][0], 4), round(uk_score[name][1], 4)]

        df = df.append(result_dict, ignore_index=True)

        save_dataframe_csv(df, table_dir, p.conf.split('.')[0])

    finally:
        experiment.end()
