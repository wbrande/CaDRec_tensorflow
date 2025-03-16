import argparse

import numpy as np
import time
import math

# import torch
# import torch.optim as optim

# if torch.cuda.is_available():
#     import torch.cuda as T
# else:
#     import torch as T

import tensorflow as tf

import optuna

import Constants as C
from utils import Utils, metric

from utils.Dataset import Dataset as dataset
from Model.Models import Model
from tqdm import tqdm
import shutil
import os

## Carl - Added to keep track of trial number while running optuna optimizations
# trialNum = 1

## Carl - any reference to Scheduler can be removed as it is not used by tensorflow.

def train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt):
    """ Epoch operation in training phase using Tensorflow. """

    pre, rec, map_, ndcg = [[] for i in range(4)]
    for batch in tqdm(user_dl, mininterval=2, desc='  - (Training)   ', leave=False,):
        ## Carl - Unpack Batch
        """ Prepare DataZ """
        user_idx, event_type, event_time, test_label = batch
        with tf.GradientTape() as tape:

            ## Carl - Forward pass
            """ Forward """
            prediction, user_embeddings, pop_vector = model(user_idx, event_type, adj_matrix, pop_encoding, evaluation=False, training = True)
            ## Carl - Backward pass
            """ Backward """
            loss = Utils.type_loss(prediction, event_type, event_time, test_label, opt)
            eta = 0.1 if C.DATASET not in C.eta_dict else C.eta_dict[C.DATASET]

            if C.ABLATION != 'w/oNorm':
                loss += Utils.l2_reg_loss(eta, model, event_type)

        """ Update Parameters """
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        ## Compute Metrics for batch
        """ Compute Metric """
        metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

    ## Compute Metrics for all batches
    pre_avg = [np.around(np.mean(x), 5) for x in pre]
    rec_avg = [np.around(np.mean(x), 5) for x in rec]
    map_avg = [np.around(np.mean(x), 5) for x in map_]
    ndcg_avg = [np.around(np.mean(x), 5) for x in ndcg]

    return pre_avg, rec_avg, map_avg, ndcg_avg


def eval_epoch(model, user_valid_dl, adj_matrix, pop_encoding, opt):
    """ Epoch operation in evaluation phase. """

    pre, rec, map_, ndcg = [[] for i in range(4)]
    for batch in tqdm(user_valid_dl, mininterval=2, desc='  - (Validation) ', leave=False):
        """ prepare test data """
        user_idx, event_type, event_time, test_label = batch

        """ forward """
        prediction, _, _ = model(user_idx, event_type, adj_matrix, pop_encoding, evaluation=False, training = False)  # X = (UY+Z) ^ T

        # valid_user_embeddings[user_idx] = users_embeddings

        """ compute metric """
        metric.pre_rec_top(pre, rec, map_, ndcg, prediction, test_label, event_type)

    pre_avg = [np.around(np.mean(x), 5) for x in pre]
    rec_avg = [np.around(np.mean(x), 5) for x in rec]
    map_avg = [np.around(np.mean(x), 5) for x in map_]
    ndcg_avg = [np.around(np.mean(x), 5) for x in ndcg]
    return pre_avg, rec_avg, map_avg, ndcg_avg


def train(model, data, optimizer, scheduler, opt):
    """ Start training. """

    best_metrics = [np.zeros(4) for i in range(4)]
    (user_valid_dl, user_dl, adj_matrix, pop_encoding) = data
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i + 1, ']')

        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        # start = time.time()
        [pre, rec, map_, ndcg] = train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt)
        # print('\r(Training)  P@k:{pre},    R@k:{rec}, \n'
        #       '(Training)map@k:{map_}, ndcg@k:{ndcg}, '
        #       'elapse:{elapse:3.3f} min'
        #       .format(elapse=(time.time() - start) / 60, pre=pre, rec=rec, map_=map_, ndcg=ndcg))

        start = time.time()
        ## Carl - Run training epoch
        train_metrics = train_epoch(model, user_dl, adj_matrix, pop_encoding, optimizer, opt)
        print('\r(Training)  P@k:{pre},    R@k:{rec}, \n(Training)map@k:{map_}, ndcg@k:{ndcg}, elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=train_metrics[0], rec=train_metrics[1], map_=train_metrics[2], ndcg=train_metrics[3]))

        start = time.time()
        ## Carl - Run evaluation epoch
        eval_metrics = eval_epoch(model, user_valid_dl, adj_matrix, pop_encoding, opt)
        print('\r(Test)  P@k:{pre},    R@k:{rec}, \n(Test)map@k:{map_}, ndcg@k:{ndcg}, elapse:{elapse:3.3f} min'
              .format(elapse=(time.time() - start) / 60, pre=eval_metrics[0], rec=eval_metrics[1], map_=eval_metrics[2], ndcg=eval_metrics[3]))

        ## Carl - Manually decay learning rate in place of scheduler every 10 epochs
        if (epoch_i +1) % 10 == 0:
            new_lr = optimizer.learning_rate * .5
            optimizer.learning_rate.assign(new_lr)
        ## Carl - Update best metric if ndcg at k = 5 is greater than previous best
        if best_metrics[3][1] < eval_metrics[3][1]:
            best_metrics = [eval_metrics[0], eval_metrics[1], eval_metrics[2], eval_metrics[3]]

    ## Prints the best epoch from defined number of epochs (default = 30)
    print('\n', '-' * 40, 'BEST', '-' * 40)
    print('k', C.Ks)
    print('\rP@k:{pre},    R@k:{rec}, \n(Best)map@k:{map_}, ndcg@k:{ndcg}'
          .format(pre=best_metrics[0], rec=best_metrics[1], map_=best_metrics[2], ndcg=best_metrics[3]))
    print('-' * 40, 'BEST', '-' * 40, '\n')

    ## Carl - Return the metrics of the best epoch at ndcg at k = 5
    return best_[-1][1]


## Carl - Not needed for the current implementation
# def get_user_embeddings(model, user_dl, opt):
#     """ Epoch operation in training phase. """

#     valid_user_embeddings = torch.zeros((C.USER_NUMBER, opt.d_model), device='cuda:0')

#     for batch in tqdm(user_dl, mininterval=2, desc='  - (Computing user embeddings)   ', leave=False, disable = True):
#         """ prepare data """
#         user_idx, event_type, event_time, test_label = map(lambda x: x.to(opt.device), batch)

#         """ forward """
#         prediction, users_embeddings = model(event_type)  # X = (UY+Z) ^ Tc
#         valid_user_embeddings[user_idx] = users_embeddings

#     return valid_user_embeddings


def pop_enc(in_degree, d_model):
    ## Carl - Create the popularity or positional encoding
    """
    Input: batch*seq_len.
    Output: batch*seq_len*d_model.
    """

    pop_vec = tf.constant([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)], dtype = tf.float32)

    result = tf.expand_dims(in_degree, -1) / pop_vec
    ## Carl - Create boolean mask for even indices
    indices = tf.range(d_model)
    even_mask = tf.equal(tf.math.mod(indices, 2), 0)
    even_mask = tf.tile(tf.expand_dims(even_mask, 0), [tf.shape(result)[0], 1])
    ## Carl - even indices use sin, odd use cosine
    result = tf.where(even_mask, tf.sin(result), tf.cos(result))

    return result


def main(trial):
    """ Main function. """
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    # # # optuna setting for tuning hyperparameters
    # opt.n_layers = trial.suggest_int('n_layers', 2, 2)
    # opt.d_inner_hid = trial.suggest_int('n_hidden', 512, 1024, 128)
    # opt.d_k = trial.suggest_int('d_k', 512, 1024, 128)
    # opt.d_v = trial.suggest_int('d_v', 512, 1024, 128)
    # opt.n_head = trial.suggest_int('n_head', 1, 5, 1)
    # # opt.d_rnn = trial.suggest_int('d_rnn', 128, 512, 128)
    # opt.d_model = trial.suggest_int('d_model', 128, 1024, 128)
    # opt.dropout = trial.suggest_uniform('dropout_rate', 0.5, 0.7)
    # opt.smooth = trial.suggest_uniform('smooth', 1e-2, 1e-1)
    # opt.lr = trial.suggest_uniform('learning_rate', 0.00008, 0.0002)

    DATASET = C.DATASET
    if DATASET == 'Foursquare':
        beta, lambda_ = 0.3256, 0.4413  # 0.4, 0.4   # 0.4, 0.5  # 0.35, 0.5  # 0.5, 1
    elif DATASET == 'Gowalla':
        beta, lambda_ = 1.5, 4  # 0.38, 1  # 1.5, 4
    elif DATASET == 'Yelp2018':
        beta, lambda_ = 2.2977, 7.0342  # 1.8, 4  # 0.35, 1  # 1, 4
    elif DATASET == 'douban-book':
        beta, lambda_ = 0.9802, 0.7473
    elif DATASET == 'ml-1M':
        beta, lambda_ = 0.4645, 0.4098  # 0.9, 1
    else:
        beta, lambda_ = 0.5, 1
    opt.beta, opt.lambda_ = beta, lambda_

    opt.lr = 0.01
    opt.epoch = 30
    opt.n_layers = 1
    opt.batch_size = 32
    opt.dropout = 0.5
    opt.smooth = 0.03

    if DATASET == 'Foursquare':
        opt.d_model, opt.n_head = 768, 1
    elif DATASET == 'Gowalla':
        opt.d_model, opt.n_head = 512, 1
    elif DATASET == 'douban-book':
        opt.d_model, opt.n_head = 512, 1
    elif DATASET == 'Yelp2018':
        opt.d_model, opt.n_head = 512, 1
    elif DATASET == 'ml-1M':
        opt.d_model, opt.n_head = 512, 2
    else:
        opt.d_model, opt.n_head = 512, 1

    print('[Info] parameters: {}'.format(opt))
    num_types = C.ITEM_NUMBER
    num_user = C.USER_NUMBER

    """ prepare model """
    model = Model(
        num_types=num_types,
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
    )

    """ loading data"""
    ds = dataset()
    print('[Info] Loading data...')
    train_data = ds.user_data
    valid_data = ds.user_valid

    def gen_train():
        for item in train_data:
            yield item

    def gen_valid():
        for item in valid_data:
            yield item

    output_signature = (
        tf.TensorSpec(shape = (), dtype = tf.int32),
        tf.TensorSpec(shape = (None,), dtype = tf.int32),
        tf.TensorSpec(shape = (None,), dtype = tf.int32),
        tf.TensorSpec(shape = (None,), dtype = tf.int32),
    )
    train_dataset = tf.data.Dataset.from_generator(gen_train, output_signature = output_signature)
    valid_dataset = tf.data.Dataset.from_generator(gen_valid, output_signature = output_signature)

    ## Carl - Pad each batch along sequence dimensions
    train_dataset = train_dataset.padded_batch(opt.batch_size, 
                                               padded_shapes = ([], [None], [None], [None]), 
                                               padding_values = (0, C.PAD, C.PAD, C.PAD))
    valid_dataset = valid_dataset.padded_batch(opt.batch_size, 
                                               padded_shapes = ([], [None], [None], [None]), 
                                               padding_values = (0, C.PAD, C.PAD, C.PAD))

    # user_dl = ds.get_user_dl(opt.batch_size)
    # user_valid_dl = ds.get_user_valid_dl(opt.batch_size)


    in_degree = ds.get_in_degree()
    pop_encoding = pop_enc(in_degree, opt.d_model)
    adj_matrix = ds.ui_adj

    data = (valid_dataset, train_dataset, adj_matrix, pop_encoding)
    """ optimizer and scheduler """
    # parameters = [
    #               {'params': model.parameters(), 'lr': opt.lr},
    #               ]
    optimizer = tf.keras.optimizers.Adam(learning_rate = opt.lr)  # , weight_decay=0.01
    ## Carl - Not needed by tensorflow
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ number of parameters """
    num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    return train(model, data, optimizer, None, opt)

## Carl - Added to run through trials to search for better hyperparameters
## Carl - While the setup was still using pytorch
# def objective(trial):
#     global trialNum
#     print("\n trial:", trialNum, "\n")

#     parser = argparse.ArgumentParser()
#     opt = parser.parse_args()
#     opt.device = torch.device('cuda')

#     # # # optuna setting for tuning hyperparameters
#     opt.n_layers = trial.suggest_int('n_layers', 1, 4)
#     opt.d_inner_hid = trial.suggest_int('n_hidden', 512, 1024, 128)
#     opt.d_k = trial.suggest_int('d_k', 512, 1024, 128)
#     opt.d_v = trial.suggest_int('d_v', 512, 1024, 128)
#     opt.n_head = trial.suggest_int('n_head', 1, 8, 1)
#     opt.d_rnn = trial.suggest_int('d_rnn', 128, 512, 128)
#     opt.d_model = trial.suggest_categorical('d_model', [128, 256, 512, 1024, 2048])
#     opt.dropout = trial.suggest_uniform('dropout_rate', 0.1, 0.7)
#     opt.smooth = trial.suggest_uniform('smooth', 1e-2, 1e-1)
#     opt.lr = trial.suggest_uniform('learning_rate', 0.000008, 0.1)
#     opt.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

#     DATASET = C.DATASET
#     if DATASET == 'Foursquare':
#         beta, lambda_ = 0.3256, 0.4413  # 0.4, 0.4   # 0.4, 0.5  # 0.35, 0.5  # 0.5, 1
#     elif DATASET == 'Gowalla':
#         beta, lambda_ = 1.5, 4  # 0.38, 1  # 1.5, 4
#     elif DATASET == 'Yelp2018':
#         beta, lambda_ = 2.2977, 7.0342  # 1.8, 4  # 0.35, 1  # 1, 4
#     elif DATASET == 'douban-book':
#         beta, lambda_ = 0.9802, 0.7473
#     elif DATASET == 'ml-1M':
#         beta, lambda_ = 0.4645, 0.4098  # 0.9, 1
#     else:
#         beta, lambda_ = 0.5, 1
#     opt.beta, opt.lambda_ = beta, lambda_

#     opt.epoch = 30

#     if DATASET == 'Foursquare': opt.d_model, opt.n_head = 768, 1
#     elif DATASET == 'Gowalla': opt.d_model, opt.n_head = 512, 1
#     elif DATASET == 'douban-book': opt.d_model, opt.n_head = 512, 1
#     elif DATASET == 'Yelp2018': opt.d_model, opt.n_head = 512, 1
#     elif DATASET == 'ml-1M': opt.d_model, opt.n_head = 512, 2
#     else: opt.d_model, opt.n_head = 512, 1

#     print('[Info] parameters: {}'.format(opt))
#     num_types = C.ITEM_NUMBER
#     num_user = C.USER_NUMBER

#     """ prepare model """
#     model = Model(
#         num_types=num_types,
#         d_model=opt.d_model,
#         n_layers=opt.n_layers,
#         n_head=opt.n_head,
#         dropout=opt.dropout,
#         device=opt.device
#     )
#     model = model.cuda()

#     """ loading data"""
#     ds = dataset()
#     print('[Info] Loading data...')
#     user_dl = ds.get_user_dl(opt.batch_size)
#     user_valid_dl = ds.get_user_valid_dl(opt.batch_size)
#     in_degree = ds.get_in_degree()
#     pop_encoding = pop_enc(in_degree, opt.d_model)
#     adj_matrix = ds.ui_adj

#     data = (user_valid_dl, user_dl, adj_matrix, pop_encoding)
#     """ optimizer and scheduler """
#     parameters = [
#                   {'params': model.parameters(), 'lr': opt.lr},
#                   ]
#     optimizer = torch.optim.Adam(parameters)  # , weight_decay=0.01
#     scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

#     """ number of parameters """
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('[Info] Number of parameters: {}'.format(num_params))

#     trialNum = trialNum + 1

#     """ train the model """
#     return train(model, data, optimizer, scheduler, opt)


if __name__ == '__main__':
    main(None)

    # if you want to tune hyperparameters, please comment out main(None) and use the following code
    ## study = optuna.create_study(direction="maximize")
    ## n_trials = 100
    ## study.optimize(objective, n_trials=n_trials)



