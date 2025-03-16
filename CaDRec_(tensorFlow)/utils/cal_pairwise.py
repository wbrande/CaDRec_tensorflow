import numpy as np
# import torch
import tensorflow as tf

import sys
sys.path.append("..")
import Constants as C
import os

import time


def read_interaction_by_trajectory(user_trajectories):
    """ Process user trajectories to create an interaction and item matrix """

    # for temporal feature
    start_time = time.time()
    print(start_time)

    directory_path = f'../data/{C.DATASET}/'
    # train_file = 'train.txt'.format(dataset=C.DATASET)
    # train_data = open(directory_path + train_file, 'r').readlines()
    count = 0

    ## Carl - Create a numpy array to collect data
    interaction_matrix_np = np.zeros((C.USER_NUMBER, C.ITEM_NUMBER), dtype = np.float32)
    item_matrix_np = np.zeros((C.ITEM_NUMBER, C.ITEM_NUMBER), dtype = np.float32)

    # print(interaction_matrix.size())
    # print(item_matrix.size())
    ## Carl - Populate the interaction matrix using user trajectories
    for uid, user_traj in enumerate(user_trajectories):
        for lid in user_traj:
            interaction_matrix_np[uid, lid] = 1
        count += 1
        if count % 10000 == 0:
            print(count, time.time() - start_time)

    ## Carl - Populate the item matrix for each user, for each item interacted with, and mark co-interacted items.
    for i in range(C.USER_NUMBER):
        nwhere = np.where(interaction_matrix_np[i] == 1)[0]
        for j in nwhere:
            item_matrix_np[j, nwhere] = 1

    # print(nwhere)
    # print(item_matrix)
    np.save(directory_path + 'item_matrix.npy', item_matrix_np)


def read_interaction(train_data=None, directory_path=None):
    """ Reads training and tuning data, creates interaction and item matrix from them. """

    # for temporal feature
    start_time = time.time()
    # print(start_time)

    if directory_path is None:
        directory_path = f'./data/{C.DATASET}/'
    if train_data is None:
        train_file = f'{C.DATASET}_train.txt'
        tune_file = f'{C.DATASET}_tune.txt'
        train_data = open(directory_path + train_file, 'r').readlines()
        train_data.extend(open(directory_path + tune_file, 'r').readlines())
    count = 0

    ## Carl - Create numpy arrays to collect data
    interaction_matrix_np = np.zeros((C.USER_NUMBER, C.ITEM_NUMBER), dtype = np.float32)
    item_matrix_np = np.zeros((C.ITEM_NUMBER, C.ITEM_NUMBER), dtype = np.float32)

    # print(interaction_matrix.size())
    for eachline in train_data:
        uid, lid, timestamp = eachline.strip().split()
        uid, lid, timestamp = int(uid), int(lid), int(timestamp)
        if C.DATASET == 'Yelp2018':
            lid = lid -1
        
        # print(uid, lid)
        interaction_matrix_np[uid, lid] = 1
        count += 1
        if count % 500000 == 0:
            print(count, time.time() - start_time)

    for i in range(C.USER_NUMBER):
        nwhere = np.where(interaction_matrix_np[i] == 1)[0]
        for j in nwhere:
            item_matrix_np[j, nwhere] = 1

    # print(nwhere)
    # print(ITEM_matrix)
    np.save(directory_path + 'item_matrix.npy', item_matrix_np)




# def main():
#     read_interaction()
#
#
# if __name__ == '__main__':
#     main()



