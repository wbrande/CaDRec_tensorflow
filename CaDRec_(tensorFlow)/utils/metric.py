import math
# import torch
import tensorflow as tf
import Constants as C


def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    """ Compute Precision, Recall, MAP, and NDCG at k. """
    idcg_k = 0
    dcg_k = 0
    map_v = 0
    ap = 0

    # Please check consistency with your baselines.
    # All baselines and this method followed this setting to compute the ideal DCG through a truncation.
    # (See Line. 94 (https://github.com/Coder-Yu/SELFRec/blob/main/util/evaluation.py))

    ## Carl - Compute Ideal DCG
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        map = ap / count

    precision = float(count) / k
    recall = float(count) / len(test_matrix)
    ndcg = float(dcg_k / idcg_k) if idcg_k != 0 else 0.0

    return precision, recall, map_v, ndcg


def vaild(prediction, label, top_n, pre, rec, map_, ndcg):
    """ For each sample in batch, compute metrics using top-n recommendations. """

    top_values, top_indices = tf.math.top_k(prediction, k = top_n, sorted=True)
    ## Carl - Convert to numpy array
    top_indices_np = top_indices.numpy()
    label_np = label.numpy()

    for top, l in zip(top_indices_np, label_np):
        try:
            l_filtered = l[l != 0] - 1
        except Exception as e:
            l_filtered = l[l != 0]
        recom_list = top
        ground_list = l_filtered
        if len(ground_list) == 0:
            continue
        # map2, mrr, ndcg2 = metric.map_mrr_ndcg(recom_list, ground_list)
        pre2, rec2, map2, ndcg2 = precision_recall_ndcg_at_k(top_n, recom_list, ground_list)
        pre.append(pre2), rec.append(rec2), map_.append(map2), ndcg.append(ndcg2)


def pre_rec_top_old(pre, rec, map_, ndcg, prediction, label, event_type):
    """ Filter out items already visited and compute metrics for each k in C.Ks. """

    ## Carl - Create a target mask
    batch_size = tf.shape(event_type)[0]
    # filter out the visited ITEM
    target_ = tf.ones((batch_size, C.ITEM_NUMBER), dtype = tf.float64)
    # Carl - Convert event_type to numpy
    event_type_np = event_type.numpy()
    for i, e in enumerate(event_type_np):
        ## Carl - Remove padding
        e_filtered = e[e != 0] - 1
        ## Carl - Create indices to set to zero
        indices = [[i, int(idx)] for idx in e_filtered]
        updates = [0.0] * len(e_filtered)
        ## Carl - Update mask tensor
        indices = tf.constant(indices, dtype=tf.int32)
        target_ = tf.tensor_scatter_nd_update(target_, indices, updates)
    ## Carl - Multiply prediction by target mask to zero visited items
    prediction_masked = prediction * tf.cast(target_, prediction.dtype)

    for i, topN in enumerate(C.Ks):
        if len(pre) <= i: pre.append([])
        if len(rec) <= i: rec.append([])
        if len(map_) <= i: map_.append([])
        if len(ndcg) <= i: ndcg.append([])
        vaild(prediction_masked, label, topN, pre[i], rec[i], map_[i], ndcg[i])

def pre_rec_top(pre, rec, map_, ndcg, prediction, label, event_type):
    """ Filter out items already visited and compute metrics for each k in C.Ks. """
    batch_size = tf.shape(event_type)[0]
    target_ = tf.ones((batch_size, C.ITEM_NUMBER), dtype=tf.float64)
    # Convert event_type to numpy for iteration
    event_type_np = event_type.numpy()
    for i, e in enumerate(event_type_np):
        # Remove padding and adjust indices (subtract 1)
        e_filtered = e[e != 0] - 1
        # Only update if there are valid indices
        if len(e_filtered) > 0:
            indices = [[i, int(idx)] for idx in e_filtered]
            updates = [0.0] * len(e_filtered)
            # Cast indices to int32
            indices = tf.constant(indices, dtype=tf.int32)
            target_ = tf.tensor_scatter_nd_update(target_, indices, updates)
    prediction_masked = prediction * tf.cast(target_, prediction.dtype)
    for i, topN in enumerate(C.Ks):
        if len(pre) <= i: pre.append([])
        if len(rec) <= i: rec.append([])
        if len(map_) <= i: map_.append([])
        if len(ndcg) <= i: ndcg.append([])
        vaild(prediction_masked, label, topN, pre[i], rec[i], map_[i], ndcg[i])


# def map_mrr_ndcg(rankedlist, test_matrix):
#     ap = 0
#     map = 0
#     dcg = 0
#     idcg = 0
#     mrr = 0
#     for i in range(len(test_matrix)):
#         idcg += 1 / math.log(i + 2, 2)
#
#     b1 = rankedlist
#     b2 = test_matrix
#     s2 = set(b2)
#     hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
#     count = len(hits)
#
#     for c in range(count):
#         ap += (c + 1) / (hits[c][0] + 1)
#         dcg += 1 / math.log(hits[c][0] + 2, 2)
#
#     if count != 0:
#         mrr = 1 / (hits[0][0] + 1)
#
#     if count != 0:
#         map = ap / count
#
#     return map, mrr, float(dcg / idcg)
