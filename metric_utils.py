import math
import numpy as np
import random
import tensorflow as tf

@tf.function
def dcg_recall(datax, target_pred, k=100, r1=20, r2=50):
    '''
    Computes NDCG@k, recall@r1 and recall@r2
    Args:
        datax: ground truth test observations (users by items)
        target_pred: predictions from a model for each user and item
        k: compute ndcg at k
        r1,r2: compute recall at r1 and r2
    Returns:
        dcg: mean NDCG@k (mean over all users)
        final_recall_1: mean recall@r1
        final_recall_2: mean recall@r2
        dcg_by_row: NDCG@k by user (one element per user)
        recall_1: recall@r1 by user
        recall_2: recall@r2 by user

    '''
    print('computing dcg and recall\n')
    target_pred = target_pred -100*tf.cast(datax < 0, dtype=tf.float32)
    top_k_pred_sort = tf.math.top_k(target_pred, k=k)
    top_k_label_sort = tf.math.top_k(datax, k=k)

    row_index = tf.reshape(tf.constant([[j for _ in range(0,k)] for j in range(0,datax.shape[0])]), shape=(-1,1))
    col_index = tf.reshape(top_k_pred_sort.indices, shape=(-1,1))
    top_k_pred_sort_labels = tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))

    discount_weights = tf.constant([ math.log(2.0 + j)/math.log(2.0) for j in range(0,k)])
    dcg_by_row = tf.reduce_sum(tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))/discount_weights, axis=1)

    col_index = tf.reshape(top_k_label_sort.indices, shape=(-1,1))
    ideal_dcg_by_row = tf.reduce_sum(tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))/discount_weights, axis=1)
    ind = ideal_dcg_by_row > 0

    dcg = tf.reduce_mean(dcg_by_row[ind]/ideal_dcg_by_row[ind])
    dcg_by_row = dcg_by_row[ind]/ideal_dcg_by_row[ind]

    recall_1 = tf.reduce_sum(tf.gather(top_k_pred_sort_labels, [_ for _ in range(0,r1)], axis=1), axis=1)
    recall_2 = tf.reduce_sum(tf.gather(top_k_pred_sort_labels, [_ for _ in range(0,r2)], axis=1), axis=1)
    label_sum = tf.reduce_sum(datax*tf.cast(datax >0, dtype=tf.float32), axis=1)
    ind = label_sum > 0
    final_recall_1 = tf.reduce_mean(recall_1[ind]/tf.math.minimum(label_sum[ind], r1))
    final_recall_2 = tf.reduce_mean(recall_2[ind]/tf.math.minimum(label_sum[ind], r2))
    recall_1 = recall_1[ind]/tf.math.minimum(label_sum[ind], r1)
    recall_2 = recall_2[ind]/tf.math.minimum(label_sum[ind], r2)
    return (dcg, final_recall_1, final_recall_2, dcg_by_row, recall_1, recall_2)

@tf.function
def get_dcg(datax, target_pred, k=100):

    '''
    Computes NDCG@k 
    Args:
        datax: ground truth test observations (users by items)
        target_pred: predictions from a model for each user and item
        k: compute ndcg at k
    Returns:
        dcg: mean NDCG@k (mean over all users)
        dcg_by_row: NDCG@k by user (one element per user)
    '''

    target_pred = target_pred -100*tf.cast(datax < 0, dtype=tf.float32)
    top_k_pred_sort = tf.math.top_k(target_pred, k=k)
    top_k_label_sort = tf.math.top_k(datax, k=k)

    row_index = tf.reshape(tf.constant([[j for _ in range(0,k)] for j in range(0,datax.shape[0])]), shape=(-1,1))
    col_index = tf.reshape(top_k_pred_sort.indices, shape=(-1,1))
    top_k_pred_sort_labels = tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))

    discount_weights = tf.constant([ math.log(2.0 + j)/math.log(2.0) for j in range(0,k)])
    dcg_by_row = tf.reduce_sum(tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))/discount_weights, axis=1)

    col_index = tf.reshape(top_k_label_sort.indices, shape=(-1,1))
    ideal_dcg_by_row = tf.reduce_sum(tf.reshape(tf.gather_nd(datax,tf.cast(tf.concat([row_index, col_index], axis=1), dtype=tf.int64)), (datax.shape[0],k))/discount_weights, axis=1)
    ind = ideal_dcg_by_row > 0

    dcg = tf.reduce_mean(dcg_by_row[ind]/ideal_dcg_by_row[ind])
    dcg_by_row = dcg_by_row[ind]/ideal_dcg_by_row[ind]

    return (dcg,dcg_by_row)

