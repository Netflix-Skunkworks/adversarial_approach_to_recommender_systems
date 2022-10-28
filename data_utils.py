import tensorflow as tf
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np

def load_data(full_path):
    """
    Args:
        full_path: full path to .pq data to be loaded
    Return:
        Loaded data as a tensor 
    """
    return tf.constant(pq.read_table(full_path).to_pandas().to_numpy()) 

def get_all_data(path):
    """
    Args:
        path: directory path where all the data resides with specific names for train, test_tr
        test_te, valid_tr, valid_te
    Return:
        All five data as tensors.
    """
    train = load_data('{}/{}'.format(path,'train.pq'))
    test_tr = load_data('{}/{}'.format(path,'test_tr.pq'))
    test_te = load_data('{}/{}'.format(path,'test_te.pq'))
    valid_tr = load_data('{}/{}'.format(path,'valid_tr.pq'))
    valid_te = load_data('{}/{}'.format(path,'valid_te.pq'))
    return (train, test_tr, test_te, valid_tr, valid_te)

def get_test_data(path):
    """
    Args:
        path: directory path where all the data resides with specific names test_tr, test_te
    Return:
        test_tr and test_te as tensors.
    """
    test_tr = load_data('{}/{}'.format(path,'test_tr.pq'))
    test_te = load_data('{}/{}'.format(path,'test_te.pq'))
    return (test_tr, test_te) 

def sparse_sample(batch_size, train, t_temp):
    """
    Args:
        batch_size: training batch size
        train: sparse training data
        t_temp: the index at which we need to slice batch_size rows
    Return:
        iter_train: sampled data converted to dense format
    """
    if (t_temp <= train.dense_shape[0] - batch_size):
       temp = tf.sparse.slice(train, start=[t_temp,0], size=[batch_size,train.dense_shape[1].numpy()])
       iter_train = tf.sparse.to_dense(temp)
    else:
       first_batch_size = train.shape[0] - t_temp
       temp = tf.sparse.slice(train, start=[t_temp,0], size=[first_batch_size,train.dense_shape[1].numpy()])
       iter_train1 = tf.sparse.to_dense(temp)
       second_batch_size = train_params.batch_size - first_batch_size
       temp = tf.sparse.slice(train, start=[t_temp,0], size=[second_batch_size,train.dense_shape[1].numpy()])
       iter_train2 = tf.sparse.to_dense(temp)
       iter_train = tf.concat([iter_train1, iter_train2], 0)
    return iter_train
