import tensorflow as tf
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np

class Model(object):
    '''
        Model class contains specifying a model dimensionality, and
        initializing it. It also has functions to predict from the model
        to save a model to a path and then also read from a given source 
        path.
    '''
    def __init__(self, n_dim, m_dim=0):
        '''
        This model is used for the EASE model, linear adversarial model as well as layers of the neural
        network in r-larm-nn.py.
        Args:
            n_dim: number of rows in the model
            m_dim: number of columns in the model. If specified as zero, number of columns is set to n_dim.

        '''
        if m_dim==0:
            m_dim = n_dim
        initializer = tf.keras.initializers.GlorotNormal()
        self.U = tf.Variable(initializer(shape=(n_dim, m_dim)))    

    def pred(self,X):
        '''
        Gives predictions from the EASE model.
        Args:
            X: the observation matrix (users by items)
        Returns:
            Predictions from the EASE model on X.
        '''
        return tf.linalg.matmul(X, tf.linalg.set_diag(self.U, tf.zeros(self.U.shape[0],1)), a_is_sparse=True)
    
    def save_model(self,path):
        '''
        Save the model to a specified location in parquet format.
        Args:
            path: path where the model needs to be sotred
        Returns:
            nothing
        '''
        df = pd.DataFrame(self.U.numpy())
        pq.write_table(pa.Table.from_pandas(df), path)

    def read_from_s3(self,path):
        '''
        Load a model in .pq format from a sepcified path
        Args:
            path: input path
        Returns:
            nothing, but assigns the read model self.U
        '''
        model_temp = tf.constant(pq.read_table(path).to_pandas().to_numpy())
        self.U.assign(model_temp)

class TrainParams(object):
    '''
        Model training hyper-parameters are specified in this class.
    '''
    def __init__(self, C, B=0.0, learning_rate=0.001, batch_size=1024, num_iter=100, dim=5):
        '''
        Args:
            C: l2 regularization on the model parameters
            B: l2 regularization on the adversarial model parameters
            learning_rate: learning rate of the optimizer
            batch_size: batch size used in learning, we sample batch_size rows and all columns from the input matrix
            num_iter: maximum number of epoches in training
            dim: dimensionality of the hidden layer in r-larm-nn.py
        '''
        self.C = tf.constant(C)
        self.B = tf.constant(B)
        self.learning_rate = tf.constant(learning_rate)
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.dim = dim
