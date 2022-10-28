import tensorflow as tf
from data_utils import *
from model_utils import *
from metric_utils import *

@tf.function
def closed_sol(train, C):
    '''
    Closed form solution of ease
    Args:
        train: training observations (users by items and binary)
        C: Frorenius norm regularization hyper-parameter
    Returns:
        closed form solution of EASE
    '''
    size= train.shape[0]
    temp = tf.linalg.inv(
            tf.linalg.matmul(train, train, transpose_a=True, a_is_sparse=True, b_is_sparse=True)/size
              + C*tf.eye(train.shape[1])
            )
    temp = tf.eye(temp.shape[0])- temp*tf.reshape(1.0/tf.linalg.diag_part(temp), shape=(1,-1))
    return temp

def closed_loop(train, train_params):
    '''
    Just a convenience function to wrap the EASE solution in the Model class.
    Args:
        train: training observations (users by items and binary)
        train_params: An instance of Train_params class with specifically C intialized
    Returns:
        Model containing the EASE solution.
    '''
    model = Model(train.shape[1])
    model.U.assign(closed_sol(train, train_params.C))
    return model

def training(train, test_tr, test_te, valid_tr, valid_te, train_params):
    '''
    Args:
        train: A tensor containing training data (num users x num items)
        test_tr: A tensor of 'observed data' for test users (num test users x num items)
        test_te: A tensor of 'unobserved test data' for test users used for evaluation (num test users x num items)
        valid_tr: A tensor of 'observed data' for validation users (num valid users x num items)
        valid_te: A tensor of 'unobserved test data' for validation users used for evaluation (num valid users x num items)
    Returns:
        closed_model: An EASE model or an EASE-IPW model depending on the model parameters.

    '''

    #For normal EASE train_params.B is set to zero and the weights are all 1.
    #For EASE-IPW train_params.B is set to a non-zero value and the weights are determined based on it
    #EASE-IPW requires only a modification in the data to get the right solution (refer to the paper)
    weights = tf.math.pow(tf.reduce_sum(train,axis=1), -train_params.B)
    weights = tf.reshape(tf.math.sqrt(weights*train.shape[0]/tf.reduce_sum(weights)), (-1,1))
    train = weights*train

    closed_model = closed_loop(train, train_params)

    valid_pred = closed_model.pred(valid_tr)
    (valid_ndcg, valid_rc20, valid_rc50, vndcg_closed, vrc20_closed, vrc50_closed)= dcg_recall(valid_te, valid_pred)

    target_pred = closed_model.pred(test_tr)
    (test_ndcg, test_rc20, test_rc50, ndcg_closed, rc20_closed, rc50_closed)= dcg_recall(test_te, target_pred)
    print('ease_result', train_params.C.numpy(), train_params.B.numpy(), valid_ndcg.numpy(), valid_rc20.numpy(), valid_rc50.numpy(), test_ndcg.numpy(), test_rc20.numpy(), test_rc50.numpy())
    return closed_model

def load_train_and_save(C,B,base_input_path, base_output_path):
    '''
    Load all the data, call the training function and save the model on the disk.
    Args:
        C: Frobrenious norm regularization level for EASE
        B: When B=0, we get EASE. When B>0, we get EASE-IPW.
        base_input_path: directory path to look for all the data for training/validaton and test
        base_output_path: directory path to store the models trained
    '''
    (train, test_tr, test_te, valid_tr, valid_te) = get_all_data(base_input_path)
    train_params = TrainParams(C=C,B=B)
    model = training(train, test_tr, test_te, valid_tr, valid_te, train_params)

    #Saved model will contain the parameters used for training in the name.
    path = '{}/closed_model_{}_{}'.format(base_output_path, str(C), str(B))
    model.save_model(path)

if __name__ == '__main__':
    base_output_path = './toy_model'
    base_input_path = './toy_data'

    #For EASE, set B to zero and C for the amount of regularization required on the model.
    C=0.005
    B=0.0
    load_train_and_save(C, B, base_input_path, base_output_path)

    #For EASE-IPW set B to a non-zero value
    C=0.002
    B=0.6
    load_train_and_save(C, B, base_input_path, base_output_path)

