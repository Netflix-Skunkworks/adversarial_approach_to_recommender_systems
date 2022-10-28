import tensorflow as tf
from data_utils import *
from model_utils import *
from metric_utils import *

@tf.function
def get_weights(U_adv,train):
    '''
    Return normalized adversarial weights.
    Args:
        U_adv: adversarial model
        train: training mini-batch
    Returns:
        normalized adversarial weights
    '''
    all_weight = tf.math.sigmoid(tf.linalg.matmul(train, U_adv, a_is_sparse=True))
    weight_matrix =  all_weight/tf.reduce_sum(all_weight)
    return weight_matrix

@tf.function
def learner_loss(U, U_adv, train, C, B):
    '''
    Args:
        U: learner model
        U_adv: adversarial model
        train: training mini-batch
        C: l2 regularization weight on the learner
        B: l2 regularization weight on adversary
    Returns:
        adversarial loss
    '''
    pred = tf.linalg.matmul(train, tf.linalg.set_diag(U, tf.zeros(U.shape[0],1)) , a_is_sparse=True)
    loss_matrix = tf.math.pow(pred - train,2.0)
    weight_matrix = get_weights(U_adv, train)
    target_loss = tf.reduce_sum(tf.multiply(loss_matrix, weight_matrix)) + C*tf.nn.l2_loss(U) - B*tf.nn.l2_loss(U_adv)
    return target_loss

@tf.function
def learner_loss_grad(U, U_adv, target,  C, B):
    '''
    Args:
        U: learner model
        U_adv: adversarial model
        target: training mini-batch
        C: l2 regularization weight on the learner
        B: l2 regularization weight on the adversary
    Returns:
        learner model gradient, loss
    '''

    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch([U])
        l= learner_loss(U, U_adv,  target,  C, B)
    gU = t.gradient(l, U)
    return (gU,l)

@tf.function
def adv_loss_grad(U, U_adv, target,  C, B):
    '''
    Args:
        U: learner model
        U_adv: adversarial model
        target: training mini-batch
        C: l2 regularization weight on the learner
        B: l2 regularization weight on the learner
    Returns:
        adversarial model gradient, loss
    '''
    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch([U_adv])
        l= -learner_loss(U, U_adv, target,  C, B)
    [gU_adv] = t.gradient(l, [U_adv])
    return (gU_adv,l)

def adv_loop(train, train_params, test_tr, test_te, valid_tr, valid_te):
    '''
    Args:
        train: A tensor containing training data (num users x num items)
        train_params: An instance of TrainParams with hyper-parameters for learning
        test_tr: A tensor of 'observed data' for test users (num test users x num items)
        test_te: A tensor of 'unobserved test data' for test users used for evaluation (num test users x num items)
        valid_tr: A tensor of 'observed data' for validation users (num valid users x num items)
        valid_te: A tensor of 'unobserved test data' for validation users used for evaluation (num valid users x num items)
    Returns:
        A learner model and an adversarial model trained with the given hyper-parameters.
    '''

    best_perf = 0.0
    no_progress_steps = 0.0
    model = Model(train.dense_shape[1].numpy())
    best_model = Model(train.dense_shape[1].numpy())
    adv_model = Model(train.dense_shape[1].numpy(), m_dim=1)
    best_adv_model = Model(train.dense_shape[1].numpy(), m_dim=1)
    target_index = tf.data.Dataset.from_tensor_slices(tf.constant([i for i in range(0, train.shape[0] - train_params.batch_size )]))
    target_index_ds = iter(target_index.repeat().shuffle(train.shape[0]).batch(1).prefetch(1))
    optimizer = tf.optimizers.Adam(learning_rate=train_params.learning_rate)
    optimizer1 = tf.optimizers.Adam(learning_rate=train_params.learning_rate)
    tot_loss = tf.constant(1e6)

    for it in range(0,train_params.num_iter):
        tot_loss = tf.constant(0.0)
        for i in range(0, int(train.shape[0]/train_params.batch_size)):
            iter_train = sparse_sample(train_params.batch_size, train, target_index_ds.next().numpy()[0])
            (pu, l1) = learner_loss_grad(model.U, adv_model.U, iter_train, train_params.C, train_params.B)
            optimizer.apply_gradients(zip([pu], [model.U]))
            tot_loss = tot_loss + l1
            if it >= 1:
                (pu, l1) = adv_loss_grad(model.U, adv_model.U,  iter_train, train_params.C, train_params.B)
                optimizer1.apply_gradients(zip([pu], [adv_model.U]))

        if (tf.math.is_inf(tot_loss) or tf.math.is_nan(tot_loss)):
            break
        if (it+1)%1==0:
            valid_pred = model.pred(valid_tr)
            (valid_ndcg, valid_rc20, valid_rc50, vndcg_closed, vrc20_closed, vrc50_closed)= dcg_recall(valid_te, valid_pred)
            target_pred = model.pred(test_tr)
            (test_ndcg, test_rc20, test_rc50, ndcg_closed, rc20_closed, rc50_closed)= dcg_recall(test_te, target_pred)
            print('adv_inter_result', it, train_params.C.numpy(), train_params.B.numpy(), train_params.learning_rate.numpy(),  valid_ndcg.numpy(), valid_rc20.numpy(), valid_rc50.numpy(), test_ndcg.numpy(), test_rc20.numpy(), test_rc50.numpy())
            if valid_ndcg > best_perf:
                best_perf = valid_ndcg
                best_model.U.assign(model.U)
                best_adv_model.U.assign(adv_model.U)
                no_progress_steps = 0.0
            else:
                no_progress_steps = no_progress_steps + 1.0
            if no_progress_steps > 5:
                break
    return (best_model, best_adv_model)


def training(train, test_tr, test_te, valid_tr, valid_te, train_params):
    '''
    Args:
        train: A tensor containing training data (num users x num items)
        test_tr: A tensor of 'observed data' for test users (num test users x num items)
        test_te: A tensor of 'unobserved test data' for test users used for evaluation (num test users x num items)
        valid_tr: A tensor of 'observed data' for validation users (num valid users x num items)
        valid_te: A tensor of 'unobserved test data' for validation users used for evaluation (num valid users x num items)
        train_params: An instance of TrainParams with hyper-parameters for learning
    Returns:
        A learner model and an adversarial model trained with the given hyper-parameters.
    '''
    tf.random.set_seed(10)
    train = tf.random.shuffle(train)
    train_sp = tf.sparse.from_dense(train)
    del train
    (model,adv_model)  = adv_loop(train_sp, train_params,  test_tr, test_te, valid_tr, valid_te)
    valid_pred = model.pred(valid_tr)
    (valid_ndcg, valid_rc20, valid_rc50, vndcg_closed, vrc20_closed, vrc50_closed)= dcg_recall(valid_te, valid_pred)
    target_pred = model.pred(test_tr)
    (test_ndcg, test_rc20, test_rc50, ndcg_closed, rc20_closed, rc50_closed)= dcg_recall(test_te, target_pred)
    print('adv_result', train_params.C.numpy(), train_params.B.numpy(), train_params.learning_rate.numpy(), valid_ndcg.numpy(), valid_rc20.numpy(), valid_rc50.numpy(), test_ndcg.numpy(), test_rc20.numpy(), test_rc50.numpy())
    return (model, adv_model)


def load_train_and_save(C, B, lr, base_input_path, base_output_path, batch_size=1024, num_iter=100):
    '''
    Load all the data, call the training function and save the model on the disk.
    Args:
        C: Frobrenious norm regularization level for learner
        B: Regularizaton level on the adversary
        lr: learning rate
        base_input_path: directory path to look for all the data for training/validaton and test
        base_output_path: directory path to store the models trained
        batch_size: training mini-batch number of users during gradient descent/ascent
        num_iter: maximum number of epochs through the dataset
    '''

    train_params = TrainParams(C=C, B=B, batch_size=batch_size, learning_rate=lr, num_iter=num_iter)
    (train, test_tr, test_te, valid_tr, valid_te) = get_all_data(base_input_path)
    (model,  adv_model)  = training(train, test_tr, test_te, valid_tr, valid_te, train_params)
    path = '{}/adv_model_{}_{}_{}'.format(base_output_path, str(C), str(B), str(lr))  
    adv_model.save_model(path)
    path = '{}/model_{}_{}_{}'.format(base_output_path, str(C), str(B), str(lr))  
    model.save_model(path)


if __name__ == '__main__':
    base_output_path = './toy_model'
    base_input_path = './toy_data'
    C=0.5
    B=1000000.0
    lr=2e-4
    load_train_and_save(C, B, lr, base_input_path, base_output_path,  num_iter=100) 
