import tensorflow as tf
from data_utils import *
from model_utils import *
from metric_utils import *

def create_and_load_model(n, model_base_path, model_name):
    """
    Args:
        n: dimensionality of the ease model
        model_base_path: directory path where the models are located
        model_name: name of the pq file containing the model
    Returns:
        model: Model that is loaded from the paths given
    """
    model = Model(n)
    model.read_from_s3('{}{}'.format(model_base_path,model_name))
    return model

def print_and_plot(base_input_path, model_base_path, ease_model_name, ease_ipw_model_name, rlarm_model_name, rlarm_nn_model_name):
    """
    Args:
        base_input_path: path containing all the input data
        model_base_path: directory path where the models are located
        ease_model_name, ease_ipw_model_name, rlarm_model_name, rlarm_nn_model_name: Names of the files containing different models.
    Returns: nothing
    """

    #Load test data
    (test_tr, test_te) = get_test_data(base_input_path)

    #Load all four models
    models = {}
    models['ease'] = create_and_load_model(test_tr.shape[1], model_base_path, ease_model_name)
    models['ease_ipw'] = create_and_load_model(test_tr.shape[1], model_base_path, ease_ipw_model_name)
    models['rlarm'] = create_and_load_model(test_tr.shape[1], model_base_path, rlarm_model_name)
    models['rlarm_nn'] = create_and_load_model(test_tr.shape[1], model_base_path, rlarm_nn_model_name)

    #Compute metrics for all the models
    (ease_ndcg, ease_rc20, ease_rc50, ease_ndcg_list , ease_rc20_list, ease_rc50_list)= dcg_recall(test_te, models['ease'].pred(test_tr))
    (ease_ipw_ndcg, ease_ipw_rc20, ease_ipw_rc50, ease_ipw_ndcg_list , ease_ipw_rc20_list, ease_ipw_rc50_list)= dcg_recall(test_te, models['ease_ipw'].pred(test_tr))
    (rlarm_ndcg, rlarm_rc20, rlarm_rc50, rlarm_ndcg_list , rlarm_rc20_list, rlarm_rc50_list)= dcg_recall(test_te, models['rlarm'].pred(test_tr))
    (rlarm_nn_ndcg, rlarm_nn_rc20, rlarm_nn_rc50, rlarm_nn_ndcg_list , rlarm_nn_rc20_list, rlarm_nn_rc50_list)= dcg_recall(test_te, models['rlarm_nn'].pred(test_tr))

    print('NDCG: ease, ease_ipw, rlarm, rlarm_nn')
    print(ease_ndcg.numpy(), ease_ipw_ndcg.numpy(), rlarm_ndcg.numpy(), rlarm_nn_ndcg.numpy())
    print('recall@20: ease, ease_ipw, rlarm, rlarm_nn')
    print(ease_rc20.numpy(), ease_ipw_rc20.numpy(), rlarm_rc20.numpy(), rlarm_nn_rc20.numpy())
    print('recall@50: ease, ease_ipw, rlarm, rlarm_nn')
    print(ease_rc50.numpy(), ease_ipw_rc50.numpy(), rlarm_rc50.numpy(), rlarm_nn_rc50.numpy())

    ease = []
    ease_ipw = []
    rlarm = []
    rlarm_nn = []
    from scipy import stats
    for q in [  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        
        #Find the users with the worst EASE performance
        #Those users are indicated by ind and then compute
        #metrics for those users.

        ind = (ease_ndcg_list) < np.quantile(ease_ndcg_list, [q])
        ease.append(np.mean(ease_ndcg_list[ind]))
        ease_ipw.append((np.mean(ease_ipw_ndcg_list[ind])- np.mean(ease_ndcg_list[ind]))*100/np.mean(ease_ndcg_list[ind]))
        rlarm.append((np.mean(rlarm_ndcg_list[ind]) - np.mean(ease_ndcg_list[ind]))*100/np.mean(ease_ndcg_list[ind]))
        rlarm_nn.append((np.mean(rlarm_nn_ndcg_list[ind]) - np.mean(ease_ndcg_list[ind]))*100/np.mean(ease_ndcg_list[ind]))
        print(stats.ttest_rel(ease_ipw_ndcg_list[ind], rlarm_ndcg_list[ind]))
        print(stats.ttest_rel(ease_ipw_ndcg_list[ind], rlarm_nn_ndcg_list[ind]))
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] =  42
    fig=plt.figure()
    X_axis = np.arange(10)+1
    plt.bar(x=X_axis - 0.3,height=ease_ipw, width= 0.3)
    plt.bar(x=X_axis + 0,height=rlarm, width= 0.3)
    plt.bar(x=X_axis + 0.3,height=rlarm_nn, width= 0.3)
    plt.xlabel('Bottom User Bucket', fontsize=14)
    plt.ylabel('Percent NDCG improvement', fontsize=14)
    plt.xticks([1, 2,4,6,8,10], [10, 20, 40, 60, 80, 100], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['EASE-IPW', 'R-LARM', 'R-LARM(NN)'])

if __name__ == '__main__':    
    base_input_path = './ml20m_data'
    models_base_path = './ml20m_model/'
    ease_model_name = 'closed_model_0.005_0.0'
    ease_ipw_model_name = 'closed_model_0.002_0.6'
    rlarm_model_name = 'model_0.005_2000.0_5e-05'
    rlarm_nn_model_name = 'model_0.005_500.0_5_5e-05'
    print_and_plot(base_input_path, models_base_path, ease_model_name, ease_ipw_model_name, rlarm_model_name, rlarm_nn_model_name)

