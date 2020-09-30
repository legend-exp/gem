import lightgbm as lgb
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from ax import *
from ax.plot.scatter import plot_fitted
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.stats.statstools import agresti_coull_sem


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Setting reproducability
manualSeed = 158138

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)


torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

VERSION="kamnet3"
FIRST_ARM=100
IT_ARM=10
ITRATION=100

def run_trial():

    # In[136]:


    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 100,
        'learning_rate': 0.01,
        'pos_bagging_fraction': 1.0,
        'neg_bagging_fraction': float(len(signal_train))/float(len(bkg_train)),
        'bagging_freq': 5,
        'verbose': 0
    }
    print(float(len(signal_train))/float(len(bkg_train)))


    # In[137]:


    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=20)


    # In[138]:


    feature_name = ["isEnr", "gain high", "tDrift", "Final_Energy", "avse", "dcr"]

    print('7th feature name is:', feature_name[4])

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Dumping model to JSON...')
    # dump model to JSON (and save to file)
    model_json = gbm.dump_model()


    # In[139]:


    #Evaluation
    def loglikelihood(preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad, hess

    def binary_error(preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        return 'error', np.mean(labels != (preds > 0.5)), False

    # another self-defined eval metric
    # f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    # accuracy
    # NOTE: when you do customized loss function, the default prediction value is margin
    # This may make built-in evalution metric calculate wrong results
    # For example, we are doing log likelihood loss, the prediction is score before logistic transformation
    # Keep this in mind when you use the customization
    def accuracy(preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        return 'accuracy', np.mean(labels == (preds > 0.5)), True


    # In[140]:


    from sklearn.metrics import mean_squared_error
    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(Y_test, y_pred) ** 0.5)
    print(y_pred, Y_test)


    # In[141]:


    rg=np.arange(0.0,1.0,0.01)
    plt.hist(y_pred[Y_test==1], label="Signal", bins=rg, histtype="step", density=True)
    plt.hist(y_pred[Y_test==0], label="Background",bins=rg, histtype="step", density=True)
    plt.legend()
    plt.xlabel("BDT output")
    plt.ylabel("% per 0.01 bin(a.u.)")
    plt.savefig("BDT_output.png")


def main(cache_dir):
signaldata = np.genfromtxt('DS6.csv', delimiter=',')
    bkgdata = np.genfromtxt('DS6cal.csv', delimiter=',')
    # signaldata = signaldata[signaldata[:,3]>500.0]
    # bkgdata = bkgdata[bkgdata[:,3]>500.0]
    print(signaldata.shape, bkgdata.shape)
    rg = np.arange(0.0,5000,1.0)
    plt.yscale("log")
    plt.hist(signaldata[:,3],bins=rg,histtype="step")
    plt.hist(bkgdata[:,3],bins=rg,histtype="step")
    plt.title("Before energy matching")
    plt.savefig("energy_dist_before_matching.png")
    plt.cla()
    plt.clf()
    plt.close()


    # In[144]:


    sigindex=[]
    bkgindex=[]
    for energy in np.arange(500,3000,1.0):
        sig_in_range = np.where(np.logical_and(signaldata[:,3]>=energy, signaldata[:,3]<energy+1.0))[0]
        bkg_in_range = np.where(np.logical_and(bkgdata[:,3]>=energy, bkgdata[:,3]<energy+1.0))[0]
        min_entry = min(len(sig_in_range), len(bkg_in_range))
        sigindex += list(np.random.choice(sig_in_range, min_entry,replace=False))
        bkgindex += list(np.random.choice(bkg_in_range, min_entry,replace=False))
    plt.yscale("log")
    plt.hist(signaldata[sigindex,3],bins=rg,histtype="step")
    plt.hist(bkgdata[bkgindex,3],bins=rg,histtype="step")
    plt.title("Before energy matching")
    signaldata = signaldata[sigindex]
    bkgdata = bkgdata[bkgindex]
    plt.savefig("energy_dist_after_matching.png")
    plt.cla()
    plt.clf()
    plt.close()


    #split signal dataset
    test_split = 0.2
    indices = np.arange(signaldata.shape[0])
    np.random.shuffle(indices)
    train_index = indices[int(len(indices)*test_split):]
    test_index = indices[:int(len(indices)*test_split)]
    signal_train = signaldata[train_index,:-1]
    signal_test = signaldata[test_index,:-1]
    siglabel_train = np.ones(signal_train.shape[0])
    siglabel_test = np.ones(signal_test.shape[0])

    #split bkg dataset
    indices = np.arange(bkgdata.shape[0])
    np.random.shuffle(indices)
    train_index = indices[int(len(indices)*test_split):]
    test_index = indices[:int(len(indices)*test_split)]
    bkg_train = bkgdata[train_index,:-1]
    bkg_test = bkgdata[test_index,:-1]
    bkglabel_train = np.zeros(bkg_train.shape[0])
    bkglabel_test = np.zeros(bkg_test.shape[0])


    X_train = np.concatenate([signal_train, bkg_train],axis = 0)
    Y_train = np.concatenate([siglabel_train, bkglabel_train],axis = 0)
    train_index = np.arange(len(X_train))
    np.random.shuffle(train_index)
    X_train = X_train[train_index]
    Y_train = Y_train[train_index]
    X_test = np.concatenate([signal_test, bkg_test],axis = 0)
    Y_test = np.concatenate([siglabel_test, bkglabel_test],axis = 0)
    test_index = np.arange(len(X_test))
    np.random.shuffle(test_index)
    X_test = X_test[test_index]
    Y_test = Y_test[test_index]
    X_val, X_test = np.split(X_test,2)
    Y_val, Y_test = np.split(Y_test,2)


    # In[108]:


    lgb_train = lgb.Dataset(X_train, Y_train,free_raw_data=False, feature_name = ["isEnr", "gain high", "tDrift", "Final_Energy", "avse", "dcr"]
    )
    lgb_eval = lgb.Dataset(X_val, Y_val, reference=lgb_train,free_raw_data=False, feature_name = ["isEnr", "gain high", "tDrift", "Final_Energy", "avse", "dcr"]
    )

    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'metric': 'binary_logloss',
    #     'num_leaves': 100,
    #     'learning_rate': 0.01,
    #     'pos_bagging_fraction': 1.0,
    #     'neg_bagging_fraction': float(len(signal_train))/float(len(bkg_train)),
    #     'bagging_freq': 5,
    #     'verbose': 0
    # }

    #List of Parameters
    p1 = ChoiceParameter(name="boosting", values=["gbdt", "rf", "dart", "goss"], parameter_type=ParameterType.STRING)
    p2 = RangeParameter(name="num_iterations", lower=50, upper=1000, parameter_type=ParameterType.INT)
    p3  = RangeParameter(name="learning_rate", lower=1e-4, upper=0.5, parameter_type=ParameterType.FLOAT)
    p4 = RangeParameter(name="num_leaves", lower=1, upper=300, parameter_type=ParameterType.INT)


    search_space = SearchSpace(
        parameters=[p1,p2,p3,p4],
    )

    experiment = Experiment(
        name="hyper_parameter_optimization",
        search_space=search_space,
    )

    sobol = Models.SOBOL(search_space=experiment.search_space)
    generator_run = sobol.gen(FIRST_ARM)

    class MyRunner(Runner):
        def __init__(self):
            '''
            nothing
            '''

        def run(self, trial):
            arm_result = []
            for arm_name, arm in trial.arms_by_name.items():
                params = arm.parameters
                print(arm.parameters)
                # train_loader = data_utils.DataLoader(self.dataset, batch_size=params["BATCH_SIZE"], sampler=self.train_sampler, drop_last=True, num_workers = 0)
                # test_loader = data_utils.DataLoader(self.dataset, batch_size=params["BATCH_SIZE"], sampler=self.test_sampler, drop_last=True, num_workers = 0)
                auc = train_network(params, dataset, rtq_dataset)
                arm_result.append(float(auc))
                gc.collect()
                wrappers = [a for a in gc.get_objects() 
                    if isinstance(a, functools._lru_cache_wrapper)]

                for wrapper in wrappers:
                    wrapper.cache_clear()
            return {"name": str(trial.index), "auc": arm_result}

    experiment.runner = MyRunner()
    experiment.new_batch_trial(generator_run=generator_run)

    experiment.trials[0].run()

    class BoothMetric(Metric):
        def fetch_trial_data(self, trial):  
            records = []
            auc_result = trial.run_metadata["auc"]
            index = 0
            for arm_name, arm in trial.arms_by_name.items():
                params = arm.parameters
                records.append({
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "mean": auc_result[index],
                    "sem": 0.0,
                    "trial_index": trial.index
                })
                index += 1
            return Data(df=pd.DataFrame.from_records(records))

    optimization_config = OptimizationConfig(
        objective = Objective(
            metric=BoothMetric(name="booth"), 
            minimize=False,
        )
    )


    experiment.optimization_config = optimization_config

    for i in tqdm(range(1, ITRATION)):
        with open('data%d.json'%(i), 'w') as outfile:
            data = experiment.fetch_data()
            gpei = Models.GPEI(experiment=experiment, data=data)
            generator_run = gpei.gen(IT_ARM)
            experiment.new_batch_trial(generator_run=generator_run)
            experiment.trials[i].run()
            data = experiment.fetch_data()
            df = data.df
            print(df)
            best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
            best_arm = experiment.arms_by_name[best_arm_name]
            print(best_arm)
            json_field = best_arm.parameters
            json_field["roc"] = df['mean'].max()
            json.dump(json_field, outfile)
            df.to_json(r'arms.json')





