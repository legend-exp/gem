import time, pickle, json, argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

'''
Find the optimal hyperparameters for a 3D SVM.
'''

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", help="file containing training data", type=str, required=True)
args = argparser.parse_args()

# Load files

with open(args.file, 'rb') as in_data:
    data_dict = pickle.load(in_data)
    
with open('../data/hyperparameters.json', 'r') as in_hyper:
    hyperparams_dict = json.load(in_hyper)

# Define training inputs

dwts_norm = data_dict['dwt_norm']
labels = data_dict['dc_label']
SVM_hyperparams = hyperparams_dict['SVM_3D']
TSNE_hyperparams = hyperparams_dict['TSNE_3D']

# Initialize 3D TSNE with optimal hyperparameters

tsne3d = TSNE(n_components= TSNE_hyperparams['n_components'], 
                   perplexity = float(TSNE_hyperparams['perplexity']),
                   early_exaggeration= TSNE_hyperparams['early_exaggeration'],
                   random_state= TSNE_hyperparams['random_state'],
                   learning_rate= float(TSNE_hyperparams['learning_rate']), 
                   metric = TSNE_hyperparams['metric'],
                   init = TSNE_hyperparams['init'], 
                   n_jobs = -1) 

# Close input files

in_data.close()
in_hyper.close()

# Get 3D waveforms with TSNE

print("Starting 3D TSNE")
start_time = time.time()
dwts_norm_3d = tsne3d.fit_transform(dwts_norm)
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))

data_dict['dwt_norm_3d'] = dwts_norm_3d


# Initialize 3D SVM optimization

C_dist = loguniform(1e-2, 1e10)
gamma_dist = loguniform(1e-9, 1e3)
param_dist = dict(gamma=gamma_dist, C=C_dist)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

clf = SVC(random_state=SVM_hyperparams['random_state'], 
          kernel=SVM_hyperparams['kernel'], 
          decision_function_shape =SVM_hyperparams['decision_function_shape'], 
          class_weight=SVM_hyperparams['class_weight'])

grid = RandomizedSearchCV(estimator=clf,
                      param_distributions=param_dist,
                      n_jobs = -1, 
                      cv=cv,
                      n_iter=10)



print("Starting random hyperparameter search")
start_time = time.time()
grid.fit(dwts_norm_3d, labels)
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

hyperparams_dict['SVM_3D']['C'] = str(grid.best_params_['C'])
hyperparams_dict['SVM_3D']['gamma'] = str(grid.best_params_['gamma'])
hyperparams_dict['SVM_3D']['score'] = str(grid.best_score_)

with open("../data/hyperparameters.json", "w") as out_hyper:
    json.dump(hyperparams_dict, out_hyper)
    
with open(args.file, "wb") as out_data:
    pickle.dump(data_dict, out_data)
    
print("Hyperparameters and 3D waveforms saved")