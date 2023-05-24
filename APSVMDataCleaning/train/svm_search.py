import time, pickle, json, argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

'''
Use this script to find the optimal hyperparameters for a SVM.
'''

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", help="file containing training data", type=str, required=True)
args = argparser.parse_args()

# Load files

with open(args.file, 'rb') as handle:
    data_dict = pickle.load(handle)
    
with open('../data/hyperparameters.json', 'rb') as infile:
    hyperparams_dict = json.load(infile)

# Define training inputs

dwts_norm = data_dict['dwt_norm']
labels = data_dict['dc_label']
SVM_hyperparams = hyperparams_dict['SVM']


# Initialize optimization

print("Initializing optimization grid")

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
                          cv=cv,
                          n_iter=10,
                          n_jobs=-1 
                         )
# Close input file 

infile.close()

# Run optimizations

print("Starting random hyperparameter search")
start_time = time.time()
grid.fit(dwts_norm, labels)
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))

print(
    "The best parameters for the SVM are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

hyperparams_dict['SVM']['C'] = grid.best_params_['C']
hyperparams_dict['SVM']['gamma'] = grid.best_params_['gamma']
hyperparams_dict['SVM']['score'] = grid.best_score_

with open("../data/hyperparameters.json", "w") as outfile:
    json.dump(hyperparams_dict, outfile)
    
# Close output file 

outfile.close()
    
print("Hyperparameters saved")