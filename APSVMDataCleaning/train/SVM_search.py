# Load modules

import time, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

# Load dict

with open('tags_dict.pickle', 'rb') as handle:
    tags_dict = pickle.load(handle)


# Define training inputs

dwts_norm = tags_dict['dwts_norm']
labels = tags_dict['dcLabel']
tsne = TSNE(n_components=2, random_state=0)

print("Starting TSNE for dwts_norm")
start_time = time.time()
dwts_norm_2d = tsne.fit_transform(dwts_norm)
np.save('dwts_norm_2d', dwts_norm_2d)
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))


# Create binary labels

idxs = np.where(labels!=0)[0]
binary_labels = np.zeros(labels.shape[0])
for i in range(labels.shape[0]):
    if labels[i] != 0:
        binary_labels[i] = 1

# Initialize optimization

C_dist = loguniform(1e-2, 1e10)
gamma_dist = loguniform(1e-9, 1e3)
param_dist = dict(gamma=gamma_dist, C=C_dist)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = SVC(random_state=0, 
          kernel='rbf', 
          decision_function_shape ='ovr', 
          class_weight='balanced')
grid = RandomizedSearchCV(estimator=clf,
                      param_distributions=param_dist, 
                      cv=cv,
                      n_iter=15)

# Run optimizations

print("Starting random hyperparameter search for dwts")
start_time = time.time()
grid.fit(dwts_norm, labels)
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))

print(
    "The best parameters for dwts are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

print("Starting random hyperparameter search for dwts_2d")
start_time = time.time()
grid.fit(dwts_norm_2d, labels)
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))

print(
    "The best parameters for dwts_2d are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

print("Starting random hyperparameter search for dwts_2d with binary labels")
start_time = time.time()
grid.fit(dwts_norm_2d, binary_labels)
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))

print(
    "The best parameters for dwts_2d with binary labels are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)