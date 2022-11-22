# Load modules

import pickle, time, math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances

# Load data dict
with open('data_dict_train.pickle', 'rb') as handle:
    data_dict = pickle.load(handle)
data_dict 

#Load dwts
dwts_norm = data_dict['dwts_norm']


#Precompute similarity matrix 

start_time = time.time()
similarities = -pairwise_distances(dwts_norm, metric='l1')
print("--- %s minutes elapsed for similarities---" % ((time.time() - start_time)/60)

median = np.median(similarities)
minimum = np.amin(similarities)
      
print('Median = ', median)
print('Min = ', minimum)


#Run initial round of AP

init_pref = -1*((abs(median) - abs(minimum)/2)
start_time = time.time()
init_ap = AffinityPropagation(max_iter=500,
                         convergence_iter=25, 
                         verbose=True,
                         random_state=0,
                         preference=init_pref,
                         damping=0.99,
                         affinity='precomputed').fit(X=similarities)
print("--- %s minutes elapsed for initial AP---" % ((time.time() - start_time)/60)

n_exemps_init = len(ap.cluster_centers_indices_)
print("Initial number of exemplars = %" % n_exemps_init)
      
# Run optimization
      
pref = init_pref 
n_exemps = n_exemps_init 

if n_exemps >= 30 and n_exemps <= 70:
      print("Optimal preference value = %" % pref)
      break
      
elif n_exemps < 30 or n_exemps > 70:
      #Have to automate 
      


