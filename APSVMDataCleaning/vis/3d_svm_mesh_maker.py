import pickle, json, argparse, time 
import numpy as np
from sklearn.svm import SVC

'''
Predict SVM labels on a 3D mesh.
'''

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", help="file containing training data", type=str, required=True)
argparser.add_argument("--npoints", help="number of points in mesh", type=int, required=True)
args = argparser.parse_args()

# Load files

with open(args.file, 'rb') as in_data:
    data_dict = pickle.load(in_data)
    
with open('../data/hyperparameters.json', 'rb') as in_hyper:
    hyperparams_dict = json.load(in_hyper)

# Define training inputs
    
dwts_norm = data_dict['dwt_norm']
dwts_norm_3d = data_dict['dwt_norm_3d']
labels = data_dict['dc_label']
SVM_hyperparams = hyperparams_dict['SVM_3D']

# Train the 3D SVM

svm = SVC(random_state=SVM_hyperparams['random_state'], 
          kernel=SVM_hyperparams['kernel'], 
          decision_function_shape=SVM_hyperparams['decision_function_shape'],
          class_weight=SVM_hyperparams['class_weight'],
          C=SVM_hyperparams['C'],
          gamma=SVM_hyperparams['gamma'])

svm.fit(dwts_norm_3d,labels)

# Create the 3D mesh

n_points = args.n_points
x_min, x_max = dwts_norm_3d[:, 0].min() - 1, dwts_norm_3d[:, 0].max() + 1
y_min, y_max = dwts_norm_3d[:, 1].min() - 1, dwts_norm_3d[:, 1].max() + 1
z_min, z_max = dwts_norm_3d[:, 2].min() - 1, dwts_norm_3d[:, 2].max() + 1

xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, n_points),
                         np.linspace(y_min, y_max, n_points),
                         np.linspace(z_min, z_max, n_points))

# Predict labels on the 3D mesh

print("Starting mesh prediction")
start_time = time.time()
Z = svm.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))

Z = Z.reshape(xx.shape)

# Save mesh labels

with open('../data/3d_svm_mesh.pickle', 'wb') as f:
    pickle.dump((Z, n_points), f)

