import pickle, time, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

'''
Plot 3D embedded waveforms and SVM decision regions.
'''

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", help="file containing mesh data", type=str, required=True)
args = argparser.parse_args()

# Load files

with open(args.file, 'rb') as in_data:
    data_dict = pickle.load(in_data)
    
with open('../data/3d_svm_mesh.pickle', 'rb') as in_mesh:
    Z, n_points = pickle.load(in_mesh)

# Define plotting inputs 

dwts_norm_3d = data_dict['dwt_norm_3d']
labels = data_dict['dc_label']

# Set SVM decision region voxels

cubes = []

for i in range(int(labels.max()+1)):
    cube = np.where(Z == i, True, False)
    cubes.append(cube)

voxelarray = cubes[0] | cubes[1] | cubes[2] | cubes[3] | cubes[4] \
            | cubes[5] | cubes[6] | cubes[7] | cubes[8] | cubes[9] \
            | cubes[10] | cubes[11] | cubes[12] | cubes[13]

# Set categories and custom colormap

categories = ['Norm', 'NegGo', 'UpSlo', 'DownSlo', 'Spike', 
              'XTalk', 'SlowRise', 'EarlyTr', 'LateTr', 'Saturation',
              'SoftPile', 'HardPile', 'Bump', 'NoiseTr']
colors = ['blue', 'green', 'red', 'cyan', 'fuchsia', 
          'gold', 'indigo', 'grey', 'maroon', 'orange',
          'pink', 'yellow', 'sienna', 'purple']
cmap = ListedColormap(colors)

# Define voxel colors

voxcolors = np.empty(voxelarray.shape, dtype=object)

for i in range(int(labels.max()+1)):
    voxcolors[cubes[i]]  = colors[i]
    
# Define plotting mesh

plot_points = n_points + 1  # Need 1 more than the points used for the SVM prediction
x_min, x_max = dwts_norm_3d[:, 0].min()-1, dwts_norm_3d[:, 0].max()+1 
y_min, y_max = dwts_norm_3d[:, 1].min()-1, dwts_norm_3d[:, 1].max()+1
z_min, z_max = dwts_norm_3d[:, 2].min()-1, dwts_norm_3d[:, 2].max()+1

xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, plot_points),
                         np.linspace(y_min, y_max, plot_points),
                         np.linspace(z_min, z_max, plot_points))

# Plot SVM voxels and 3D embedded waveforms

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(projection='3d')
alphas = [0.4, 0.4, 0.4, 0.4, 0.4,
          0.4, 0.4, 0.4, 0.4, 0.4,
          0.4, 0.4, 0.4, 0.1]

print("Starting voxel plot")
start_time = time.time()
for i in range(int(labels.max()+1)):        
    ax.voxels(xx, yy, zz, cubes[i], 
          facecolors=voxcolors, 
          edgecolor=None, 
          alpha = alphas[i]) 
print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60))
    
scatter = ax.scatter3D(dwts_norm_3d[:, 0], 
                       dwts_norm_3d[:, 1], 
                       dwts_norm_3d[:, 2],
                       c=labels, cmap=cmap, 
                       edgecolors=None, marker='o', 
                       s=100, alpha=.9, label=labels)

ax.set_xlim3d([x_min, x_max]) 
ax.set_ylim3d([y_min, y_max])
ax.set_zlim3d([z_min, z_max])

handles = scatter.legend_elements()[0]
legend_labels = []

for i in list(set(labels)):
    legend_labels.append(categories[int(i)])

fig.legend(handles, legend_labels, fontsize=60, 
          loc='lower right', markerscale=2)

plt.savefig("3d_svm.pdf", dpi=700, format='pdf')

print("3D plot saved")


