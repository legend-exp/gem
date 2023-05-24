## Visualization Workflow 

1. Run the ```tsne_grid_maker.py``` script to generate a grid of 3D embedded waveforms with different hyperparameters of a t-distributed Stochastic Neighor Embedding (TSNE) model. The script takes in a train dataset file as a required argument  
(i.e. ```python3 tsne_grid_maker.py --file <path_to_file>```).
2. Run the ```TSNEGridPlotter.ipynb``` notebook to plot the grid of 3D embedded waveforms and choose a set hyperparameters that render the best visualization.
3. Run the ```3d_svm_search.py``` script to find optimal hyperparameters for a 3D SVM. The script takes in a train dataset file as a required argument  
(i.e. ```python3 3d_svm_search.py --file <path_to_file>```).
4. Run the ```3d_svm_mesh_maker.py``` script to create a 3D mesh of predicted decision regions by the 3D SVM and save the mesh data onto the ```data``` directory. The script takes in a train dataset file name and the number of points for the mesh as required arguments  
(i.e. ```python3 3d_svm_mesh_maker.py --file <path_to_file> --npoints 100```).
5. Run the ```3d_svm_mesh_plotter.py```script to plot the 3D embedded waveforms and the 3D SVM decision regions. The script takes in a mesh data file as a required argument  
(i.e. ```python3 3d_svm_mesh_plotter.py --file <path_to_file>```).