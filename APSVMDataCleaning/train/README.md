## Training Workflow

1. Run the ```APSearch.ipynb``` notebook to find optimal hyperparameters for AP. The optimization is done via a grid search, and each grid point iteration requires about 4 GB of RAM. For large grids (> 16 points), run the ```ap_search.py``` script in a large memory computing node. The script takes in a train dataset file and a desired number of clusters as required arguments  
(i.e. ```python3 ap_search.py --file <path_to_file> --nexemplars 100```).
2. Run the ```APBrowser.ipynb``` notebook to label the training dataset with data cleaning tags.
3. Run the ```svm_search.py``` script to find optimal hyperparameters for the SVM. The script takes in a train dataset file and a desired number of clusters as required arguments  
(i.e. ```python3 svm_search.py --file <path_to_file>```)
4. Run the ```SVMBrowser.ipynb``` notebook to train the SVM with optimal hyperparameters and obtain a 2D visualization of the model's decision regions. 

