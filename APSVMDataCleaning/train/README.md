## Training Workflow

The automated hyperparameter search for AP is under development and found in ```AP_search.py```.

1. Run the ```AP.ipynb``` notebook to find clusters and label the training dataset with data cleaning category tags
2. Run the ```APBrowser.ipynb``` notebook to look at clustering statistics 
3. Run the ```SVM_search.py``` script to find optimal hyperparameters for the SVM classifiers
4. Run the ```SVM.ipynb``` notebook to train the SVM with optimal hyperparameters and to visualize the classification in 2D
5. Copy the ```svm.sav``` file containing the trained SVM model into the ``test`` directory

