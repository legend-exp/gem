## Training Workflow

1. Run the ```ap_search.py``` script to find optimal hyperparameters for AP. This optimization is memory intensive and parallelized, so we need to allocate enough memory and time using SLURM. Thus, we submit a job in the queue with the ```slurm_ap.sh``` script. To run this, open the terminal in NERSC and run the following command:

```bash
sbatch slurm_ap.sh
```

2. Open the ```APBrowser.ipynb``` notebook to label the training dataset with ML data cleaning tags.

3. Run the ```svm_search.py``` script to find optimal hyperparameters for SVM. This optimization is memory intensive and parallelized, so we need to allocate enough memory and time using SLURM. Thus, we submit a job in the queue with the ```slurm_svm.sh``` script. To run this, open the terminal in NERSC and run the following command:

```bash
sbatch slurm_svm.sh
```

4. Open the ```SVMBrowser.ipynb``` notebook to train the SVM with optimal hyperparameters and save the model. 
