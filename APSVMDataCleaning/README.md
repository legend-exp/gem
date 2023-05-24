 AP-SVM Data Cleaning
=============

The Affinity Propagation - Support Vector Machine (AP-SVM) data cleaning model code for LEGEND is found here.

## Software requirements

- Python >= 3.9
- Jupyter
- Machine learning packages: scikit-learn
- Other packages: [pygama](https://github.com/legend-exp/pygama), ipywidgets, seaborn

## General workflow

1. Open the  ```data``` directory. There you will find notebooks to create train/test datasets, .json configuration files, and other pickled data files. The first step in the model is to create a train dataset.

2. Open the ```train``` directory. There you will find notebooks and scripts to train AP and SVM. The trained models are also stored in the ```data``` directory. This directory also contains a notebook that renders a 2D visualization of the trained SVM.

3. Open the ```vis``` directory. There you will find notebooks and scripts to produce a 3D visualization of the trained SVM model. 

4. Open the ```test``` directory. There you will find notebooks to analyze the performance of the trained SVM model and compare it to traditional data cleaning cuts. 

## Contact

For any questions or concerns, contact Esteban León via Slack or email (esleon97@unc.edu).
