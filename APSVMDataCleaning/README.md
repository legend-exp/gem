## AP-SVM Data Cleaning

1. Request NERSC access to a GPU node on JupyterHub:

    - Link to [JupyterHub](https://jupyter.nersc.gov/)
    - Under the NERSC Help Portal, go to ‘Service Catalog → Request Forms → GPU node access’ and fill out the form to get GPU access

2. Create a custom Conda environment and kernel (on Cori terminal/command line):

    ```bash
    module load pytorch/v1.5.0−gpu
    conda create −n pytorch_gpu_env python =3.7 ipykernel numpy scipy
    source activate pytorch_gpu_env
    python −m ipykernel install −−user −−name pytorch_gpu_env −−display−name pytorch_gpu_kernel
    ```
    - More information here about creating custom Conda environments and kernels [here](https://www.nersc.gov/assets/Uploads/13-Using-Jupyter-20200616.pdf)

3. Install necessary packages to run the MNIST and waveform convolutional autoencoder Jupyter notebooks:

    ```bash
    pip install torch pytorch_model_summary gzip−reader pickle5 pathlib requests matplotlib tsnecuda
    ```

    - Can also install pygama inside this Conda environment with:

        ```bash 
        git clone https://github.com/legend−exp/pygama.git
        pip install −e <path_to_local_pygama_directory>
        ```
4. Once you have access to a GPU node, log into JupyterHub and start a “Shared GPU Node.” Make sure the Jupyter notebooks are in some directory within Cori and open them. Switch the kernel (on the upper right section of the screen) to “pytorch_gpu_kernel” and run the notebook cells.
