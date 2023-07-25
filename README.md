# Code for Cell Migration Classifiers
> Image-based classifiers for predicting probability of migration of a cell

Setting up `fastai` environment (together with jupyter kernels):

```bash
# assuming your base environment has jupyter
# and nb_conda_kernels is installed
#

# base env must have nb_conda_kernels:
# conda install nb_conda_kernels

# conda environment name: "fastai" with python3.9
conda create --name fastai python=3.9
source activate fastai

# for macos/linux use fastchan channel
conda install -c fastchan fastai
# for accessing "fastai" environment:
conda install ipykernel

# for opening tiff stacks etc.:
pip install --upgrade pip
pip install --upgrade scikit-image
```

Once all above installed, you should see a new kernel (environment) in jupyter named "fastai".

---