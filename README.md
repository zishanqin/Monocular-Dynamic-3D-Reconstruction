# Monocular 4D Reconstruction of Non-Rigid Scenes using Neural Radiance Fields


## Setup 

**Preparing the dataset**

Get the training data here: http://sintel.is.tue.mpg.de/downloads

unzip, and move to the ``data`` folder

Run ``python data_split.py`` in the ``data`` folder

**Virtual environment**

``cd`` to the root directory. To support ``open3d`` we use python 3.8. Create a virtual environment by

``conda create -n mono python=3.8``

Then install the requirements

``pip install -r requirements.txt``


## Training

Our config files are provided in the `plenoxels/configs/final/mpi` directory. These config files may be updated with the desired scene name, experiment name and other hyperparameters. To train a model, run
```
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/mpi/TEST_mpi_explicit_alley_2.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/mpi/TEST_mpi_explicit_cave_2.py
```

## Visualization/Evaluation

The results can be viewed in the corresponding ``log/{scene}/{experiment}`` folder.

To generate the point cloud for rendered RGB image and depth image, run in the root directory

``python pcd_vis.py {rgb_pth} {depth_pth} {K_alley/K_cave pth} {pcd_pth}``
