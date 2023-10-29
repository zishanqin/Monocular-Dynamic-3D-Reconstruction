# Monocular 4D Reconstruction of Non-Rigid Scenes using Neural Radiance Fields


## Setup 



## Training

Our config files are provided in the `configs` directory, organized by dataset and explicit vs. hybrid model version. These config files may be updated with the location of the downloaded data and your desired scene name and experiment name. To train a model, run
```
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/mpi/TEST_mpi_explicit_alley_2.py
```

Note that for DyNeRF scenes it is recommended to first run for a single iteration at 4x downsampling to pre-compute and store the ray importance weights, and then run as usual at 2x downsampling. This is not required for other datasets.

## Visualization/Evaluation

The `main.py` script also supports rendering a novel camera trajectory, evaluating quality metrics, and rendering a space-time decomposition video from a saved model. These options are accessed via flags `--render-only`, `--validate-only`, and `--spacetime-only`, and a saved model can be specified via `--log-dir`.
