# Model Monitoring and Dynamic Model Selection in Travel Time-series Forecasting

This repository contains the code and the useful links to reproduce the experiments in the paper [Model Monitoring and Dynamic Model Selection in Travel Time-series Forecasting](https://arxiv.org/abs/2003.07268).

The M4 competition dataset, together with the monitored models used in the framework, are available in the original [M4 repository](https://github.com/Mcompetitions/M4-methods).

The implementation of the two meta-learning approaches used for the comparison with our method can be found in the following links:

- FFORMS: https://github.com/thiyangt/WorkingPaper1
- ADE: https://github.com/vcerqueira/tsensembler

## Getting started

To use one of the available monitoring models, you can run the corresponding python script, specifying the path of the needed files.
Example for LSTM model:

`python lstm.py --observations=<OBSERVATIONS_PATH> --true_values=<TRUE_VALUES_PATH> --forecasts=<FORECASTS_PATH>`

Parameters:

- `OBSERVATIONS_PATH`: path of csv file containing the observations
- `TRUE_VALUES_PATH`: path of csv file containing the true_values
- `FORECASTS_PATH`: path of csv file containing the forecasts

To perform dynamic model selection, you can run the script `dynamic_model_selection.py`, specifying the path of the needed files:

`python dynamic_model_selection.py --observations=<OBSERVATIONS_PATH> --true_values=<TRUE_VALUES_PATH> --forecasts_folder=<FORECASTS_FOLDER>`

Parameters:

- `OBSERVATIONS_PATH`: path of csv file containing the observations
- `TRUE_VALUES_PATH`: path of csv file containing the true_values
- `FORECASTS_FOLDER`: path of the folder containing the csv files of forecasts

## Citation

Please cite it as follows:

`@misc{c2020model,
    title={Model Monitoring and Dynamic Model Selection in Travel Time-series Forecasting},
    author={Rosa Candela and Pietro Michiardi and Maurizio Filippone and Maria A. Zuluaga},
    year={2020},
    eprint={2003.07268},
    archivePrefix={arXiv},
    primaryClass={stat.AP}
}`


