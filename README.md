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

    @inproceedings{c2020model,
        year =        {2020},
        title =       {{M}odel monitoring and dynamic model selection in travel time-series forecasting},
        author =      {{C}andela, {R}osa and  {M}ichiardi, {P}ietro and  {F}ilippone, {M}aurizio and  {Z}uluaga, {M}aria {A}},
        booktitle =   {{ECML}-{PKDD} 2020, {T}he {E}uropean {C}onference on {M}achine {L}earning and {P}rinciples and {P}ractice of                           {K}nowledge {D}iscovery in {D}atabases, 14-18 {S}eptember 2020, {G}hent, {B}elgium},
        address =     {{G}hent, {BELGIUM}},
        month =       {09},
        url =         {https://arxiv.org/abs/2003.07268}
    }

    

