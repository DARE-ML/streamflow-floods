
# Evaluation-of-Deep-Learning-Models-for-Extreme-Floods-in-Australia


We offer a study that assesses various deep learning techniques for forecasting multi-step ahead time series. Our focus is on hydrological data from diverse catchments across Australia, aiming to enhance predictions for extreme flood events. After evaluating several methods, we introduce a novel approach that integrates both time series and static features using Quantile LSTM.


## Data

We make use of data from the Australian edition of the Catchment Attributes and Meteorology for Large-sample Studies (CAMELS) series of datasets at the [link](https://doi.pangaea.de/10.1594/PANGAEA.921850). 

The data includes time-series as well as summary(static) features for the 222 catchments(stations) spread across Australia.The complete dataset we have used can also be accessed at the [Dropbox Link](https://www.dropbox.com/home/Evaluation-of-Deep-Learning-Models-for-Extreme-Floods-in-Australia%3A%20Data). 

<!-- ## Code

We include a structured code for the datasets used. The code contains comments to guide through the multiple stages of experiments. The python notebook for implementation can be found at: 
[Data Processing](https://github.com/DARE-ML/streamflow-floods/blob/main/data_processing.ipynb) and [Model and Architecture Building](https://github.com/DARE-ML/streamflow-floods/blob/main/models_architectures.ipynb) -->

## Experiments

The following scripts are provided:

1. `evaluate_strategy.py` - Evaluate various catchment strategies with Multi-LSTM model for South Australia region
2. `evaluate_architecture.py` - Evaluate various model architectures with individual strategy for South Australia region
3. `train_quantile_ensemble_individual.py` - Train the Quantile-Ensemble model with individual strategy for selected state's top 5 catchments based on runoff-ratio.
4. `train_quantile_ensemble_batch.py` - Train the Quantile-Ensemble model for all-catchments in the selected state using batch strategy.


*Note: for help on available runtime-arguments for the script use the help command: `$python train_quantile_ensemble_individual.py --help`* 

**Flood Risk Indicator**

`/notebooks/flood_risk_indicator.ipynb` provides code on how the flood risk indicator is generated.
