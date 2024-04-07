
# Evaluation-of-Deep-Learning-Models-for-Extreme-Floods-in-Australia


We offer a study that assesses various deep learning techniques for forecasting multi-step ahead time series. Our focus is on hydrological data from diverse catchments across Australia, aiming to enhance predictions for extreme flood events. After evaluating several methods, we introduce a novel approach that integrates both time series and static features using Quantile LSTM. This strategy demonstrates superior performance compared to previous methods.


## Code

We include a structured code for the datasets used. The code contains comments to guide through the multiple stages of experiments. The python notebook for implementation can be found at: 
[Code](https://github.com/DARE-ML/streamflow-floods/tree/main/Code)

## Data

We make use of data from the Australian edition of the Catchment Attributes and Meteorology for Large-sample Studies (CAMELS) series of datasets at the [link](https://doi.pangaea.de/10.1594/PANGAEA.921850). 

The data includes time-series as well as summary(static) features for the 222 catchments(stations) spread across Australia.The complete dataset we have used can also be accessed at the [Dropbox Link](https://www.dropbox.com/home/Evaluation-of-Deep-Learning-Models-for-Extreme-Floods-in-Australia%3A%20Data). 

## Experiments

We have conducted 3 stages of experiments which are specified in the code with appropriate divisions.

Stage 1: Comparing Architectures for the LSTM Model for South Australia.

Stage 2: Comparing various Deep Learning Models across the Indivudal Architecture for South Australia.

Stage 3: Implementing the Quantile LSTM Model for South Australia and then continue for all the states in Australia.

For each experiment we conducted 30 runs to minimize distorted results. The results can be found for all the stages at [Results](https://github.com/DARE-ML/streamflow-floods/tree/main/Results)

