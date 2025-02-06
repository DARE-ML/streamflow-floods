#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tqdm

from src.data import PrepareData
from src.data import plot_catchments, read_data_from_file
from src.window import MultiNumpyWindow, MultiWindow, WindowGenerator
from src.model import Switch_Model, QuantileEnsemble



# Argument Parser
parser = argparse.ArgumentParser(description='Train Stage 3')
parser.add_argument('--data-dir', type=str, default='/srv/scratch/z5370003/data/camels-dropbox/', help='Path to the data directory')
parser.add_argument('--num-runs', type=int, default=1, help='Number of runs')
parser.add_argument('--state', type=str, default='SA', help='State to train the model on')
parser.add_argument('--input-width', type=int, default=5, help='Input width for the window')
parser.add_argument('--output-width', type=int, default=5, help='Output width for the window')
parser.add_argument('--shift', type=int, default=5, help='Shift for the window')



# Parse the arguments
args = parser.parse_args()
data_dir = args.data_dir
num_runs = args.num_runs

# Read timeseries and summary data from data dir
timeseries_data, summary_data = read_data_from_file(data_dir)

# Create Dataset
camels_data = PrepareData(timeseries_data, summary_data)

# Plot catchments on map
plot_catchments(camels_data, data_dir)


camels_data.summary_data = camels_data.summary_data.T.drop_duplicates().T

char = camels_data.summary_data[['state_outlet', 'station_name', 'river_region', 
                                 'Q5', 'Q95', 'catchment_area', 'lat_outlet', 
                                 'long_outlet', 'runoff_ratio']]
char_selected = char[(char.state_outlet==args.state)]
selected_stations = char_selected.sort_values('runoff_ratio', ascending=False).head().index.tolist()

variable_ts = ['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']

# variable_ts_switch = ['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']
# variable_static = ['q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq']

train_df, test_df = camels_data.get_train_val_test(source=variable_ts, stations=selected_stations)

summary_cols = ['SER_1%', 'SER_2%', 'SER_5%', 'SER_10%', 'SER_25%', 'SER_50%', 'SER_75%', 'RMSE']

summary_data_q05 = []
summary_data_q95 = []
summary_data_reg = []

quantile_ensemble = {}

for n in tqdm.trange(args.num_runs):

    print(f'Starting run {n+1} of {args.num_runs}.')

    results_regular = []
    results_q05 = []
    results_q95 = []
    
    for station in selected_stations:
        multi_window = MultiWindow(input_width=args.input_width,
                           label_width=args.output_width,
                           shift=args.shift,
                           train_df=train_df,
                           test_df=test_df,
                           stations=[station],
                           label_columns=['streamflow_MLd_inclInfilled'],
                           batch_size=32)
        
        print(f'Training Quantile Ensembles for station {station}.')
        quantile_ensemble[station] = QuantileEnsemble(window=multi_window, CONV_WIDTH=args.output_width)
        summary_regular, summary_q05, summary_q95 = quantile_ensemble[station].summary(station)
        results_regular.append(summary_regular)
        results_q05.append(summary_q05)
        results_q95.append(summary_q95)

    results_reg = pd.DataFrame(results_regular)[summary_cols].mean().to_dict()
    results_q05 = pd.DataFrame(results_q05)[summary_cols].mean().to_dict()
    results_q95 = pd.DataFrame(results_q95)[summary_cols].mean().to_dict()

    summary_data_reg.append(results_reg)
    summary_data_q05.append(results_q05)
    summary_data_q95.append(results_q95)


results_reg_df = pd.DataFrame(summary_data_reg)
results_q05_df = pd.DataFrame(summary_data_q05)
results_q95_df = pd.DataFrame(summary_data_q95)


os.makedirs(f'results/Stage 6/{args.state}-{args.output_width}', exist_ok=True)
results_reg_df.to_csv(f'results/Stage 6/{args.state}-{args.output_width}/results_reg.csv', index=True)
results_q05_df.to_csv(f'results/Stage 6/{args.state}-{args.output_width}/results_q05.csv', index=True)
results_q95_df.to_csv(f'results/Stage 6/{args.state}-{args.output_width}/results_q95.csv', index=True)


for idx, station in enumerate(selected_stations):

    s = 1200
    e = 1600
    # Plot the streamflow
    fig = plt.figure(figsize=(20,5))
    plt.title(station, fontsize= 20)
    plt.ylabel('Streamflow', fontsize=18)
    # plt.ylim(0, 1.1)


    min_ = camels_data.scaler_test.min_[idx*len(variable_ts)+1]
    scale_ = camels_data.scaler_test.scale_[idx*len(variable_ts)+1]


    pred = (quantile_ensemble[station].predictions(station)[s:e] - min_)/scale_
    actual = (multi_window.test_windows(station)[s:e] - min_)/scale_

    plt.rcParams.update({'font.size': 15})

    df_date = multi_window.test_df[station].reset_index()
    date_values = df_date['date']

    ax1 = plt.plot(date_values[s:e], pred[:, 0, 0], color='blue', label='Predicted')
    ax2 = plt.plot(date_values[s:e], actual[:, 0], color='red', label='Actual')
    ax3 = plt.fill_between(date_values[s:e], pred[:, 0, 1], pred[:, 0, 2], color='blue', alpha=0.4, label='90% Confidence Interval')

    red_patch = mpatches.Patch(color='red', label='Actual')
    blue_patch = mpatches.Patch(color='blue', label='Predicted')

    plt.legend()
    plt.savefig(f'results/Stage 6/{args.state}-{args.output_width}/{station}-streamflow.png')







