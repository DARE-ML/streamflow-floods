import os
import argparse
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.data import PrepareData
from src.data import plot_catchments, read_data_from_file
from src.window import MultiNumpyWindow, MultiWindow
from src.model import Switch_Model


# Argument Parser
parser = argparse.ArgumentParser(description='Train Stage 3')
parser.add_argument('--data-dir', type=str, default='/srv/scratch/z5370003/data/camels-dropbox/', help='Path to the data directory')
parser.add_argument('--num-runs', type=int, default=1, help='Number of runs')
parser.add_argument('--state', type=str, default='SA', help='State to train the model on')
parser.add_argument('--input-width', type=int, default=5, help='Input width for the window')
parser.add_argument('--output-width', type=int, default=1, help='Output width for the window')
parser.add_argument('--shift', type=int, default=1, help='Shift for the window')

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
selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == args.state].index)


# Quantile LSTM
results_switch=[]
variable_ts = ['streamflow_MLd_inclInfilled', 'flood_prob_acc', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']
variable_ts_switch = ['flood_prob_acc', 'streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']

variable_static = ['q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq']

train_df, test_df = camels_data.get_train_val_test(source=variable_ts, stations=selected_stations)

multi_window = MultiWindow(input_width=args.input_width,
                           label_width=args.output_width,
                           shift=args.shift,
                           train_df=train_df,
                           test_df=test_df,
                           stations=selected_stations,
                           label_columns=['streamflow_MLd_inclInfilled'])

np_window = MultiNumpyWindow(input_width=args.input_width, 
                             label_width=args.output_width,
                             shift=args.shift,
                             camels_data=camels_data,
                             timeseries_source=variable_ts_switch,
                             summary_source=variable_static,
                             summary_data=camels_data.summary_data,
                             stations=selected_stations,
                             label_columns=['flood_prob_acc'])

model_switch = Switch_Model(window_switch=np_window, window_regular=multi_window, CONV_WIDTH=args.output_width) 

for station in selected_stations:
    results_switch.append(model_switch.summary(station))

Switch_res = pd.DataFrame(results_switch)
print(Switch_res)
# # Switch_res = Switch_res.mean()
# Switch_res = Switch_res.to_dict()
# combined.append(Switch_res)



# Switch_res = pd.concatenate(combined)
os.makedirs(f'results/Stage 3/{args.state}-{args.output_width}', exist_ok=True)
Switch_res.to_csv(f'results/Stage 3/{args.state}-{args.output_width}/Quantilelstm_{args.state}.csv', index=False)

# Get visualizations of predicted vs actual for quantile lstm model
# Enter the station name and the horizons as per need 

for station in selected_stations:

    fig, axes = plt.subplots(args.output_width, 1, figsize=(20, 5*args.output_width))
    
    
    plt.title(station, fontsize= 20)
    plt.ylabel('Flood Probability', fontsize=18)
    plt.ylim(0, 1.1)

    plt.rcParams.update({'font.size': 15})

    df_date = np_window.test_df[station].reset_index()
    date_values = df_date['date']
    # print(date_values.min(), date_values.max(), len(date_values))

    s=800
    e=1600

    for i in range(args.output_width):
        ax1 = axes[i].plot(date_values[s:e], model_switch.switch.predictions(station)[:,i][s:e], color='blue')
        ax2 = axes[i].plot(date_values[s:e], np_window.test_windows(station)[:,i][s:e], color='red')

    red_patch = mpatches.Patch(color='red', label='Actual')
    blue_patch = mpatches.Patch(color='blue', label='Predicted')

    plt.legend(handles=[red_patch, blue_patch])
    plt.savefig(f'results/Stage 3/{args.state}-{args.output_width}/{station}-floodprob.png')
    plt.close(fig)

    # Plot the streamflow
    fig, axes = plt.subplots(args.output_width, 1, figsize=(20, 5*args.output_width))
    
    plt.title(station, fontsize=20)
    plt.ylabel('Streamflow', fontsize=18)
    plt.ylim(0, 1.1)

    plt.rcParams.update({'font.size': 15})

    df_date = multi_window.test_df[station].reset_index()
    date_values = df_date['date']
    # print(date_values.min(), date_values.max(), len(date_values))

    for i in range(args.output_width):
        ax1 = axes[i].plot(date_values[s:e], model_switch.predictions(station)[:,i][s:e], color='blue')
        ax2 = axes[i].plot(date_values[s:e], multi_window.test_windows(station)[:,i][s:e], color='red')

    red_patch = mpatches.Patch(color='red', label='Actual')
    blue_patch = mpatches.Patch(color='blue', label='Predicted')

    plt.legend(handles=[red_patch, blue_patch])
    plt.savefig(f'results/Stage 3/{args.state}-{args.output_width}/{station}-streamflow.png')
    plt.close(fig)