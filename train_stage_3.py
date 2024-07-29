import itertools
import pandas as pd
import matplotlib.patches as mpatches

from datetime import datetime

from src.data import PrepareData
from src.data import plot_catchments, read_data_from_file
from src.window import WindowGenerator, MultiNumpyWindow, MultiWindow
from src.model import Base_Model, Ensemble_Static, Switch_Model


# Read timeseries and summary data from data dir
data_dir = '/srv/scratch/z5370003/data/camels-dropbox/'
timeseries_data, summary_data = read_data_from_file(data_dir)

# Create Dataset
camels_data = PrepareData(timeseries_data, summary_data)

# Plot catchments on map
plot_catchments(camels_data, data_dir)


camels_data.summary_data = camels_data.summary_data.T.drop_duplicates().T
selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'SA'].index)


# Quantile LSTM
combined=[]
for i in range(3):
    print('RUN',i)
    results_switch=[]
    variable_ts = ['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']
    variable_ts_switch = ['flood_probabilities', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']

    variable_static = ['q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq']

    train_df, test_df = camels_data.get_train_val_test(source=variable_ts, stations=selected_stations)

    multi_window = MultiWindow(input_width=5,
                               label_width=5,
                               shift=5,
                               train_df=train_df,
                               test_df=test_df,
                               stations=selected_stations,
                               label_columns=['streamflow_MLd_inclInfilled'])

    np_window = MultiNumpyWindow(input_width=5, 
                                 label_width=5,
                                 shift=5,
                                 camels_data=camels_data,
                                 timeseries_source=variable_ts_switch,
                                 summary_source=variable_static,
                                 summary_data=camels_data.summary_data,
                                 stations=selected_stations,
                                 label_columns=['flood_probabilities'])

    model_switch = Switch_Model(window_switch=np_window, window_regular=multi_window, CONV_WIDTH=5) 

    for station in selected_stations:
                results_switch.append(model_switch.summary(station))
    
    Switch_SA= pd.DataFrame(results_switch)
    # Switch_SA= Switch_SA.mean()
    Switch_SA= Switch_SA.to_dict()
    combined.append(Switch_SA)



Switch_SA_bdlstm = pd.DataFrame(combined)
# Switch_SA_bdlstm.to_csv('Quantilelstm_SA.csv')

# Get visualizations of predicted vs actual for quantile lstm model
# Enter the station name and the horizons as per need 

fig = plt.figure(figsize=(20,5))
plt.title('SA-A5030502', fontsize= 20)
plt.ylabel('Flood Probability', fontsize=18)
plt.ylim(0, 1.1)

plt.rcParams.update({'font.size': 15})

df_date = np_window.test_df['A5030502'].reset_index()
date_values = df_date['date']

s=800
e=1600

ax1 = plt.plot(date_values[s:e], model_switch.predictions('A5030502')[:,0][s:e], color='blue')
ax2 = plt.plot(date_values[s:e], multi_window.test_windows('A5030502')[:,0][s:e], color='red')

red_patch = mpatches.Patch(color='red', label='Actual')
blue_patch = mpatches.Patch(color='blue', label='Predicted')

plt.legend(handles=[red_patch, blue_patch])
plt.show()