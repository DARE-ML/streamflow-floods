import itertools
from datetime import datetime
import pandas as pd

from src.data import PrepareData
from src.data import plot_catchments, read_data_from_file
from src.window import WindowGenerator, MultiNumpyWindow, MultiWindow
from src.model import Base_Model



# Read timeseries and summary data from data dir
data_dir = '/srv/scratch/z5370003/data/camels-dropbox/'
timeseries_data, summary_data = read_data_from_file(data_dir)

# Create Dataset
camels_data = PrepareData(timeseries_data, summary_data)

# Plot catchments on map
plot_catchments(camels_data, data_dir)


camels_data.summary_data = camels_data.summary_data.T.drop_duplicates().T
selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'SA'].index)

# Enter the model you wish to run or multiple as per requirements. The models can be accessed through the following names:
#['multi-LSTM', 'multi-linear','multi-CNN', 'multi-Bidirectional-LSTM']


# Individual LSTM-SA
combined= []
for i in range(0, 3):
    print('RUN', i)
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']]

    permutations_base = list(itertools.product(*[input_widths, label_widths, selected_stations, models, variables]))

    results_baseModels_variables = []
    models_baseModels_variables = []
    errors_baseModels_variables = []

    for input_width, label_width, station, model_name, variable in permutations_base:
        if input_width < label_width:
            continue

        train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations)

        try:
            print('input_width:{}, label_width:{}, station:{}, model:{}, variables:{}'.format(input_width, label_width, station, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            window = WindowGenerator(input_width=input_width,
                                     label_width=label_width,
                                     shift=label_width,
                                     train_df=train_df,
                                     test_df=test_df,
                                     station=station,
                                     label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=window, CONV_WIDTH=label_width)

            results_baseModels_variables.append(model.summary())

            pd.DataFrame(results_baseModels_variables).to_csv('results_files/results_ensemble_all_1.csv')

        except:
            errors_baseModels_variables.append([input_width, label_width, station, model])
        
        
        break

    Individual_SA= pd.DataFrame(results_baseModels_variables)
    # Individual_SA= Individual_SA.mean()
    # Individual_SA= Individual_SA.to_dict()
    # combined.append(Individual_SA)


# Batch-Indicator LSTM-SA
combined= []
for i in range(0, 3):
    print('RUN',i)
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']]

    permutations_base_batch = list(itertools.product(*[input_widths, label_widths, models, variables]))

    results_baseModels_batch = []
    errors_baseModels_batch = []

    for input_width, label_width, model_name, variable in permutations_base_batch:
        try:
            if input_width < label_width:
                continue

            train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations, discard=0.5)
            multi_window = MultiWindow(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_df=train_df,
                                       test_df=test_df,
                                       stations=selected_stations,
                                       label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=multi_window, CONV_WIDTH=label_width)

            print('input_width:{}, label_width:{}, model:{}, variables:{}'.format(input_width, label_width, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            for station in selected_stations:
                results_baseModels_batch.append(model.summary(station=station))

            pd.DataFrame(results_baseModels_batch).to_csv('results_files/results_batch_all_1.csv')
        except:
            errors_baseModels_batch.append([input_width, label_width, model_name]) 
    Batch_SA= pd.DataFrame(results_baseModels_batch)
    Batch_SA= Batch_SA.mean()
    Batch_SA= Batch_SA.to_dict()
    combined.append(Batch_SA)



# Batch-Static LSTM-SA
combined=[]
for i in range(0, 3):
    print('RUN', i)
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP', 'q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq']]

    permutations_base_batch = list(itertools.product(*[input_widths, label_widths, models, variables]))

    results_baseModels_batch = []
    errors_baseModels_batch = []

    for input_width, label_width, model_name, variable in permutations_base_batch:
        try:
            if input_width < label_width:
                continue

            train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations)
            multi_window = MultiWindow(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_df=train_df,
                                       test_df=test_df,
                                       stations=selected_stations,
                                       label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=multi_window, CONV_WIDTH=label_width)

            print('input_width:{}, label_width:{}, model:{}, variables:{}'.format(input_width, label_width, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            for station in selected_stations:
                results_baseModels_batch.append(model.summary(station=station))

            pd.DataFrame(results_baseModels_batch).to_csv('results_files/results_batch_static_all_1.csv')
        except:
            errors_baseModels_batch.append([input_width, label_width, model])
    Batch_SA= pd.DataFrame(results_baseModels_batch)
    Batch_SA= Batch_SA.mean()
    Batch_SA= Batch_SA.to_dict()
    combined.append(Batch_SA)