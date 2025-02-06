from __future__ import absolute_import

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy import array
from sklearn.preprocessing import MinMaxScaler


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, test_df, 
                 station, filtered=None, label_columns=None):

        # Store the raw data.
        self.train_df = train_df[station]
        self.test_df = test_df[station]
        self.station = station
        self.filtered = filtered
        
        # validation
        if self.filtered == 'upper_soil_filter':
            assert('upper_soil_indicator' in list(self.train_df.columns))
        
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, filtered=None, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=shuffle,
                batch_size=32,)
        
        ds = ds.map(self.split_window)
        
        if filtered == 'upper_soil_filter':
            indicator_index = list(self.train_df.columns).index('upper_soil_indicator')
            ds = ds.unbatch().filter(lambda x, y: tf.math.reduce_sum(x[:, indicator_index]) > 0).batch(32)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, filtered=self.filtered)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)
           
    @property
    def test_windows(self):
        total_size = self.test_array.shape[0]
        convolution_size = self.input_width
        prediction_size = self.label_width
        
        a = []
        
        for i in range(convolution_size, (total_size-prediction_size+1)):
            index_list = list(range(i, i+prediction_size))
            a.append(self.test_array[index_list])
        
        return np.squeeze(array(a), axis=2)
    
    @property
    def test_array(self):
        return array(self.test_df[self.label_columns])
       
    def test_indicator(self, filtered):
        if filtered == 'upper_soil_filter':
            return array(self.test_df['upper_soil_indicator'])       

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
            
        return result
       
    @property
    def num_features(self):
        return self.train_df.shape[1]
    
    def test_example(self, index):
        return np.array(self.test_df.iloc[index]).reshape(1, 1, self.num_features)

    def plot(self, model=None, plot_col='streamflow_MLd_inclInfilled', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def station_percentile(self, cut, variable='streamflow_MLd_inclInfilled', station=None):
        if station==None:
            return np.percentile(self.test_df[variable], 100-cut)
        else:
            return np.percentile(self.test_df[station][variable], 100-cut)








class MultiWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df,test_df, stations, 
                 statics='separate', filtered=-1000, 
                 label_columns=None, batch_size=512):
     
        # Store the raw data.
        self.train_df = train_df       
        self.test_df = test_df
        self.stations = stations
        self.total_stations = len(stations) 

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.label_columns = label_columns
        
        self.filtered = filtered

        self.batch_size = batch_size
           
        self.trains = []
        self.tests = []        

        for i, s in enumerate(stations):
            window = WindowGenerator(input_width=input_width,
                                     label_width=label_width,
                                     shift=shift,
                                     train_df=train_df,
                                     test_df=test_df,
                                     station=s,
                                     label_columns=label_columns)

            padding = np.zeros((input_width, self.total_stations), dtype=np.float32)
            padding[:, i] = 1

            self.trains.append(window.train.unbatch().map(lambda x, y: (tf.concat([x, tf.convert_to_tensor(padding)], axis=1),y)))
            self.tests.append(window.test.unbatch().map(lambda x, y: (tf.concat([x, tf.convert_to_tensor(padding)], axis=1),y)))
            
    @property
    def train(self):
        ds = tf.data.Dataset.from_tensor_slices(self.trains)
        concat_ds = ds.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE).filter(lambda x, y: tf.math.reduce_max(y) > self.filtered)
        
        return concat_ds.batch(self.batch_size)


    @property
    def test(self):
        ds = tf.data.Dataset.from_tensor_slices(self.tests)
        concat_ds = ds.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
        
        return concat_ds.batch(self.batch_size)


    def test_windows(self, station=None):
        total_size = self.test_array(station).shape[0]
        convolution_size = self.input_width
        prediction_size = self.label_width
        
        a = []
        
        for i in range(convolution_size, (total_size-prediction_size+1)):
            index_list = list(range(i, i+prediction_size))
            a.append(self.test_array(station)[index_list])
        
        return np.squeeze(array(a), axis=2)
    
    def test_array(self, station):
        return array(self.test_df[station][self.label_columns])
    
    def station_percentile(self, cut, variable='streamflow_MLd_inclInfilled', station=None):
        if station==None:
            return np.percentile(self.test_df[variable], 100-cut)
        else:
            return np.percentile(self.test_df[station][variable], 100-cut)



class MultiNumpyWindow():
    def __init__(self, input_width, label_width, shift, camels_data,
                 timeseries_source, summary_source, summary_data,
                 stations, label_columns=None):

        # Get train and test data
        train_df, test_df = camels_data.get_train_val_test(source=timeseries_source,
                                                           stations=stations)

        # Store the raw data.
        self.train_df = train_df
        self.test_df = test_df
        self.stations = stations
        self.total_stations = len(stations)
        
        self.num_timeseries_features = len(timeseries_source)
        self.num_static_features = len(summary_source)     
        self.timeseries_source = timeseries_source
        self.summary_source = summary_source      
    
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.label_columns = label_columns
        
        self.train_timeseries = []
        self.test_timeseries = []  
        
        self.train_static = []
        self.test_static = [] 
        
        self.train_y = []
        self.test_y = []
        
        for i, s in enumerate(stations):
            # process timeseries
            window = WindowGenerator(input_width=input_width,
                                     label_width=label_width,
                                     shift=shift,
                                     train_df=train_df,
                                     test_df=test_df,
                                     station=s,
                                     label_columns=label_columns)
            
            x_train, y_train = self.mapdata_tonumpy(window.train)
            x_test, y_test = self.mapdata_tonumpy(window.test)
            
            self.train_timeseries.extend(x_train) 
            self.test_timeseries.extend(x_test) 
            
            self.train_y.extend(y_train) 
            self.test_y.extend(y_test)      
            
            # process static
            static = summary_data[summary_source].loc[s].to_numpy()         
            padding = np.zeros((self.total_stations, ), dtype=np.float32)
            padding[i] = 1
            
            static = np.concatenate([static, padding], axis=0)
            
            self.train_static.extend([static for _ in range(x_train.shape[0])])
            self.test_static.extend([static for _ in range(x_test.shape[0])])
            
        self.train_timeseries = np.array(self.train_timeseries)
        self.test_timeseries = np.array(self.test_timeseries) 
        
        self.train_static = np.array(self.train_static)
        self.test_static = np.array(self.test_static)
        
        scaler = MinMaxScaler()
        scaler.fit(self.train_static)
        
        self.train_static = scaler.transform(self.train_static)
        self.test_static = scaler.transform(self.test_static)
                
        #self.train_static = np.ones((self.train_static.shape[0], self.train_static.shape[1]))
        
        self.train_y = np.array(self.train_y) 
        self.train_y = np.swapaxes(self.train_y, 1, 2)

        self.test_y = np.array(self.test_y)
        self.test_y = np.swapaxes(self.test_y, 1, 2)

    def mapdata_tonumpy(self, map_ds):
        map_ds = map_ds.unbatch()

        x_array = []
        y_array = []

        for ts in map_ds:
            x = ts[0]
            y = ts[1]

            x_array.append(x.numpy())
            y_array.append(y.numpy())

        return np.array(x_array), np.array(y_array) 
    
    @property
    def train(self):
        return self.train_timeseries, self.train_static, self.train_y


    @property
    def test(self):
        return self.test_timeseries, self.test_static, self.test_y


    def test_windows(self, station=None):
        total_size = self.test_array(station).shape[0]
        convolution_size = self.input_width
        prediction_size = self.label_width
        
        a = []
        
        for i in range(convolution_size, (total_size-prediction_size+1)):
            index_list = list(range(i, i+prediction_size))
            a.append(self.test_array(station)[index_list])
        
        return np.squeeze(array(a), axis=2)
    
    def test_array(self, station):
        return array(self.test_df[station][self.label_columns])
