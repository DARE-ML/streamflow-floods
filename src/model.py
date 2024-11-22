from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import RepeatVector
from keras.layers import Flatten

from tensorflow.keras.optimizers import SGD

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from .window import WindowGenerator


class CustomLoss():  
    def qloss_95(y_true, y_pred, q=0.95):
        e = (y_true-y_pred)    
        return tf.square(y_true-y_pred) + tf.math.maximum(q*e, (q-1)*e)
        
    def qloss_90(y_true, y_pred, q=0.9):
        e = (y_true-y_pred)    
        return tf.square(y_true-y_pred) + tf.math.maximum(q*e, (q-1)*e)
    
    def qloss_70(y_true, y_pred, q=0.7):
        e = (y_true-y_pred)    
        return tf.square(y_true-y_pred) + tf.math.maximum(q*e, (q-1)*e)
    
    def qloss_50(y_true, y_pred, q=0.5):
        e = (y_true-y_pred)    
        return tf.square(y_true-y_pred) + K.maximum(q*e, (q-1)*e)
    
class Model():
    MAX_EPOCHS = 150
    
    def __init__(self, window):
        # Store the raw data.
        self.window = window
        
        self.train_df = self.window.train_df
        self.test_df = self.window.test_df
             
    def compile_and_fit(self, model, window, loss_func, patience=10):

        model.compile(loss=loss_func,
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=self.MAX_EPOCHS,
                            verbose=0)

        return history

    def num_flood_events(self, cut=1):
        actuals = np.squeeze(self.window.test_windows, axis=2) 
        cut_percentile = np.percentile(actuals.flatten(), cut)
        locs = np.unique(np.where(actuals<cut_percentile)[0])

        events = np.split(locs, np.cumsum( np.where(locs[1:] - locs[:-1] > 1) )+1)

        return len(events)
    
    def summary(self, station=None):
        summary_dict = {}
        
        summary_dict['model_name'] = self.model_name
        summary_dict['input_width'] = self.window.input_width
        summary_dict['label_width'] = self.window.label_width
             
        if station == None:
            summary_dict['station'] = self.window.station
            summary_dict['inputs'] = str(list(self.train_df.columns))
            summary_dict['NSE'] = self.get_NSE()
        else:
            summary_dict['station'] = station
            example_station = self.train_df.columns.get_level_values(0)[0]
            summary_dict['inputs'] = str(list(self.train_df[example_station].columns))
            summary_dict['NSE'] = self.get_NSE(station)  
                             
        summary_dict['SER_1%'] = self.average_model_error(station, cut=1)
        summary_dict['SER_2%'] = self.average_model_error(station, cut=2)    
        summary_dict['SER_5%'] = self.average_model_error(station, cut=5)        
        summary_dict['SER_10%'] = self.average_model_error(station, cut=10)  
        summary_dict['SER_25%'] = self.average_model_error(station, cut=25)  
        summary_dict['SER_50%'] = self.average_model_error(station, cut=50)  
        summary_dict['SER_75%'] = self.average_model_error(station, cut=75)  
        summary_dict['RMSE'] = self.average_model_error(station, cut=100)
        

        summary_dict['f1_score_individual_1%'] = self.binary_metrics(station=station, cut=1, metric='f1_score', evaluation='individual')        
        summary_dict['f1_score_individual_2%'] = self.binary_metrics(station=station, cut=2, metric='f1_score', evaluation='individual')  
        summary_dict['f1_score_individual_5%'] = self.binary_metrics(station=station, cut=5, metric='f1_score', evaluation='individual') 
        summary_dict['f1_score_individual_10%'] = self.binary_metrics(station=station, cut=10, metric='f1_score', evaluation='individual') 
        summary_dict['f1_score_individual_25%'] = self.binary_metrics(station=station, cut=25, metric='f1_score', evaluation='individual') 
        summary_dict['f1_score_individual_50%'] = self.binary_metrics(station=station, cut=50, metric='f1_score', evaluation='individual')  
        summary_dict['f1_score_individual_75%'] = self.binary_metrics(station=station, cut=75, metric='f1_score', evaluation='individual') 
        summary_dict['f1_score_individual_all'] = self.binary_metrics(station=station, cut=100, metric='f1_score', evaluation='individual') 
          
        return summary_dict
            
    def print_model_error(self, station=None, cut=0):
        if station != None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
            test_array = self.window.test_array(station)
        else:
            preds = self.predictions(station)
            actuals = self.window.test_windows  
            test_array = self.window.test_array
            
        cut_percentile = np.percentile(actuals.flatten(), cut)

        locs = np.unique(np.where(actuals>cut_percentile)[0])
        preds = preds[locs]
        actuals = actuals[locs]

        for window_pred, window_actual, loc in zip(preds, actuals, locs):
            print("time: {}".format(loc))
            print("Input: {}".format(test_array[loc:loc+self.window.input_width].flatten()))
            print("Predicted: {}".format(window_pred))
            print("Actual: {}".format(window_actual))
            print("-------------------------")
            
    def model_predictions_less_than_cut(self, cut=100):
        
        preds = self.predictions
        actuals =self.window.test_windows

        cut_percentile = np.percentile(actuals.flatten(), cut)

        num_predicted = (preds.flatten() < cut_percentile).sum()
        num_actual = (actuals.flatten() < cut_percentile).sum()

        return num_predicted, num_actual
        
    def average_model_error(self, station=None, cut=100):
        if self.window.label_columns[0] == 'streamflow_MLd_inclInfilled':
            cut = 100 - cut
            
        if station != None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
        else:
            preds = self.predictions()
            actuals = self.window.test_windows         

        cut_percentile = np.percentile(actuals.flatten(), cut)

        locs = np.where(actuals>cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]

        avg_error = 0

        for window_pred, window_actual in zip(preds, actuals):
            avg_error += np.sum((window_pred - window_actual)**2)
        

        avg_error = avg_error/actuals.shape[0]*actuals.shape[1]


        return avg_error
    
    def get_NSE(self, station=None, type='cast'):
        if station != None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
        else:
            preds = self.predictions()
            actuals = self.window.test_windows
        
        NSE = []

        for i in range(self.window.label_width):
            numer = np.sum(np.square(preds[:, i] - actuals[:, i]))
            denom = np.sum(np.square(actuals[:, i] - np.mean(actuals[:, i])))
        
            NSE.append(1-(numer/denom))
        
        if type=='cast':
            return np.mean(NSE)
        else:
            return NSE

    def binary_metrics(self, cut, metric, evaluation='whole', station=None):
        percentile_cut = self.window.station_percentile(station=station, cut=cut)
        
        if station==None:
            preds_pre = self.predictions()
            actuals_pre = self.window.test_windows
        else:        
            preds_pre = self.predictions(station)
            actuals_pre = self.window.test_windows(station)
            
        if evaluation=='whole':  
            preds = np.array([int(any(x > percentile_cut)) for x in preds_pre])
            actuals = np.array([int(any(x > percentile_cut)) for x in actuals_pre])
        else:
            preds = np.array([int(x > percentile_cut) for x in preds_pre.flatten()])           
            actuals = np.array([int(x > percentile_cut) for x in actuals_pre.flatten()])

        if metric=='accuracy':
            return accuracy_score(actuals, preds)
        elif metric=='precision':
            return precision_score(actuals, preds)
        elif metric=='recall':
            return recall_score(actuals, preds)
        elif metric=='f1_score':
            return f1_score(actuals, preds)
     

    @property
    def test_loss(self):
        return self.model.evaluate(self.window.test, verbose=0)[0]

    def predictions(self, station=None):
        tf_test = self.window.test

        if station != None:
            filter_index = self.window.stations.index(station)
            num_inputs = len(self.window.train_df.columns.levels[1])
            tf_test = tf_test.unbatch().filter(lambda x, y: tf.math.reduce_sum(x[:, num_inputs + filter_index]) > 0).batch(32)

        return np.squeeze(self.model.predict(tf_test), axis=2)
        
class Base_Model(Model):  
    def __init__(self, model_name, window, CONV_WIDTH, output_activation='sigmoid', loss_func=tf.losses.MeanSquaredError()):
        super().__init__(window)
        
        self.model_name = model_name
        self.mix_type_name = None
        self.loss_func = loss_func
        
        if self.model_name == 'multi-linear':          
            self.model = tf.keras.Sequential([
                            # Take the last time step.
                            # Shape [batch, time, features] => [batch, 1, features]
                            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                            # Shape => [batch, 1, dense_units]
                            tf.keras.layers.Dense(20, activation='relu'),
                            # Shape => [batch, out_steps*features]
                            tf.keras.layers.Dense(CONV_WIDTH, activation=output_activation, 
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1]
                            tf.keras.layers.Reshape([CONV_WIDTH, 1])
                        ])
            
        elif self.model_name == 'multi-CNN':
            self.model = tf.keras.Sequential([
                            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
                            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
                            # Shape => [batch, 1, conv_units]
                            tf.keras.layers.Conv1D(64, activation='relu', kernel_size=(CONV_WIDTH)),
                            # Shape => [batch, 1,  out_steps*features]
                            tf.keras.layers.Dense(CONV_WIDTH, activation=output_activation, 
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1]
                            tf.keras.layers.Reshape([CONV_WIDTH, 1])
                        ])
            
        elif self.model_name == 'multi-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            LSTM(20, return_sequences=False),
                            # Shape => [batch, out_steps*features].
                            Dense(CONV_WIDTH, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([CONV_WIDTH, 1])
                        ])

            
        elif self.model_name == 'multi-ED-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            LSTM(20, return_sequences=True,),
                            # Shape => [batch, out_steps*features].
                            Dropout(0.2),
                            Flatten(),
                            RepeatVector(5),
                            LSTM(20, return_sequences=False), 
                            
                            Dropout(0.2),                
                            Dense(CONV_WIDTH, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([CONV_WIDTH, 1])
                        ])    
            
        elif self.model_name == 'multi-Bidirectional-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            Bidirectional(LSTM(20, return_sequences=False)),
                            # Shape => [batch, out_steps*features].
                            Dense(CONV_WIDTH, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([CONV_WIDTH, 1])
                        ])  
            
        elif self.model_name == 'multi-deep-Bidirectional-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            Bidirectional(LSTM(64, return_sequences=True
                                              )),
                            Dropout(0.2),
                            Bidirectional(LSTM(32, return_sequences=False)),
                            Dropout(0.2),
                            # Shape => [batch, out_steps*features].
                            Dense(CONV_WIDTH, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([CONV_WIDTH, 1])
                        ]) 
            
        self.compile_and_fit(self.model, window, loss_func)



class Mixed_Model(Base_Model):
    threshold = 0.2
    
    def __init__(self, model_name, mix_type_name, window, CONV_WIDTH):
        super().__init__(model_name, window, CONV_WIDTH)
        self.mix_type_name = mix_type_name
               
        if self.mix_type_name == 'simple-two_model-onestepAR':
            window_simple = WindowGenerator(input_width=1,
                                             label_width=1,
                                             shift=1,
                                             train_df=train_df.loc[:,train_df.columns.get_level_values(1).isin(self.window.label_columns)] ,
                                             test_df=test_df.loc[:,test_df.columns.get_level_values(1).isin(self.window.label_columns)] ,
                                             station=self.window.station,
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = Base_Model(model_name=model_name, window=window_simple, CONV_WIDTH=1)
            
        elif self.mix_type_name == 'simple-two_model-multistep':
            window_simple = WindowGenerator(input_width=1,
                                             label_width=self.window.label_width,
                                             shift=self.window.label_width,
                                             train_df=train_df,
                                             test_df=test_df,
                                             station=self.window.station,
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = Base_Model(model_name=model_name, window=window_simple, CONV_WIDTH=self.window.label_width)
        elif self.mix_type_name == 'upper_soil-two_model-multistep':
            window_simple = WindowGenerator(input_width=self.window.input_width,
                                             label_width=self.window.label_width,
                                             shift=self.window.label_width,
                                             train_df=train_df,
                                             test_df=test_df,
                                             station=self.window.station,
                                            filtered='upper_soil_filter',
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = Base_Model(model_name=model_name, window=window_simple, CONV_WIDTH=self.window.label_width)
            
            
    @property
    def predictions(self):
        if self.mix_type_name == 'simple':
            preds = super().predictions
            test_array = self.window.test_array[self.window.input_width:]
            new_pred=[]
       
            for pred, actual_before in zip(preds, test_array):
                if actual_before < self.threshold:
                    pred = np.full((self.window.label_width,), actual_before)

                new_pred.append(pred)  
            
            
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'simple-two_model-onestepAR':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # test array starts 1 time unit before predictions
            test_array = self.window.test_array[self.window.input_width:]
            
            new_pred=[]

            for pred, actual_before in zip(preds, test_array):
                if actual_before < self.threshold:
                    pred = []
                                      
                    input_value = np.array(actual_before).reshape(1,1,1)
                    
                    for j in range(self.window.label_width):
                        pred_simple = self.model_simple.model.predict(input_value).item()
                        pred.append(pred_simple)
                        
                        input_value = np.array(pred_simple).reshape(1,1,1)
                                         
                    pred = np.array(pred)

                new_pred.append(pred)  
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'simple-two_model-multistep':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # test array starts 1 time unit before predictions
            test_array = self.window.test_array[self.window.input_width:]
            
            new_pred=[]

            for i, (pred, actual_before) in enumerate(zip(preds, test_array)):
                if actual_before < self.threshold:                                
                    input_value = self.window.test_example(i+self.window.input_width)
                                              
                    pred = self.model_simple.model.predict(input_value).flatten()

                new_pred.append(pred)  
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'upper_soil-two_model-multistep':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # upper soil indicator 1 time unit before predictions
            upper_soil_indicator = self.window.test_indicator(filtered='upper_soil_filter')
                      
            new_pred=[]

            for i, (pred, indicator) in enumerate(zip(preds, upper_soil_indicator)):
                if indicator == 1:                                
                    input_value = self.window.test_example(i+self.window.input_width)
                                              
                    pred = self.model_simple.model.predict(input_value).flatten()

                new_pred.append(pred)              
            
            return np.array(new_pred)  




## Without Changes

class Ensemble_Static():
    epochs = 100
    patience = 5
    def __init__(self, numpy_window, batch_size=32):
        num_timesteps = numpy_window.input_width
        num_timeseries_features = numpy_window.num_timeseries_features
        num_static_features = numpy_window.num_static_features + numpy_window.total_stations
          
        num_predictions = numpy_window.label_width
        
        self.batch_size = batch_size
        self.stations = numpy_window.stations
        self.n_stations = numpy_window.total_stations
        self.numpy_window = numpy_window
        # RNN + SLP Model
        # Define input layer

        recurrent_input = Input(shape=(num_timesteps, num_timeseries_features),name="TIMESERIES_INPUT")
        static_input = Input(shape=(num_static_features,),name="STATIC_INPUT")

        # RNN Layers
        # layer - 1
        rec_layer_one = LSTM(20, name ="BIDIRECTIONAL_LAYER_1", return_sequences=True)(recurrent_input)
        rec_layer_one = Dropout(0.1,name ="DROPOUT_LAYER_1")(rec_layer_one)
        
        # layer - 2
        rec_layer_two = LSTM(20, name ="BIDIRECTIONAL_LAYER_2", return_sequences=False)(rec_layer_one)
        rec_layer_two = Dropout(0.1,name ="DROPOUT_LAYER_2")(rec_layer_two)      
        

        # SLP Layers
        static_layer_one = Dense(20, activation='relu',name="DENSE_LAYER_1")(static_input)
        # Combine layers - RNN + SLP
        combined = Concatenate(axis= 1,name = "CONCATENATED_TIMESERIES_STATIC")([rec_layer_two, static_layer_one])
        combined_dense_two = Dense(20, activation='relu',name="DENSE_LAYER_2")(combined)
        output = Dense(num_predictions, name="OUTPUT_LAYER", activation='sigmoid')(combined_dense_two)

      
        # Compile ModeL
        self.model = keras.models.Model(inputs=[recurrent_input, static_input], outputs=[output])
        # MSE
        
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.train_timeseries_x, self.train_static_x, self.train_y = numpy_window.train     
        self.test_timeseries_x, self.test_static_x, self.test_y = numpy_window.test 
        
        self.model.summary()
        
    def train(self):
        self.model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['MeanAbsoluteError'])
        
        
        self.model.fit([self.train_timeseries_x, self.train_static_x], 
                       self.train_y, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size, 
                       verbose=1)


    @property
    def test_loss(self):
        return self.model.evaluate(self.window.test, verbose=0)[0]
    
    def predictions(self, station):  
        filter_index = self.stations.index(station)
        n_observations = int(self.test_static_x.shape[0]/self.n_stations)

        start = int(filter_index*n_observations)
        end = int((filter_index+1)*n_observations)
        
        print("Timeseries input shape:", self.test_timeseries_x[start:end].shape)
        print("Static input shape:", self.test_static_x[start:end].shape)
        
        return self.model.predict([self.test_timeseries_x[start:end,:], self.test_static_x[start:end,:]])
    
    
    def actuals(self, station):
        filter_index = self.stations.index(station)
        n_observations = int(self.test_static_x.shape[0]/self.n_stations)

        start = int(filter_index*n_observations)
        end = int((filter_index+1)*n_observations)
        print('TSET Y SHAPE',self.test_y.shape)
        

        return self.test_y.reshape(self.test_y.shape[0], self.test_y.shape[2])[start:end, :]

    
    def average_model_error(self, station, cut=100):
        preds = self.predictions(station)
        actuals = self.actuals(station)
        
        cut_percentile = np.percentile(actuals.flatten(), 100-cut)

        locs = np.where(actuals > cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]

        avg_error = 0

        for window_pred, window_actual in zip(preds, actuals):
            avg_error += np.sum((window_pred - window_actual)**2)
        
        if avg_error==0:
            return 0
        print(avg_error)
        avg_error = avg_error/actuals.shape[0]*actuals.shape[1]

        return avg_error 
    
    def print_model_windows(self, station, cut=100):
        preds = self.predictions(station)
        actuals = self.actuals(station)
        cut_percentile = np.percentile(actuals.flatten(), 100-cut)

        locs = np.where(actuals > cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]
        
        for pred, actual, loc in zip(preds, actuals, locs):
            print("time: {}".format(loc))
            print("Input: {}".format(self.test_y[loc:loc+self.numpy_window.input_width+1].flatten()))
            print("Predicted: {}".format(pred))
            print("Actual: {}".format(actual))
            print("-------------------------")        

    def summary(self, station):
        summary_dict = {}
        
        summary_dict['station'] = station
        summary_dict['input_width'] = self.numpy_window.input_width
        summary_dict['label_width'] = self.numpy_window.label_width
        summary_dict['num_timeseries_features'] = self.numpy_window.num_timeseries_features 
        summary_dict['num_static_features'] = self.numpy_window.num_static_features        
        summary_dict['timeseries_inputs'] = self.numpy_window.timeseries_source
        summary_dict['static_inputs'] = self.numpy_window.summary_source     

   
        summary_dict['SERA_1%'] = self.average_model_error(station, cut=1)
        summary_dict['SERA_2%'] = self.average_model_error(station, cut=2)    
        summary_dict['SERA_5%'] = self.average_model_error(station, cut=5)        
        summary_dict['SERA_10%'] = self.average_model_error(station, cut=10)  
        summary_dict['SERA_25%'] = self.average_model_error(station, cut=25)  
        summary_dict['SERA_50%'] = self.average_model_error(station, cut=50)  
        summary_dict['SERA_75%'] = self.average_model_error(station, cut=75)  
        summary_dict['SERA_all'] = self.average_model_error(station, cut=100)
        
        return summary_dict





class Switch_Model(Model):
    threshold = 0.7
    
    def __init__(self, window_switch, window_regular, CONV_WIDTH):
        self.window_switch = window_switch
        self.window = window_regular
        
        assert(window_switch.input_width == self.window.input_width)
        
        self.switch = Ensemble_Static(window_switch)
        
        self.regular = Base_Model(model_name='multi-LSTM', window=window_regular, CONV_WIDTH=CONV_WIDTH)
        self.q70 = Base_Model(model_name='multi-LSTM', window=window_regular, CONV_WIDTH=CONV_WIDTH, loss_func=CustomLoss.qloss_70)
        self.q95 = Base_Model(model_name='multi-LSTM', window=window_regular, CONV_WIDTH=CONV_WIDTH, loss_func=CustomLoss.qloss_95)
        
    def predictions(self, station):
        preds_switch = self.switch.predictions(station)     
        
        preds_regular = self.regular.predictions(station)
        preds_q70 = self.q70.predictions(station)        
        preds_q95 = self.q95.predictions(station)
        
        test_array = self.window.test_windows(station)   

        new_pred=[]
        
        for pred_switch, pred_regular, pred_q70, pred_q95 in zip(preds_switch, preds_regular, preds_q70, preds_q95):

                
            switch_condition = pred_switch > 0.95
            q95_condition = pred_switch > 0.7
            q70_condition = pred_switch <= 0.7  # You might want to specify this condition differently

            new_pred.append(np.where(switch_condition, pred_q95, np.where(q95_condition, pred_q70, pred_regular)))
                
        return np.array(new_pred)
        

    def test_MSE(self, station=None):
        preds = self.predictions(data='test', station=station)
        test_array = self.window.test_array(station)[self.window.input_width:]

        return mean_squared_error(test_array, preds)
    
    def test_ROCAUC(self, station, level=0.05):
        preds = self.predictions(data='test', station=station)
        test_array = (self.window.test_array(station)[self.window.input_width:] < level).astype(int)
        
        return roc_auc_score(test_array, preds)

    def summary(self, station=None):
        summary_dict = {}
        
        summary_dict['input_width'] = self.window.input_width
        summary_dict['label_width'] = self.window.label_width
        
        # summary_dict['station'] = station

        summary_dict['NSE'] = self.get_NSE(station)       
                  
        summary_dict['SER_1%'] = self.average_model_error(station, cut=1)
        summary_dict['SER_2%'] = self.average_model_error(station, cut=2)    
        summary_dict['SER_5%'] = self.average_model_error(station, cut=5)        
        summary_dict['SER_10%'] = self.average_model_error(station, cut=10)  
        summary_dict['SER_25%'] = self.average_model_error(station, cut=25)  
        summary_dict['SER_50%'] = self.average_model_error(station, cut=50)  
        summary_dict['SER_75%'] = self.average_model_error(station, cut=75)  
        summary_dict['RMSE'] = self.average_model_error(station, cut=100)
        
        return summary_dict  