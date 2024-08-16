import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import folium
import os
import glob
import random

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


def read_data_from_file(data_dir):
    timeseries_dfs = []
    summary_dfs = []
    
    # Read all csv files from directory
    # Sort files into timeseries and summary data
    for file_path in glob.glob(data_dir + '**/*.csv', recursive=True):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # print(file_path, file_name)
        df = pd.read_csv(file_path, low_memory=False) 
    
        #skip these files
        if file_name in ['streamflow_QualityCodes']:
            continue
    
        if 'year' in df.columns:    
            df['source'] = file_name
            df= df[df['year'] > 1990]
            df= df.drop_duplicates(['year','month','day'])
            timeseries_dfs.append(df)
        else:
            df = df.rename({'ID':'station_id'}, axis=1)
            df = df.set_index('station_id')
            summary_dfs.append(df)
    
    timeseries_data = pd.concat(timeseries_dfs, axis=0, ignore_index=True)
    timeseries_data['date'] = pd.to_datetime(timeseries_data[['year', 'month', 'day']])
    timeseries_data = timeseries_data.drop(['year', 'month', 'day'], axis=1)
    
    summary_data = pd.concat(summary_dfs, axis=1)

    return timeseries_data, summary_data


def plot_catchments(camels_data, data_dir):
    
    cd_sd = camels_data.summary_data.loc[:,~camels_data.summary_data.columns.duplicated()]

    # Define the list of cities and their latitudes/longitudes
    cities = cd_sd['station_name']
    lats = cd_sd['lat_outlet']
    longs = cd_sd['long_outlet']
    
    # Generate a random priority for each city between 1 and 5
    priority = np.random.randint(1, 6, size=len(cities))
    state= cd_sd['state_outlet']
    
    # Create the DataFrame with the city data
    data = {'cityname': cities,
            'lats': lats,
            'longs': longs,
            'States': state,
            'priority': priority}
    
    df = pd.DataFrame(data)

    state_mapping = {'QLD': 1, 'NSW': 2, 'SA': 3, 'VIC': 4, 'ACT': 5, 'WA': 6, 'NT': 7, 'TAS': 8}
    df['state_num'] = df['States'].map(state_mapping)

    # Load the shapefile of Australia
    # australia = gpd.read_file('STE_2021_AUST_SHP_GDA2020/STE_2021_AUST_GDA2020.shp')
    shape_file = data_dir + '02_location_boundary_area/shp/bonus data/Australia_boundaries.shp'
    australia = gpd.read_file(shape_file)
    
    # Define the CRS of the shapefile manually
    australia.crs = 'epsg:7844'
    
    # Create a GeoDataFrame from the DataFrame of cities
    gdf_cities = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longs, df.lats))
    
    # Set the CRS of the GeoDataFrame to EPSG 7844
    # https://epsg.io/7844
    gdf_cities.crs = 'epsg:7844'
    
    # Reproject the GeoDataFrame of cities to match the CRS of the shapefile
    gdf_cities = gdf_cities.to_crs(australia.crs)
    
    # Perform a spatial join to link the cities to their corresponding polygons in the shapefile
    gdf_cities = gpd.sjoin(gdf_cities, australia, predicate='within')
    
    # Set up the plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # # Define a custom dark color palette
    custom_palette = sns.color_palette(['darkblue', 'black', 'purple',
                                        'darkred', 'darkgreen', 'darkorange',
                                        'brown' , 'blue'], 
                                       n_colors=len(df['state_num'].unique()))
    
    # Plot the cities colored by priority with adjustments
    sns.scatterplot(ax=ax, data=gdf_cities, x='longs', y='lats', hue='States',
                    s=15, palette=custom_palette, edgecolor='black',
                    alpha=0.8, legend='full', zorder=2)
    
    
    # Set x-axis limits
    ax.set_xlim(110, 160)
    
    # Add the shapefile of Australia as a background map
    australia.plot(ax=ax, color='lightgrey', edgecolor='white', zorder=1)
    
    # Set the plot title and axis labels
    ax.set_title('Catchments across Australia')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    return fig




class PrepareData():
    
    def __init__(self, timeseries_data, summary_data):
        ### Data Cleaning
        self.timeseries_data = timeseries_data.replace(-99.99,np.NaN)
        
        ### Feature Engineering
        # get precipitation deficit
        actualTransEvap_data = self.timeseries_data[self.timeseries_data['source'] == 'et_morton_actual_SILO'].drop(['source'], axis=1)
        precipitation_data = self.timeseries_data[self.timeseries_data['source'] == 'precipitation_AWAP'].drop(['source'], axis=1)
         
        actualTransEvap_data = actualTransEvap_data[actualTransEvap_data['date'].isin(precipitation_data['date'])].reset_index(drop=True)
        precipitation_data = precipitation_data[precipitation_data['date'].isin(actualTransEvap_data['date'])].reset_index(drop=True)
        
        self.precipitation_deficit = precipitation_data.drop(['date'], axis=1).subtract(actualTransEvap_data.drop(['date'], axis=1))
        self.precipitation_deficit['source'] = 'precipitation_deficit'
        self.precipitation_deficit['date'] = precipitation_data['date']
        
        # get flood probabilities
        self.streamflow_data = self.timeseries_data[timeseries_data['source'] == 'streamflow_MLd_inclInfilled'].drop(['source'], axis=1)
        self.streamflow_data = self.streamflow_data.set_index('date')
        
        self.flood_probabilities = self.streamflow_data.apply(self.flood_extent, axis=0)
        self.flood_probabilities['source'] = 'flood_probabilities'
        self.flood_probabilities['date'] = self.streamflow_data.index
        
        self.flood_indicator = self.flood_probabilities.applymap(lambda x: int(x <0.05) if pd.isnull(x) == False and isinstance(x, float) else x)
        self.flood_indicator['source'] = 'flood_indicator'
        self.flood_indicator['date'] = self.flood_probabilities['date']        
        
        # turn date into sin and cos function 
        date_min = np.min(self.flood_probabilities['date'])
        year_seconds = 365.2425*24*60*60
        year_sin = self.flood_probabilities['date'].apply(lambda x: np.sin((x-date_min).total_seconds() * (2 * np.pi / year_seconds)))
        year_cos = self.flood_probabilities['date'].apply(lambda x: np.cos((x-date_min).total_seconds() * (2 * np.pi / year_seconds)))
        all_stations = list(self.flood_probabilities.drop(columns=['source', 'date'], axis=1).columns) 
        
        df_sin = []     
        for value in year_sin:
            df_sin.append({k:value for k in all_stations})
            
        df_sin = pd.DataFrame(df_sin)
        df_sin['source'] = 'year_sin'
        df_sin['date'] = self.flood_probabilities['date']
 
        df_cos = []
        for value in year_cos:
            df_cos.append({k:value for k in all_stations})
            
        df_cos = pd.DataFrame(df_cos)
        df_cos['source'] = 'year_cos'
        df_cos['date'] = self.flood_probabilities['date']
            
        ### Return
        self.timeseries_data = pd.concat([self.timeseries_data, self.precipitation_deficit, self.flood_probabilities, df_sin, df_cos, self.flood_indicator], axis=0).reset_index(drop=True)
        self.summary_data = summary_data
        
    def get_timeseries_data(self, source, stations):      
        # filter by source
        self.data_filtered = self.timeseries_data[self.timeseries_data['source'].isin(source)]
        # pivot data by station
        self.data_filtered = self.data_filtered[['date', 'source'] + stations].pivot(index='date', columns='source', values=stations)
        # get rows with no nan
        self.data_filtered = self.data_filtered[~self.data_filtered.isnull().any(axis=1)]
        
        return self.data_filtered
        
        
    def get_data(self, source, stations):
        summary_source = [i for i in source if i in list(self.summary_data.columns)]
        timeseries_source = [i for i in source if i not in list(self.summary_data.columns)]
     
        # filter by source
        self.data_filtered = self.timeseries_data[self.timeseries_data['source'].isin(timeseries_source)]
        # pivot data by station
        self.data_filtered = self.data_filtered[['date', 'source'] + stations].pivot(index='date', columns='source', values=stations)
        # get rows with no nan
        self.data_filtered = self.data_filtered[~self.data_filtered.isnull().any(axis=1)]
        
        for station in stations:
            for variable in summary_source:
                value = self.summary_data.loc[station][variable]
                self.data_filtered[station, variable] = value
        
        return self.data_filtered.sort_index(axis=1)
    

    
    def get_train_val_test(self, source, stations, 
                           scaled=True, target=['streamflow_MLd_inclInfilled'],
                           start=None, end=None,
                           discard=0.05, train=0.6, test=0.4):
        assert 0<=discard<=1
        assert (train + test) == 1
     
        summary_source = [i for i in source if i in list(self.summary_data.columns)]
        timeseries_source = [i for i in source if i not in list(self.summary_data.columns)]        
        
        all_data = self.get_timeseries_data(timeseries_source, stations).loc[start:end]
        n_rows_all = len(all_data)
        
        all_data_discarded = all_data.iloc[int(n_rows_all*discard):]
        n_rows_discarded = len(all_data_discarded)
        
        train_df = all_data_discarded[:int(n_rows_discarded*train)]
        test_df = all_data_discarded[-int(n_rows_discarded*(test)):]
        
        if scaled == True:
            scaler = MinMaxScaler()
            scaler.fit(train_df)
            
            scaler_test = MinMaxScaler()
            scaler_test.fit(test_df)
            
            train_df = pd.DataFrame(scaler.transform(train_df), index=train_df.index, columns=train_df.columns)
            test_df = pd.DataFrame(scaler_test.transform(test_df), index=test_df.index, columns=test_df.columns)
            
     
        for station in stations:
            for variable in summary_source:
                value = self.summary_data.loc[station][variable]
                
                train_df[station, variable] = value                
                test_df[station, variable] = value 
                                  
        return train_df.sort_index(axis=1), test_df.sort_index(axis=1) 
    
    def flood_extent(self, streamflow_ts):
        station_name = streamflow_ts.name

        flow_data = pd.DataFrame(streamflow_ts)  
        na_values = flow_data[flow_data[station_name].isna()][station_name]

        flow_data = flow_data.dropna().sort_values(by=station_name, ascending=False).reset_index()
        flow_data['probability'] = (flow_data.index + 1)/(1+len(flow_data)) 
        flow_data = flow_data.sort_values(by='date').drop(['date', station_name], axis=1)['probability']
        flow_data = pd.concat([na_values, flow_data]).reset_index(drop=True) 
        flow_data.name = station_name  

        return flow_data 



if __name__ == '__main__':

    # Read timeseries and summary data from data dir
    data_dir = '/srv/scratch/z5370003/data/camels-dropbox/'
    timeseries_data, summary_data = read_data_from_file(data_dir)

    # Create Dataset
    camels_data = PrepareData(timeseries_data, summary_data)

    # Plot catchments on map
    plot_catchments(camels_data, data_dir)


