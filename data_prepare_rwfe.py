'''
@siyu yang 
@11/18/2021
generate the data files for training and testing data for module.
including average speed data attributes (but not distance to average speed data)
for driver A 10 trips
based on synced data
'''

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import column_or_1d

def fill(dataframe, attribute):
    dataframe[attribute].fillna(method = 'pad',inplace=True)
    dataframe[attribute].fillna(method = 'backfill',inplace=True)    

def generate_final_frame(path, filename, RoadTypes):
    dataframe = pd.read_csv(path + '/'+filename,index_col=None, header=0, 
            usecols=['Accelerator pedal operation','Steering angle','Longitudinal acceleration','Transversal acceleration','Vehicle speed',
                        'Average_Speed_Data_0','Average_Speed_Data_1','Slope_Data_0','Slope_Data_1', 'Curvature_Data_0', 'Curvature_Data_1','Road type'])
    dataframe.replace({'Average_Speed_Data_0':1023,'Average_Speed_Data_1':1023, 'Slope_Data_0':51.2,'Slope_Data_1':51.2, 'Curvature_Data_0':1023, 'Curvature_Data_1':1023}, np.nan, inplace=True) # delete 1023km/h (change to nan)
   
    # impute road type average speed data 0&1
    fill(dataframe,'Road type')
    fill(dataframe,'Average_Speed_Data_0')
    fill(dataframe,'Average_Speed_Data_1')
    fill(dataframe,'Slope_Data_0')
    fill(dataframe,'Slope_Data_1')
    fill(dataframe,'Curvature_Data_0')
    fill(dataframe,'Curvature_Data_1')

    dataframe.dropna(axis=0,subset=['Accelerator pedal operation'],inplace=True)   # drop na
    frame = dataframe.loc[dataframe['Road type'].isin(RoadTypes) ] # select road type

    order = ['Average_Speed_Data_0','Average_Speed_Data_1','Slope_Data_0','Slope_Data_1', 'Curvature_Data_0', 'Curvature_Data_1','Accelerator pedal operation','Steering angle','Longitudinal acceleration','Transversal acceleration','Vehicle speed']
    frame = frame[order]
    print(frame)
    return frame

if __name__ == '__main__':

    path = '..\\Jose Salazar’s files\\Shared\\NissanProj\\SyncedData\\RWFE_course'
    RoadTypes = [5,6]
    # select road type = [0] M1
    # select road type = [2,3,4] M2
    # select road type = [5,6] M3

    filenames = os.listdir(path)
    print(filenames)

    for filename in filenames:
        frame = generate_final_frame(path, filename, RoadTypes)
        frame.to_csv('..\\speed_prediction_repo\\output_of_rwfe\\'+ filename,index = None)
    
