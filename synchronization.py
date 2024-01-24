import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def get_length(file1, file2):
    speed1 = file1['Vehicle speed']
    speed1[0] = 0 # set the first cell to 0 if empty
    speed2 = file2['Vehicle speed']
    speed2[0] = 0 # set the first cell to 0 if empty
    d1 = next((i for i, x in enumerate(speed1) if x), None)
    d2 = next((i for i, x in enumerate(speed2) if x), None)
    
    return len(speed1), len(speed2), d1, d2


def gen_sync_file(out_dir, data1_pth, data2_pth):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print("Processing file " + data1_pth[-10:-6])
    data1 = pd.read_csv(data1_pth)
    data2 = pd.read_csv(data2_pth)
    cols = [0, 1, 2]
    l1, l2, d1, d2 = get_length(data1, data2)
    if d2 >= d1:
        data2 = data2.iloc[abs(d1-d2): , : ]
        if len(data1) >= len(data2):
            data1.drop(data1.tail(abs(l1-d1-l2+d2)).index, inplace=True)
        else:
            data2.drop(data2.tail(abs(l1-d1-l2+d2)).index, inplace=True)
    else:
        data1 = data1.iloc[abs(d1-d2): , : ]
        if len(data1) >= len(data2):
            data1.drop(data1.tail(abs(l1-d1-l2+d2)).index, inplace=True)
        else:
            data2.drop(data2.tail(abs(l1-d1-l2+d2)).index, inplace=True)
    
    
    data2.drop(data2.columns[cols], axis = 1, inplace = True) 
    assert len(data1) == len(data2), "length of data files not same"
    data1.index = np.arange(0, len(data1))
    data2.index = np.arange(0, len(data1))

    sync_data = pd.concat([data1, data2], axis=1)

    trip_id = data1_pth[-10:-6]
    sync_data.to_csv(out_dir + 'sync_{}.csv'.format(trip_id), index=False)


if __name__ == "__main__": 

    data_file1_dir = 'C:\\Users\\user\\Dropbox (University of Michigan)\\zixuan_nissan\\backup\\2021\\NMC-0001_Learning_Data_Set_FY20\\Mountain_course 2_Data_1\\'
    data_file2_dir = 'C:\\Users\\user\\Dropbox (University of Michigan)\\zixuan_nissan\\backup\\2021\\NMC-0001_Learning_Data_Set_FY20\\Mountain_course 2_Data_2\\'
    output_dir = '.\\Synced_Mountain_2_new\\'

    data1_list = os.listdir(data_file1_dir)
    data2_list = os.listdir(data_file2_dir)

    def order(x):
        return(x[-9:-6])

    sorted(data1_list, key=order)
    sorted(data2_list, key=order)

    for idx, data_file in enumerate(data1_list):
        data1 = data_file1_dir + data_file
        data2 = data_file2_dir + data2_list[idx]
        gen_sync_file(output_dir, data1, data2)
