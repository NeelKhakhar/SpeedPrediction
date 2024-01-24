'''
author: Zixuan Zhao@ zxzhao@umich.edu
function: find the optimal interpolation method for fixing the incomplete data
'''
import os
import pandas as pd
import numpy as np
import itertools,operator
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class Imputation:
    _defaults = {
        "data_path"    : 'type0_170.csv',
        "attribute"    : 'Vehicle speed',
        "drop_rate"    : 0.1,
        "sample_rate"  : 50
    }


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.signal = self._get_signal()
        self.ref = self.gen_reference()
        self.tg = self.gen_target()
        self.best_interpolate()


    def _get_signal(self):
        assert self.sample_rate in [50, 10, 1], 'Sample rate must be 50, 10 or 1'

        data_path = os.path.expanduser(self.data_path)
        assert data_path.endswith('.csv')

        data = pd.read_csv(data_path)
        attribute = self.attribute
        signal = data[attribute]
        return signal


    def gen_reference(self):
        '''
        get the longest non-nan segment
        '''
        signal = self.signal
        non_na = signal.dropna(how = 'any').index
        ref_id = max(np.split(non_na, np.where(np.diff(non_na) != 1)[0]+1), key = len)
        ref = signal.iloc[ref_id]
        print('The reference set is generated based on the longest non-na segment'\
            ' which is of length {}.'.format(len(ref)))
        return ref


    def gen_target(self):
        ref = self.ref
        drop_rate = self.drop_rate
        nan_mat = np.random.random(ref.shape) < drop_rate
        print('Totally {} points out of {} points are randomly replaced with "NaN"'.format(
            nan_mat.sum(), len(ref)))
        tg = ref.mask(nan_mat)
        return tg


    def best_interpolate(self):
        ref = self.ref
        tg = self.tg
        
        df = pd.DataFrame()
        df = df.assign(Target = tg)
        df = df.assign(InterpolateLinear    = tg.interpolate(method='linear'))
        df = df.assign(InterpolateQuadratic = tg.interpolate(method='quadratic'))
        df = df.assign(InterpolateCubic     = tg.interpolate(method='cubic'))
        df = df.assign(InterpolateSLinear   = tg.interpolate(method='slinear'))
        df = df.assign(InterpolateAkima     = tg.interpolate(method='akima'))
        df = df.assign(InterpolatePoly5     = tg.interpolate(method='polynomial', order=5)) 
        df = df.assign(InterpolatePoly7     = tg.interpolate(method='polynomial', order=7))
        df = df.assign(InterpolateSpline3   = tg.interpolate(method='spline', order=3))
        df = df.assign(InterpolateSpline4   = tg.interpolate(method='spline', order=4))
        df = df.assign(InterpolateSpline5   = tg.interpolate(method='spline', order=5))

        results = [(method, r2_score(ref, df[method])) for method in list(df)[1:]]
        results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])
        results_df.sort_values(by='R_squared', ascending=True)
        print('-------------------- results --------------------')
        print(results_df)
        
        styles=['k--', 'bo-', 'r*', 'y^-']
        df[['InterpolateSLinear', 'Target']].plot( figsize=(20,10));
        plt.ylabel('Vehicle speed (km/h)');
        plt.xlabel('Time (s)');
        plt.show()

        return results_df 


    def close_session(self):
        self.sess.close()


if __name__ == "__main__":
    imputation = Imputation()
    tg = imputation.tg
    print(type(tg))
