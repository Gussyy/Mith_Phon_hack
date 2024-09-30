from autogluon.tabular import TabularPredictor
import pandas as pd

class RAI_ML:
    def __init__(self, file_path='data_test.csv') -> None:
        df = pd.read_csv('data_test.csv')
        df['avg_wvp'] = df['avg_wvp']/df['area']
        df['avg_ndvi'] = df['avg_ndvi']/df['area']
        self.df = df

    def __call__(self, rai_num):
        print(rai_num)
        return self.check_rai(rai_num)

    def check_rai(self, rai_num):
        index = self.df[self.df['Unnamed: 0'] == rai_num].index
        text = 'Your field status:\n'
        print(self.df['y'][index])
        if self.df['y'][index].item() >= 0.6: #yield mean
            text += 'Your crop is very healthy (Above avg yield)'
            return text
        elif self.df['avg_ndvi'][index].item() <= 5.0: #ndvi mean
            text += '-You need to check you crop. They are less green\n'
        else:
            text += 'Your crop is very normal (Avg yield)'
            return 
        if self.df['avg_wvp'][index].item() <= 2.6: #wvp mean
            text += '-You need to give your plant more water\n'
        if self.df['avg_ndvi'][index].item() >= 6.0: #above ndvi mean
            text += '-You need to check you crop. They are too green. There might be a weed\n'

        return text
        