import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from data_clean import Cleaner  #Shane's class for cleaning

class MeanVisualization():
    ''' 
    For use with the ridesharing dataset/case study: 
    https://github.com/gSchool/dsi-churn-case-study
    
    Using Shane's Cleaner class, this class will clean 
    the data and then visualize the means of each
    feature broken down by churned vs. did not churn.

    Attributes:
    ---------------
    filepath: string with the filepath of the data to import (must be csv)
              a filepath must be passed in each time this class is instantiated

    df: pandas dataframe created automatically once filepath in passed in

    Methods:
    ---------------
    raw_viz: return barplot with means from raw (unscaled) data
    standardized_viz: return barplot with means from standardized data
    
    '''

    def __init__(self, filepath):
        self.filepath = filepath  #should be a string with the filepath of the raw data (must be csv)
        self.df = pd.read_csv(self.filepath, parse_dates=['last_trip_date','signup_date'])  #parse_dates is hard coded for our specific reqs.

    def raw_viz(self):
        '''
        Creates barplot of the means using raw (unscaled) data.
        '''
        #cleaning the data
        df = self.df
        clnr = Cleaner()
        clnr.stage_cleaning(df)
        clean_data = clnr.clean(df)
        #creating binary column for churn
        clean_data.loc[df['last_trip_date'] > '2014-05-31', 'churn'] = 0  # did not churns
        clean_data.loc[df['last_trip_date'] < '2014-06-01', 'churn'] = 1  # did churns
        #creating pivot table with means to visualize, re-formatting for easy seaborn barplotting
        pivot_mean = pd.pivot_table(clean_data, columns='churn', aggfunc=np.mean)
        churn_col = pd.DataFrame(pivot_mean[1])
        churn_col['churn'] = 1
        churn_col.columns = ['mean', 'churn']
        non_churn_col = pd.DataFrame(pivot_mean[0])
        non_churn_col['churn'] = 0
        non_churn_col.columns = ['mean', 'churn']
        new_df = churn_col.append(non_churn_col)
        new_df.reset_index(level=0, inplace=True)
        # if they exist, dropping acct_age and inactive_days columns
        # because they're uninformative and skew the y-axis of the plot
        if 'inactive_days' in new_df:
            viz_df = new_df[(new_df['index'] != 'acct_age') & (new_df['index'] != 'inactive_days')]
        else:
            viz_df = new_df
        #plotting
        fig, ax = plt.subplots(figsize=(15, 5))
        ax = sns.barplot(x='index', y='mean', hue='churn', data=viz_df)
        plt.title('Unscaled means')


    def standardized_viz(self):
        '''
        Creates barplot of the means using scaled data.
        Uses standard scaler from sklearn on all features.
        '''
        #cleaning the data
        df = self.df
        clnr = Cleaner()
        clnr.stage_cleaning(df)
        clean_data_unscaled = clnr.clean(df)
        #scaling the data
        x = clean_data_unscaled.values  #returns a numpy array
        scaler = preprocessing.StandardScaler()
        x_scaled = scaler.fit_transform(x)
        clean_data = pd.DataFrame(x_scaled)
        clean_data.columns = list(clean_data_unscaled)  #replaces column names lost during scaling
        #creating binary churn column
        clean_data.loc[df['last_trip_date'] > '2014-05-31', 'churn'] = 0  # did not churns
        clean_data.loc[df['last_trip_date'] < '2014-06-01', 'churn'] = 1  # did churns
        #creating pivot table with means to visualize, re-formatting for easy seaborn barplotting
        pivot_mean = pd.pivot_table(clean_data, columns='churn', aggfunc=np.mean)
        churn_col = pd.DataFrame(pivot_mean[1])
        churn_col['churn'] = 1
        churn_col.columns = ['mean', 'churn']
        non_churn_col = pd.DataFrame(pivot_mean[0])
        non_churn_col['churn'] = 0
        non_churn_col.columns = ['mean', 'churn']
        new_df = churn_col.append(non_churn_col)
        new_df.reset_index(level=0, inplace=True)
        # if they exist, dropping acct_age and inactive_days columns
        # because they're uninformative and skew the y-axis of the plot
        if 'inactive_days' in new_df:
            viz_df = new_df[(new_df['index'] != 'acct_age') & (new_df['index'] != 'inactive_days')]
        else:
            viz_df = new_df
        #plotting
        fig, ax = plt.subplots(figsize=(15, 5))
        ax = sns.barplot(x='index', y='mean', hue='churn', data=viz_df)
        plt.title('Scaled means')