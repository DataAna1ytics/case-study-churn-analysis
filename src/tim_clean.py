import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Cleaner:

    def __init__(self):
        self.len_train = 0
        self.feature_names = []
        self.today = pd.to_datetime('7/1/2014')

    def stage_and_clean(self, df):
        # send data to staging for initialization and separation
        X, y = self.stage(df)

        # clean x and y
        X = self.cleanX(X)
        y = self.cleany(y)
        return X, y

    def stage(self, df):
        # save null values
        self.rider_null = df['avg_rating_by_driver'].median()
        self.driver_null = df['avg_rating_of_driver'].median()

        df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
        df['signup_date'] = pd.to_datetime(df['signup_date'])

        # separate x and y
        y = df.drop('last_trip_date', axis=1)
        X = df
        return X, y

    def cleany(self, y):
        inactive_days = (pd.Timedelta self.today - y).apply(lambda x: x.days)
        churn_mask = inactive_days > 30
        return churn_mask.astype('int64')

    def cleanX(self, X):
        # first create clean df with android iphone dummy
        clean_X = self.phone_types(X)

        # convert datetimes to datetimes
        clean_X['acct_age'] = (self.today - X['signup_date'])\
                                .apply(lambda x: x.days)

        # map dummies
        clean_X = clean_X.merge(self.cities(X),
                                      left_index=True, right_index=True)
        clean_X['black'] = X['luxury_car_user'].astype('int64')

        # ensure full columns in rating columns, using staged values as needed
        clean_X['rider_rtg'] = self.rider_rtg(X)
        clean_X['driver_rtg'] = self.driver_rtg(X)

        # observed to be non-null in training data
        # will write cleaning methods if null appears in test
        clean_X['avg_dist'] = X['avg_dist']
        clean_X['avg_surge'] = X['avg_surge']
        clean_X['surge_pct'] = X['surge_pct']
        clean_X['first_30'] = X['trips_in_first_30_days']
        clean_X['weekday_pct'] = X['weekday_pct']

        # save cleaned changes
        return clean_X

    def phone_types(self,df):
        return pd.get_dummies(df['phone'].map({'Android':'Android','iPhone':'iPhone'}))

    def rider_rtg(self,df):
        return df['avg_rating_by_driver'].apply(lambda x: self.rider_null if pd.isnull(x) else x)

    def driver_rtg(self,df):
        return df['avg_rating_of_driver'].apply(lambda x: self.driver_null if pd.isnull(x) else x)

    def cities(self,df):
        return pd.get_dummies(df['city'].map({'Astapor':'Astapor','Winterfell':'Winterfell'}))
