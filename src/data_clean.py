import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


class Cleaner:
    def __init__(self):
        self.len_train = 0
        self.feature_names = []
        self.today = pd.datetime(2014,7,1)

    def stage_cleaning(self,train_data):

        self.rider_null =  train_data['avg_rating_by_driver'].median()
        self.driver_null =  train_data['avg_rating_of_driver'].median()

        self.dist_null =  train_data['avg_dist'].median()
        self.dist_mu =  train_data['avg_dist'].mean()
        self.dist_sigma =  train_data['avg_dist'].std()

        self.surge_null =  train_data['avg_surge'].median()
        self.surgepct_null =  train_data['surge_pct'].median()
        self.first30_null =  train_data['trips_in_first_30_days'].median()
        self.black_null =  0
        self.weekday_null =  train_data['weekday_pct'].median()
        return None

    def clean(self,data):
        #first create DF with android iphone dummy
        clean_data = self.phone_types(data)
        #account age, days since last trip
        #clean_data['inactive_days']=self.inactive_days(data)
        #clean_data['acct_age']=self.acct_age(data)
        clean_data['rider_rtg']=self.rider_rtg(data)
        clean_data['driver_rtg']=self.driver_rtg(data)
        #observed non-null in training, will write cleaning methods if null appears in test
        clean_data['avg_dist']=data['avg_dist'].apply(lambda x: (self.dist_null-self.dist_mu)/self.dist_sigma if pd.isnull(x) else (x-self.dist_mu)/self.dist_sigma)
        clean_data['avg_surge']=data['avg_surge'].apply(lambda x: self.surge_null if pd.isnull(x) else x)
        clean_data['surge_pct']=data['surge_pct'].apply(lambda x: self.surgepct_null/100 if pd.isnull(x) else x/100)
        clean_data['first_30']=data['trips_in_first_30_days'].apply(lambda x: self.first30_null if pd.isnull(x) else x)
        clean_data['black']= data['luxury_car_user'].astype('int64').apply(lambda x: self.black_null if pd.isnull(x) else x)
        clean_data['weekday_pct']= data['weekday_pct'].apply(lambda x: self.weekday_null/100 if pd.isnull(x) else x/100)
        #clean_data['surge_bool']=data['surge_pct'].where(data['surge_pct']==0,1)
        clean_data['weekend_signup']=data['signup_date'].apply(lambda x: x.weekday()>4)
        clean_data['weekday_pct']= data['weekday_pct'].apply(lambda x: self.weekday_null/100 if pd.isnull(x) else x/100)

        #map cities
        clean_data = clean_data.merge(self.cities(data),
                                    left_index=True,right_index = True)
        clean_data = clean_data.drop('iPhone',axis = 1)
        return clean_data



        #

    def phone_types(self,df):
        return pd.get_dummies(df['phone'].map({'Android':'Android','iPhone':'iPhone'}))

    def inactive_days(self,df):
        return (self.today-df['last_trip_date']).apply(lambda x: x.days)

    def acct_age(self,df):
        return (self.today-df['signup_date']).apply(lambda x: x.days)

    def rider_rtg(self,df):
        return df['avg_rating_by_driver'].apply(lambda x: self.rider_null if pd.isnull(x) else x)

    def driver_rtg(self,df):
        return df['avg_rating_of_driver'].apply(lambda x: (self.driver_null) if pd.isnull(x) else x)
    def cities(self,df):
        return pd.get_dummies(df['city'].map({'Astapor':'Astapor','Winterfell':'Winterfell'}))


if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv',parse_dates = ['last_trip_date','signup_date'])
    from data_clean import Cleaner

    today = pd.datetime(2014,7,1)
    churn_thresh = pd.Timedelta(30, 'D')
    churn =(today-df['last_trip_date'] >churn_thresh).astype('int64').values

    clnr = Cleaner()
    clnr.stage_cleaning(df)
    clean_data = clnr.clean(df)
    gb = GradientBoostingClassifier(learning_rate=0.005,max_depth=10,
                                n_estimators = 1000,min_samples_leaf=50, max_features = 5)
    df_test = pd.read_csv('data/churn_train.csv',parse_dates = ['last_trip_date','signup_date'])

    churn_test =(today-df_test['last_trip_date'] >churn_thresh).astype('int64').values
    clean_data_test = clnr.clean(df_test)

    gb.fit(clean_data,churn)
    print(gb.score(clean_data_test,churn_test))
