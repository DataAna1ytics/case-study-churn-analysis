import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from data_clean import Cleaner

sns.set(style="ticks")
df = pd.read_csv('data/churn_train.csv')
dcc = Cleaner()

df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
df['signup_date'] = pd.to_datetime(df['signup_date'])
dcc.stage_cleaning(df)
df = dcc.clean(df)
# sns.pairplot(df)
# plt.savefig('scatter_matrix.png')

print(df.head())


# plt.hist(df['driver_rtg'])
# plt.show()

# high_driver_rating = df[df['driver_rtg'] == 5.0]
# low_driver_rating = df[df['driver_rtg'] < 5.0]

y = df.pop('surge_bool')
X = df


# kf = KFold(n_splits=5, shuffle=True)
# preds = [] 
# for i, (train_index, test_index) in enumerate(kf.split(y)):
    # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    # y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # log_reg = LogisticRegression(solver='lbfgs')
    # log_reg.fit(X_train, y_train)
    # preds.append(log_reg.predict(X_test))
    

# print(preds)
# preds = np.array([np.mean(preds[:, i]) for i in range(len(preds[0]))])


clf = LogisticRegressionCV(cv=5, random_state=0, solver='liblinear', n_jobs=-1).fit(X, y)
print("Logistic Regression score: ", clf.score(X, y))
print(clf.predict_proba(X))
