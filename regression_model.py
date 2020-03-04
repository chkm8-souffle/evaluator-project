
import pandas as pd
import numpy as np
import textprep as tp
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.compose import TransformedTargetRegressor
import joblib


# randomized search for parameter optimization
# from sklearn.model_selection import RandomizedSearchCV


df = pd.read_pickle('./data/complete_dataset.pickle')
df = df[df['Pageviews'] < df['Pageviews'].quantile(.95)].reset_index()
print(len(df))

df.loc[:,'18-24'] = df.loc[:,'18-24'].fillna(0)
df.loc[:,'25-34'] = df.loc[:,'25-34'].fillna(0)
df.loc[:,'35-44'] = df.loc[:,'35-44'].fillna(0)
df.loc[:,'45-54'] = df.loc[:,'45-54'].fillna(0)
df.loc[:,'55-64'] = df.loc[:,'55-64'].fillna(0)
df.loc[:,'65+'] = df.loc[:,'65+'].fillna(0)
print(len(df))

print(sum(df['18-24']))
print((df['18-24']).mean())
print(max(df['18-24']))
print((df['18-24']).median())
drop_columns= ['index','fk_grade_level','category','published_time','year','title','url','description','content','tags','year_month','content_tm','title_tm','tags_tm','predicted_sentiment_title','predicted_sentiment_content','title_nouns','title_verbs','title_adj','title_pronouns','content_nouns','content_verbs','content_adj','content_pronouns','tags_nouns','tags_verbs','tags_adj','tags_pronouns','25-34','35-44','45-54','55-64','65+','topic','Document_No','Dominant_Topic','Topic_Perc_Contrib','Keywords','word_count_tags','char_count_tags', 'tags_nouns_count', 'tags_verbs_count', 'tags_adj_count', 'tags_pronouns_count']
df_regression = df.drop(columns=drop_columns, axis=1)

X1 = df_regression.drop(columns=['18-24','Pageviews'])
y1 = df_regression['18-24']

X2 = df_regression.drop(columns=['18-24','Pageviews'])
y2 = df_regression['Pageviews']



X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.4, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.4, random_state=42)

#model for 18-24
model1 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.9040922615763338, learning_rate = 0.033979488347959955,
                max_depth = 2, alpha = 10, n_estimators = 113, subsample = 0.9233589392465844, gamma = 0.2252496259847715, num_rounds=1000)

#model for general
model2 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.790263492945030, learning_rate = 0.041066084206359835,
                max_depth = 2, alpha = 10, n_estimators = 101, subsample = 0.8010716092915446, gamma =  0.1424202471887338, num_rounds=1000)


regr_trans1 = TransformedTargetRegressor(regressor=model1,func=np.log1p, inverse_func=np.expm1)
regr_trans2 = TransformedTargetRegressor(regressor=model2,func=np.log1p, inverse_func=np.expm1)


regr_trans1.fit(X_train1, y_train1)
regr_trans2.fit(X_train2, y_train2)

y_pred1 = regr_trans1.predict(X_test1)
y_pred2 = regr_trans2.predict(X_test2)
print(np.sqrt(mse(y_test1, y_pred1)))
print(np.sqrt(mse(y_test2, y_pred2)))
df_regression.to_pickle('./data/regression_dataset.pickle')
joblib.dump(regr_trans1, './models/xgboost_model1.sav')
joblib.dump(regr_trans2, './models/xgboost_model2.sav')



# from xgboost import plot_importance
# from matplotlib import pyplot
# plot_importance(model)
# pyplot.show()

# def report_best_scores(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")

# xgb_model = xgb.XGBRegressor()

# params = {
#     "colsample_bytree": uniform(0.7, 0.3),
#     "gamma": uniform(0, 0.5),
#     "learning_rate": uniform(0.03, 0.3), # default 0.1 
#     "max_depth": randint(2, 6), # default 3
#     "n_estimators": randint(100, 150), # default 100
#     "subsample": uniform(0.6, 0.4)
# }

# search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True)

# search.fit(X_train, y_train)

# report_best_scores(search.cv_results_, 1)
