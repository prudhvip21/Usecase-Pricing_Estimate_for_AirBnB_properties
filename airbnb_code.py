import pandas as pd
import os
import geopy
from geopy.distance import great_circle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir("C:\\Users\\Prudhvi\\Documents\\data\\airbnb_data")
from sklearn.preprocessing import LabelEncoder
from __future__ import division
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import  train_test_split

train = pd.read_csv('train.csv')
cor_m = train.corr()
f, ax = plt.subplots(figsize=(14,8))
#ax.axes

plt.xticks(rotation=70)
#ax.tick_params(axis='y', rotation=45)
sns.heatmap(cor_m , xticklabels=cor_m.columns.values, yticklabels=cor_m.columns.values)
plt.yticks(rotation = 0)
plt.show()
plt.savefig('corr.png')
test = pd.read_csv('test.csv')

data = pd.concat([train,test])
data.dtypes

char_cols = data.dtypes.pipe(lambda x: x[x == 'object']).index
for c in char_cols:
    data[c]= data[c].astype('category')

data.dtypes
data.describe(include = 'all')
some_csv = data.describe(include = 'all')

data.isnull().any()

data.loc[:,['bathrooms','bedrooms','beds','cleaning_fee','host_total_listings_count','price','review_scores_accuracy','review_scores_checkin','review_scores_cleanliness','review_scores_communication','review_scores_location','review_scores_rating','review_scores_value','reviews_per_month','security_deposit']]= data.loc[:,['bathrooms','bedrooms','beds','cleaning_fee','host_total_listings_count','price','review_scores_accuracy','review_scores_checkin','review_scores_cleanliness','review_scores_communication','review_scores_location','review_scores_rating','review_scores_value','reviews_per_month','security_deposit']].fillna(0)

data.isnull().any()

data.price.value_counts()
data['price_bins'] = pd.cut(data.price , bins = [0,100,200,300,600] , labels= ['low','medium','high','very_high'])
counts = data.price_bins.value_counts()

""" Accomodation analysis """

data['accom_bins'] = pd.cut(data.accommodates , bins = [0,1,2,4,600] , labels= ['one','two','3&4','>4'])
accom = pd.crosstab(data.accom_bins,data.price_bins)
accom['sum']= accom.sum(axis = 1)
accom.loc[:,'low':'very_high'] = accom.loc[:,'low':'very_high'].div(accom['sum'],axis = 0)*100
# the problem with range 0-40. So three categories i,2,3 and 4 and >5
del accom['sum']
print accom


""" Accom plotting"""
f, ax = plt.subplots(figsize=(14,8))
accom.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 150)
plt.xticks(rotation=70 , fontsize= 16)
plt.savefig('accom.png')
plt.show()



""" Cancellation analysis """
canc = pd.crosstab(data.cancellation_policy,data.price_bins)
su = canc.sum(axis = 0)
canc = canc.loc[:,'low':'very_high'].div(su,axis = 1)*100
print canc


""" Cancellation plotting """
canc = canc.T
f, ax = plt.subplots(figsize=(14,8))
canc.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 150)
plt.xticks(rotation=70 , fontsize= 12)
plt.savefig('canc.png')
plt.show()

""" Host verified"""

host_ver = pd.crosstab(data.host_identity_verified,data.price_bins)
su =  host_ver.sum(axis = 0)
host_ver = host_ver.loc[:,'low':'very_high'].div(su,axis = 1)*100
print host_ver
host_ver = host_ver.T

""" Host plotting """

f, ax = plt.subplots(figsize=(14,8))
host_ver.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 150)
plt.xticks(rotation=70 , fontsize= 12)
plt.savefig('host_ver.png')
plt.show()

"""Business travel ready %"""

b_trav = pd.crosstab(data.is_business_travel_ready,data.price_bins)
su = b_trav.sum(axis = 0)
b_trav = b_trav.loc[:,'low':'very_high'].div(su,axis = 1)*100
print b_trav
b_trav = b_trav.T

""" B travel plots """

f, ax = plt.subplots(figsize=(14,8))
b_trav.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 150)
plt.xticks(rotation=70 , fontsize= 12)
plt.savefig('b_travel.png')
plt.show()


print pd.crosstab(data.minimum_nights,data.price_bins)
data['min_night_bins'] = pd.cut(data.minimum_nights , bins = [0,1,2,3,4,5,10,600] , labels= ['1','2','3','4','5','6-10','>10'])
# minimum nights dependendant on time
min_nig =  pd.crosstab(data.min_night_bins,data.price_bins)
min_nig_sum= min_nig.sum(axis = 0)
min_nig = min_nig.loc[:,'low':'very_high'].div(min_nig_sum,axis = 1)*100
print min_nig

""" Min nights plotting """
min_nig = min_nig.T
f, ax = plt.subplots(figsize=(14,8))
min_nig.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 180)
plt.xticks(rotation=70 , fontsize= 12)
plt.savefig('min_nig.png')
plt.show()

""" Hosts listings analysis"""
data['host_bins'] = pd.cut(data.host_total_listings_count , bins = [0,1,2,3,10,50,600] , labels= ['1','2','3','3-10','10-50','50-600'])
host =  pd.crosstab(data.host_bins,data.price_bins)
host_sum= host.sum(axis = 0)
host = host.loc[:,'low':'very_high'].div(host_sum,axis = 1)*100

host = host.T
f, ax = plt.subplots(figsize=(14,8))
host.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 150)
plt.xticks(rotation=70 , fontsize= 12)
plt.savefig('host_listings_count.png')
plt.show()

""" Bathrooms per person"""

data['bathroom_per_person'] = data.bathrooms/data.accommodates
data['bpp_bins'] = pd.cut(data.bathroom_per_person , bins = [0,0.51,0.76,1,2,9])
bpp = pd.crosstab(data.bpp_bins,data.price_bins)
bpp.to_csv('bathrooms_per_person_table.csv')


data['bedroom_per_person'] = data.bedrooms/data.accommodates
data['bepp_bins'] = pd.cut(data.bedroom_per_person , bins = [0,0.51,1,2,9])
bdp = pd.crosstab(data.bepp_bins,data.price_bins)
bdp.to_csv('bedrooms_per_person.csv')


print pd.crosstab(data.instant_bookable,data.price_bins)
pd.crosstab(data.instant_bookable,data.price_bins).to_csv('instant_bookable_table.csv')

""" Room type analysis"""
room_type =  pd.crosstab(data.room_type,data.price_bins)
room_su = room_type.sum(axis = 0)
room_type = room_type.loc[:,'low':'very_high'].div(room_su,axis = 1)*100 


""" Room type plotting """

room_type = room_type.T
f, ax = plt.subplots(figsize=(14,8))
room_type.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 150)
plt.xticks(rotation=70 , fontsize= 12)
plt.savefig('room_type.png')
plt.show()

""" Security deposit analysis """

data['sd_bins'] = pd.cut(data.security_deposit , bins = [0,200,300,500,1000,8000], labels= ['<200','200-300','300-500','500-1000','>1000'])
sd = pd.crosstab(data.sd_bins,data.price_bins)
sd_su= sd.sum(axis = 0)
sd = sd.loc[:,'low':'very_high'].div(sd_su,axis = 1)*100

""" Sd plotting"""

sd = sd.T
f, ax = plt.subplots(figsize=(14,8))
sd.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 150)
plt.xticks(rotation=70 , fontsize= 12)
plt.savefig('sd.png')
plt.show()


cooge = (-33.9190,151.2555)
# Manly beach
manly = (-33.7932,151.2876)
parra = (-33.8150,151.0011)

def dist(row):
    d1 = great_circle(cooge,(row['latitude'],row['longitude'])).kilometers
    d2 = great_circle(manly,(row['latitude'],row['longitude'])).kilometers
    d3 = great_circle(parra, (row['latitude'], row['longitude'])).kilometers
    return np.mean([d1,d2,d3])

data['avg_dist'] = data.apply(lambda x : dist(x),axis = 1)
data['avg_dist_bins'] = pd.cut(data.avg_dist, bins= [12.55,13.077,13.58,14.82,53.88])
di_ana = pd.crosstab(data.avg_dist_bins,data.price_bins)
dis_su= di_ana.sum(axis = 0)
di_ana = di_ana.loc[:,'low':'very_high'].div(dis_su,axis = 1)*100

""" Dist analysis """

di_ana = di_ana.T
f, ax = plt.subplots(figsize=(14,8))
di_ana.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.ylim(ymax = 150)
plt.xticks(rotation=70 , fontsize= 12)
plt.savefig('distance_analysis.png')
plt.show()



""" Remaining """
 #beds
 #clng fee
 #extra people
 #num of reviews per month
 # review score values






""" Modelling part """
data.host_is_superhost = data.host_is_superhost.cat.add_categories(['Missing'])
data.host_is_superhost.fillna(value ='Missing' , inplace= True)

data.host_identity_verified = data.host_identity_verified.cat.add_categories(['Missing'])
data.host_identity_verified.fillna('Missing' , inplace= True)

cat_var = ['cancellation_policy','host_identity_verified','host_is_superhost','instant_bookable','is_business_travel_ready','property_type','room_type','min_night_bins','host_bins','accom_bins']

num_var = ['bathrooms','bedrooms','beds','cleaning_fee','extra_people','host_total_listings_count','minimum_nights','number_of_reviews','reviews_per_month','security_deposit','review_scores_value','review_scores_accuracy']

le = LabelEncoder()
data.loc[:,cat_var] = data.loc[:,cat_var].apply(lambda x : le.fit_transform(x), axis = 0 )
enc = OneHotEncoder(sparse= False)
encode = pd.DataFrame(enc.fit_transform(data.loc[:,cat_var]))
cat_features = []
for i, name in enumerate(cat_var):
    l = len(np.unique(data[name]))
    for k in range(l):
        f_nm = str(name) + '@' + str(k + 1)
        cat_features.append(f_nm)
all_features = cat_features + num_var

""" scaling numerical variables """
#numeric = Imputer().fit_transform(trans_static.loc[:,num_var_list])
from sklearn.preprocessing import StandardScaler
numeric = data.loc[:,num_var]
numeric  = pd.DataFrame(MinMaxScaler().fit_transform(numeric))
numeric  = pd.DataFrame(StandardScaler().fit_transform(numeric))

final_df = encode.merge(numeric , left_index= True , right_index=  True)

final_df = pd.concat([encode,numeric] , axis= 1)

x_train = final_df.head(1000)
y_train = train['price']

x_test = final_df.tail(7967)

#x_train,x_test,y_train,y_test = train_test_split(final_df,data.price, shuffle = False, train_size=1000)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

""" Random Forest model """

model = RandomForestRegressor(n_estimators=300 , max_features=8, max_depth= 80)
model.fit(x_train,y_train)
preds = model.predict(x_train)
model.score(x_train,y_train)
sqrt(mse(y_train,preds))

rand_test_pred = model.predict(x_test)
sub = pd.DataFrame(columns= ['Id','price'])
sub['Id'] = test['Id']
sub['price'] = test_pred
sub.to_csv('sub.csv' , ignore_index = True)

""" Ada boost model """
boost_model = AdaBoostRegressor(base_estimator= model , n_estimators= 100 ,  learning_rate= 0.5)
boost_model.fit(x_train,y_train)
adaboost_preds = boost_model.predict(x_train)
model.score(x_train,y_train)
sqrt(mse(y_train,adaboost_preds))

ada_boost_preds = boost_model.predict(x_test)
sub = pd.DataFrame(columns= ['Id','price'])
sub['Id'] = test['Id']
sub['price'] = test_pred
sub.to_csv('sub.csv' , ignore_index = True)


""" Lasso Regression Model """

model = Lasso( alpha= 0.75)
model.fit(x_train,y_train)
preds = model.predict(x_train)
model.score(x_train,y_train)
sqrt(mse(y_train,preds))

test_pred = model.predict(x_test)
sub = pd.DataFrame(columns= ['Id','price'])
sub['Id'] = test['Id']
sub['price'] = test_pred
sub.to_csv('sub.csv' , ignore_index = True)

""" SVR model """

model = SVR( epsilon= 0.4 , kernel= 'linear' , C= 3)
model.fit(x_train,y_train)
preds = model.predict(x_train)
model.score(x_train,y_train)
sqrt(mse(y_train,preds))

svr_test_pred = model.predict(x_test)
sub = pd.DataFrame(columns= ['Id','price'])
sub['Id'] = test['Id']
sub['price'] = test_pred
sub.to_csv('sub.csv' , ignore_index = True)


""" Stacking model """

sub = pd.DataFrame(columns= ['Id','price'])
sub['Id'] = test['Id']
sub['random_forest'] = rand_test_pred
sub['adaboost_pred'] = ada_boost_preds
sub['svr_preds'] = svr_test_pred
sub['price'] = (sub['random_forest'] + sub['adaboost_pred'] +sub['svr_preds'])/3
sub.to_csv('sub.csv' , ignore_index = True)
