
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
get_ipython().magic('matplotlib inline')


# In[2]:


pwd


# In[3]:


cd '/Users/Chanti/Desktop/Cookbook/Chapter 10'


# In[4]:


pwd


# In[5]:


dataframe = pd.read_csv("kc_house_data.csv", header='infer')


# In[6]:


list(dataframe)


# In[7]:


dataframe.head()


# In[8]:


dataframe.tail()


# In[9]:


dataframe.describe()


# In[10]:


dataframe['bedrooms'].value_counts().plot(kind='bar')
plt.title('No. of bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# In[11]:


dataframe['bedrooms'].value_counts().plot(kind='pie')
plt.title('No. of bedrooms')


# In[12]:


dataframe['floors'].value_counts().plot(kind='bar')
plt.title('Number of floors')
plt.xlabel('No. of floors')
plt.ylabel('Count')
sns.despine


# In[13]:


plt.figure(figsize=(20,20))
sns.jointplot(x=dataframe.lat.values, y=dataframe.long.values, size=9)
plt.xlabel('Longitude', fontsize=10)
plt.ylabel('Latitude', fontsize=10)
plt.show()
sns.despine()


# In[14]:


plt.figure(figsize=(20,20))
sns.jointplot(x=dataframe.lat.values, y=dataframe.long.values, size=9)
plt.xlabel('Longitude', fontsize=10)
plt.ylabel('Latitude', fontsize=10)
plt.show()
sns.despine()


# In[15]:


plt.figure(figsize=(8,8))
plt.scatter(dataframe.price, dataframe.sqft_living)
plt.xlabel('Price')
plt.ylabel('Square feet')
plt.show()


# In[16]:


plt.figure(figsize=(5,5))
plt.bar(dataframe.condition, dataframe.price)
plt.xlabel('Condition')
plt.ylabel('Price')
plt.show()


# In[17]:


plt.figure(figsize=(8,8))
plt.scatter(dataframe.zipcode, dataframe.price)
plt.xlabel('Zipcode')
plt.ylabel('Price')
plt.show()


# In[18]:


plt.figure(figsize=(10,10))
plt.scatter(dataframe.grade, dataframe.price)
plt.xlabel('Grade')
plt.ylabel('Price')
plt.show()


# In[19]:


x_df = dataframe.drop(['id','date',], axis = 1)
x_df


# In[20]:


y = dataframe[['price']].copy()
y_df = pd.DataFrame(y)
y_df


# In[21]:


print('Price Vs Bedrooms: %s' % x_df['price'].corr(x_df['bedrooms']))
print('Price Vs Bathrooms: %s' % x_df['price'].corr(x_df['bathrooms']))
print('Price Vs Living Area: %s' % x_df['price'].corr(x_df['sqft_living']))
print('Price Vs Plot Area: %s' % x_df['price'].corr(x_df['sqft_lot']))
print('Price Vs No. of floors: %s' % x_df['price'].corr(x_df['floors']))
print('Price Vs Waterfront property: %s' % x_df['price'].corr(x_df['waterfront']))
print('Price Vs View: %s' % x_df['price'].corr(x_df['view']))
print('Price Vs Grade: %s' % x_df['price'].corr(x_df['grade']))
print('Price Vs Condition: %s' % x_df['price'].corr(x_df['condition']))
print('Price Vs Sqft Above: %s' % x_df['price'].corr(x_df['sqft_above']))
print('Price Vs Basement Area: %s' % x_df['price'].corr(x_df['sqft_basement']))
print('Price Vs Year Built: %s' % x_df['price'].corr(x_df['yr_built']))
print('Price Vs Year Renovated: %s' % x_df['price'].corr(x_df['yr_renovated']))
print('Price Vs Zipcode: %s' % x_df['price'].corr(x_df['zipcode']))
print('Price Vs Latitude: %s' % x_df['price'].corr(x_df['lat']))
print('Price Vs Longitude: %s' % x_df['price'].corr(x_df['long']))


# In[22]:


x_df.corr().iloc[:,-19]


# In[23]:


sns.pairplot(data=x_df,
                  x_vars=['price'],
                  y_vars=['bedrooms', 'bathrooms', 'sqft_living', 
                          'sqft_lot', 'floors', 'waterfront','view',
                          'grade','condition','sqft_above','sqft_basement',
                          'yr_built','yr_renovated','zipcode','lat','long'],
            size = 5)


# In[24]:


x_df2 = x_df.drop(['price'], axis = 1)


# In[25]:


reg=linear_model.LinearRegression()


# In[26]:


x_train,x_test,y_train,y_test = train_test_split(x_df2,y_df,test_size=0.4,random_state=4)


# In[27]:


reg.fit(x_train,y_train)


# In[28]:


reg.coef_


# In[29]:


predictions=reg.predict(x_test)
predictions


# In[30]:


reg.score(x_test,y_test)


# In[31]:


import xgboost


# In[91]:


new_model = xgboost.XGBRegressor(n_estimators=750, learning_rate=0.01, gamma=0, subsample=0.55, colsample_bytree=1, max_depth=10)


# In[92]:


from sklearn.model_selection import train_test_split


# In[93]:


traindf, testdf = train_test_split(x_train, test_size = 0.2)
new_model.fit(x_train,y_train)


# In[94]:


from sklearn.metrics import explained_variance_score
predictions = new_model.predict(x_test)
print(explained_variance_score(predictions,y_test))

