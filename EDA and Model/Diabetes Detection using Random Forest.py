#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[3]:


data = "C:/Users/HP/Downloads/Diabetes_dataset.csv"


# In[4]:


df = pd.read_csv(data, sep = ";")
display(df)


# In[40]:


df.columns


# In[24]:


df.head()


# In[26]:


df.shape


# ## This dataset contains  520  observations with  17  characteristics.

# In[30]:


# Get the number of missing data points per column
missing_values_count = df.isnull().sum()
# Look at the missing points
missing_values_count.to_frame()


# ## Data do not have any missing values.

# In[5]:


ndf = df


# In[6]:


ndf = ndf.drop('class', axis = 1)


# In[7]:


ndf['male'] = ndf['gender']
ndf['female'] = ndf['gender']


# In[8]:


def male(x):
    return 1 if x == "Male" else 0

def female(x):
    return 1 if x == "Female" else 0

male_list = list(ndf['male'])
female_list = list(ndf['female'])


# In[9]:


nmale_list = [male(g) for g in male_list]
nfemale_list = [female(g) for g in female_list]


# In[10]:


ndf['male'] = nmale_list
ndf['female'] = nfemale_list


# In[11]:


ndf = ndf.drop('gender', axis = 1)


# In[12]:


ndf['class'] = df['class']
display(ndf)


# In[13]:


ndf.info()


# In[37]:


plt.figure(figsize=(24, 14))
sns.heatmap(df.corr(), annot=True);


# In[32]:


df.corr()['class'].sort_values(ascending=False) # the correlation


# ## Above are the features that associate with diabetic risk.

# In[34]:


# checking skewness value
skew_value = df.skew().sort_values(ascending=False)
skew_value


# ## If the skewness value lies between -0.5 to 0.5  then it is normal otherwise skewed. There are many features that are skewed.

# In[15]:


drops = ["itching", "delayed_healing", "obesity"]
for item in drops:
    ndf = ndf.drop(f"{item}", axis = 1)
    
display(df)


# In[35]:


plt.figure(figsize=(24, 8)) 
sns.countplot(x='gender', data=df)
plt.title('Gender', fontname='monospace', fontweight='bold', fontsize=15)
plt.xlabel('Gender');


# ## Most distribution is showing male. This shows that men are more likely to develop diabetes than women.

# In[39]:


plt.figure(figsize=(24, 8))
sns.distplot(df['age'], kde=True, color="red");


# ## This shows that most people have diabetes in the age group of 25 to 75 but children, teens, and young adults are also developing it.

# ## Random Forest Classifier

# In[ ]:


x = ndf.drop("class", axis = 1)
y = ndf["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[18]:


randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)


# In[19]:


randomforest_test = randomforest.predict(x_test)


# In[20]:


randomforest_r2 = metrics.r2_score(y_test, randomforest_test)
randomforest_error = metrics.mean_squared_error(y_test, randomforest_test)

print(randomforest_r2)
print(randomforest_error)


# ## This model have an accuracy of 80%+, pretty stable at 90%+.

# ## The error here is very low

# In[21]:


comparison_df = pd.DataFrame()
comparison_df["Confirmed Diabetes"] = y_test
comparison_df["Predicted Diabetes"] = randomforest_test
comparison_df = comparison_df.reset_index(drop = True)
comparison_df.head()


# In[22]:


plt.title("Number of cases of diabetes")
sns.countplot(data = comparison_df, y = y_test)
plt.show()


# In[23]:


plt.title("Number of cases of diabetes")
sns.countplot(data = comparison_df, y = randomforest_test)
plt.show()


# In[ ]:





# In[ ]:




