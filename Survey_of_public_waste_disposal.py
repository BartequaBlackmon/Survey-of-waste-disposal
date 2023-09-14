#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries .
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# reading the csv file
df=pd.read_csv("C:\\Users\\black\\OneDrive\\Desktop\\Omdena_project\\32121-0002 Survey of public waste disposal.csv")


# In[3]:


df.shape


# In[4]:


df.head(20)


# In[5]:


df.info()


# In[6]:


df['Year'] = df['Year'].astype(str)


# In[7]:


df.describe().T


# In[8]:


def data_profiling(df):
    data_profile = []
    columns = df.columns
    for col in columns:
        dtype = df[col].dtypes
        nunique = df[col].nunique()
        null = df[col].isnull().sum()
        duplicates = df[col].duplicated().sum()
        data_profile.append([col,dtype,nunique,null,duplicates])
    data_profile_finding = pd.DataFrame(data_profile)
    data_profile_finding.columns = ['column','dtype','nunique','null','duplicates']
    return data_profile_finding


# In[9]:


data_profiling(df)


# In[10]:


import pandas as pd

# Assuming df is your DataFrame
df['Amount of household waste per inhabitant'] = df['Amount of household waste per inhabitant'].astype(float)


# In[11]:


df.isna().sum()


# In[12]:


# Count number of zeros in all columns of Dataframe
for column_name in df.columns:
    column = df[column_name]
    # Get the count of zeros in column
    count = (column == 0).sum()
    print('Count of zeros in column ', column_name, ' is : ', count )


# In[13]:


for i in range(len(df.columns)):
    print(df.columns[i])
    print(df[df.columns[i]].unique())
    print('**' * 20)


# In[14]:


numeric_features = [i for i in df.columns if df[i].dtype!='0']
categorical_features = [i for i in df.columns if df[i].dtype=='object']


# In[16]:


numeric_features = [col for col in numeric_features if col != 'State']


# In[17]:


print(len(numeric_features))


# In[18]:


print(len(categorical_features))


# In[19]:


for i in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=i,palette='Set2')
    plt.title('Count of ' + i)
    plt.xticks(rotation=90)
    plt.show()


# In[20]:


for feature in numeric_features:
    plt.figure(figsize=(10,4))
    sns.histplot(data=df, x=feature, kde=True)
    plt.title('Distribution of ' + feature)
    plt.show()


# In[21]:


plt.figure(figsize=(8,4))
plt.xticks(rotation=90)
sns.boxplot(data=df)


# In[22]:


for i in numeric_features:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=df[i])
    plt.title('Count of ' + i)
    plt.xticks(rotation=90)
    plt.show()


# In[23]:


plt.figure(figsize=(8,4))
sns.barplot(data=df,x="State", y="generation of household waste")
plt.xticks(rotation=90)


# In[24]:


plt.figure(figsize=(8,4))
sns.barplot(data=df,x="State", y="Amount of household waste per inhabitant")
plt.xticks(rotation=90)


# In[25]:


plt.figure(figsize=(8,4))
sns.barplot(data=df,x="State", y="Household waste disposed of at the first recipient")
plt.xticks(rotation=90)


# In[26]:


plt.figure(figsize=(8,4))
sns.barplot(data=df,x="State", y="Household waste recycled at the first recipient")
plt.xticks(rotation=90)


# In[27]:


sns.lineplot(data=df, x=df['Year'], y=df['Household waste recycled at the first recipient'])


# In[28]:


sns.lineplot(data=df, x=df['Year'], y=df['Household waste disposed of at the first recipient'])


# In[29]:


sns.lineplot(data=df, x=df['Year'], y=df['Amount of household waste per inhabitant'])


# In[30]:


sns.pairplot(df)


# In[31]:


# Compute the correlation matrix
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(15, 12))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap')
plt.show()


# 
