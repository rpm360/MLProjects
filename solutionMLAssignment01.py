#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
train_df=pd.read_csv("train.csv")

test_df=pd.read_csv("test.csv")
combine=[train_df,test_df]

## combining the both the dataset in to variable x
x=pd.concat(combine,sort=True)




print("## Q1 solution")
print(x.columns.values)
print()




print("## Q5")
tot = x.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([total], axis=1, keys=['Total Missing value'])
print(missing_data.head(12))
print()


print("## Q6")
print(x.info())
print()

### printing all the numeric data with count mean ...max
## here we also get columns of numeric datatype i.e question 3
print("## Q7 and Q3")
print(x.describe())
print()

### printing all the categorial data with count,unique ...feq
### here we also get columns of categoriacal datatype  i.e question 2

print("## Q8 & Q2")
print(x.describe(include=['O']))
print()


print("Q4")
print(x.head())
print()


