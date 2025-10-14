# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:


import pandas as pd 

from scipy import stats 

import numpy as np 

df = pd.read_csv("bmi.csv")

df.head()

<img width="304" height="122" alt="image" src="https://github.com/user-attachments/assets/97f3af47-13a2-4fa1-ad42-5380f3086ad0" />


max_values = np.max(np.abs(df[['Height','Weight']]))
max_values


<img width="120" height="34" alt="image" src="https://github.com/user-attachments/assets/f4e207e0-795f-429a-9204-0b41be49088a" />



from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

df[['Height','Weight']] = sc.fit_transform(df[['Height','Weight']])

df.head(10)

<img width="354" height="227" alt="image" src="https://github.com/user-attachments/assets/e82ba202-27e9-44aa-a917-23e8a34d52d2" />


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[["Height",'Weight']] = scaler.fit_transform(df[['Height','Weight']])

df.head(10)

<img width="351" height="231" alt="image" src="https://github.com/user-attachments/assets/b8a782f4-4fbe-49a8-9c74-b41a386d272c" />



from sklearn.preprocessing import Normalizer

scaler = Normalizer()

df[["Height",'Weight']] = scaler.fit_transform(df[['Height','Weight']])

df.head(10)

<img width="348" height="231" alt="image" src="https://github.com/user-attachments/assets/d7ed79ce-948b-41bd-a941-fcb501089329" />


from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])

df.head()

<img width="345" height="132" alt="image" src="https://github.com/user-attachments/assets/08963ba9-3856-47ac-aeb8-e5d6ac087406" />



from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])

df.head()

<img width="347" height="131" alt="image" src="https://github.com/user-attachments/assets/0c1bc584-4ad4-46b7-a04a-2aac2f52998a" />


df = pd.read_csv("titanic_dataset.csv")

df.columns

<img width="692" height="65" alt="image" src="https://github.com/user-attachments/assets/7ab3ed10-898a-4159-8286-a5836a0616f1" />




df = df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)

df.columns

<img width="762" height="48" alt="image" src="https://github.com/user-attachments/assets/169edc21-d547-4f1b-a72f-80a0cbccbeab" />


df["Age"].isnull().sum()

<img width="199" height="30" alt="image" src="https://github.com/user-attachments/assets/54f2d674-40fe-4454-afd4-fa8d699d785e" />



df["Age"].fillna(method="ffill")

<img width="351" height="246" alt="image" src="https://github.com/user-attachments/assets/b7b36b5e-6611-4a0e-a8aa-06b27818a742" />


cols = df.columns.tolist()

cols[-1], cols[1] = cols[1],cols[-1]

df= df[cols]

df.columns

<img width="780" height="55" alt="image" src="https://github.com/user-attachments/assets/c4034265-94a6-486a-9d75-7973d3dfb4a0" />



x = df.iloc[:,0:6]
y = df.iloc[:,6]

x.columns

<img width="690" height="46" alt="image" src="https://github.com/user-attachments/assets/9aaa5db1-3b2e-437d-87fc-af44831c91d2" />



y = y.to_frame()


y.columns

<img width="410" height="35" alt="image" src="https://github.com/user-attachments/assets/920fa8d8-d1c8-4698-86dc-8333f8163b7a" />

import pandas as pd 
from sklearn.feature_selection import chi2

data = pd.read_csv("titanic_dataset.csv")


data =data.dropna()

X = data.drop(['Survived','Name','Ticket'],axis=1)

y = data["Survived"]

X

<img width="754" height="245" alt="image" src="https://github.com/user-attachments/assets/4158dbf0-803e-4aa6-84f5-2a9e6458cdfb" />

data["Sex"] = data["Sex"].astype("category")
data["Cabin"] = data["Cabin"].astype("category")
data["Embarked"] = data["Embarked"].astype("category")

data

<img width="661" height="256" alt="image" src="https://github.com/user-attachments/assets/8d06274b-fd86-4bee-83af-1d98cb731937" />


x.info()


<img width="393" height="211" alt="image" src="https://github.com/user-attachments/assets/285ece97-e833-455c-914d-dc04fef2c15d" />






# INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
       # INCLUDE YOUR RESULT HERE
