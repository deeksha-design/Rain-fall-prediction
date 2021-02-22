#!/usr/bin/env python
# coding: utf-8

# BASED ON THE GIVEN DATASET:WE HAVE TO PREDICT WHEATHER IT WILL RAIN TOMORROW OR NOT IN THE AUSTRALIA AND ITS CITIES

# In[1]:


#first import all neccessary libraries


# In[60]:


import pandas as pd


# In[61]:


import numpy as np


# In[62]:


import seaborn as sns


# In[63]:


import matplotlib.pyplot as plt


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


data=pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/weatherAUS.csv")


# In[66]:


data.head()


# In[67]:


data.shape


# In[68]:


data.dtypes


# In[69]:


data.info()


# In[70]:


data.describe()


# In[71]:


data.columns


# In[72]:


data['RainTomorrow'].value_counts()


# In[73]:


data['RainToday'].value_counts()


# In[74]:


data['Evaporation'].value_counts()


# In[75]:


data['Sunshine'].value_counts()


# In[76]:


data.head(1)


# In[77]:


data['MinTemp'].value_counts()


# In[78]:


#VISUALIZATION:


# In[79]:


import missingno as msno
msno.matrix(data)


# In[80]:


data.isnull().sum()


# In[81]:


data.head(2)


# In[82]:


data['RainToday'].replace({'No':0,'Yes':1},inplace=True)


# In[83]:


data['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)


# In[84]:


data.head(2)


# AS we checked the count of rain tomorrow,it is completely imbalanced dataset,lets make it balance

# In[85]:


import matplotlib.pyplot as plt
fig=plt.figure(figsize=(8,5))
data.RainTomorrow.value_counts(normalize=True).plot(kind='bar',color=['Red','Black'],alpha=0.9,rot=0)
#full_data.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')
plt.show()


# In[86]:


#lets make it balanced dataset


# In[87]:


from sklearn.utils import resample


# In[88]:


no = data[data.RainTomorrow == 0]
yes = data[data.RainTomorrow == 1]
yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
df1 = pd.concat([no, yes_oversampled])
fig = plt.figure(figsize = (8,5))
df1.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['orange','Black'], alpha = 0.9, rot=0)
plt.title('RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)')
plt.show()


# In[89]:


##now the dataset is completely balanced,lets impute the missing values:


# In[90]:


data.isnull().sum()


# In[91]:


data.dtypes


# In[94]:


data.head(2)


# In[95]:


#FIRST IMPUTE THE CATEGORICAL MISSING VALUES


# In[96]:


df1.head(2)


# In[97]:


df1['WindGustDir']=df1['WindGustDir'].fillna(df1['WindGustDir'].mode()[0])


# In[98]:


df1.isnull().sum()


# In[99]:


df1['WindGustDir'].isnull().sum()


# In[100]:


df1['WindDir9am']=df1['WindDir9am'].fillna(df1['WindDir9am'].mode()[0])


# In[101]:


df1['WindDir9am'].isnull().sum()


# In[102]:


df1.head(2)


# In[103]:


df1['Cloud3pm'].value_counts()


# In[104]:


df1['WindDir3pm'] = df1['WindDir3pm'].fillna(df1['WindDir3pm'].mode()[0])


# In[105]:


df1['WindDir3pm'].isnull().sum()


# In[106]:


df1.dtypes


# In[107]:


data.head(2)


# In[108]:


data['Sunshine'].value_counts()


# In[109]:


df1.select_dtypes(include=['object']).columns


# In[110]:


df1.select_dtypes(include=['float']).columns


# In[111]:


#Impute categorical var with Mode
df1['Date'] = df1['Date'].fillna(df1['Date'].mode()[0])
df1['Location'] = df1['Location'].fillna(df1['Location'].mode()[0])
df1['WindGustDir'] = df1['WindGustDir'].fillna(df1['WindGustDir'].mode()[0])
df1['WindDir9am'] = df1['WindDir9am'].fillna(df1['WindDir9am'].mode()[0])
df1['WindDir3pm'] = df1['WindDir3pm'].fillna(df1['WindDir3pm'].mode()[0])


# In[112]:


mean=df1.Sunshine.mean()
mean


# In[113]:


mean=df1.Evaporation.mean()
mean


# In[114]:


mean=df1.MinTemp.mean()
mean


# In[115]:


mean=df1.MaxTemp.mean()
mean


# In[116]:


mean=df1.Rainfall.mean()
mean


# In[117]:


mean=df1.Humidity3pm.mean()
mean


# In[118]:


mean=df1.WindGustSpeed.mean()
mean


# In[119]:


mean=df1.Pressure3pm.mean()
mean


# In[120]:


mean=df1.Cloud9am.mean()
mean


# In[121]:


mean=df1.Cloud3pm.mean()
mean


# In[122]:


mean=df1.Temp9am.mean()
mean


# In[123]:


mean=df1.RISK_MM.mean()
mean


# In[124]:


from sklearn.preprocessing import LabelEncoder


# In[125]:


lencoders={}


# In[126]:


for col in df1.select_dtypes(include=['object']).columns:
    lencoders[col]=LabelEncoder()
    df1[col]=lencoders[col].fit_transform(df1[col])


# In[127]:


df1.head()


# In[128]:


##imputation of missing values:


# In[129]:


import warnings
warnings.filterwarnings("ignore")


# In[130]:


df1.MinTemp=df1.MinTemp.fillna(mean)
df1.MaxTemp=df1.MaxTemp.fillna(mean)
df1.Rainfall=df1.Rainfall.fillna(mean)
df1.Evaporation=df1.Evaporation.fillna(mean)
df1.WindGustSpeed=df1.WindGustSpeed.fillna(mean)
df1.Humidity9am=df1.Humidity9am.fillna(mean)
df1.Humidity3pm=df1.Humidity3pm.fillna(mean)
df1.Pressure9am=df1.Pressure9am.fillna(mean)
df1.Pressure3pm=df1.Pressure3pm.fillna(mean)
df1.Cloud9am=df1.Cloud9am.fillna(mean)
df1.Cloud3pm=df1.Cloud3pm.fillna(mean)
df1.Temp9am=df1.Temp9am.fillna(mean)
df1.Temp3pm=df1.Temp3pm.fillna(mean)
df1.RISK_MM=df1.RISK_MM.fillna(mean)
df1.Sunshine=data.Sunshine.fillna(mean)


# In[131]:


df1.isnull().sum()


# In[132]:


df1.WindSpeed9am=df1.WindSpeed9am.fillna(mean)    


# In[133]:


df1.isnull().sum()


# In[134]:


df1.WindSpeed3pm=df1.WindSpeed3pm.fillna(mean)    


# In[135]:


df1.RainToday=df1.RainToday.fillna(mean)    


# In[136]:


df1.isnull().sum()


# In[137]:


from sklearn.preprocessing import LabelEncoder


# In[138]:


leencoders={}


# In[139]:


for col in df1.select_dtypes(include=['object']).columns:
        leencoders[col] = LabelEncoder()
        df1[col] = leencoders[col].fit_transform(df1[col])


# In[140]:


df1.head(3)


# In[141]:


df2 = df1.copy(deep=True) 


# In[142]:


df2.head(2)


# In[ ]:


TO FIND THE OUTLIERS IN THE DATA BY USING Z SCORE :method


# from scipy import stats
# import numpy as np
# z = np.abs(stats.zscore(df2))
# print(z)

# In[144]:


threshold = 3
print(np.where(z > 3))


# In[146]:


print(z[220624][1]) ##it is an outlier


# In[ ]:


FINDING THE OUTLIERS BY USING THE INTERQUARTILE RANGE


# In[148]:


Q1 = df2.quantile(0.25)
Q3 = df2.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[149]:


print(df2 < (Q1 - 1.5 * IQR)) |(df2 > (Q3 + 1.5 * IQR))


# In[150]:


##df2 = df2[(z < 3).all(axis=1)]


# In[151]:


#df2.shape


# In[ ]:


WITH THE BELOW LINE OF CODE :WE CAN FOUND THE OUTLIERS COUNT:


# In[161]:


df2out = df2[~((df2 < (Q1 - 1.5 * IQR)) |(df2 > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[162]:


df2out.shape


# In[163]:


df2out.head()


# In[ ]:


#TO FIND THE CORRELATION :


# In[164]:


df2_out.corr().head()


# In[165]:


##feature selection 


# In[169]:


from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(df2out)
modified_data = pd.DataFrame(r_scaler.transform(df2out), index=df2out.index, columns=df2out.columns)


# In[170]:


modified_data.head()


# In[171]:


# Feature Importance using Filter Method (Chi-Square)
from sklearn.feature_selection import SelectKBest, chi2
X = modified_data.loc[:,modified_data.columns!='RainTomorrow']
y = modified_data[['RainTomorrow']]
selector = SelectKBest(chi2, k=10)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])


# In[172]:


features = df2[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']]
target = df2['RainTomorrow']


# In[173]:


from sklearn.model_selection import train_test_split


# In[174]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=12345)


# In[175]:


##from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[176]:


##def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[177]:


from sklearn.linear_model import LogisticRegression


# In[178]:


logistic_regression= LogisticRegression() 


# In[179]:


logistic_regression.fit(X_train,y_train)


# In[180]:


y_pred=logistic_regression.predict(X_test)


# In[181]:


from sklearn import metrics


# In[182]:


cnf_matrix = metrics.confusion_matrix(y_test,y_pred)


# In[183]:


print(cnf_matrix)


# In[184]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[185]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[188]:


y_pred = rf.predict(X_test)


# In[189]:


from sklearn.metrics import accuracy_score,classification_report


# In[190]:


cnf_matrix = metrics.confusion_matrix(y_test,y_pred)


# In[191]:


print(cnf_matrix)


# In[192]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[193]:


from sklearn.metrics import classification_report


# In[194]:


c_report=metrics.classification_report(y_test,y_pred)


# In[195]:


print(c_report)


# In[196]:


from sklearn.ensemble import GradientBoostingClassifier


# In[197]:


model=GradientBoostingClassifier()


# In[198]:


model.fit(X_train, y_train)


# In[199]:


y_pred=model.predict(X_test)


# In[200]:


from sklearn.metrics import accuracy_score,classification_report


# In[201]:


cnf_matrix = metrics.confusion_matrix(y_test,y_pred)


# In[202]:


print(cnf_matrix)


# In[203]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[204]:


from sklearn.metrics import classification_report


# In[205]:


c_report=metrics.classification_report(y_test,y_pred)


# In[206]:


print(c_report)


# In[ ]:





# In[ ]:




