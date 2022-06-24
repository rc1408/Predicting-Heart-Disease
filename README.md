# Predicting-Heart-Disease
Predicting Heart Disease The Goal is to predict the binary class heart_disease_present, which represents whether or not a patient has heart disease

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('labels.csv')
df = pd.DataFrame(df)
df1 = pd.read_csv('values.csv')
df1 = pd.DataFrame(df1)
df1.drop(['patient_id'],axis=1,inplace = True)

data = pd.concat([df,df1],axis=1)
data.shape

data.head()

data['thal'].value_counts()

thal_ranked = {
    'normal':0,
    'reversible_defect':1,
    'fixed_defect':2
}

data['thal_rank']= data['thal'].map(thal_ranked)

data.drop(['thal'],axis=1,inplace=True)

data.isnull().sum()

data.drop(['thal'],axis=1,inplace=True)

# Checking Correlation Between Independent Features

corr = data.corr()
plt.figure(figsize=(15,5))
sns.heatmap(corr,annot=True,cmap='BuPu')

threshold = 0.7
def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
    
 correlation(data,threshold)
 
# Droping Constant Variable

from sklearn.feature_selection import VarianceThreshold
VT = VarianceThreshold(threshold=0.8)
X = data.drop(['heart_disease_present','patient_id'],axis=1)
VT.fit_transform(X)
VT

VT.get_support()

X.columns[VT.get_support()]

#This is my Constant Column
constant_column = [column for column in X.columns
                  if column not in X.columns[VT.get_support()]]
print(constant_column)


# Checking Correlation Between Dependent and Independent Feature

X = data.drop(['patient_id','heart_disease_present'],axis=1)
y = data.heart_disease_present

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=14)

X_train = pd.DataFrame(X_train,columns=X.columns)

# Model Selection and Building, Model Evaluation

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
# NOTE BorutaPy accepts numpy arrays only, if X_train and y_train #are pandas dataframes, then add .values attribute X_train.values in #that case
X_train = X_train
y_train = y_train
# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
# find all relevant features - 5 features should be selected
feat_selector.fit(X_train, y_train)
# check selected features - first 5 features are selected
feat_selector.support_
# check ranking of features
feat_selector.ranking_
# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X_train)
#To get the new X_train now with selected features
# X_train.columns[feat_selector.support_]
X_train.columns[(X_train.get_support())]




