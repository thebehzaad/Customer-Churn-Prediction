"""*****************************************************************************************

            Customer Churn Prediction Using XGBoost (imbalanced classification)


Methods to deal with the imbalance problem:

1- Giving more weights to the samples of the smaller class during the training process
2- Upsampling the smaller class with replacement 

*****************************************************************************************"""
#%% Importing Libraries

# General
import numpy as np
import pandas as pd
import random
# Visualization (EDA)
import matplotlib.pyplot as plt 
import seaborn as sns
# Data Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# Model Building
from sklearn.utils import resample #Upsampling for the Positive Class
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve

#%% Reading the dataset

dataset=pd.read_csv("./telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
dataset.head()
dataset.info()
dataset.describe()

print("Num of Rows: {}".format(dataset.shape[0]))
print("Num of Columns: {}".format(dataset.shape[1]))
print("Features: {}".format(dataset.columns.tolist()))
print("Missing Values: {}".format(dataset.isnull().sum().sum()))
print("Unique Values:\n{}".format(dataset.nunique()))

#%% Data Manipulation (Missing Values, Duplicates, and Data Cleaning)

# Missing Values: Method1 (replacing missing values with zeros)------------------------
print('Number of Empty Cells: {}'.format(dataset.applymap(lambda x:x==' ').sum().sum()))
dataset.replace(' ', 0, inplace=True)

# Missing Values: Method2 (dropping rows with missing values)
"""
dataset.replace(' ', np.nan,inplace=True)
print("Missing Values: {}".format(dataset.isnull().sum().sum()))
dataset.dropna(inplace=True)
dataset.reset_index(drop=True,inplace=True)
"""

# Data Cleaning----------------------------------------------------------------
dataset.replace({'No internet service' : 'No'}, inplace=True)
dataset['TotalCharges']=dataset['TotalCharges'].astype(float)
dataset["SeniorCitizen"] = dataset["SeniorCitizen"].replace({1:"Yes",0:"No"})


#%% Exploratory Data Analysis (Data Visualization Using Seaborn)

# Identifying Catagorical  and Numerical Columns--------------------------------
Id_col     = ['customerID']
target_col = ['Churn']
cat_cols   = dataset.columns[dataset.nunique() < 6].tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
bin_cols   = dataset.columns[dataset.nunique() == 2].tolist()    # Categorical Columns with two values
multi_cols = [i for i in cat_cols if i not in bin_cols]          # Categorical Columns with more than two values
num_cols   = [x for x in dataset.columns if x not in cat_cols + target_col + Id_col] # numerical columns

# Imbalanced classes
plt.figure()
dataset['Churn'].value_counts().plot(kind='pie')

# Probability of feature given class
"""
plt.figure()
sns.catplot(x="Churn", kind="count", data=dataset, palette="deep", height=5, aspect=0.6, orient='h')

for feature in cat_cols:
    plt.figure()
    color_pallettes=["muted", "pastel", "bright", "deep", "colorblind", "dark"]
    color=random.choice(color_pallettes)
    sns.catplot(x="Churn", kind='count', hue=feature, palette=color, data=dataset)

for feature in num_cols:
    plt.figure()
    sns.catplot(x="Churn", y=feature, kind='violin', palette="bright", data=dataset)
"""


"""
plt.figure()
sns.catplot(x="Churn", y="tenure", kind='violin', hue='gender', palette="bright", split=True, data=dataset)

plt.figure()
sns.catplot(x="Contract", y="MonthlyCharges", kind='box', palette="bright", data=dataset)

plt.figure()
sns.catplot(x="Churn", y="MonthlyCharges", kind="box", hue="Contract", palette="bright", data=dataset, height=4.2, aspect=1.4)
"""


# Distribution Plots (Histograms, and Kernel Density Estimation)---------------
"""
plt.figure() # Histogram and Kernel Density Estimation
sns.distplot(dataset['TotalCharges'][dataset['Churn']=='Yes'])
sns.distplot(dataset['TotalCharges'][dataset['Churn']=='No'])

plt.figure() # Histogram Only
sns.distplot(dataset['TotalCharges'][dataset['Churn']=='Yes'], kde=False, label="Churn: Yes")
sns.distplot(dataset['TotalCharges'][dataset['Churn']=='No'], kde=False, label="Churn: No")

plt.figure() # Kernel Density Estimation Only
sns.distplot(dataset['tenure'][dataset['Churn']=='No'], hist=False, label="Churn: No")
sns.distplot(dataset['tenure'][dataset['Churn']=='Yes'], hist=False , label="Churn: Yes")

plt.figure() # Kernel Density Estimation Only
sns.kdeplot(dataset['tenure'][dataset['Churn']=='Yes'], shade=True, label="Churn: Yes")
sns.kdeplot(dataset['tenure'][dataset['Churn']=='No'], shade=True, label="Churn: No")
"""

# Scatter Plots----------------------------------------------------------------
"""
plt.figure()
sns.scatterplot(x="tenure", y="MonthlyCharges", hue='Churn', data=dataset)

plt.figure()
sns.scatterplot(x="tenure", y="TotalCharges", hue='Churn', size='gender', data=dataset)

plt.figure()
sns.scatterplot(x="tenure", y="TotalCharges", hue='Churn', style='gender', data=dataset)
"""
# Pair Grid--------------------------------------------------------------------
"""
g=sns.PairGrid(dataset[num_cols+target_col], hue='Churn', palette='muted')
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()
"""

# Joint Plot-------------------------------------------------------------------
"""
sns.jointplot('tenure','TotalCharges',kind='kde', data=dataset)
sns.jointplot('tenure','TotalCharges',kind='hex', data=dataset)
"""

#%% Data Preprocessing

dataset_org=dataset.copy()

# Label Encoding for Binary columns
le = LabelEncoder()
for i in bin_cols:
    dataset[i] = le.fit_transform(dataset[i])
for i in multi_cols:
    dataset[i] = le.fit_transform(dataset[i])


#%% Model Building and Evaluation using scale postive weights

# Train-Test Splits
train, test = train_test_split(dataset, test_size = .25 ,random_state = 111)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

cols    = [i for i in dataset.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]

# model with cross validation--------------------------------------------------
f1_scores=[]
auc_scores=[]
weights=[1, 2, 3, 4, 5, 6, 7, 8]
for weight in weights:
    model=XGBClassifier(scale_pos_weight=weight)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    f1scores = cross_val_score(model, train_X, train_Y, scoring='f1', cv=cv, n_jobs=-1)
    aucscores = cross_val_score(model, train_X, train_Y, scoring='roc_auc', cv=cv, n_jobs=-1)
    f1_scores.append(f1scores.mean())
    auc_scores.append(aucscores.mean())

f1_bestweight=weights[np.argmax(f1_scores)]
auc_bestweight=weights[np.argmax(auc_scores)]
print("f1_bestweight=%f" %f1_bestweight)
print("auc_bestweight=%f" %auc_bestweight)

model=XGBClassifier(scale_pos_weight=f1_bestweight)
model.fit(train_X,train_Y)
predictions = model.predict(test_X)
probabilities = model.predict_proba(test_X)

# feature importance-----------------------------------------------------------
feature_imp = list(model.feature_importances_)
cols=list(train_X.columns)
feature_sumry  = pd.DataFrame({'features':cols, 'importance':feature_imp})
feature_sumry = feature_sumry.sort_values(by = "importance",ascending = False)

# performance metrics----------------------------------------------------------
print (model)
print ("Classification report: ",classification_report(test_Y,predictions))
print ("Accuracy Score: ", accuracy_score(test_Y,predictions)) 
print ("Area under curve: ",roc_auc_score(test_Y,probabilities[:,1]))
print ("F1 Score: ", f1_score(test_Y,predictions))
print ("Precision Score: ", precision_score(test_Y,predictions))
print ("Recall Score: ", recall_score(test_Y,predictions))

conf_matrix = confusion_matrix(test_Y,predictions) #confusion matrix
fpr,tpr,thresholds = roc_curve(test_Y,probabilities[:,1]) #roc curve

# performance plots
plt.figure()
chart=sns.catplot(x="features", y='importance', kind='bar', palette="bright", data=feature_sumry)
chart.set_xticklabels(rotation=90)
plt.figure()
sns.lineplot(x=fpr, y=tpr)

#%% Model Building and Evaluation using Upsampling

# Train-Test Splits
train, test = train_test_split(dataset, test_size =.25 ,random_state = 111)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

train_n=train.iloc[np.where(train['Churn']==0)[0],:]
train_p=train.iloc[np.where(train['Churn']==1)[0],:]
train_p_upsampled=resample(train_p,
                           replace=True,                    # sample with replacement
                           n_samples=train_n.shape[0],      # to match majority class
                           random_state=123)                # reproducible results

train=pd.concat([train_n,train_p_upsampled])
train=train.sample(frac=1) # shuffling the dataset
train.reset_index(drop=True, inplace=True)

cols    = [i for i in dataset.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]

model=XGBClassifier()
model.fit(train_X,train_Y)
predictions = model.predict(test_X)
probabilities = model.predict_proba(test_X)

# feature importance-----------------------------------------------------------
feature_imp = list(model.feature_importances_)
cols=list(train_X.columns)
feature_sumry  = pd.DataFrame({'features':cols, 'importance':feature_imp})
feature_sumry = feature_sumry.sort_values(by = "importance",ascending = False)

# performance metrics----------------------------------------------------------
print (model)
print ("Classification report: ",classification_report(test_Y,predictions))
print ("Accuracy Score: ", accuracy_score(test_Y,predictions)) 
print ("Area under curve: ",roc_auc_score(test_Y,probabilities[:,1]))
print ("F1 Score: ", f1_score(test_Y,predictions))
print ("Precision Score: ", precision_score(test_Y,predictions))
print ("Recall Score: ", recall_score(test_Y,predictions))

conf_matrix = confusion_matrix(test_Y,predictions) #confusion matrix
fpr,tpr,thresholds = roc_curve(test_Y,probabilities[:,1]) #roc curve

# performance plots
plt.figure()
chart=sns.catplot(x="features", y='importance', kind='bar', palette="bright", data=feature_sumry)
chart.set_xticklabels(rotation=90)
plt.figure()
sns.lineplot(x=fpr, y=tpr)


