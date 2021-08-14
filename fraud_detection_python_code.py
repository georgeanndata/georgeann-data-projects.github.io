########################################
# Classification Prediction
########################################


#####################################################
# Import required packages
#####################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import RFECV
import seaborn as sea

#####################################################
# Import sample data
#####################################################


## Function to check if file does NOT exist>  If it does, print exits and data is loaded (load_data function)
def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print("OOPS!", filepath, "file does not exist!")
    else:
        print(filepath, "exists.")
        print("Beginning to load data....")
        load_data(filepath)

## Function to load data
def load_data(ms_file):
    global data_for_model 
    data_for_model = pd.read_csv(ms_file)
    print("Data has completed loading")

## data path variable in case it ever changes
path = "data/"

date_file_name  = path+"insurance_claims.csv"

check_file_exists(date_file_name)

#####################################################
# Data Cleaning & Preparation
#####################################################

# 1. Handling missing values

## Check if there are any na
na_cols = data_for_model.isna().any()
for index, value in na_cols.items():
    if value == True:
        print(f"{index} contains NAs")

## Column contains all NAs
data_for_model.columns[data_for_model.isna().all()]

## Drop all columns that have all NAs
data_for_model.dropna(axis = 1, inplace = True)

## Make sure there are none remaining.
data_for_model.isna().any().sum() # display number of missing data

# 2. Duplicate data

## Check for duplicates
data_for_model.duplicated().value_counts()

## no duplicates found

# 3. Incorrect & Irrelevant Data 

## noticed some columns have ? as a value
v = ['?', '-', '*']
for ext_cols in data_for_model:
    extended_chars = data_for_model[ext_cols].isin(v).any()
    #print(ext_cols, extended_chars)
    if extended_chars == True:
        print(ext_cols)

# 1.  'collision_type'
data_for_model['collision_type'].value_counts()

## all ? collision types were either Parked Car or Vehicle Theft

data_for_model.groupby(['collision_type','incident_type'])["incident_type"].count()

## update collision_type to None
data_for_model.loc[(data_for_model.collision_type == '?'),'collision_type']='None'
data_for_model['collision_type'].value_counts()

# 2.  'property_damage'
data_for_model['property_damage'].value_counts()

data_for_model.groupby(['property_damage','incident_type', 'collision_type'])["incident_type"].count()

## update property_damage to No
data_for_model.loc[(data_for_model.property_damage == '?'),'property_damage']='NO'
data_for_model['property_damage'].value_counts()

# 3.  'police_report_available'
data_for_model['police_report_available'].value_counts()

## update police_report_available to No
data_for_model.loc[(data_for_model.police_report_available == '?'),'police_report_available']='NO'
data_for_model['police_report_available'].value_counts()


## look for unique values
for uni_cols in data_for_model:
    u_length = len(pd.unique(data_for_model[uni_cols]))
    p_of_total = u_length / len(data_for_model)
    perc_format = "{:.0%}". format(p_of_total) 
    print(f"{uni_cols} has {perc_format}")
    
## too many unique values
data_for_model.drop(columns=['policy_number','incident_location'],inplace = True)


#check for any other columns that  won't add value
data_for_model.drop(columns=['policy_state','insured_zip','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year'],inplace = True)


## Check for dates types, non are as datetimes
data_for_model.dtypes

## Looking through the data, found policy_bind_date

## to evalute if you should use OHE vs toordinal, check out many unique values there are

## add date columns into a list
date_cols = ['policy_bind_date']

## loop through to print out count for each individual columns   
for cols in date_cols:
     no_unique = len(pd.unique(data_for_model[cols]))
     print(f"{cols} has {no_unique}")

## In this case, use toordinal for policy_bind_date

toordinal_col  = date_cols[0]
  
import datetime as dt

#convert column to datatime
data_for_model[toordinal_col]= pd.to_datetime(data_for_model[toordinal_col])

## convert toordinal
data_for_model[toordinal_col] = data_for_model[toordinal_col].map(dt.datetime.toordinal)

data_for_model['policy_bind_date']

# 4. Outliers

# Logisitc Regression is affected by outliers,

outlier_investigation = data_for_model.describe()

outlier_columns = data_for_model.select_dtypes(exclude="object").columns

## Verify and drop outliers
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
    

#####################################################
# Data Exploration Vizualizations
#####################################################
##age, police_report_available, tota_claim_amount (3d coloumn chart)

##'AGE'  

#color_blind_safe = sea.set_palette("colorblind", 4)
############################
## Create new dataframe to include just 'age' and fraud_reported
#del age_fraud_reported

from matplotlib.ticker import PercentFormatter


#   -AGE
#######################################
canvas_size = (16.7, 10.27)
fig, ax = plt.subplots(figsize=canvas_size)


# just select age and fraud_reported
age_fraud_reported = data_for_model[['age', 'fraud_reported']]
# split into two df's (just easier sometimes...)
fraud_df = age_fraud_reported.loc[age_fraud_reported["fraud_reported"] == 'Y']
non_fraud_df = age_fraud_reported.loc[age_fraud_reported["fraud_reported"] == 'N']

# plot each using "density=True" to give % rather than count
plt.hist(fraud_df["age"], density = True, alpha = 0.5, label = "Fraud",  lw=0, color = "#0072B2" )
plt.hist(non_fraud_df["age"], density = True, alpha = 0.5, label = "Non Fraud",  lw=0, color = "#CC79A7")
plt.title("Distribution by Age", fontdict = {'fontsize': 20, 'fontweight':'bold'}, pad = 15)
plt.legend(title = "Claims", labels = ["Fraud","Non-Fraud"],fancybox=True, framealpha=1, shadow=True)
plt.xticks(fontsize=14)
plt.xlabel("\nAge", fontsize=16)
plt.xticks(fontsize=14)
plt.ylabel("Percent\n", fontsize=16)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()


#######################################
 

## 'INSURED_SEX',
import matplotlib.ticker as mtick

#del gender_fraud_reported
############################
## Create new dataframe to include just 'insured_sex' and fraud_reported
#Sdel gender_fraud_reported
gender_fraud_reported = data_for_model[['insured_sex', 'fraud_reported']]
#data_for_model.groupby(['witnesses','fraud_reported'])["fraud_reported"].count()

gender_fraud_reported.value_counts()

##### GENDER
#######################################
# just select age and fraud_reported
reset_color = sea.color_palette(palette=None)
sea.set(color_codes=True)
canvas_size = (6.7, 6.27)
#canvas_size = (16.7, 10.27)

fig, ax = plt.subplots(figsize=canvas_size)

gender_fraud_reported = data_for_model[['insured_sex', 'fraud_reported']]
# split into two df's (just easier sometimes...)
fraud_df = gender_fraud_reported.loc[gender_fraud_reported["fraud_reported"] == 'Y']
non_fraud_df = gender_fraud_reported.loc[gender_fraud_reported["fraud_reported"] == 'N']
# plot each using "density=True" to give % rather than count
plt.hist(fraud_df["insured_sex"], density = True, alpha = 1.0, label = "Fraud",  lw=0, align='left')
plt.hist(non_fraud_df["insured_sex"], density = True, alpha = 1.0, label = "Non Fraud",  lw=0, align='right')
plt.title("Fraud Status \nby Gender", fontdict = {'fontsize': 20, 'fontweight':'bold'}, pad = 15)
plt.legend(title = "Claims", labels = ["Fraud", "Non-Fraud"],fancybox=True, framealpha=1, shadow=True, loc="upper center")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=10.0, decimals=None, symbol='%', is_latex=False))
plt.xticks(fontsize=14)
plt.xlabel("\nGender", fontsize=16)
plt.ylabel("Percent\n", fontsize=16)
plt.tight_layout()
plt.show()
############### ###############


##############################3
## 'WITNESSES
#del witness_fraud_reported

##### Andrew's code--WITNESSES
#######################################

witness_fraud_reported = data_for_model[['witnesses', 'fraud_reported']]



reset_color = sea.color_palette("Set3_r")
sea.set_palette(sea.color_palette(reset_color))

canvas_size = (6.7, 6.27)
fig, ax = plt.subplots(figsize=canvas_size)

witness_fraud_reported = data_for_model[['witnesses', 'fraud_reported']]
witness_fraud_reported['witnesses'] = witness_fraud_reported['witnesses'].apply(lambda x: 'No' if x <= 0 else 'Yes')

# split into two df's (just easier sometimes...)
fraud_df = witness_fraud_reported.loc[witness_fraud_reported["fraud_reported"] == 'Y']
non_fraud_df = witness_fraud_reported.loc[witness_fraud_reported["fraud_reported"] == 'N']
# plot each using "density=True" to give % rather than count
plt
plt.hist(fraud_df["witnesses"], density = True, alpha = 1, label = "Fraud",  lw=0, align='left')
plt.hist(non_fraud_df["witnesses"], density = True, alpha = 1, label = "Non Fraud",  lw=0, align='right')
plt.title("Fraud Status \nby Witnesses", fontdict = {'fontsize': 20, 'fontweight':'bold'}, pad = 15)
plt.legend(title = "Claims", labels = ["Fraud", "Non-Fraud"],fancybox=True, framealpha=1, shadow=True, loc="best")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=10, decimals=None, symbol='%', is_latex=False))
plt.xticks(fontsize=14)
plt.xlabel("\nWitnesses", fontsize=16)
plt.ylabel("Percent\n", fontsize=16)
plt.tight_layout()
plt.show()
##############################



##### POLICE REPPORT AVAILABILITY
#######################################

#del police_fraud_reported
############################
## Create new dataframe to include just police_report_available' and 'fraud_reported'
police_fraud_reported = data_for_model[['police_report_available','fraud_reported']]

reset_color = sea.color_palette("coolwarm")
sea.set_palette(sea.color_palette(reset_color))

canvas_size = (6.7, 6.27)
fig, ax = plt.subplots(figsize=canvas_size)


police_fraud_reported = data_for_model[['police_report_available', 'fraud_reported']]

# split into two df's (just easier sometimes...)
fraud_df = police_fraud_reported.loc[police_fraud_reported["fraud_reported"] == 'Y']
non_fraud_df = police_fraud_reported.loc[police_fraud_reported["fraud_reported"] == 'N']
# plot each using "density=True" to give % rather than count
plt.hist(fraud_df["police_report_available"], density = True, alpha = 1.0, label = "Fraud",  lw=0, align='left')
plt.hist(non_fraud_df["police_report_available"], density = True, alpha = 1.0, label = "Non Fraud",  lw=0, align='right')
plt.title("Fraud Status \nby Police Report Availability", fontdict = {'fontsize': 20, 'fontweight':'bold'}, pad = 15)
plt.legend(title = "Claims", labels = ["Fraud", "Non-Fraud"],fancybox=True, framealpha=1, shadow=True, loc="best")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=10.0, decimals=None, symbol='%', is_latex=False))
plt.xticks(fontsize=14)
plt.xlabel("\nPolice Report Availability", fontsize=16)
plt.ylabel("Percent\n", fontsize=16)
plt.tight_layout()
plt.show()
##############################




#####################################################
# Split Input Variables & Output Variables
#####################################################
    
# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)


X = data_for_model.drop(["fraud_reported"], axis = 1)
y = data_for_model["fraud_reported"]


    
#####################################################
# Split out Training & Test sets
#####################################################
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)##test_size percentage allocated to test_set, random_state = shuffling 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42, stratify = y)

# 5. Categorical Variables

# which columns are not numbers

categorical_vars = data_for_model.select_dtypes(include="object").columns

categorical_vars = categorical_vars[:-1]

# initialize one hot encoder
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first") # returns an array instead of object (sparse matrix)

## Fit 
## TRAIN data
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
## TEST data
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])


## what are each of these columns are
encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)

## put above in dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)

## put it back with other dataframe
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1) 

## drop original columns
X_train.drop(categorical_vars, axis = 1, inplace = True)

## put above in dataframe
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)

## put it back with other dataframe
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1) 

## drop original 
X_test.drop(categorical_vars, axis = 1, inplace = True)

# 6. Feature Scaling 

scale_standared = StandardScaler()

X_train = pd.DataFrame(scale_standared.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_standared.transform(X_test), columns = X_test.columns)

# 7. Feature Engineering & Selection

#####################################################
# Feature Selection
#####################################################


## Initializa logistic regression model
##Use Logistic Regression
clf = LogisticRegression(random_state = 42, max_iter = 1000) 

feature_selector = RFECV(clf)## default is 5 chunks but can specify other

## fit feature to our data to train our model 
fit = feature_selector.fit(X_train,y_train)

## to find optimal # of features and count
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features:  {optimal_feature_count}")


plt.plot(range(1, len(fit.grid_scores_) + 1), fit.grid_scores_, marker = "o", markersize=12, linewidth=5 )#, 
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFECV \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.grid_scores_), 4)})")
plt.tight_layout()
plt.show()

## see the best features (what they actually are)
print('Best features :', X_train.columns[fit.support_])
best_features = X_train.columns[fit.support_]
for features in best_features:
    print(features)
#print('Original features :', X_train.columns)

## updated X with only selected variables above (optimal)
## for logistic classification
X_train_LR = X_train.loc[: , feature_selector.get_support()]
X_test_LR = X_test.loc[: , feature_selector.get_support()]



#####################################################
# Model Training
#####################################################

#REFIT after feature importance
clf.fit(X_train_LR, y_train)


#####################################################
# Model Assessment
#####################################################

## Assess model accuracy

y_pred_class = clf.predict(X_test_LR)

## for probabilities
y_pred_prob = clf.predict_proba(X_test_LR)[:,1]##only with column index of 1

## Confusion matrix
#del conf_matrix
conf_matrix = confusion_matrix(y_pred_class, y_test, labels=['Y','N'])##flip for classfication models 


## Plot confusion matrix
import seaborn as sea
plt.figure(figsize=(16,12))
group_names = ['True Frauds','False Frauds','False Non-Frauds','True Non-Frauds']
group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sea.set(font_scale=3)
sea.heatmap(conf_matrix, annot=labels, fmt='', cmap="coolwarm", xticklabels=False, yticklabels=False, cbar=False)
plt.title("Confusion Matrix", pad = 15)
plt.ylabel("Predicted")
plt.xlabel("Actual Class")
plt.show()



 ## Accuracy (the number of correct classification out of all attempted classifications)

accuracy_score_r = accuracy_score(y_pred_class, y_test)

## Precision (of all observations that were predicted as positive, how many were actually positive)
precision_score_r = precision_score(y_test, y_pred_class, pos_label='Y')

## Recall (of all positive observations, how many did we predict as positive)
recall_score_r = recall_score(y_test, y_pred_class, pos_label='Y')

## F1 score (harmonic mean of precision and recall)

f1_score_r = f1_score(y_test, y_pred_class, pos_label='Y')

print("\nAccuracy Score: {0:.2%}".format(accuracy_score_r))
print("Precision Score: {0:.2%}".format(precision_score_r))
print("Recall Score: {0:.2%}".format(recall_score_r))
print("F1 Score: {0:.2%}".format(f1_score_r))

#####################################################
# Finding the optimal threshold
#####################################################
#del y_test_t
y_test_numbers = y_test.map({'Y': 1, 'N': 0}).astype(int)


thresholds = np.arange(0, 1, 0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    
    pred_class = (y_pred_prob >= threshold) * 1
    
    precision = precision_score(y_test_numbers, pred_class, zero_division = 0)
    precision_scores.append(precision)
    
    recall = recall_score(y_test_numbers, pred_class)
    recall_scores.append(recall)
    
    f1 = f1_score(y_test_numbers, pred_class)
    f1_scores.append(f1)


max_f1 = max(f1_scores)
max_f1_indx = f1_scores.index(max_f1)

## display optimal threshold score

plt.style.use('seaborn-poster')
plt.plot(thresholds, precision_scores, label = "Precision", linestyle = "--")
plt.plot(thresholds, recall_scores, label = "Recall", linestyle = "--")
plt.plot(thresholds, f1_scores, label = "F1", linewidth = 5)
plt.title(f"Finding the Optimal Threshold for Classification Model \n Max F1 {round(max_f1,2)} (Threshold = {round(thresholds[max_f1_indx],2)})")
plt.xlabel("Threshold")
plt.ylabel("Assessment Score")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()

optimal_threshold = 0.31#0.13##round(thresholds[max_f1_indx],2)
y_pred_class_opt_thresh = (y_pred_prob >= optimal_threshold) * 1

 ##Confusion Matrix with updated toptimal hreshold

conf_matrix_threshold = confusion_matrix(y_pred_class_opt_thresh, y_test_numbers, labels=[1,0])
## flipt tp and tn

print(classification_report(y_test, y_pred_class_opt_thresh))


## Plot confusion matrix
import seaborn as sea
plt.figure(figsize=(16,12))
group_names = ['True Frauds','False Frauds','False Non-Frauds','True Non-Frauds']
group_counts = ["{0:0.0f}".format(value) for value in conf_matrix_threshold.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in conf_matrix_threshold.flatten()/np.sum(conf_matrix_threshold)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sea.set(font_scale=3)
sea.heatmap(conf_matrix_threshold, annot=labels, fmt='', cmap= 'Spectral_r', xticklabels=False, yticklabels=False, cbar=False)
plt.title("Confusion Matrix with Optimal Threshold", pad = 15)
plt.ylabel("Predicted")
plt.xlabel("Actual Class")
plt.show()

## POST THRESHOLD Accuracy (the number of correct classification out of all attempted classifications)


accuracy_score_2 = accuracy_score(y_pred_class_opt_thresh, y_test_numbers)

## Precision (of all observations that were predicted as positive, how many were actually positive)
precision_score_2 = precision_score(y_test_numbers, y_pred_class_opt_thresh) #pos_label='Y')#, pos_label='Y')#33/(21+33)

## Recall (of all positive observations, how many did we predict as positive)
recall_score_2 = recall_score(y_test_numbers, y_pred_class_opt_thresh)#, pos_label='Y')#33/(42+33)

## F1 score (harmonic mean of precision and recall)

f1_score_2 = f1_score(y_test_numbers, y_pred_class_opt_thresh)#, pos_label='Y')

print("\nAccuracy Score: {0:.2%}".format(accuracy_score_2))
print("Precision Score: {0:.2%}".format(precision_score_2))
print("Recall Score: {0:.2%}".format(recall_score_2))
print("F1 Score: {0:.2%}".format(f1_score_2))




######################################################
## Use Random Forest instead
#####################################################

############################################ start from here
## Initializa logistic regression model

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 42, n_estimators = 2000)
rfc.fit(X_train, y_train)


# Feature Importance

feature_importance = pd.DataFrame(rfc.feature_importances_)
feature_names = pd.DataFrame(X_train.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace = True)

plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

## get column names
rf_features_columns = []
for cols in feature_importance_summary['input_variable']:
    #print(cols)
    rf_features_columns.append(cols)

for x in rf_features_columns:
    print(x)


# Permutation Importance 
from sklearn.inspection import permutation_importance


result = permutation_importance(rfc, X_test, y_test, n_repeats = 10, random_state = 42)
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names_perm = pd.DataFrame(X_train.columns)
permutation_importance_summary = pd.concat([feature_names_perm,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)


plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

#negative importance, in this case, means that removing a given feature from the model actually improves the performance.


rf_perm = permutation_importance_summary[permutation_importance_summary['permutation_importance'] > 0]
rf_perm.sort_values(by = "permutation_importance", ascending = False, inplace = True)


## get column names
rf_perm_columns = []
for cols in rf_perm['input_variable']:
    #print(cols)
    rf_perm_columns.append(cols)

for x in rf_perm_columns:
    print(x)

## updated X with only selected variables above (optimal)
## for random forest
X_train_RF = X_train.loc[: , rf_perm_columns]
X_test_RF = X_test.loc[: , rf_perm_columns]


#REFIT after feature importance
rfc.fit(X_train_RF, y_train)

#####################################################
# Model Assessment
#####################################################

## Assess model accuracy

y_pred_class_rf = rfc.predict(X_test_RF)

## for probabilities
y_pred_prob_rf = rfc.predict_proba(X_test_RF)[:,1]##only with column index of 1

### Confusion Matrix
conf_matrix_rf = confusion_matrix(y_pred_class_rf, y_test, labels=['Y','N'])

## Plot confusion matrix
import seaborn as sea
plt.figure(figsize=(16,12))
#sea.heatmap(conf_matrix/np.sum(conf_matrix), fmt = '.2%', annot = True, cmap="coolwarm")
#group_names = ['Non-Frauds Correctly','Non-Frauds Incorrectly','Frauds Incorrectly','Frauds Correctly']
group_names = ['True Frauds','False Frauds','False Non-Frauds','True Non-Frauds']
group_counts = ["{0:0.0f}".format(value) for value in conf_matrix_rf.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in conf_matrix_rf.flatten()/np.sum(conf_matrix_rf)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sea.set(font_scale=3)
sea.heatmap(conf_matrix_threshold, annot=labels, fmt='', cmap= 'Oranges_r', xticklabels=False, yticklabels=False, cbar=False)#icefire threshold
#sea.heatmap(conf_matrix_threshold, annot=labels, fmt='', cmap="Spectral", xticklabels=False, yticklabels=False, cbar=False)
plt.title("Confusion Matrix Random Forest \n Before Threshold", pad = 15)
plt.ylabel("Predicted")
plt.xlabel("Actual Class")
plt.show()


accuracy_score_rf_ = accuracy_score(y_pred_class_rf, y_test)

## Precision (of all observations that were predicted as positive, how many were actually positive)
precision_score_rf_ = precision_score(y_test, y_pred_class_rf, pos_label='Y')#,)#33/(21+33)

## Recall (of all positive observations, how many did we predict as positive)
recall_score_rf_ = recall_score(y_test, y_pred_class_rf, pos_label='Y')#, pos_label='Y')#33/(42+33)

## F1 score (harmonic mean of precision and recall)

f1_score_rf_ = f1_score(y_test, y_pred_class_rf,pos_label='Y')#, pos_label='Y')

print("\nAccuracy Score: {0:.2%}".format(accuracy_score_rf_))
print("Precision Score: {0:.2%}".format(precision_score_rf_))
print("Recall Score: {0:.2%}".format(recall_score_rf_))
print("F1 Score: {0:.2%}".format(f1_score_rf_))

#############################################################

####OPTIMAL THRESHOLD
## remap to 1, 0 as int
y_test_t = y_test.map({'Y': 1, 'N': 0}).astype(int)

thresholds_rf = np.arange(0, 1, 0.01)

precision_scores_rf = []
recall_scores_rf = []
f1_scores_rf = []

for threshold in thresholds_rf:
    
    pred_class_rf = (y_pred_prob_rf >= threshold) * 1
    
    precision_rf = precision_score(y_test_t, pred_class_rf, zero_division = 0)
    precision_scores_rf.append(precision_rf)
    
    recall_rf = recall_score(y_test_t, pred_class_rf)
    recall_scores_rf.append(recall_rf)
    
    f1_rf = f1_score(y_test_t, pred_class_rf)
    f1_scores_rf.append(f1_rf)


max_f1_rf = max(f1_scores_rf)
max_f1_indx_rf = f1_scores_rf.index(max_f1_rf)

optimal_threshold_rf = 0.31#0.13#0.13##round(thresholds[max_f1_indx],2)
y_pred_class_opt_thresh_rf = (y_pred_prob_rf >= optimal_threshold_rf) * 1

## display optimal threshold score

plt.style.use('seaborn-poster')
plt.plot(thresholds_rf, precision_scores_rf, label = "Precision", linestyle = "--")
plt.plot(thresholds_rf, recall_scores_rf, label = "Recall", linestyle = "--")
plt.plot(thresholds_rf, f1_scores_rf, label = "F1", linewidth = 5)
plt.title(f"Finding the Optimal Threshold for Random Forest Model \n Max F1 {round(max_f1_rf,2)} (Threshold = {round(thresholds_rf[max_f1_indx],2)})")
plt.xlabel("Threshold")
plt.ylabel("Assessment Score")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()


 ##Confusion Matrix with updated toptimal hreshold

conf_matrix_threshold_rf = confusion_matrix(y_pred_class_opt_thresh_rf, y_test_t, labels=[1,0])
## flipt tp and tn

# print classification report
print(classification_report(y_test, y_pred_class_opt_thresh_rf))


## Plot confusion matrix
import seaborn as sea
plt.figure(figsize=(16,12))
group_names = ['True Frauds','False Frauds','False Non-Frauds','True Non-Frauds']
group_counts = ["{0:0.0f}".format(value) for value in conf_matrix_threshold_rf.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in conf_matrix_threshold_rf.flatten()/np.sum(conf_matrix_threshold_rf)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sea.set(font_scale=3)
sea.heatmap(conf_matrix_threshold_rf, annot=labels, fmt='', cmap= 'icefire', xticklabels=False, yticklabels=False, cbar=False)#icefire threshold
plt.title("Confusion Matrix for Random Forest \n with Optimal Threshold", pad = 15)
plt.ylabel("Predicted")
plt.xlabel("Actual Class")
plt.show()

## POST THRESHOLD Accuracy (the number of correct classification out of all attempted classifications)


accuracy_score_rf = accuracy_score(y_pred_class_opt_thresh_rf, y_test_t)

## Precision (of all observations that were predicted as positive, how many were actually positive)
precision_score_rf = precision_score(y_test_t, y_pred_class_opt_thresh_rf)#, pos_label='Y')#33/(21+33)

## Recall (of all positive observations, how many did we predict as positive)
recall_score_rf = recall_score(y_test_t, y_pred_class_opt_thresh_rf)#, pos_label='Y')#33/(42+33)

## F1 score (harmonic mean of precision and recall)

f1_score_rf = f1_score(y_test_t, y_pred_class_opt_thresh_rf)#, pos_label='Y')

print("\nAccuracy Score: {0:.2%}".format(accuracy_score_rf))
print("Precision Score: {0:.2%}".format(precision_score_rf))
print("Recall Score: {0:.2%}".format(recall_score_rf))
print("F1 Score: {0:.2%}".format(f1_score_rf))

#AUC: when data is imbalanced. ROC can be misleading
#################################

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test_numbers, y_pred_prob)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_t, y_pred_prob_rf)


##plot ROC curves
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr_lr, tpr_lr, label= "Logistic Regression", linewidth=4)
plt.plot(fpr_rf, tpr_rf, label= "Random Forest", linewidth=4)
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Receiver Operating Characteristic')
plt.show()

# auc scores
auc_log_reg = roc_auc_score(y_test_numbers, y_pred_class_opt_thresh)#clf.predict_proba(X_test_LR)[:,1])#y_pred_class_opt_thresh
auc_random_forest = roc_auc_score(y_test_t, y_pred_class_opt_thresh_rf) #rfc.predict_proba(X_test_RF)[:,1])

line = "----" * 10
print(f"\nAUC Scores\n{line}")
print("Logistic Regression: {0:.2%}".format(auc_log_reg))
print("Random Forest: {0:.2%}".format(auc_random_forest))


#######################3

#Precision-Recall Curves (PR) and AUC: when data is imbalanced. ROC can be misleading
#################################

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test_numbers,  y_pred_class_opt_thresh)#,  pos_label='Y'
precision_rf, recall_rf, _ = precision_recall_curve(y_test_t,  y_pred_class_opt_thresh_rf)#,  pos_label='Y'
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Logistic Regression', linewidth=4)
plt.plot(recall_rf, precision_rf, linestyle='--', label='Random Forest', linewidth=4)
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
# show the legend
plt.legend()
# show the plot
plt.show()


# auc scores
from sklearn.metrics import auc

auc_log_reg_pr = auc(recall, precision)#, y_pred_class_opt_thresh)
auc_random_forest_pr = auc(recall_rf, precision_rf)#, y_pred_class_opt_thresh)

line = "----" * 10
print(f"\nPrecision Recall AUC Scores\n{line}")
print("Logistic Regression: {0:.2%}".format(auc_log_reg_pr))
print("Random Forest: {0:.2%}".format(auc_random_forest_pr))





