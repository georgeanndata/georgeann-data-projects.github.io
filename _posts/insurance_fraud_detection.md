---
layout: post
title: Auto Insurance Fraud Detection
image: "/posts/.jpg"
tags: [fraud detection, Logistic Regression, Random Forest]
---

# Table of Contents
* [Business Problem](#business-problem)
* [Potential Solution](#potential-solution)
* [Results](#results)
* [Conclusion](#conclusion)
* [Resources](#resources)
* [Data](#data)
* [Data Cleaning and Exploration](#data-cleaning-exploration)
  * [Missing Data](#missing-data)
  * [Duplicate Data](#duplicate-data)
  * [Incorrect & Irrelevant Data](#incorrect-data)
  *   [Unique Values](#unique-values)
  *   [Date Times](#date-times)
  *   [Outliers](#outliers)
* [Data Exploration and Visualizations](#data-exploration-visualizations)
  * [Age](#age)
  * [Gender](#gender)
  * [Witnesses](#witnesses)
  * [Police Report Available](#police)
* [Split Input Variables & Oupput Variables](#split)
* [Split Out Training & Test Sets](#split_tt)
* [Categorical Variables](#categorical-variables)
* [Feature Scaling](#feature-scaling)
* [Logistic Regression](#log-reg)
  * [Feature Selection](#feature-selection)
  * [Model Training](#model-training)
  * [Model Assessment](#model-assessment)
    * [Confusion Matrix](#confusion-matrix)
    * [Accuracy, Precision, Recall and F1 Scores](#a_p_r_f_1)
  * [Handling Data Imbalance](#imbalance)
    * [Optimal Threshold](#opt_thresh) 
    * [Confusion Matrix Post Threshold](confusion-matrix_t)
    * [Accuracy, Precision, Recall and F1 Scores Post Threshold](#a_p_r_f_2)
  * [Model Assessment of 60:40 Split](#model-assessment_6040)
    * [Updated Feature Selection](#opt_fs_6040)
    * [Optimal Threshold](#opt_thresh_6040) 
    * [Confusion Matrix Post Threshold](#confusion-matrix_6040)
    * [Accuracy, Precision, Recall and F1 Scores Post Threshold](#a_p_r_f_6040)
 * [Random Forest Model](#random)
   * [Feature Selection](#feature-selection_rf)
    * [Built-in Random Forest Importance](#built-in) 
    * [Permutation Importance](#permutation)
   * [Model Training](#model-training_rf)
   * [Model Assessment](#model-assessment_rf)
    * [Confusion Matrix Post Pre-Threshold](#confusion-matrix_6040_rf)
    * [Optimal Threshold](#opt_thresh_6040_rf) 
    * [Confusion Matrix Post Pre-Threshold](confusion-matrix_6040_rf_2)
    * [Accuracy, Precision, Recall and F1 Scores Pre & Post Threshold](#a_p_r_f_6040_rf) 
 * [ROC-AUC](#roc-auc)
    * [Receiver Operator Characteristic](#graph)
    * [AUC Scores](#auc)
 * [Precision Recall Curves](#pr_curves)
    * [PR Curve Graph](#pr_graph)
    * [PR AUC Scores](#pr_auc)
 


## Overview of Fraud on Auto Accident Insurance Claims Project

 <a id="business-problem"></a> 
 ### Business Problem:
The Covid-19 pandemic has caused an increase in the number of drivers on the road, resulting in a rise in auto accident claims. Insurance companies, struggling with vast numbers, are in desperate need of finding a way to manage them. They need a way to quickly determine a preliminary fraud status so their employees can focus their immediate attention on claims that are actually fraudulent and process the non-fraud ones later. 
<a id="potential-solution"></a> 
### Potential Solution:
Machine learning is perfect for this kind of problem.  With the use of a high performing algorithm, claims can be quickly perdicted as fraud or non-fraud, allowing employees the ability to get ahead of the backlog.

This type of data science problem is considered a classification problem because it is trying to answer the question of 'how would you **classify** this claim? Fraud or non-fraud? There are many classification algorithms that can be used, but I've limited my research to the Logistic Regression model and the Random Forest for Classification model. I will test both and provide a recommendation based on which offered the best performance and accuracy.
<a id="results"></a> 
### Results

Logistic Regression and Random Forest for Classificaiton models were assessed with the insurance claims data. Feature Selection as well as threshold optimization were utilitized on both models to obtain the best performance metrics.   

The performance metrics are as follows:

![alt text](/img/posts/fraud_prod/ss/perform_m_graph.png)

**Accuracy Score** 

The accuracy score is the percentage of all predictions that were correct, whether they were correctly predicted as fraud or correctly predicted as non-fraud, divided by the total number of predictions.
  
Logistic Regression: **79.00%** <br>
Random Forest:  72.41%
  
The Logistic Regression model returned the best accuracy score.

**Precision**  
 
The precision score of a model is a score given for how well the model did on predicting clasess correctly. 
 
Logistic Regression:  **54.76%**<br>
Random Forest: 43.16%

The Logistic Regression model returned the best precision score.

**Recall**  

The recall score is how well the model did in labeling fraud claims as fraud.

Logistic Regression: **61.33%**<br>
Random Forest: 54.67%

The Logistic Regression model returned the best recall score.


**F1 score**

The F1 score is the blending of precision and recall, an harmonic mean.

Logistic Regression:  **57.86%**<br>
Random Forest:  48.24%

The Logistic Regression model returned the best F1 score.

**ROC-AUC**

ROC-AUC is the measure across all possible thresholds.

Logistic Regression: **72.88%**<br>
Random Forest:  66.27%

The Logistic Regression model returned the best ROC-AUC score.

**Precision-Recall AUC**

Precision-Recall AUC is the measure across all possible thresholds.

Logistic Regression: **62.69%**<br>
Random Forest:  54.24%

The Logistic Regression model returned the best Precision-Recall score.

 <a id="conclusion"></a> 
 ### Conclusion

After assessing both models, I determined that the Logistic Regression model was the best performing model for determing fraud status.  I based this decision on the performance metrics, ROC-AUC and Precision-Recall AUC scores of both models. The performance metrics from the 60:40 split of the data was used, as an 80:20 split would have resulted in overfitting and unable to predict correctly on unforseen data.
### Background
 <a id="resources"></a> 
### Resources
- Data Source: [insurance_claims.csv](data/insurance_claims.csv)
- Software: Python 3.7.6, Spyder 4.0.0
 <a id="data"></a> 
### Data

The insurance_claims.csv file contains a mixture of 1,000 fraudulent and non-fraudulent insurance claims, 247 and 753 respectively.

The data points included in the file consists of the following:

![alt text](/img/posts/fraud_prod/ss/data_points_in_file.png) 

 <a id="data-cleaning-exploration"></a> 
## Data Cleaning and Exploration

Since not all the data is needed to determine fraud status, and too much and/or incorrect data can  skew the results, the cleaning of data is imperative.  Some of the ways I cleaned the data was by handling:

1.  Missing data
2.  Duplicate data
3.  Incorrect or irrelevant data
4.  Outliers  

<a id="missing-data"></a> 
### 1. Missing Data

There are many options when it comes to missing data.  You can choose to ignore it, replace it with something else or remove it. If an entire row or column is empty, it is better to remove the entire row or column as it offers no value.  I found the entire _c39 column was empty, so I removed it from the dataset.

```
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
```
<a id="duplicate-data"></a> 
### 2. Duplicate Data

I found no duplicate data in the dataset, so nothing needed to be done.

```
# 2. Duplicate data

## Check for duplicates
data_for_model.duplicated().value_counts()
```
<a id="incorrect-data"></a> 
### 3. Incorrect & Irrelevant Data

If information is not known, besides leaving it blank, another character will sometimes be entered in its place.  When I looked through the dataset, I found a '?' for some collision types, property damage and police report available columns.  Just like the missing data, you could choose to ignore it, replace it with something else or remove it.  

#### 1. collision_type column

The data revealed that for all the claims that contained '?' for collision type, they had an incident type of Parked Car or Vehicle Theft, not exactly a collision.  

![alt text](/img/posts/fraud_prod/ss/collision_type.png)

Additionally, the number of '?' collision type claims was equal to 18% of the entire dataset, making ignoring or removing them not an option.  I decided to instead replace '?' with None.

~~~
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

~~~


#### 2. property_damage column

For property damage, I didn't see any patterns to possibly explain why there may have been '?' entered so I decided to replace all '?' to NO.  

![alt text](/img/posts/fraud_prod/ss/property_type.png)

```
# 2.  'property_damage'
data_for_model['property_damage'].value_counts()

data_for_model.groupby(['property_damage','incident_type', 'collision_type'])["incident_type"].count()

## update property_damage to No
data_for_model.loc[(data_for_model.property_damage == '?'),'property_damage']='NO'
data_for_model['property_damage'].value_counts()
```
#### 3. police_report_available column

Again, I saw no patterns as to why there may be a '?' entered so I decided to replace all '?' to NO.

![alt text](/img/posts/fraud_prod/ss/police_report.png)

```
# 3.  'police_report_available'
data_for_model['police_report_available'].value_counts()

## update property_damage to No
data_for_model.loc[(data_for_model.police_report_available == '?'),'police_report_available']='NO'
data_for_model['police_report_available'].value_counts()
```
<a id="unique-values"></a> 
### Unique values

If any data is unique to a particular claim, it offers no value.  Querying the dataset, I found a few unique values and felt comfortable removing, such as policy_number and incident location.  

![alt text](/img/posts/fraud_prod/ss/unique.png)
```
## look for unique values
for uni_cols in data_for_model:
    u_length = len(pd.unique(data_for_model[uni_cols]))
    p_of_total = u_length / len(data_for_model)
    perc_format = "{:.0%}". format(p_of_total) 
    print(f"{uni_cols} has {perc_format}")
    
## too many unique values
data_for_model.drop(columns=['policy_number','incident_location'],inplace = True)

```

I also decided to remove policy_state, insured_zip, incident_date, incident_state, incident_city, insured_hobbies, auto_make, auto_model, auto_year as I also thought they offered no value.

```
#check for any other columns that  won't add value
data_for_model.drop(columns=['policy_state','insured_zip','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year'],inplace = True)
```
<a id="date-times"></a> 
### Date times

Datetime data can prove to be very useful but needs to be in number format to work with the algorithm. There is a *toordinal* function that will convert datetime to the Gregorian ordinal of a date which is a number. In this dataset I found the policy_bind_date came in as an object, so I had to change it to a datetime data type and then convert it using the toordinal function.  

```
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
```
<a id="outliers"></a> 
## 4. Outliers

Outliers are a challenge for some algorithms, Logistic Regression being one of them. Before deciding to remove them, I wanted to see how many there were and for which features.

![alt text](/img/posts/fraud_prod/ss/outliers.png)

After reviewing the list, I decided that it was probably safe to remove them.
```
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
    
```
<a id="data-exploration-visualizations"></a> 
## Data Exploration Visualizations

I was curious to see information about age, gender, witnesses, and police report available for fraud claims.
 <a id="age"></a>
### Age

![alt text](/img/posts/fraud_prod/graphs/Distribution_by_age.png)

Evn thought Non-Frauds just happens to be slightly larger, both Non-Fraud and Frauds actually have a pretty similar age distribution.
 <a id="gender"></a>
### Gender
<p align="center">
  <img src="/img/posts/fraud_prod/graphs/status_by_gender_c3.png" />
</p>
The number of fraud claims by gender is close to being evenly split, although females have a hair more.  All of all auto accident claims, combined fraud and non-faud, females seem to have more auto accident claimes than males. So much for putting to rest the age old adage about females not having the best driving skills.  
 <a id="witnesses"></a>

### Witnesses
<p align="center">
  <img src="/img/posts/fraud_prod/graphs/status_by_witnesses_c3.png" />
</p>

Surprisingly, fraud claims are more apt to have witnesses, than not.  
<a id="police"></a>
### Police Report Available

<p align="center">
  <img src="/img/posts/fraud_prod/graphs/status_by_Police_Report_Availability_c3.png" />
</p>

Unsurprisingly, there are less police reports available for fraud claims than not.  
<a id="split"></a>
## Split Input Variables & Output Variables

Before splitting the data, it is also best to shuffle it in case it is in some order. Shuffling adds to the randomness of the data and increase accuracy of the model.

```
# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```

Finally, we split the input (X) variables from the output (y) variables.

```
X = data_for_model.drop(["fraud_reported"], axis = 1)
y = data_for_model["fraud_reported"]

```
<a id="split_tt"></a>
## Split Out Training & Test Sets

I split out the training and test splits to the defaulted 80/20 split. I may, depending on the results, resplit it to a 60/40 split so I get a larger test set. 

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)##test_size percentage allocated to test_set, random_state = shuffling applied before split

```
<a id="categorical-variables"></a>
## Categorical Variables

Algorithms only work with numbers, so categorical variables need to be converted using One Hot Encoding.  In a nutshell, it breaks up the categorical data into its own column and populates the column values with 1 for yes and 0 for no.  For instance, if you have a gender column that is populated with the words Male and Female, you need to use OHE to convert it so it would no longer have a column for gender but would instead have one column for gender_male and one for gender_female. Each of these new columns would then have a value of 1 for yes or 0 for no accordingly.  BUT you must remove one of them due to avoid something called multicollinearity.  For this assignment, I have chosen to drop the first instance.

```
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

```

<a id="feature-scaling"></a>
## Feature Scaling

Feature Scaling is to put all the values on the same scale.  I used standard scaling.

```
# 6. Feature Scaling
scale_standared = StandardScaler()

X_train = pd.DataFrame(scale_standared.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_standared.transform(X_test), columns = X_test.columns)

```
<a id="log-reg"></a>
## Initialize Logistic Regression

Having the optimal amount of features is paramount to the effectiveness of the model.  For the Logistic Regression model I decided to use the Recursive Feature Elimination and Cross-Validation Selection (RFECV) to eliminate the irrelevant features.

```
#####################################################
# Initialize classifier
#####################################################

## Initialize logistic regression model
##Use Logistic Regression
clf = LogisticRegression(random_state = 42, max_iter = 1000)
feature_selector = RFECV(clf)## default is 5 chunks but can specify other

## fit feature to our data to train our model 
fit = feature_selector.fit(X_train,y_train)

```
<a id="feature-selection"></a>
## Feature Selection

```
#####################################################
# Feature Selection
#####################################################

## to find optimal # of features and count
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features:  {optimal_feature_count}")

```

The optimal number of features is 3. Having only 3 features can lead to overfitting so, besides addressing the class imbalance, I needed to address the small number of optimal features.

![alt text](/img/posts/fraud_prod/graphs/LR_optimal_feature_graph.png)

Here are the optimal features:

![alt text](/img/posts/fraud_prod/ss/LR_optimal.png)


Update the test and training tests with the above optimal features.

```
## updated X with only selected variables above (optimal)
## for logistic classification
X_train_LR = X_train.loc[: , feature_selector.get_support()]
X_test_LR = X_test.loc[: , feature_selector.get_support()]

```
<a id="model-training"></a>
## Model Training

Refit the model with the updated training and test sets.

```
#####################################################
# Model Training
#####################################################

clf.fit(X_train_LR, y_train)

```
<a id="model-assessment"></a>
## Model Assessment
<a id="confusion-matrix"></a>
**Confusion Matrix**

Generating a confusion matrix, we can easily see that the data is imbalanced, meaning one of the classes has a larger amount than the other(s).  In this case it is the True Non-Frauds (non-frauds predicted correctly as non-frauds), 105 to 28 True Frauds (frauds predicted correctly as frauds).

![alt text](/img/posts/fraud_prod/graphs/LR_con_matrix_before.png)

<a id="a_p_r_f_1"></a>
**Accuracy, Precision, Recall and F1 scores**

Even though the data is imbalaned and needs to be addressed, I ran the Accuracy, Precision, Recall and F1 scores so we can compare them to what they are after making adjustments for the imbalancing. (See below)

 ```
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

 ```

![alt text](/img/posts/fraud_prod/ss/a_p_r_scores_1.png)
<a id="imbalance"></a>
 ## Handling Data Imbalance
 <a id="opt_thresh"></a>
  **Optimal Threshold**
  
One way to handle the imbalancing of data is to adjust the threshold. The threshold is the line between saying a claim is fraud or non-fraud. To go deeper, when the Logisitic Regression model returns a probability score for a claim (which it does for all in the test set), it looks at the threshold amount and asks if the that probability score is above or below this line.  If it is above, it will say it is probabily fraud. If it is below, it will say it is probably non-fraud. This is why adjusting the threshold will sometimes make the data more balanced.  
  
   ```
#####################################################
# Finding the optimal threshold
#####################################################
## map test to 1, 0, cast as int
y_test = y_test.map({'Y': 1, 'N': 0}).astype(int)


thresholds = np.arange(0, 1, 0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    
    pred_class = (y_pred_prob >= threshold) * 1
    
    precision = precision_score(y_test, pred_class, zero_division = 0)
    precision_scores.append(precision)
    
    recall = recall_score(y_test, pred_class)
    recall_scores.append(recall)
    
    f1 = f1_score(y_test, pred_class)
    f1_scores.append(f1)


max_f1 = max(f1_scores)
max_f1_indx = f1_scores.index(max_f1)

max_f1 = max(f1_scores)
max_f1_indx = f1_scores.index(max_f1)

   ```

The optimal threshold is 0.13. 

 ![alt text](/img/posts/fraud_prod/graphs/LR_optimal_threshold_.png)
  
The default threshold is 0.5, so adjusting it to 0.13 may give us better results.  
<a id="confusion-matrix_t"></a>
**Confusion Matrix post threshold**

![alt text](/img/posts/fraud_prod/graphs/LR_con_matrix_AFTER.png)

<a id="a_p_r_f_2"></a>

**The Accuracy, Precision, Recall and F1 scores post threshold**

BEFORE

![alt text](/img/posts/fraud_prod/ss/a_p_r_scores_1.png)
  
AFTER

![alt text](/img/posts/fraud_prod/ss/a_p_r_scores_2.png)

Changing the threshold did not result in better performance numbers, they actually stayed the same.  The reason for this is may be due to the the model having a small test set, I initially did a 80:20 split.  I resplit the dataset with a 60:40 split to see if the model predicted better on the larger test set.  

<a id="model-assessment_6040"></a>
## Model Assessment with larger test set, 60:40 Split

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42, stratify = y)
```
<a id="opt_fs_6040"></a>
After the new split, besides re-encoding categorical variables and re-scalling the features, I reran feature selection with this larger training set and the optimal features number went from 3 to 28. The splitting of the data to make a larger test set, helped with increasing the number of optimal features, which helps with the overfitting concern.

![alt text](/img/posts/fraud_prod/graphs/LR_optimal_threshold_update.png)

The list of the 28 features.

![alt text](/img/posts/fraud_prod/ss/LR_feature_selection_2.png)

After updating the test and training sets with the 28 optimal features. I refitted and retrained the model and updated the threshold (0.31), hoping for better performance results.
<a id="opt_thresh_6040"></a>

--**Updated optimal threshold**--

![alt text](/img/posts/fraud_prod/graphs/LR_optimal_threshold_update_larger_testset.png)
<a id="confusion-matrix_6040"></a>

--**Updated confusion matrix**--

![alt text](/img/posts/fraud_prod/graphs/LR_con_matrix_AFTER_.png)

<a id="a_p_r_f_6040"></a>

--**Updated performance metrics**--
  
BEFORE

![alt text](/img/posts/fraud_prod/ss/a_p_r_scores_2.png)

AFTER

![alt text](/img/posts/fraud_prod/ss/a_p_r_scores_3.png)

**Accuracy Score** 

The accuracy score is the percentage of correctly predicted claims (whether they were fraud or non-fraud) divided by the total number of claims. It is a score that can give a good description of the model's performance but does not give an accurate picture when their is a data imbalance, which is in this case. 

<p align="center">
  <img src="/img/posts/fraud_prod/ss/accuracy_calc.png"  />
</p>
  
Before the theshold adjustment and test set size increase, the accuracy score was 83.12% and after it went down to 79.00%.  If it were not for the fact of the data being imbalanced, both these accuracy scores would indicate a good model. Any accuracy score between 70-80% is considered good and between 80-90% is considered excellent. Again, considering there is a data imbalance, we need to look at Precision, Recall and F1 scores to get a better assessment of the model.

**Precision**  

The precision score of a model is a score given for how well the model did on predicting clasess correctly. Using this project as an example, the calculation would be take the total number of times the model CORRECTLY predicted a fraud was a fraud (**True Positive** (TP)) and divide it by the total number of times the model CORRECTLY predicted a fraud was a fraud (**True Positive** (TP)) + the total number of times the model INCORRECTLY predicted it was a fraud when it was actually a non-fraud (**False Positive** (FP)). In other words, of all the claims the model predicted as frauds (correctly predicted as frauds and incorrectly predicted as frauds), what is the percentage that were correctly predicted as fraud (true frauds).  
<p align="center">
  <img src="/img/posts/fraud_prod/ss/precision_calc.png" />
</p> 

Before the theshold adjustment and test set size increase, the prcecision score was 62.22% and after it went down to 54.76%. Any amount over 50% is considered good but, again, this score is not great.

**Recall**  

A recall score is the converse of precision and if you add to the two together they equal (or should) 100%. The recall score is how well the model did in labeling fraud claims as fraud.  Again, using this project as an example, you would take the total number of times the model CORRECTLY predicted a fraud was a fraud (**True Positive** (TP))) and divide it by the total number of times the model CORRECTLY predicted a fraud was a fraud (**True Positive** (TP)) + the total number of time the model INCORRECTLY predicted it was a non-fraud when it was actually a fraud (**False Negative** (FN)). In other words, of all the claims the model were actually frauds (correctly predicted as frauds and incorrectly predicted as non-frauds when they were actually frauds), what is the percentage that were correctly predicted as fraud (true frauds).  

<p align="center">
  <img src="/img/posts/fraud_prod/ss/recall_calc.png"  />
</p>

Before the theshold adjustment and test set size increase, the recall score was 73.68% and after it went down to 61.33%. Any amount over 50% is considered good but, again, this score is not great.

**F1 score**

The F1 score is the blending of precision and recall, an harmonic mean. The higher the score, the more accurate the model did at predicting.  

<p align="center">
  <img src="/img/posts/fraud_prod/ss/f1_score_calc.png"  />
</p>

Before the theshold adjustment and test set size increase, the f1 score score was 67.47% and after it went down to 57.86%. The higher the f1 score, the better and in this case, before and after, this score it not great at all.   

Since using a Logistic Regression model did not provide great results, the next model I tried was the Random Forest as they tend to be highly accurate. 

<a id="random"></a>
########################################################

## Random Forest model

```
######################################################
## Use Random Forest instead
#####################################################

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 42, n_estimators = 2000)
rfc.fit(X_train, y_train)

```
<a id="feature-selection_rf"></a>
### Feature Selection 

There are few methods that can be used for feature selection with a Random Forest model.  I tried both the the built-in method which utilizes the Gini importance (mean decrease impurity, the more it decreases, the more important the feature) and the Permutation Importance which calculates what effective the feature has on the model accuracy by randomly re-ordering it. 
<a id="built-in"></a>
**Built-in Random Forest Importance** 

```
feature_importance = pd.DataFrame(rfc.feature_importances_)
feature_names = pd.DataFrame(X_train.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace = True)

```


![alt text](/img/posts/fraud_prod/graphs/RF_importance_2.png)

Here are the features:

![alt text](/img/posts/fraud_prod/ss/rf_features_1.png)

![alt text](/img/posts/fraud_prod/ss/rf_features_2.png)


<a id="permutation"></a>
**Permutation Importance**

```
# Permutation Importance 
y_test = y_test.map({1: 'Y', 0: 'N'})##remap back for permutation

from sklearn.inspection import permutation_importance

result = permutation_importance(rfc, X_test, y_test, n_repeats = 10, random_state = 42)
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names_perm = pd.DataFrame(X_train.columns)
permutation_importance_summary = pd.concat([feature_names_perm,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

```

![alt text](/img/posts/fraud_prod/graphs/permutation_importance_2_.png)

Any feature with a permutation importance amount less than 0 was removed as it doesn't actually improve the importance of the model.  

Here are the resulting 16 features:

![alt text](/img/posts/fraud_prod/ss/perm_features.png)

I decided to use the Permutation Importance feature selection because, although more computing intensive to calculate, it, unlike the built-in feature importance, is not biased towards high-cardinality categorical variables and is more reliable in its results. 

<a id="model-training_rf"></a>
## Model Training

The resulting Permutation Importance 16 features were then refitted to the model and the training and test sets updated.

```
rf_perm = permutation_importance_summary[permutation_importance_summary['permutation_importance'] >= 0]
rf_perm.sort_values(by = "permutation_importance", ascending = False, inplace = True)


## get column names
rf_perm_columns = []
for cols in rf_perm['input_variable']:
    #print(cols)
    rf_perm_columns.append(cols)

## updated X with only selected variables above (optimal)
## for random forest
X_train_RF = X_train.loc[: , rf_perm_columns]
X_test_RF = X_test.loc[: , rf_perm_columns]


#REFIT after feature importance
rfc.fit(X_train_RF, y_train)

```
<a id="model-assessment_rf"></a>
## Model Assessment
<a id="confusion-matrix_6040_rf"></a>
**Confusion Matrix**

![alt text](/img/posts/fraud_prod/graphs/rf_confusion_matrix_before_threshold.png)

<a id="opt_thresh_6040_rf"></a>
--**Updated optimal threshold**--

![alt text](/img/posts/fraud_prod/graphs/RF_optimal_threshold.png)

The default threshold is 0.5, so decreasing it to 0.31 may give us better results.

<a id="confusion-matrix_6040_rf_2"></a>
**Confusion Matrix Post Threshold**

![alt text](/img/posts/fraud_prod/graphs/rf_confusion_after_threshold.png)

<a id="a_p_r_f_6040_rf"></a>
**Accuracy, Precision, Recall and F1 scores**

BEFORE

![alt text](/img/posts/fraud_prod/ss/rf_a_p_r_1.png)

AFTER

![alt text](/img/posts/fraud_prod/ss/rf_a_p_r_2.png)

<a id="roc-auc"></a>
## ROC-AUC

```
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test_numbers, , y_pred_prob)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_t, , y_pred_prob_rf)

##plot ROC curves
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr_lr, tpr_lr, label= "Logistic Regression", linewidth=4)
plt.plot(fpr_rf, tpr_rf, label= "Random Forest", linewidth=4)
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Receiver Operating Characteristic')
plt.show()

```
<a id="graph"></a>

![alt text](/img/posts/fraud_prod/graphs/AUC-ROC_graph.png)

```
# auc scores
auc_log_reg = roc_auc_score(y_test_numbers, , y_pred_prob)
auc_random_forest = roc_auc_score(y_test_t, , y_pred_prob_rf)

line = "----" * 10
print(f"\nAUC Scores\n{line}")
print("Logistic Regression: {0:.2%}".format(auc_log_reg))
print("Random Forest: {0:.2%}".format(auc_random_forest))
```
<a id="auc"></a>
![alt text](/img/posts/fraud_prod/ss/AUC-ROC_scores.png)

<a id="pr_curves"></a>

## Precision-Recall Curves

```
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test_numbers,  y_pred_class_opt_thresh)
precision_rf, recall_rf, _ = precision_recall_curve(y_test_t,  y_pred_class_opt_thresh_rf)
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

```
<a id="pr_graph"></a>
![alt text](/img/posts/fraud_prod/graphs/precision_recall_curves.png)

```
# Precision-Rcall Auc scores
from sklearn.metrics import auc

auc_log_reg_pr = auc(recall, precision)
auc_random_forest_pr = auc(recall_rf, precision_rf)

line = "----" * 10
print(f"\nPrecision Recall AUC Scores\n{line}")
print("Logistic Regression: {0:.2%}".format(auc_log_reg_pr))
print("Random Forest: {0:.2%}".format(auc_random_forest_pr))
```
<a id="pr_auc"></a>

![alt text](/img/posts/fraud_prod/ss/precision_recall_auc.png)




