---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Possible variables

* is_group_X(multiple binaries)
* has_children
* years_of_edu_per_age

## Variables to transform
* Age(from birthday)
* Marital Status(simplify maybe)
* employment_sector_simplified
* Education Level(The PostGraduation Paradox)

## Bonus points ideas
* XGboost
* MBR
* Ensemble
* PCA

```python
# Libraries for manipulating and displaying data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# Utility Libraries
from datetime import datetime

# Model Libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, RFECV, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
```

## Importing Data


First, we import both the Train and Test datasets into Pandas Dataframes:

```python
data = pd.read_excel('data/Train.xlsx')
test_data = pd.read_excel('data/Test.xlsx')
```

Afterwards, let's get a look at our data.

```python
data.head()
```

Since the table already has an ID, let's use that as the DF's index.

```python
data.set_index('CITIZEN_ID',drop=True,inplace=True)
test_data.set_index('CITIZEN_ID',drop=True,inplace=True)
```

Backup the data in case we make significant changes we need to revert

```python
backup = data.copy()
```

```python
data = backup.copy()
```

Because we need to handle metric and non-metrical data in a different way, let us create a way to filter them in case it is necessary.

```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
metric_features = data.select_dtypes(include=numerics).columns
non_metric_features = data.columns.drop(metric_features).to_list()
```

```python
metric_features
```

### Pandas Profiling
To get a quick overview of the dataset, we used the pandas-profiling library to generate an automatic report. This, however, does not and will not replace an actual, in-depth, analysis.

The ProfileReport shows us the basic statistics and the general distribution for each variable, as well as a Correlation Matrix and Scatterplots for comparison between multiple attributes.

```python
# Generate a profile
def generate_report(path):
    profile = ProfileReport(
        data, 
        title='Newland Citizens Report',
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
            "cramers": {"calculate": False},
        }
     )
    profile.to_file(path)


# Export this profile to a file
# generate_report('reports/citizen_profiling.html')
```

## Missing Values

We considered that a value o '?' in a column represents a missing value.

For better treatment, we replaced them with numpy.NaN.

```python
col_with_missing = []
for column in data.columns:
    if '?' in data[column].values:
        col_with_missing.append(column)

# We repeat the analysis for the Test dataset as well, to ensure that the columns with missing values are the
# same
testcol_with_missing = []
for column in test_data.columns:
    if '?' in test_data[column].values:
        testcol_with_missing.append(column)
```

```python
print(col_with_missing == testcol_with_missing)
```

```python
test_data[testcol_with_missing] = test_data[testcol_with_missing].replace('?', np.NaN)
data[col_with_missing] = data[col_with_missing].replace('?', np.NaN)
```

```python
data.loc[:,col_with_missing].isnull().mean()
```

```python
test_data.loc[:,testcol_with_missing].isnull().mean()
```

From this, we can see that we have aproximmately the same proportion of missing values in Train and Test.

```python
test_data['Base Area'].value_counts()
```

```python
data['Base Area'].value_counts()
```

```python
data['Employment Sector'].value_counts()
```

```python
data['Role'].value_counts()
```

```python
data[data['Employment Sector'] == 'Unemployed']
```

```python
data[(data['Role'].isna()) & (data['Employment Sector'].isna())]
```

```python
data[(data['Role'].isna()) & ~(data['Employment Sector'].isna())]
```

```python
data[data['Role'].isna()]
```

```python
data['Role'].fillna('Unemployed', inplace=True)
```

```python
data['Employment Sector'].fillna('Unemployed', inplace=True)
```

```python
data['Employment Sector']=data['Employment Sector'].apply(lambda x: 'Unemployed' if x=='Never Worked' else x)
```

```python
test_data['Role'].fillna('Unemployed', inplace=True)
test_data['Employment Sector'].fillna('Unemployed', inplace=True)
test_data['Employment Sector'] = test_data['Employment Sector'].apply(
    lambda x: 'Unemployed' if x == 'Never Worked' else x)
test_data['Base Area'].fillna(test_data['Base Area'].mode()[0], inplace=True)
```

```python
data.loc[:,col_with_missing].isnull().mean()
```

```python
data['Base Area'].fillna(data['Base Area'].mode()[0],inplace=True)
```

```python
test_data.loc[:,col_with_missing].isnull().mean()
```

# Modelling


* Random Forest
* Decision Trees
* Neural Network
* Logistic Regression
* kNN
* Naive Bayes


## Target Variable definition and Train-Test Split


First we split the dataframe, in order to separate the target variable from the rest.


For this initial analysis, we'll only consider the 4 metric variables present in the initial dataset.

```python
target = data['Income']
X = data.drop(columns='Income')
```

```python
X_metrics = data[metric_features[:-1]]
X_test_metrics = test_data[metric_features[:-1]] 
```

```python
X_train, X_val, y_train, y_val = train_test_split(X_metrics,
                                                  target,
                                                  test_size=0.25,
                                                  stratify=target,
                                                  random_state=35)
```

```python
def metrics(model, X_train, X_val, y_train, y_val):
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    
    print(classification_report(y_train, pred_train))
    # print(confusion_matrix(y_train, pred_train))
    
    print(classification_report(y_val, pred_val))
    # print(confusion_matrix(y_val, pred_val))
    
def evaluate(model):
    metrics(model, X_train, X_val, y_train, y_val)
    f1_micro = f1_evaluation(model)
    print(f'The Micro Average of the F1 Score is : {f1_micro}')
    
def f1_evaluation(model):
    return f1_score(y_val, model.predict(X_val), average='micro')
```

```python
def batch_model_creation():
    model_df = pd.DataFrame(columns=['Model_Name', 'F1 Score Initial'])
    
    # Random Forest
    randForest = RandomForestClassifier(max_depth=10, random_state=0)
    randForest.fit(X_train, y_train)
    model_df.loc[0] = ['Random Forest', f1_evaluation(randForest)] 
    
    #Decision Tree
    dt_gini = DecisionTreeClassifier(max_depth = 10, random_state=0)
    dt_gini.fit(X_train, y_train)
    model_df.loc[1] = ['Decision Tree GINI', f1_evaluation(dt_gini)] 
    
    
    # Multi-layer Perceptron
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    model_df.loc[3] = ['Multi-layer Perceptron', f1_evaluation(mlp)] 
    
    # Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    model_df.loc[4] = ['Logistic Regression', f1_evaluation(log_model)] 
    
    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    model_df.loc[5] = ['K-Nearest Neighbors', f1_evaluation(knn)] 
    
    # Gaussian Model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    model_df.loc[6] = ['GaussianNB', f1_evaluation(nb_model)] 
    
    
    return model_df

def batch_model_update(model_df, score_name):
    new_scores = []
    
    # Random Forest
    randForest = RandomForestClassifier(max_depth=10, random_state=0)
    randForest.fit(X_train, y_train)
    new_scores.append(f1_evaluation(randForest)) 
    
    dt_gini = DecisionTreeClassifier(max_depth = 10, random_state=0)
    dt_gini.fit(X_train, y_train)
    new_scores.append(f1_evaluation(dt_gini)) 
    
    # Multi-layer Perceptron
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    new_scores.append(f1_evaluation(mlp)) 
    
    # Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    new_scores.append(f1_evaluation(log_model)) 
    
    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    new_scores.append(f1_evaluation(knn)) 
    
    # Gaussian Model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    new_scores.append(f1_evaluation(nb_model))
    
    new_scores = pd.Series(data=new_scores, name=score_name, index=model_df.index)
    return pd.concat([model_df, new_scores], axis=1)
    
```

```python
df_model = batch_model_creation()
df_model
```

## Results Exporting

```python
def export_results(model, nversion,test_data):
    results=pd.concat([test_data, pd.DataFrame(model.predict(test_data), index=test_data.index)],axis=1)
    results=results.iloc[:,-1:]
    results.index.rename('CITIZEN_ID',inplace=True)
    results.rename(columns={0:'Income'},inplace=True)
    results.to_csv(path_or_buf='subs/Group48_Version'+str(nversion)+'.csv')
```

```python
randForest = RandomForestClassifier(max_depth=10, random_state=0)
randForest.fit(X_train, y_train)
#export_results(randForest,1,X_test_metrics)
```

```python
def plot_tree(model):
    dot_data = export_graphviz(model,
                               feature_names=X_train.columns,  
                               class_names=["No Diabetes", "Diabetes"],
                               filled=True)
    pydot_graph = pydotplus.graph_from_dot_data(dot_data)
    pydot_graph.set_size('"20,20"')
    return graphviz.Source(pydot_graph.to_string())
```

## Outlier Analysis


### Numeric Vars Histograms

```python
%matplotlib inline
# All Numeric Variables' Histograms in one figure
sns.set()

# Prepare figure. Create individual axes where each histogram will be placed
fig, axes = plt.subplots(2, int(len(metric_features) / 2), figsize=(20, 11))
# Plot data

# Iterate across axes objects and associate each histogram (hint: use the ax.hist() instead of plt.hist()):
for ax, feat in zip(axes.flatten(), metric_features): # Notice the zip() function and flatten() method
    ax.hist(data[feat])
    ax.set_title(feat, y=-0.13)
    
# Layout
# Add a centered title to the figure:
title = "Numeric Variables' Histograms"
plt.suptitle(title)
plt.show()
```

### Numeric Boxplots

```python
# All Numeric Variables' Box Plots in one figure
sns.set()
# Prepare figure. Create individual axes where each box plot will be placed
fig, axes = plt.subplots(2, int(len(metric_features) / 2), figsize=(20, 11))

# Plot data
# Iterate across axes objects and associate each box plot (hint: use the ax argument):
for ax, feat in zip(axes.flatten(), metric_features): # Notice the zip() function and flatten() method
    sns.boxplot(x=data[feat], ax=ax)
    
# Layout
# Add a centered title to the figure:
title = "Numeric Variables' Box Plots"
plt.suptitle(title)
plt.show()
```

### Vars Exclusive to some Citizens
The vars 'Money Received' and 'Ticket Price' assume a value of zero if they are not applicable for the citizen in question. That will make real values look like outliers. By observing the distributions with and without the 0 values we will be able to better identify outliers.

```python
%matplotlib inline
# All Numeric Variables' Histograms in one figure
sns.set()

# Prepare figure. Create individual axes where each histogram will be placed
fig, axes = plt.subplots(2, 2, figsize=(20, 11))
# Plot data

# Iterate across axes objects and associate each histogram (hint: use the ax.hist() instead of plt.hist()):
features=['Money Received','Ticket Price','Money Received','Ticket Price']
count=0
for ax, feat in zip(axes.flatten(), features): # Notice the zip() function and flatten() method
    if count<2:
        ax.hist(data[data[feat]!=0][feat])
        ax.set_title(feat, y=-0.13)
    else:
        sns.boxplot(x=data[data[feat]!=0][feat], ax=ax)
        ax.set_title(feat, y=-0.13)
    count+=1
    
# Layout
# Add a centered title to the figure:
title = "Money Received and Ticket Price Histograms and BoxPlots"
plt.suptitle(title)
plt.show()
```

Years of Education Outliers

```python
len(data[(data['Years of Education']<=7.5) | (data['Years of Education']>=20)])/len(data)
```

```python
len(data[(data['Years of Education']<=5) | (data['Years of Education']>=20)])/len(data)
```

```python
len(data[data['Years of Education']<=5])/len(data)
```

```python
data[(data['Years of Education']<=5) | (data['Years of Education']>=20)].groupby('Years of Education')['Education Level'].value_counts()
```

Working Hours per Week

```python
len(data[(data['Working Hours per week']>=60) | (data['Working Hours per week']<=20)]  )/len(data)
```

```python
len(data[(data['Working Hours per week']>=80)])/len(data)
```

Money Received

```python
len(data[data['Money Received']>=40000])/len(data[data['Money Received']>0])
```

```python
len(data[data['Money Received']>=120000])/len(data[data['Money Received']>0])
```

Ticket Price

```python
len(data[data['Ticket Price']>=4000])/len(data[data['Ticket Price']>0])
```

```python
len(data[data['Ticket Price']>=120000])/len(data[data['Ticket Price']>0])
```

Outlier Removal

```python
filters1 = (
    (data['Years of Education']>5)
    &
    (data['Working Hours per week']<=80)
    &
    ((data['Ticket Price']>=150) | (data['Ticket Price']==0))
    &
    ((data['Money Received']>=134) | (data['Money Received']==0))
    
)
len(data[filters1]) / len(data)
```

```python
data_no_outliers=data[filters1]
```

#### Modeling no Outliers

```python
target = data_no_outliers['Income']
X = data_no_outliers.drop(columns='Income')
```

```python
X_metrics = data_no_outliers[metric_features[:-1]]
X_test_metrics = test_data[metric_features[:-1]] 
```

```python
X_train, X_val, y_train, y_val = train_test_split(X_metrics,
                                                  target,
                                                  test_size=0.25,
                                                  stratify=target,
                                                  random_state=35)
```

```python
df_model=batch_model_update(df_model, 'Post-Outliers F1 Score')
```

```python
df_model
```

```python
randForest = RandomForestClassifier(max_depth=10, random_state=0)
randForest.fit(X_train, y_train)
evaluate(randForest)
```

```python
#export_results(randForest,2,X_test_metrics)
```

## Variables to transform
* Age(from birthday)
* Marital Status(simplify maybe)
* employment_sector_simplified
* Education Level(The PostGraduation Paradox)
*Lives with
*Continent

```python
data['is_Married']= data['Marital Status'].apply(lambda x: 1 if x in ['Married - Spouse in the Army','Married', 'Married - Spouse Missing'] else 0)
test_data['is_Married']= test_data['Marital Status'].apply(lambda x: 1 if x in ['Married - Spouse in the Army','Married', 'Married - Spouse Missing'] else 0)
```

```python
data['Private Sector'] = data['Employment Sector'].apply(lambda x: 1 if 'Private' in x else 0)
test_data['Private Sector'] = test_data['Employment Sector'].apply(lambda x: 1 if 'Private' in x else 0)
```

```python
data['Public Sector'] = data['Employment Sector'].apply(lambda x: 1 if 'Public' in x else 0)
test_data['Public Sector'] = test_data['Employment Sector'].apply(lambda x: 1 if 'Public' in x else 0)
```

```python
data['Self Employed'] = data['Employment Sector'].apply(lambda x: 1 if 'Self' in x else 0)
test_data['Self Employed'] = test_data['Employment Sector'].apply(lambda x: 1 if 'Self' in x else 0)
```

```python
data['Unemployed'] = data['Employment Sector'].apply(lambda x: 1 if 'Unemployed' in x else 0)
test_data['Unemployed'] = test_data['Employment Sector'].apply(lambda x: 1 if 'Unemployed' in x else 0)
```

```python
data['Employment Sector'].value_counts()
```

```python
data.groupby('Education Level')['Years of Education'].value_counts()
```

```python
data['Professional School']= data['Education Level'].apply(lambda x: 1 if 'Professional' in x else 0)
test_data['Professional School']= test_data['Education Level'].apply(lambda x: 1 if 'Professional' in x else 0)
```

```python
data['Native Continent'].value_counts()
```

```python
data=data.drop(columns='Native Continent').merge(pd.get_dummies(data['Native Continent'],prefix='Continent').iloc[:,:-1],on=data.index, left_index=True)
test_data=test_data.drop(columns='Native Continent').merge(pd.get_dummies(test_data['Native Continent'],prefix='Continent').iloc[:,:-1],on=test_data.index, left_index=True)
```

```python
data.drop(columns='key_0',inplace=True)
test_data.drop(columns='key_0',inplace=True)
```

```python
data['Lives with'].value_counts()
```

```python
data['Lives_Spouse']=data['Lives with'].apply(lambda x: 1 if x in['Husband','Wife'] else 0)
test_data['Lives_Spouse']=test_data['Lives with'].apply(lambda x: 1 if x in['Husband','Wife'] else 0)
```

```python
data['Lives_Children']=data['Lives with'].apply(lambda x: 1 if x in 'Children' else 0)
test_data['Lives_Children']=test_data['Lives with'].apply(lambda x: 1 if x in 'Children' else 0)
```

```python
data['Lives_Other']=data['Lives with'].apply(lambda x: 1 if 'Other' in x else 0)
test_data['Lives_Other']=test_data['Lives with'].apply(lambda x: 1 if 'Other' in x else 0)
```

```python
data['Lives_Northbury']=data['Base Area'].apply(lambda x: 1 if 'Northbury' in x else 0)
test_data['Lives_Northbury']=test_data['Base Area'].apply(lambda x: 1 if 'Northbury' in x else 0)
```

```python
data=data.drop(columns='Role').merge(pd.get_dummies(data['Role'],prefix='Role').iloc[:,:-1],on=data.index, left_index=True)
data.drop(columns='key_0',inplace=True)
test_data=test_data.drop(columns='Role').merge(pd.get_dummies(test_data['Role'],prefix='Role').iloc[:,:-1],on=test_data.index, left_index=True)
test_data.drop(columns='key_0',inplace=True)
```

```python
data_encoded=data.drop(columns=['Name','Birthday','Marital Status','Lives with','Base Area','Education Level','Employment Sector'])
```

```python
test_data_encoded=test_data.drop(columns=['Name','Birthday','Marital Status','Lives with','Base Area','Education Level','Employment Sector'])
```

```python
target=data_encoded['Income']
data_encoded.drop(columns='Income',inplace=True)
```

## Model with Encoded Vars

```python
X_train, X_val, y_train, y_val = train_test_split(data_encoded,
                                                  target,
                                                  test_size=0.25,
                                                  stratify=target,
                                                  random_state=35)
```

```python
df_model=batch_model_update(df_model,'Encoded Vars F1 Score')
df_model
```

```python
randForest = RandomForestClassifier(max_depth=10, random_state=0)
randForest.fit(X_train, y_train)
evaluate(randForest)
```

```python
#export_results(randForest,3,test_data_encoded)
```

### Encoded Vars no outliers

```python
data_encoded_out=data_encoded[filters1].copy()
target_no_out=target.loc[data_encoded_out.index]
```

```python
X_train, X_val, y_train, y_val = train_test_split(data_encoded_out,
                                                  target_no_out,
                                                  test_size=0.25,
                                                  stratify=target_no_out,
                                                  random_state=35)
```

```python
df_model=batch_model_update(df_model,'Encoded Vars no Outliers F1 Score')
df_model
```

# Feature Extraction

* is_group_X(multiple binaries)
* has_children
* years_of_edu_per_age
* Education Level(The PostGraduation Paradox)
* gender (from the title that comes before the name) 

```python
data['Age'] = data['Birthday'].apply(lambda x: datetime.strptime(x[-4:], "%Y").date()).astype('datetime64[ns]')
data['Age'] = data['Age'].apply(lambda x: 2048 - x.year)
```

```python
test_data['Age'] = test_data['Birthday'].apply(lambda x: datetime.strptime(x[-4:], "%Y").date()).astype('datetime64[ns]')
test_data['Age'] = test_data['Age'].apply(lambda x: 2048 - x.year)
```

```python
data['Education per Age'] = data['Years of Education'] / data['Age']
```

```python
test_data['Education per Age'] = test_data['Years of Education'] / test_data['Age']
```

```python
data['is_Male']= data['Name'].apply(lambda x: 1 if x.split(' ')[0]=='Mr.' else 0)
```

```python
test_data['is_Male']= test_data['Name'].apply(lambda x: 1 if x.split(' ')[0]=='Mr.' else 0)
```

```python
data['is_group_a'] = data['Ticket Price'] + data['Money Received']
data['is_group_a'] = data['is_group_a'].apply(lambda x: 1 if x == 0 else 0)
data['is_group_b'] = data['Money Received'].apply(lambda x: 1 if x > 0 else 0)
data['is_group_c'] = data['Ticket Price'].apply(lambda x: 1 if x > 0 else 0)
data[['is_group_a', 'is_group_b', 'is_group_c']]
```

```python
test_data['is_group_a'] = test_data['Ticket Price'] + test_data['Money Received']
test_data['is_group_a'] = test_data['is_group_a'].apply(lambda x: 1 if x == 0 else 0)
test_data['is_group_b'] = test_data['Money Received'].apply(lambda x: 1 if x > 0 else 0)
test_data['is_group_c'] = test_data['Ticket Price'].apply(lambda x: 1 if x > 0 else 0)
```

```python
data_transformed=data.drop(columns=['Name','Birthday','Marital Status','Lives with','Base Area','Education Level','Employment Sector'])
```

```python
test_transformed=test_data.drop(columns=['Name','Birthday','Marital Status','Lives with','Base Area','Education Level','Employment Sector'])
```

```python
target_transformed=data_transformed['Income']
data_transformed=data_transformed.drop(columns='Income')
```

### Model with new Transformed Variables

```python
X_train, X_val, y_train, y_val = train_test_split(data_transformed,
                                                  target_transformed,
                                                  test_size=0.25,
                                                  stratify=target_transformed,
                                                  random_state=35)
```

```python
df_model=batch_model_update(df_model,'Transformed Vars F1 Score')
df_model
```

```python
randForest = RandomForestClassifier(max_depth=10, random_state=0)
randForest.fit(X_train, y_train)
evaluate(randForest)
```

```python
#export_results(randForest,5,test_transformed)
```

## Model with Transformed Vars with no Outliers

```python
data_transformed_no_out=data_transformed[filters1]
target_transformed_no_out=data.loc[data_transformed_no_out.index]['Income']

X_train, X_val, y_train, y_val = train_test_split(data_transformed_no_out,
                                                  target_transformed_no_out,
                                                  test_size=0.25,
                                                  stratify=target_transformed_no_out,
                                                  random_state=35)
```

```python
df_model=batch_model_update(df_model,'Transformed Vars No Outliers F1 Score')
df_model
```

```python
randForest = RandomForestClassifier(max_depth=10, random_state=0)
randForest.fit(X_train, y_train)
evaluate(randForest)
```

## Correlation Analysis

```python
# Prepare figure
fig = plt.figure(figsize=(20, 20))
# Obtain correlation matrix. Round the values to 2 decimal cases. Use the DataFrame corr() and round() method.
corr = np.round(data.corr(method="pearson"), decimals=2)
# Build annotation matrix (values above |0.5| will appear annotated in the plot)
mask_annot = np.absolute(corr.values) >= 0.5
annot = np.where(mask_annot, corr.values, np.full(corr.shape,"")) # Try to understand what this np.where() does
# Plot heatmap of the correlation matrix
sns.heatmap(data=corr, annot=annot, cmap=sns.diverging_palette(220, 10, as_cmap=True), 
            fmt='s', vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
# Layout
fig.subplots_adjust(top=0.95)
fig.suptitle("Correlation Matrix", fontsize=20)
plt.show()
```

```python
data.corr(method='spearman')['Income'].abs().sort_values(ascending=False)
```

```python
##Come back to the previous Step
data_transformed=data.drop(columns=['Name','Birthday','Marital Status','Lives with','Base Area','Education Level','Employment Sector'])
test_transformed=test_data.drop(columns=['Name','Birthday','Marital Status','Lives with','Base Area','Education Level','Employment Sector'])

target_transformed= data_transformed['Income']
data_transformed.drop(columns='Income', inplace=True)

```

```python
data_t_uncorr=data_transformed.drop(columns=['is_group_c', 'is_Married'])
```

```python
X_train, X_val, y_train, y_val = train_test_split(data_t_uncorr,
                                                  target_transformed,
                                                  test_size=0.25,
                                                  stratify=target_transformed,
                                                  random_state=35)
```

```python
df_model=batch_model_update(df_model,'Transformed Vars Uncorrelated F1 Score')
df_model
```

## Data Standartization

```python
scaler=MinMaxScaler()
data_t_scaled=pd.DataFrame(scaler.fit_transform(data_t_uncorr), index=data_t_uncorr.index, columns=data_t_uncorr.columns)
data_t_scaled
```

```python
X_train, X_val, y_train, y_val = train_test_split(data_t_scaled,
                                                  target_transformed,
                                                  test_size=0.25,
                                                  stratify=target_transformed,
                                                  random_state=35)
```

```python
df_model=batch_model_update(df_model,'Transformed Vars, Uncorrelated Scaled F1 Score')
df_model
```

```python
scaler=MinMaxScaler()
test_t_scaled=pd.DataFrame(scaler.fit_transform(test_transformed), index=test_transformed.index, columns=test_transformed.columns)
test_t_scaled
```

```python
#export_results(randForest,6,test_t_scaled)
```

## Data Standartization no Outliers

```python
scaler=MinMaxScaler()
data_t_scaled_out=pd.DataFrame(scaler.fit_transform(data_transformed_no_out), index=data_transformed_no_out.index, columns=data_transformed_no_out.columns)
data_t_scaled_out
```

```python
X_train, X_val, y_train, y_val = train_test_split(data_t_scaled_out,
                                                  target_transformed_no_out,
                                                  test_size=0.25,
                                                  stratify=target_transformed_no_out,
                                                  random_state=35)
```

```python
df_model=batch_model_update(df_model,'Transformed Vars Scaled No Out F1 Score')
df_model
```

## Feature Selection


#### Standardization First, VarianceThreshold After

```python
X_train, X_val, y_train, y_val = train_test_split(data_t_uncorr,
                                                  target_transformed,
                                                  test_size=0.25,
                                                  stratify=target_transformed,
                                                  random_state=35)
```

```python
var_pipe=Pipeline([
    ('Standardization', MinMaxScaler()),
    ('Feature Selection', VarianceThreshold(threshold = .8 * (1 - .8))),
    ('Classifier', RandomForestClassifier(max_depth = 10, random_state = 0))
])
    
var_pipe.fit(X_train,y_train)
```

```python
evaluate(var_pipe)
```

```python
data_transformed.iloc[:,var_pipe.named_steps['Feature Selection'].get_support(indices=True)]
```

#### VarianceThreshold First, Standardization After

```python
var_pipe=Pipeline([
    ('Feature Selection', VarianceThreshold(threshold = .8 * (1 - .8))),
    ('Standardization', MinMaxScaler()),
    ('Classifier', RandomForestClassifier(max_depth = 10, random_state = 0))
])
    
var_pipe.fit(X_train,y_train)
```

```python
evaluate(var_pipe)
```

```python
df_model=batch_model_update(df_model,'Transformed Vars Uncorrelated Pipe VarThresh F1 Score')
df_model
```

```python
data_transformed.iloc[:,var_pipe.named_steps['Feature Selection'].get_support(indices=True)]
```

```python
test_t_uncorr=test_transformed.drop(columns=['is_group_c','is_Married'])
```

```python
#export_results(var_pipe,7,test_t_uncorr)
```

#### Std first, RFE after

```python
var_pipe=Pipeline([ 
    ('Standardization', MinMaxScaler()),
    ('Feature Selection', RFECV(estimator=SVC(kernel='linear'),step=1,cv=StratifiedKFold(2), scoring='f1_micro')),
    ('Classifier', RandomForestClassifier(max_depth = 10, random_state = 0))
])
    
var_pipe.fit(X_train,y_train)
```

```python
evaluate(var_pipe)
```

```python
df_model=batch_model_update(df_model,'Transformed Vars Uncorrelated Pipe STD RFE F1 Score')
df_model
```

```python
data_transformed.iloc[:,var_pipe.named_steps['Feature Selection'].get_support(indices=True)]
```

#### Tree Based Feature Selection

```python
var_pipe=Pipeline([ 
    ('Impurity Based Feature Selection',SelectFromModel(ExtraTreesClassifier(n_estimators=10))),
    ('Classifier', RandomForestClassifier(max_depth = 10, random_state = 0))
])
    
var_pipe.fit(X_train,y_train)
```

```python
evaluate(var_pipe)
```

```python
df_model=batch_model_update(df_model,'Transformed Vars Uncorrelated Pipe Tree Based F1 Score')
df_model
```

```python
data_transformed.iloc[:,var_pipe.named_steps['Impurity Based Feature Selection'].get_support(indices=True)]
```

## Feature Importance

```python
feature_importance=pd.DataFrame([test_transformed.columns,randForest.feature_importances_])
feature_importance=feature_importance.T
feature_importance.rename(columns={0:'Feature',1:'Importance'},inplace=True)
feature_importance

##Feature Importance Graph
plt.figure(figsize=(10,10))
sns.barplot(x='Feature',y='Importance',data=feature_importance, order=feature_importance.sort_values('Importance', ascending=False).Feature)
ax_ticks=plt.xticks(rotation='vertical')
```

# Fine Tuning


## Random Forest


### Max Depth

```python
max_depths=range(1,30)
f1_scores_val=[]
f1_scores_train=[]
for i in max_depths:    
    randForest = RandomForestClassifier(max_depth=i, random_state=0)
    randForest.fit(X_train, y_train)
    f1_scores_train.append(f1_score(y_train, randForest.predict(X_train), average='micro'))
    f1_scores_val.append(f1_evaluation(randForest))
```

```python
depths={'Max Depth': max_depths, 'F1 Score Train': f1_scores_train, 'F1 Score Validation': f1_scores_val}
depths=pd.DataFrame(depths)
depths['diff']=depths['F1 Score Train']-depths['F1 Score Validation']
depths
```

```python
fig, ax = plt.subplots()

depths.plot(x = 'Max Depth', y = 'F1 Score Train', ax = ax) 
depths.plot(x = 'Max Depth', y = 'F1 Score Validation', ax = ax)
```

#### MaxDepth==9

```python
randForest = RandomForestClassifier(max_depth=9, random_state=0,)
randForest.fit(X_train, y_train)
evaluate(randForest)
```

```python
#export_results(randForest,8,test_t_uncorr)
```

#### MaxDepth==11

```python
randForest = RandomForestClassifier(max_depth=11, random_state=0,)
randForest.fit(X_train, y_train)
evaluate(randForest)
```

```python
#export_results(randForest,9,test_t_uncorr)
```

### Number Estimators (Number of Trees)

```python
n_estimator=[10,20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 150, 200]
f1_scores_val=[]
f1_scores_train=[]
for i in n_estimator:    
    randForest = RandomForestClassifier(max_depth=11, n_estimators=i, random_state=0)
    randForest.fit(X_train, y_train)
    f1_scores_train.append(f1_score(y_train, randForest.predict(X_train), average='micro'))
    f1_scores_val.append(f1_evaluation(randForest))
```

```python
NEstimator={'N Estimators': n_estimator, 'F1 Score Train': f1_scores_train, 'F1 Score Validation': f1_scores_val}
NEstimator=pd.DataFrame(NEstimator)
NEstimator['diff']=NEstimator['F1 Score Train']-NEstimator['F1 Score Validation']
NEstimator
```

```python
fig, ax = plt.subplots()

NEstimator.plot(x = 'N Estimators', y = 'F1 Score Train', ax = ax) 
NEstimator.plot(x = 'N Estimators', y = 'F1 Score Validation', ax = ax)
```

## Decision Tree


### Max Depth

```python
max_depths=range(1,30)
f1_scores_val=[]
f1_scores_train=[]
for i in max_depths:    
    dt_gini = DecisionTreeClassifier(max_depth = i, random_state=0)
    dt_gini.fit(X_train, y_train)
    f1_scores_train.append(f1_score(y_train, dt_gini.predict(X_train), average='micro'))
    f1_scores_val.append(f1_evaluation(dt_gini))
```

```python
depths={'Max Depth': max_depths, 'F1 Score Train': f1_scores_train, 'F1 Score Validation': f1_scores_val}
depths=pd.DataFrame(depths)
depths['diff']=depths['F1 Score Train']-depths['F1 Score Validation']
depths
```

```python
fig, ax = plt.subplots()

depths.plot(x = 'Max Depth', y = 'F1 Score Train', ax = ax) 
depths.plot(x = 'Max Depth', y = 'F1 Score Validation', ax = ax)
```

### Max Depth==7

```python
dt_gini = DecisionTreeClassifier(max_depth = 7, random_state=0)
dt_gini.fit(X_train, y_train)
evaluate(dt_gini)
```

### Max Depth==8

```python
dt_gini = DecisionTreeClassifier(max_depth = 8, random_state=0)
dt_gini.fit(X_train, y_train)
evaluate(dt_gini)
```

## Multi Layer Perceptron

```python
hidden_layer=range(1,30)
f1_scores_val=[]
f1_scores_train=[]
for i in max_depths:    
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    f1_scores_train.append(f1_score(y_train, dt_gini.predict(X_train), average='micro'))
    f1_scores_val.append(f1_evaluation(dt_gini))
```

```python
depths={'Max Depth': max_depths, 'F1 Score Train': f1_scores_train, 'F1 Score Validation': f1_scores_val}
depths=pd.DataFrame(depths)
depths['diff']=depths['F1 Score Train']-depths['F1 Score Validation']
depths
```

```python
fig, ax = plt.subplots()

depths.plot(x = 'Max Depth', y = 'F1 Score Train', ax = ax) 
depths.plot(x = 'Max Depth', y = 'F1 Score Validation', ax = ax)
```

```python

```

```python

```

```python

```

```python
 # Random Forest
    randForest = RandomForestClassifier(max_depth=10, random_state=0,)
    randForest.fit(X_train, y_train)
    new_scores.append(f1_evaluation(randForest)) 
    
    dt_gini = DecisionTreeClassifier(max_depth = 10, random_state=0)
    dt_gini.fit(X_train, y_train)
    new_scores.append(f1_evaluation(dt_gini)) 
    
    # Multi-layer Perceptron
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    new_scores.append(f1_evaluation(mlp)) 
    
    # Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    new_scores.append(f1_evaluation(log_model)) 
    
    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    new_scores.append(f1_evaluation(knn)) 
    
    # Gaussian Model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    new_scores.append(f1_evaluation(nb_model))
```
