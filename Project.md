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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
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
    model_df = pd.DataFrame(columns=['Model_Name', 'F1_Score_Initial'])
    
    # Random Forest
    randForest = RandomForestClassifier(max_depth=10, random_state=0)
    randForest.fit(X_train, y_train)
    model_df.loc[0] = ['Random Forest', f1_evaluation(randForest)] 
    
    dt_gini = DecisionTreeClassifier(max_depth = 10, random_state=0)
    dt_gini.fit(X_train, y_train)
    model_df.loc[1] = ['Decision Tree GINI', f1_evaluation(dt_gini)] 
    
    dt_ent = DecisionTreeClassifier(max_depth = 10, random_state=0)
    dt_ent.fit(X_train, y_train)
    model_df.loc[2] = ['Decision Tree Entropy', f1_evaluation(dt_ent)] 
    
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
    
    dt_ent = DecisionTreeClassifier(max_depth = 10, random_state=0)
    dt_ent.fit(X_train, y_train)
    new_scores.append(f1_evaluation(dt_ent)) 
    
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
```

## Results Exporting

```python
results=pd.concat([test_data1, pd.DataFrame(randForest.predict(test_data1), index=test_data1.index)],axis=1)
```

```python
results=results.iloc[:,-1:]
```

```python
results.rename(columns={index:'CITIZEN\_ID'},inplace=True)
```

```python
results.index.rename('CITIZEN_ID',inplace=True)
```

```python
results.to_csv(path_or_buf='results.csv')
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
        ax.hist(np.log10(data[data[feat]!=0][feat]))
        ax.set_title(feat, y=-0.13)
    else:
        sns.boxplot(x=np.log10(data[data[feat]!=0][feat]), ax=ax)
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
    ((data['Ticket Price']>=150) | (data['Ticket Price']==10000))
    &
    ((data['Money Received']>=134) | (data['Money Received']==0))
    
)
len(data[filters1]) / len(data)
```

```python
data = data[filters1]
```

```python
reset_model_data()
```

```python
batch_model_update(df_model, 'Post-Outliers F1 Score')
```

## Variables to transform
* Age(from birthday)
* Marital Status(simplify maybe)
* employment_sector_simplified
* Education Level(The PostGraduation Paradox)
*Lives with
*Continent

```python
data['Age'] = data['Birthday'].apply(lambda x: datetime.strptime(x[-4:], "%Y").date()).astype('datetime64[ns]')
data['Age'] = data['Age'].apply(lambda x: 2048 - x.year)
```

```python
data['Marital Status'].value_counts()
```

```python
data['is_Married']= data['Marital Status'].apply(lambda x: 1 if x in ['Married - Spouse in the Army','Married', 'Married - Spouse Missing'] else 0)
```

```python
data['is_Married'].value_counts()
```

```python
data['Private Sector'] = data['Employment Sector'].apply(lambda x: 1 if 'Private' in x else 0)
```

```python
data['Public Sector'] = data['Employment Sector'].apply(lambda x: 1 if 'Public' in x else 0)
```

```python
data['Self Employed'] = data['Employment Sector'].apply(lambda x: 1 if 'Self' in x else 0)
```

```python
data['Unemployed'] = data['Employment Sector'].apply(lambda x: 1 if 'Unemployed' in x else 0)
```

```python
data['Employment Sector'].value_counts()
```

```python
data.groupby('Education Level')['Years of Education'].value_counts()
```

```python
data['Professional School']= data['Education level'].apply(lambda x: 1 if 'Professional' in x else 0)
```

## Possible variables

* is_group_X(multiple binaries)
* has_children
* years_of_edu_per_age
* Education Level(The PostGraduation Paradox)
* gender (from the title that comes before the name) 

```python
data['years_of_edu_per_age'] = data['Years of Education'] / data['Age']
```

```python
data['Name'].apply(lambda x: x.split(' ')[0]).value_counts()
```

```python
data['is_Male']= data['Name'].apply(lambda x: 1 if x.split(' ')[0]=='Mr.' else 0)
```

```python
data['is_group_a'] = data['Ticket Price'] + data['Money Received']
data['is_group_a'] = data['is_group_a'].apply(lambda x: 1 if x == 0 else 0)
data['is_group_b'] = data['Money Received'].apply(lambda x: 1 if x > 0 else 0)
data['is_group_c'] = data['Ticket Price'].apply(lambda x: 1 if x > 0 else 0)
data[['is_group_a', 'is_group_b', 'is_group_c']]
```
