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
<<<<<<< Updated upstream
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold, RFECV, SelectFromModel
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
=======
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier, BaggingClassifier
from sklearn.feature_selection import VarianceThreshold, RFE, RFECV, SelectFromModel
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
>>>>>>> Stashed changes
from sklearn.svm import SVC
from sklearn.decomposition import PCA


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.tree import export_graphviz
import graphviz
import pydotplus

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
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

Because we need to handle metric and non-metrical data in a different way, let us create a way to filter them in case it is necessary.

```python
get_metric_features = FunctionTransformer(lambda x: x.select_dtypes(include=np.number))
metric_features = get_metric_features.fit_transform(data).columns
```

```python
get_metric_features.fit_transform(data).head()
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
data.groupby('Base Area')['Income'].value_counts().sort_values()
```

```python
data['Employment Sector'].value_counts()
```

```python
def Sectorsector(string):
    if ' - ' in string:
        return string.split(' - ')[1]
    elif ' (' in string:
        return string.split('(')[1][:-1]
    else:
        return string
```

```python
data['Employment Sector'].apply(lambda x: Sectorsector(x)).value_countscounts()
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
data['Working Hours per week'].map(lambda x: x>data['Working Hours per week'].mean())
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
def metrics(model, X_train, X_val, y_train, y_val):
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    
    print(classification_report(y_train, pred_train))
    # print(confusion_matrix(y_train, pred_train))
    
    print(classification_report(y_val, pred_val))
    # print(confusion_matrix(y_val, pred_val))
    
def evaluate(model, X_train, X_val, y_train, y_val):
    metrics(model, X_train, X_val, y_train, y_val)
    f1_micro = f1_evaluation(X_val, y_val, model)
    print(f'The Micro Average of the F1 Score is : {f1_micro}')
    
def f1_evaluation(X_val, y_val, model):
    return f1_score(y_val, model.predict(X_val), average='micro')
```

```python
def batch_model_creation():
    model_df = pd.DataFrame(columns=['Model_Name', 'Initial'])

    input_data = get_metric_features.fit_transform(data)
    target = input_data['Income']
    X = input_data.drop(columns='Income')

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      target,
                                                      test_size=0.25,
                                                      stratify=target,
                                                      random_state=35)
    # Random Forest
    randForest = RandomForestClassifier(max_depth=10, random_state=0)
    randForest.fit(X_train, y_train)
    model_df.loc[0] = [
        'Random Forest',
        f1_evaluation(X_val, y_val, randForest)
    ]

    #Decision Tree
    dt_gini = DecisionTreeClassifier(max_depth=10, random_state=0)
    dt_gini.fit(X_train, y_train)
    model_df.loc[1] = [
        'Decision Tree GINI',
        f1_evaluation(X_val, y_val, dt_gini)
    ]

    # Multi-layer Perceptron
    mlp = MLPClassifier(random_state=0)
    mlp.fit(X_train, y_train)
    model_df.loc[2] = [
        'Multi-layer Perceptron',
        f1_evaluation(X_val, y_val, mlp)
    ]

    # Logistic Regression
    log_model = LogisticRegression(random_state=0)
    log_model.fit(X_train, y_train)
    model_df.loc[3] = [
        'Logistic Regression',
        f1_evaluation(X_val, y_val, log_model)
    ]

    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    model_df.loc[4] = ['K-Nearest Neighbors', f1_evaluation(X_val, y_val, knn)]

    # Gaussian Model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    model_df.loc[5] = ['GaussianNB', f1_evaluation(X_val, y_val, nb_model)]

    # AdaBoost
    ada_model = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada_model.fit(X_train, y_train)
    model_df.loc[6] = ['AdaBoost', f1_evaluation(X_val, y_val, ada_model)]

    # Gradiant Boosting
    grad_model = GradientBoostingClassifier(random_state=0)
    grad_model.fit(X_train, y_train)
    model_df.loc[7] = [
        'GradientBoost',
        f1_evaluation(X_val, y_val, grad_model)
    ]

    return model_df


def batch_model_update(data_steps, model_steps, model_df, score_name):
    new_scores = []

    input_data = data_steps.fit_transform(data)

    target = input_data['Income']
    X = input_data.drop(columns='Income')

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      target,
                                                      test_size=0.25,
                                                      stratify=target,
                                                      random_state=35)

    # Random Forest
    randForest = Pipeline([('Classifier',
                            RandomForestClassifier(max_depth=10,
                                                   random_state=0))])
    for i, s in enumerate(model_steps):
        randForest.steps.insert(i, (str(i), s))
    print(randForest.steps)
    randForest.fit(X_train, y_train)
    new_scores.append(f1_evaluation(X_val, y_val, randForest))

    # Decision Tree
    dt_gini = Pipeline([('Classifier',
                         DecisionTreeClassifier(max_depth=10,
                                                random_state=0))])
    for i, s in enumerate(model_steps):
        dt_gini.steps.insert(i, (str(i), s))
    dt_gini.fit(X_train, y_train)
    new_scores.append(f1_evaluation(X_val, y_val, dt_gini))

    # Multi-layer Perceptron
    mlp = Pipeline([('Classifier', MLPClassifier(random_state=0))])
    for i, s in enumerate(model_steps):
        mlp.steps.insert(i, (str(i), s))
    mlp.fit(X_train, y_train)
    new_scores.append(f1_evaluation(X_val, y_val, mlp))

    # Logistic Regression
    log_model = Pipeline([('Classifier', LogisticRegression(random_state=0))])
    for i, s in enumerate(model_steps):
        log_model.steps.insert(i, (str(i), s))
    log_model.fit(X_train, y_train)
    new_scores.append(f1_evaluation(X_val, y_val, log_model))

    # K-Nearest Neighbors
    knn = Pipeline([('Classifier', KNeighborsClassifier())])
    for i, s in enumerate(model_steps):
        knn.steps.insert(i, (str(i), s))
    knn.fit(X_train, y_train)
    new_scores.append(f1_evaluation(X_val, y_val, knn))

    # Gaussian Model
    nb_model = Pipeline([('Classifier', GaussianNB())])
    for i, s in enumerate(model_steps):
        nb_model.steps.insert(i, (str(i), s))
    nb_model.fit(X_train, y_train)
    new_scores.append(f1_evaluation(X_val, y_val, nb_model))

    # AdaBoost
    ada_model = Pipeline([('Classifier',
                           AdaBoostClassifier(n_estimators=100,
                                              random_state=0))])
    for i, s in enumerate(model_steps):
        ada_model.steps.insert(i, (str(i), s))
    ada_model.fit(X_train, y_train)
    new_scores.append(f1_evaluation(X_val, y_val, ada_model))

    # Gradient Boosting
    grad_model = Pipeline([('Classifier',
                            GradientBoostingClassifier(random_state=0))])
    for i, s in enumerate(model_steps):
        grad_model.steps.insert(i, (str(i), s))
    grad_model.fit(X_train, y_train)
    new_scores.append(f1_evaluation(X_val, y_val, grad_model))

    new_scores = pd.Series(data=new_scores,
                           name=score_name,
                           index=model_df.index)
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
title = "Test Numeric Variables' Histograms"
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

```python
# All Numeric Variables' Box Plots in one figure
sns.set()
# Prepare figure. Create individual axes where each box plot will be placed
fig, axes = plt.subplots(2, int(len(metric_features) / 2), figsize=(20, 11))

# Plot data
# Iterate across axes objects and associate each box plot (hint: use the ax argument):
for ax, feat in zip(axes.flatten(), metric_features): # Notice the zip() function and flatten() method
    sns.boxplot(x=test_data[feat], ax=ax)
    
# Layout
# Add a centered title to the figure:
title = "Test Numeric Variables' Box Plots"
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
len(data[filters1])
```

```python
outlier_filter_transformer = FunctionTransformer(lambda x: x[filters1])
```

#### Modeling no Outliers

```python
no_outliers_pipeline = Pipeline([
    ('Outlier Remove', outlier_filter_transformer),
    ('Get Metric Features', get_metric_features)
])
```

```python
df_model = batch_model_update(model_steps=[],
                   data_steps=no_outliers_pipeline,
                   model_df=df_model,
                   score_name='No Outliers')
df_model
```

## Variables to transform
* Age(from birthday)
* Marital Status(simplify maybe)
* employment_sector_simplified
* Education Level(The PostGraduation Paradox)
*Lives with
*Continent

```python
def is_married(df):
    df = df.copy()
    df['is_Married']= df['Marital Status'].apply(lambda x: 1 if x in ['Married - Spouse in the Army','Married', 'Married - Spouse Missing'] else 0)
    return df.drop(columns='Marital Status')

is_married_transformer = FunctionTransformer(is_married)
```

```python
def sector_binning(df):
    df = df.copy()
    df['Private Sector'] = df['Employment Sector'].apply(lambda x: 1 if 'Private' in x else 0)
    df['Public Sector'] = df['Employment Sector'].apply(lambda x: 1 if 'Public' in x else 0)
    df['Self Employed'] = df['Employment Sector'].apply(lambda x: 1 if 'Self' in x else 0)
    df['Unemployed'] = df['Employment Sector'].apply(lambda x: 1 if 'Unemployed' in x else 0)
    return df.drop(columns=['Employment Sector'])

sector_binning_transformer = FunctionTransformer(sector_binning)
```

```python
data.groupby('Education Level')['Years of Education'].value_counts()
```

```python
def is_professional(df):
    df = df.copy()
    df['Professional School'] = df['Education Level'].apply(lambda x: 1 if 'Professional' in x else 0)
    return df.drop(columns=['Education Level'])

is_professional_transformer = FunctionTransformer(is_professional)
```

```python
data['Native Continent'].value_counts()
```

```python
def continent_dummies(df):
    df = df.copy()
    df = df.drop(columns='Native Continent').merge(pd.get_dummies(
        df['Native Continent'], prefix='Continent').iloc[:, :-1],
                                                       on=df.index,
                                                       left_index=True)
    return df.drop(columns = ['key_0'])

continent_encoder = FunctionTransformer(continent_dummies)
```

```python
def lives_with_bins(df):
    df = df.copy()
    df['Lives_Spouse']=df['Lives with'].apply(lambda x: 1 if x in['Husband','Wife'] else 0)
    df['Lives_Children']=df['Lives with'].apply(lambda x: 1 if x in 'Children' else 0)
    df['Lives_Other']=df['Lives with'].apply(lambda x: 1 if 'Other' in x else 0)
    return df.drop(columns = 'Lives with')

lives_with_encoder = FunctionTransformer(lives_with_bins)
```

```python
def lives_northbury(df):
    df = df.copy()
    df['Lives_Northbury']=df['Base Area'].apply(lambda x: 1 if 'Northbury' in x else 0)
    return df.drop(columns = 'Base Area')

lives_northbury_encoder = FunctionTransformer(lives_northbury)
```

```python
def AbvAvgWH(df):
    df['Abv Avg WH per Role']=0

    for val in df['Role'].unique():
        index=df[df['Role']==val].index
        mean=df[df['Role']==val]['Working Hours per week'].mean()
        df.loc[index,'Abv Avg WH per Role'] =df.loc[index,'Working Hours per week'].apply(lambda x: 1 if x>mean else 0)
    return df

abv_avg_wh_role=FunctionTransformer(AbvAvgWH)
```

```python
def AbvAvgYoE(df):
    df['Abv Avg YoE per Role']=0

    for val in df['Role'].unique():
        index=df[df['Role']==val].index
        mean=df[df['Role']==val]['Years of Education'].mean()
        df.loc[index,'Abv Avg YoE per Role'] =df.loc[index,'Years of Education'].apply(lambda x: 1 if x>mean else 0)
    return df

abv_avg_yoe_role=FunctionTransformer(AbvAvgYoE)
```

```python
def role_dummies(df):
    df = df.copy()
    df = df.drop(columns='Role').merge(pd.get_dummies(
        df['Role'], prefix='Role').iloc[:, :-1],
                                       on=df.index,
                                       left_index=True)
    
    return df.drop(columns='key_0')

role_enconder = FunctionTransformer(role_dummies)
```

```python
variable_encoder = Pipeline([
    ('is_married', is_married_transformer),
    ('Sector', sector_binning_transformer),
    ('Professional School', is_professional_transformer),
    ('Native Continent', continent_encoder),
    ('Lives With', lives_with_encoder),
    ('Lives in Northbury', lives_northbury_encoder),
    ('Above Average Working Hours per role', abv_avg_wh_role),
    ('Above Average YoE per role', abv_avg_yoe_role),
    ('Role', role_enconder)
])

variable_encoder.fit_transform(data).columns
```

## Model with Encoded Vars

```python
variable_encoding_pipeline = Pipeline([
    ('Variable Encoding', variable_encoder),
    ('Metric Features', get_metric_features)
])

df_model = batch_model_update(data_steps=variable_encoding_pipeline,
                   model_steps=[],
                   model_df=df_model,
                   score_name='Variable Encoding')
df_model
```

```python
# export_results(randForest,3,test_data_encoded)
```

### Encoded Vars no outliers

```python
encoded_no_out_pipeline = Pipeline([
    ('Remove Outliers', outlier_filter_transformer),
    ('Encode Variables', variable_encoder),
    ('Outlier Remove', outlier_filter_transformer),
    ('Get Metric Features', get_metric_features)
])
```

```python
df_model = batch_model_update(data_steps = encoded_no_out_pipeline,
                   model_steps = [],
                   model_df = df_model,
                   score_name = 'No Outliers Encoded')
df_model
```

# Feature Extraction

* is_group_X(multiple binaries)
* has_children
* years_of_edu_per_age
* Education Level(The PostGraduation Paradox)
* gender (from the title that comes before the name) 

```python
def calculate_age(df):
    df = df.copy()
    df['Age'] = df['Birthday'].apply(lambda x: datetime.strptime(x[-4:], "%Y").date()).astype('datetime64[ns]')
    df['Age'] = df['Age'].apply(lambda x: 2048 - x.year)
    return df.drop(columns='Birthday')

age_transformer = FunctionTransformer(calculate_age)
```

```python
def calculate_edu_per_age(df):
    df = df.copy()
    df['Education per Age'] = df['Years of Education'] / df['Age']
    return df

edu_per_age_transformer = FunctionTransformer(calculate_edu_per_age)
```

```python
def calculate_gender(df):
    df = df.copy()
    df['is_Male']= df['Name'].apply(lambda x: 1 if x.split(' ')[0]=='Mr.' else 0)
    return df.drop(columns='Name')

gender_transformer = FunctionTransformer(calculate_gender)
```

```python
def calculate_group(df):
    df = df.copy()
    df['is_group_a'] = df['Ticket Price'] + df['Money Received']
    df['is_group_a'] = df['is_group_a'].apply(lambda x: 1 if x == 0 else 0)
    df['is_group_b'] = df['Money Received'].apply(lambda x: 1 if x > 0 else 0)
    df['is_group_c'] = df['Ticket Price'].apply(lambda x: 1 if x > 0 else 0)
    return df

group_transformer = FunctionTransformer(calculate_group)
```

```python
variable_transformer = Pipeline([
    ('Variable Encoding', variable_encoder),
    ('Age', age_transformer),
    ('Education Per Age', edu_per_age_transformer),
    ('Gender', gender_transformer),
    ('Groups', group_transformer)
])

variable_transformer.fit_transform(data).columns
```

### Model with new Transformed Variables

```python
df_model = batch_model_update(data_steps = variable_transformer,
                             model_steps = [],
                             model_df = df_model,
                             score_name = 'Transformed Variables')
df_model
```

```python
# export_results(randForest,5,test_transformed)
```

## Model with Transformed Vars with no Outliers

```python
transformed_no_out_pipeline = Pipeline([
    ('Variable Transformation', variable_transformer),
    ('Outlier Removal', outlier_filter_transformer)
])
```

```python
df_model = batch_model_update(data_steps=transformed_no_out_pipeline,
                              model_steps=[],
                              model_df=df_model,
                              score_name='Transformed Variables No Outliers')
df_model
```

```python
df_model.drop(columns=['No Outliers Encoded', 'Transformed Variables No Outliers'], inplace=True)
```

## Correlation Analysis

```python
data_transformed_pipe = Pipeline([
    ('Variable Transformation', variable_transformer)
])

data_transformed = data_transformed_pipe.fit_transform(data)
```

```python
# Prepare figure
fig = plt.figure(figsize=(20, 20))
# Obtain correlation matrix. Round the values to 2 decimal cases. Use the DataFrame corr() and round() method.
corr = np.round(data_transformed.corr(method="pearson"), decimals=2)
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
corr['Money Received'].abs().sort_values(ascending=False)
```

```python
corr['Income'].abs().sort_values(ascending=False)
```

```python
remove_corr = FunctionTransformer(lambda x: x.drop(columns=['is_group_c', 'is_Married', 'Money Received per YoE', 'Money Received per Age', 'Ticket Price per YoE', 'Ticket Price per Age', 'Private Sector']))
```

```python
remove_uncorrelated = FunctionTransformer(lambda x: x.drop(columns=['Role_Army','Continent_Asia']))
```

# Adaboost 0.8626

```python
data_transformed = Pipeline([
    ('Variable Transformation', variable_transformer),
    ('Correlation Removal', remove_corr),
]).fit_transform(data)

target = data_transformed['Income']
X = data_transformed.drop(columns='Income')

X_train, X_val, y_train, y_val = train_test_split(X,
                                                  target,
                                                  test_size=0.25,
                                                  stratify=target,
                                                  random_state=35)
```

```python
AdaBoost = Pipeline([
    ('Scaling', MinMaxScaler()),
    ('Classifier', AdaBoostClassifier(n_estimators=1000,
                                      base_estimator=DecisionTreeClassifier(max_depth=1),
                                      algorithm="SAMME.R"))
])
AdaBoost.fit(X_train,y_train)

evaluate(AdaBoost,X_train, X_val, y_train, y_val)
```

```python
test_data_transformed = Pipeline([
    ('Transformation', variable_transformer),
    ('Correlation Removal', remove_corr),
]).fit_transform(test_data)
```

```python
<<<<<<< Updated upstream
#export_results(AdaBoost, 12, test_data_transformed)
=======
#generate_report(data_transformed, 'reports/citizen_profiling_after_transformation')
```

# Peixoto
# Gradiend Grid Search

```python

```

```python

```

```python

```

```python

```

```python
GradBoost = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',GradientBoostingClassifier(random_state=10,
                                                               n_estimators=70,
                                                               min_samples_split=300,
                                                              min_samples_leaf=40,
                                                              max_depth=9,
                                                              max_features=11,
                                                               learning_rate=0.2,
                                                               subsample=0.8))
                     ])

parameters = {
    'Classifier__subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
    
    
}

search = GridSearchCV(GradBoost, parameters, estimator=f1_score, verbose=10)
search.fit(X, target)
```

```python
search.best_params_, search.best_score_
```

# Tuned GradBoost

```python
GradBoost = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',GradientBoostingClassifier(random_state=10,
                                                               n_estimators=1400,
                                                               min_samples_split=300,
                                                              min_samples_leaf=60,
                                                              max_depth=7,
                                                              max_features=11,
                                                               learning_rate=0.01,
                                                               subsample=0.8))
                     ])

GradBoost.fit(X_train, y_train)
evaluate(GradBoost, X_train, X_val, y_train, y_val)
```

```python
#export_results(GradBoost, 21, test_data_transformed)
```

# Adaboost Grid Search

```python
ada_model = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=5, n_estimators = 10, random_state=0),
                              n_estimators=50,
                              random_state=0)
ada_model.fit(X_train, y_train)

evaluate(ada_model, X_train, X_val, y_train, y_val)
```

```python
AdaBoost = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',AdaBoostClassifier(n_estimators=30,
                                                       random_state=10,
                                                       learning_rate=0.2,
                                                      base_estimator=RandomForestClassifier(max_depth=5,
                                                                                            n_estimators = 10,
                                                                                            random_state=10)))
                     ])

parameters = {
    'Classifier__base_estimator': [RandomForestClassifier(max_depth=5,n_estimators = 10,random_state=10),
                                  RandomForestClassifier(max_depth=15,n_estimators = 60,random_state=10),
                                  RandomForestClassifier(max_depth=10,n_estimators = 30,random_state=10)]
    
    
}

search = GridSearchCV(AdaBoost, parameters, estimator=f1_score, verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

# Tuned Adaboost

```python
AdaBoost = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',AdaBoostClassifier(n_estimators=100,
                                                       random_state=10,
                                                       learning_rate=0.2,
                                                      base_estimator=RandomForestClassifier(max_depth=5, 
                                                                                            n_estimators=70, 
                                                                                            random_state=10)))
                     ])

AdaBoost.fit(X_train, y_train)

evaluate(AdaBoost, X_train, X_val, y_train, y_val)
```

# HistGrad Grid Search

```python
hist_grad = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', HistGradientBoostingClassifier(random_state=10,
                                                  max_iter=50,
                                                  learning_rate=0.2
                                                 ))
])

parameters = {
    'Classifier__max_iter' : range(20,81,10)
}

search = GridSearchCV(hist_grad, parameters,scoring=f1_scorer, verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

# HistGrad Tuned

```python
hist_grad = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', HistGradientBoostingClassifier(random_state=10,max_iter=50,learning_rate=0.2))
])

hist_grad.fit(X_train, y_train)
evaluate(hist_grad, X_train, X_val, y_train, y_val)
```

# Rand Forest GridSearch

```python
RandForest = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', RandomForestClassifier(random_state=10,
                                          n_estimators=100, 
                                          max_depth=9,
                                          min_samples_split=70,
                                      
                                         ))
])

parameters = {
    'Classifier__n_estimators' : range(130,155,10),
    'Classifier__max_features' : range(30,34,1),
    'Classifier__max_depth' : range(15,20,2),
    
}

search = GridSearchCV(RandForest, parameters,scoring=f1_scorer, verbose=10,cv=4)
search.fit(X_train, y_train)

search.best_params_, search.best_score_
```

```python
RandForest = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', RandomForestClassifier(random_state=10,
                                          n_estimators=130, 
                                          max_depth=19,
                                          min_samples_split=80,
                                          max_features=31,
                                         ))
])

RandForest.fit(X_train, y_train)
evaluate(RandForest, X_train, X_val, y_train, y_val)
```

# GradBoost Bagging

```python
grad=Pipeline([('Scaler', MinMaxScaler()),
               ('Classifier',GradientBoostingClassifier(random_state=10,
                                                        n_estimators=70,
                                                        learning_rate=0.2,
                                                        warm_start=True,
                                                       min_samples_split=350,
                                                       min_samples_leaf=60,
                                                       max_depth=9,
                                                       max_features=11,
                                                       subsample=0.8,
                                                       ccp_alpha=0.00001
                                                                      ))
                     ])

grad_bag = BaggingClassifier(base_estimator=grad,
                           random_state=10, 
                           n_estimators = 3, 
                           bootstrap=True)

grad_bag.fit(X_train, y_train)
evaluate(grad_bag, X_train, X_val, y_train, y_val)
```

```python
GradBoost = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',GradientBoostingClassifier(random_state=10,
                                                                       n_estimators=70,
                                                                        learning_rate=0.2,
                                                                        warm_start=True,
                                                                       min_samples_split=350,
                                                                       min_samples_leaf=60,
                                                                       max_depth=9,
                                                                       max_features=11,
                                                                       subsample=0.8,
                                                                       ccp_alpha=0.00001
                                                                      ))
                     ])

GradBoost.fit(X_train, y_train)
evaluate(GradBoost, X_train, X_val, y_train, y_val)
```

# Adaboost com Bagging La dentro

```python
AdaBoost = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',AdaBoostClassifier(n_estimators=10,
                                                       random_state=10,
                                                       learning_rate=0.2,
                                                      base_estimator=grad_bag))
                     ])

AdaBoost.fit(X_train, y_train)

evaluate(AdaBoost, X_train, X_val, y_train, y_val)
```

# Ensemble

```python
ensembler = StackingClassifier([
    ('GradBoost', grad_bag),
    ('AdaBoost', AdaBoost),
    ('Hist Grad Boost', hist_grad),
    #('RandForest', RandForest),
    
])
ensembler.fit(X_train, y_train)
evaluate(ensembler, X_train, X_val, y_train, y_val)
```

```python
ensembler = StackingClassifier([
    ('Grad_Boost', GradBoost),
    ('AdaBoost', AdaBoost),
    ('Hist Grad Boost', hist_grad),
    ('RandForest', RandForest),
    
])
ensembler.fit(X_train, y_train)
evaluate(ensembler, X_train, X_val, y_train, y_val)
```

```python
export_results(ensembler, 24, test_data_transformed)
```

# Bagging Weak Learners

```python
log_reg = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',LogisticRegression(random_state=10,))
                     ])

weak_bag=BaggingClassifier(base_estimator=log_reg,
                           random_state=10,
                           n_estimators =10,
                           warm_start=True,
                           bootstrap=True)

weak_bag.fit(X_train, y_train)
evaluate(weak_bag, X_train, X_val, y_train, y_val)
```

```python
mlp_pipe = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',MLPClassifier(activation='relu', hidden_layer_sizes=(30,),random_state=10))
                     ])

weak_bag=BaggingClassifier(base_estimator=mlp_pipe,
                           random_state=10,
                           n_estimators =5,
                           warm_start=True,
                           bootstrap=True)

weak_bag.fit(X_train, y_train)
evaluate(weak_bag, X_train, X_val, y_train, y_val)
>>>>>>> Stashed changes
```

# Adaboost not Tested

```python
<<<<<<< Updated upstream
AdaBoost = Pipeline([
    ('Scaling', MinMaxScaler()),
    ('Classifier', AdaBoostClassifier(n_estimators=1000,
                                      base_estimator=DecisionTreeClassifier(max_depth=1),
                                      algorithm="SAMME.R"))
])
AdaBoost.fit(X_train,y_train)

evaluate(AdaBoost,X_train, X_val, y_train, y_val)
=======
ensembler = StackingClassifier([
    ('Grad_Boost', GradBoost),
    ('MLP', mlp_pipe),
    ('Hist Grad Boost', hist_grad),
    ('LogReg', log_reg),
    
    
    
])
ensembler.fit(X_train, y_train)
evaluate(ensembler, X_train, X_val, y_train, y_val)
```

```python
export_results(ensembler, 29, test_data_transformed)
```

```python

```

# Peixoto End

```python
mlp = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', MLPClassifier(random_state=0))
])
parameters = {
    'Classifier__hidden_layer_sizes' : [(30, 20), (30,20, 10)],
    'Classifier__activation' : ['tanh'],
    'Classifier__learning_rate' : ['adaptive'],
    'Classifier__max_iter' : [100, 200],
    'Classifier__early_stopping' : [True]
}

search = GridSearchCV(estimator=mlp, parameters, estimator=f1_score, verbose=10)
search.fit(X, target)
>>>>>>> Stashed changes
```

```python
test_data_transformed = Pipeline([
    ('Transformation', variable_transformer),
    ('Correlation Removal', remove_corr)
]).fit_transform(test_data)
```

```python
#export_results(AdaBoost, 12, test_data_transformed)
```

```python
data_transformed = Pipeline([
    ('Variable Transformation', variable_transformer),
    ('Correlation Removal', remove_corr),
]).fit_transform(data)

target = data_transformed['Income']
X = data_transformed.drop(columns='Income')

grid_pipe = Pipeline([
    ('Scaling', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state = 0))
])

parameters = {
    'Classifier__learning_rate': [0.01, 0.05, 0.1],
    'Classifier__n_estimators' : [30, 50, 100, 150, 200, 500],
    'Classifier__criterion' : ['friedman_mse', 'mse', 'mae'],
    'Classifier__max_depth' : [3, 5, 9, 11, 20],
    'Classifier__max_features' : ['auto', 'sqrt', 'log2', None],
    
}

search = GridSearchCV(grid_pipe, parameters, n_jobs=-1, estimator=f1_score)
search.fit(X, target)
```

## Data Standartization

```python
data_pipeline = Pipeline([
    ('Variable Transformation', variable_transformer),
    ('Correlation Removal', remove_corr)
])

model_pipeline = Pipeline([
    ('Scaler', MinMaxScaler()),
])

df_model = batch_model_update(data_steps=data_pipeline,
                              model_steps=model_pipeline,
                              model_df=df_model,
                              score_name='Scaled')
df_model
```

## Data Standartization no Outliers

```python
data_pipeline = Pipeline([
    ('Variable Transformation', variable_transformer),
    ('Correlation Removal', remove_corr)
    ('Outlier Removal', outlier_filter_transformer)
])

model_pipeline = Pipeline([
    ('Scaler', MinMaxScaler()),
])

df_model = batch_model_update(data_steps=data_pipeline,
                              model_steps=model_pipeline,
                              model_df=df_model,
                              score_name='Scaled NO')
df_model
```

## Feature Selection

```python
data_pipeline = Pipeline([
    ('Variable Transformation', variable_transformer),
])

model_pipeline = Pipeline([
    ('Impurity Based Feature Selection', SelectFromModel(ExtraTreesClassifier(n_estimators=100))),
])

batch_model_update(data_steps=data_pipeline,
                              model_steps=model_pipeline,
                              model_df=df_model,
                              score_name='Selection with Tree')

```

#### Standardization First, VarianceThreshold After

```python
data_pipeline = Pipeline([
    ('Variable Transformation', variable_transformer),
])

model_pipeline = Pipeline([
    ('Standardization', MinMaxScaler()),
    ('Feature Selection', VarianceThreshold(threshold=.8 * (1 - .8))),
])

batch_model_update(data_steps=data_pipeline,
                   model_steps=model_pipeline,
                   model_df=df_model,
                   score_name='STD-VT')
```

#### VarianceThreshold First, Standardization After

```python
data_pipeline = Pipeline([
    ('Variable Transformation', variable_transformer),
])

model_pipeline = Pipeline([
    ('Feature Selection', VarianceThreshold(threshold=.8 * (1 - .8))),
    ('Standardization', MinMaxScaler()),
])

batch_model_update(data_steps=data_pipeline,
                   model_steps=model_pipeline,
                   model_df=df_model,
                   score_name='VT-STD')
```

#### Std first, RFE after

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


### Grid Search

```python
randForest = RandomForestClassifier()

parameters = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [5, 9, 10, 11, 20],
    'warm_start' : [True, False],
    'max_features' : ['auto', 'sqrt', 'log2'],
    'bootstrap' : [True, False],
    'min_impurity_decrease' : [0, 0.01, 0.1, 0.2, 0.5]
}

search = GridSearchCV(randForest, parameters, estimator = f1_score)
search.fit(X, target)
```

```python
RandomForestClassifier().get_params().keys()
```

### Max Depth

```python
max_depths=range(1,30)
f1_scores_val=[]
f1_scores_train=[]
for i in max_depths:    
    randForest = RandomForestClassifier(max_depth=i, random_state=0,)
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

### Warm Start

```python
warmStart=[True, False]
f1_scores_val=[]
f1_scores_train=[]
for start in warmStart:    
    randForest = RandomForestClassifier(max_depth=11, warm_start=start)
    randForest.fit(X_train, y_train)
    f1_scores_train.append(f1_score(y_train, randForest.predict(X_train), average='micro'))
    f1_scores_val.append(f1_evaluation(randForest))
```

```python
start={'Warm Start': warmStart, 'F1 Score Train': f1_scores_train, 'F1 Score Validation': f1_scores_val}
start=pd.DataFrame(start)
start['diff']=start['F1 Score Train']-start['F1 Score Validation']
start
```

```python

```

## Decision Tree


### Max Depth

```python
max_depths=range(1,30)
f1_scores_val=[]
f1_scores_train=[]
for i in max_depths:    
    dt_gini = DecisionTreeClassifier(max_depth = i, random_state=10)
    dt_gini.fit(X_train, y_train)
    f1_scores_train.append(f1_score(y_train, dt_gini.predict(X_train), average='micro'))
    f1_scores_val.append(f1_evaluation(X_val,y_val, model=dt_gini))
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


Grid Search

```python
mlp = MLPClassifier()

parameters={
    'activation': ['relu','logistic','tanh'],
    'hidden_layer_sizes' : [(10,10,10),(30,),(20,10),(30,10),(100,)],
    'alpha':[0.0001,0.001,0.01,0.05],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.0001,0.001,0.01,0.05]
}

search = GridSearchCV(mlp, parameters, estimator = f1_score)
search.fit(X, target)
```

```python

```

### Activation

```python
activations=['relu','logistic','tanh']
f1_scores_val=[]
f1_scores_train=[]
for funct in activations:    
    mlp = MLPClassifier(activation=funct)
    mlp.fit(X_train, y_train)
    f1_scores_train.append(f1_score(y_train, mlp.predict(X_train), average='micro'))
    f1_scores_val.append(f1_evaluation(mlp))
```

```python
Activation={'Function': activations, 'F1 Score Train': f1_scores_train, 'F1 Score Validation': f1_scores_val}
Activation=pd.DataFrame(Activation)
Activation['diff']=Activation['F1 Score Train']-Activation['F1 Score Validation']
Activation
```

```python
fig, ax = plt.subplots()

Activation.plot(x = 'Function', y = 'F1 Score Train', ax = ax) 
Activation.plot(x = 'Function', y = 'F1 Score Validation', ax = ax)
```

### Hidden Layer Size

```python
hidden_layers=[(30,), (10,10,10), (20,10),(30,10,10), (100,)]
hidden=[]
func=[]
activations=['relu','logistic','tanh']

f1_scores_val=[]
f1_scores_train=[]
for funct in activations:
    for layers in hidden_layers:
        mlp = MLPClassifier(activation=funct, hidden_layer_sizes=layers)
        mlp.fit(X_train, y_train)
        f1_scores_train.append(f1_score(y_train, mlp.predict(X_train), average='micro'))
        f1_scores_val.append(f1_evaluation(mlp))
        hidden.append(layers)
        func.append(funct)
```

```python
Activation={'Function': func,'Hidden Layers': hidden, 'F1 Score Train': f1_scores_train, 'F1 Score Validation': f1_scores_val}
Activation=pd.DataFrame(Activation)
Activation['diff']=Activation['F1 Score Train']-Activation['F1 Score Validation']
Activation
```

```python
fig, ax = plt.subplots()

Activation.plot(x = 'Hidden Layers', y = 'F1 Score Train', ax = ax) 
Activation.plot(x = 'Hidden Layers', y = 'F1 Score Validation', ax = ax)
```

### Learning Rate, Solver, Learning Rate init

```python
parameter_space = {
    'solver': ['sgd', 'adam'],
    'learning_rate_init': list(np.linspace(0.00001,0.1,5)),
    'learning_rate': ['constant','adaptive']
}
```

```python
mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(30,))

clf = GridSearchCV(mlp, parameter_space, estimator=f1_score)
clf.fit(X_train, y_train)
```

```python
clf.best_params_
```

```python
clf.best_score_
```

```python
pd.DataFrame(clf.cv_results_)
```

### Solver lbfgs

```python
mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(30,), solver='lbfgs')
mlp.fit(X_train, y_train)
```

```python
evaluate(mlp)
```

# Boosting


# Adaboost 0.8626

```python
data_transformed = Pipeline([
    ('Variable Transformation', variable_transformer),
    ('Correlation Removal', remove_corr),
]).fit_transform(data)

target = data_transformed['Income']
X = data_transformed.drop(columns='Income')

X_train, X_val, y_train, y_val = train_test_split(X,
                                                  target,
                                                  test_size=0.25,
                                                  stratify=target,
                                                  random_state=35)
```

```python
AdaBoost = Pipeline([
    ('Scaling', MinMaxScaler()),
    ('Classifier', AdaBoostClassifier(n_estimators=1000,
                                      base_estimator=DecisionTreeClassifier(max_depth=1),
                                      algorithm="SAMME.R"))
])
AdaBoost.fit(X_train,y_train)

evaluate(AdaBoost,X_train, X_val, y_train, y_val)
```

```python
test_data_transformed = Pipeline([
    ('Transformation', variable_transformer),
    ('Correlation Removal', remove_corr)
]).fit_transform(test_data)
```

```python
#export_results(AdaBoost, 12, test_data_transformed)
```

# Gradboost

```python
GradBoost = Pipeline([
    ('Scaling', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=10,
                                              n_iter_no_change=100,
                                             learning_rate=0.1,
                                             n_estimators=500,
                                             max_depth=3,
                                             max_features='sqrt',
                                             validation_fraction=0.2,
                                             tol=0.001
                                             ))
])
GradBoost.fit(X_train,y_train)

evaluate(GradBoost,X_train, X_val, y_train, y_val)
```

```python
GradBoost['Classifier'].feature_importances_
```

## Feature Importance

```python
feature_importance=pd.DataFrame([test_data_transformed.columns,HistBoost['Histogram-Based Gradient Boosting'].feature_importances_])
feature_importance=feature_importance.T
feature_importance.rename(columns={0:'Feature',1:'Importance'},inplace=True)
feature_importance

##Feature Importance Graph
plt.figure(figsize=(10,10))
sns.barplot(x='Feature',y='Importance',data=feature_importance, order=feature_importance.sort_values('Importance', ascending=False).Feature)
ax_ticks=plt.xticks(rotation='vertical')
```

```python
remove_unimportant = FunctionTransformer(lambda x: x.drop(columns=['Role_Household Services', 'Role_Army', ]))
```

```python
def avg_score_GB(method,model,X,y):
    score_train = []
    score_test = []
    for train_index, test_index in method.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = model.fit(X_train,y_train)
        value_train = f1_evaluation(X_train, y_train, model)
        value_test = f1_evaluation(X_test,y_test, model)
        score_train.append(value_train)
        score_test.append(value_test)

    print('Train:', np.mean(score_train))
    print('Test:', np.mean(score_test))
```

```python
data_transformed = Pipeline([
    ('Variable Transformation', variable_transformer),
    ('Correlation Removal', remove_corr),
    ('Remove Unimportant', remove_unimportant)
    
]).fit_transform(data)

target = data_transformed['Income']
X = data_transformed.drop(columns='Income')



X_train, X_val, y_train, y_val = train_test_split(X,
                                                  target,
                                                  test_size=0.25,
                                                  stratify=target,
                                                random_state=35)

from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_train_resampled, y_train_resampled = cc.fit_resample(X_train, y_train)

```

```python
data_transformed = Pipeline([
    ('Variable Transformation', variable_transformer),
    ('Correlation Removal', remove_corr)
    
]).fit_transform(data)

target = data_transformed['Income']
X = data_transformed.drop(columns='Income')



X_train, X_val, y_train, y_val = train_test_split(X,
                                                  target,
                                                  test_size=0.25,
                                                  stratify=target,
                                                random_state=35)

```

```python
HistBoost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Histogram-Based Gradient Boosting', HistGradientBoostingClassifier(loss='binary_crossentropy',
                                                                        random_state=10,
                                                                        max_iter=700,
                                                                        max_depth=2,
                                                                        n_iter_no_change=400,
                                                                        tol=0.1,
                                                                        validation_fraction=0.05,
                                                                         learning_rate=0.15,
                                                                        warm_start=True
                                                                         ))
])


skf=StratifiedKFold(n_splits=3,shuffle=False,random_state=0)

#avg_score_GB(skf,GradBoost,X, target)

HistBoost.fit(X_train,y_train)

evaluate(HistBoost,X_train, X_val, y_train, y_val)
```

```python
HistBoost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Histogram-Based Gradient Boosting', HistGradientBoostingClassifier(loss='binary_crossentropy',
                                                                        random_state=10,
                                                                        max_iter=500,
                                                                        max_depth=2,
                                                                        n_iter_no_change=300,
                                                                        tol=0.01,
                                                                        validation_fraction=0.1,
                                                                        learning_rate=0.2,
                                                                         warm_start=True
                                                                         ))
])


skf=StratifiedKFold(n_splits=3,shuffle=False,random_state=0)

#avg_score_GB(skf,GradBoost,X, target)

HistBoost.fit(X_train,y_train)

evaluate(HistBoost,X_train, X_val, y_train, y_val)
```

```python
test_data_transformed = Pipeline([
    ('Transformation', variable_transformer),
    ('Correlation Removal', remove_corr),
    ('Remove Unimportant', remove_unimportant)
    
]).fit_transform(test_data)
```

```python
export_results(GradBoost, 18, test_data_transformed)
```
