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
# Libraries for manipulating and displaying data import pandas as pd import numpy as np import seaborn as sns import matplotlib.pyplot as plt from pandas_profiling import ProfileReport  # Utility Libraries from datetime import datetime  # Model Libraries from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# Utility Libraries
from datetime import datetime

# Model Libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier, BaggingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.feature_selection import VarianceThreshold, RFE, RFECV, SelectFromModel
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
import xgboost as xgb


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn import tree
from sklearn.metrics import make_scorer
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

```python
data[data['Employment Sector'].isna()][['Income']].value_counts()
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
def generate_report(data, path):
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
# generate_report(data, 'reports/citizen_profiling.html')
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
role_na_index=data[data['Role'].isna()].index
```

```python
emp_sector_na_index=data[data['Employment Sector'].isna()].index
```

```python
def fill_missing_values(df):
    df = df.copy()
    df['Role'].fillna('Unemployed', inplace=True)
    df['Employment Sector'].fillna('Unemployed', inplace=True)
    df['Employment Sector'] = df['Employment Sector'].apply(lambda x: 'Unemployed' if x == 'Never Worked' else x)
    # index_to_change = df[df['Employment Sector'] == 'Unemployed'].index
    # df.loc[index_to_change,'Working Hours per week'] = 0
    df['Base Area'].fillna(df['Base Area'].mode()[0], inplace=True)
    return df
    
fill_na_transformer = FunctionTransformer(fill_missing_values)
data_no_missing = fill_na_transformer.fit_transform(data)
```

```python
data[data['Employment Sector'] == '?'][['Working Hours per week', 'Employment Sector']].value_counts()
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

    input_data = Pipeline([
        ('Fill Missing Values', fill_na_transformer),
        ('Get the Metric Features', get_metric_features)
    ]).fit_transform(data)
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
len(data[filters1]) /len(data)
```

```python
outlier_filter_transformer = FunctionTransformer(lambda x: x[filters1])
```

#### Modeling no Outliers

```python
no_outliers_pipeline = Pipeline([
    ('Fill Missing Values', fill_na_transformer),
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
def Sectorsector(string):
    if ' - ' in string:
        return string.split(' - ')[1]
    elif ' (' in string:
        return string.split('(')[1][:-1]
    else:
        return string

def sector_binning(df):
    df = df.copy()
    df['Private Sector'] = df['Employment Sector'].apply(lambda x: 1 if 'Private' in x else 0)
    df.loc[emp_sector_na_index,'Private Sector']=np.nan
    df['Public Sector'] = df['Employment Sector'].apply(lambda x: 1 if 'Public' in x else 0)
    df.loc[emp_sector_na_index,'Public Sector']=np.nan
    df['Self Employed'] = df['Employment Sector'].apply(lambda x: 1 if 'Self' in x else 0)
    df.loc[emp_sector_na_index,'Self Employed']=np.nan
    df['Unemployed'] = df['Employment Sector'].apply(lambda x: 1 if 'Unemployed' in x else 0)
    df.loc[emp_sector_na_index,'Unemployed']=np.nan
    
    df['Sector'] = df['Employment Sector'].apply(lambda x: Sectorsector(x))
    df = df.drop(columns='Sector').merge(pd.get_dummies(df['Sector'], prefix='Sector').iloc[:, :-1],
                                                       on=df.index,
                                                       left_index=True)
    df.loc[emp_sector_na_index,['Sector_Company', 'Sector_Government', 'Sector_Individual','Sector_Others', 'Sector_Services ']]=np.nan
    
    df.rename(columns={'Sector_Services ': 'Sector_Services'}, inplace=True)
    
    return df.drop(columns=['Employment Sector', 'key_0'])

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
def role_dummies(df):
    df = df.copy()
    df = df.drop(columns='Role').merge(pd.get_dummies(
        df['Role'], prefix='Role').iloc[:, :-1],
                                       on=df.index,
                                       left_index=True)
    df.loc[role_na_index,['Role_Administratives',
       'Role_Agriculture and Fishing', 'Role_Army', 'Role_Cleaners & Handlers',
       'Role_Household Services', 'Role_IT',
       'Role_Machine Operators & Inspectors', 'Role_Management',
       'Role_Other services', 'Role_Professor', 'Role_Repair & constructions',
       'Role_Sales', 'Role_Security', 'Role_Transports']]=np.nan
    return df.drop(columns='key_0')

role_enconder = FunctionTransformer(role_dummies)
```

```python
def fill_job_na(df):
    df = df.copy()
    columns_to_use=['Role_Administratives',
       'Role_Agriculture and Fishing', 'Role_Army', 'Role_Cleaners & Handlers',
       'Role_Household Services', 'Role_IT',
       'Role_Machine Operators & Inspectors', 'Role_Management',
       'Role_Other services', 'Role_Professor', 'Role_Repair & constructions',
       'Role_Sales', 'Role_Security', 'Role_Transports','Sector_Company', 
    'Sector_Government', 'Sector_Individual','Sector_Others', 'Sector_Services','Years of Education',
                    'Working Hours per week','Private Sector','Public Sector','Self Employed','Unemployed']
    imputer = KNNImputer(n_neighbors=1)
    KNN=imputer.fit_transform(df[columns_to_use])
    to_merge=pd.DataFrame(KNN, index=df.index, columns=columns_to_use)
    df[columns_to_use]=to_merge[columns_to_use]
    return df

fill_job_na_transformer = FunctionTransformer(fill_job_na)
```

```python
variable_encoder = Pipeline([
    ('Fill Missing Values', fill_na_transformer),
    ('is_married', is_married_transformer),
    ('Sector', sector_binning_transformer),
    ('Professional School', is_professional_transformer),
    ('Native Continent', continent_encoder),
    ('Lives With', lives_with_encoder),
    ('Lives in Northbury', lives_northbury_encoder),
    ('Role', role_enconder),
    ('Fill Job Na',fill_job_na_transformer)
    
])

variable_encoder.fit_transform(data).isna().sum()
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
    ('Encode Variables', variable_encoder),
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
def money_ratios(df):
    df = df.copy()
    df['Ticket Price per Age'] = df['Ticket Price'] / df['Age']
    df['Money Received per Age'] = df['Money Received'] / df['Age']
    df['Ticket Price per YoE'] = df['Ticket Price'] / df['Years of Education']
    df['Money Received per YoE'] = df['Money Received'] / df['Years of Education']
    df['Ticket Price per WHpW'] = df['Ticket Price'] / df['Working Hours per week']
    df['Money Received per WHpW'] = df['Money Received'] / df['Working Hours per week']
    return df

money_ratios_transformer = FunctionTransformer(money_ratios)
```

```python
variable_transformer = Pipeline([
    ('Variable Encoding', variable_encoder),
    ('Age', age_transformer),
    ('Education Per Age', edu_per_age_transformer),
    ('Gender', gender_transformer),
    ('Groups', group_transformer),
    ('Money Rations', money_ratios_transformer)
    
])

variable_transformer.fit_transform(data).isna().sum()
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
#df_model.drop(columns=['No Outliers Encoded', 'Transformed Variables No Outliers'], inplace=True)
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
corr['Income'].apply(np.abs).sort_values(ascending=False)
```

```python
remove_corr = FunctionTransformer(lambda x: x.drop(columns=['is_group_c', 'is_Married', 'Continent_Africa',
                                                           'Ticket Price per Age', 'Money Received per Age',
                                                           'Ticket Price per YoE', 'Money Received per YoE',
                                                           'Ticket Price per WHpW', 'Money Received per WHpW',
                                                           'Education per Age', 'Sector_Services']))
```

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
test_data_transformed = Pipeline([
    ('Transformation', variable_transformer),
    ('Correlation Removal', remove_corr)
]).fit_transform(test_data)

f1_scorer = make_scorer(f1_score, average='micro')
```

```python
# generate_report(data_transformed, 'reports/citizen_profiling_after_transformation')
```

```python
mlp = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', MLPClassifier(random_state=0))
])
parameters = {
    'Classifier__hidden_layer_sizes' : [(30, 20), (30,20, 10)],
    'Classifier__activation' : [ 'tanh'],
    'Classifier__learning_rate' : ['adaptive'],
    'Classifier__max_iter' : [100, 200],
    'Classifier__early_stopping' : [True]
}

search = GridSearchCV(mlp, parameters, estimator=f1_score, verbose=10)
search.fit(X, target)
```

```python
GradBoost = Pipeline([('Scaler', MinMaxScaler()),
                      ('Classifier',GradientBoostingClassifier(random_state=92,
                                                              n_iter_no_change=200,
                                                              learning_rate=0.1,
                                                              n_estimators=200,
                                                              max_depth=5,
                                                              max_features='auto',
                                                              validation_fraction=0.2,
                                                              tol=0.01,
                                                              min_impurity_decrease=0.01,
                                                              min_samples_split=3))
                     ])

parameters = {
    'Scaler' : [MinMaxScaler()]
    'Classifier__max_features' : ['sqrt', 'auto'],
    'Classifier__learning_rate' : [0.5, 0.2, 0.1],
    'Classifier'
    
}

search = GridSearchCV(GradBoost, parameters, estimator=f1_score, verbose=10)
search.fit(X, target)
```

```python
optimized_randForest = RandomForestClassifier(random_state=0,
                                             bootstrap=True,
                                             max_depth=20,
                                             max_features='sqrt',
                                             min_impurity_decrease=0,
                                             n_estimators=100,
                                             warm_start=True)

grid_pipeline = Pipeline([
    ('Classifier', AdaBoostClassifier(random_state=0))
])

parameters = {
    'Classifier__n_estimators' : [50],
    'Classifier__base_estimator' : [optimized_randForest],
    'Classifier__learning_rate' : [0.1]
}

search = GridSearchCV(grid_pipeline, parameters, estimator = f1_score, verbose=10)
search.fit(X, target)
```

```python
search.best_score_
```

```python
search.best_params_
```

```python
evaluate(search.best_estimator_, X_train, X_val, y_train, y_val)
```

```python
test_data_transformed = Pipeline([
    ('Transformation', variable_transformer),
    ('Correlation Removal', remove_corr)
]).fit_transform(test_data)
#export_results(search.best_estimator_, 17, test_data_transformed)
```

## Feature Importance

```python
feature_importance=pd.DataFrame([data_transformed.drop(columns='Income').columns,ada_model.feature_importances_])
feature_importance=feature_importance.T
feature_importance.rename(columns={0:'Feature',1:'Importance'},inplace=True)
feature_importance

##Feature Importance Graph
plt.figure(figsize=(10,10))
sns.barplot(x='Feature',y='Importance',data=feature_importance, order=feature_importance.sort_values('Importance', ascending=False).Feature)
ax_ticks=plt.xticks(rotation='vertical')
```

```python
remove_unimportant = FunctionTransformer(lambda x: x.drop(columns=['Role_Household Services', 'Role_Army']))
```

```python
# data_transformed = Pipeline([
#    ('Variable Transformation', variable_transformer),
#    ('Correlation Removal', remove_corr),
#    ('Remove Unimportant', remove_unimportant)
# ]).fit_transform(data)

target = data_transformed['Income']
X = data_transformed.drop(columns='Income')

X_train, X_val, y_train, y_val = train_test_split(X,
                                                  target,
                                                  test_size=0.25,
                                                  stratify=target,
                                                  random_state=35)
```

```python
ada_model = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=5, n_estimators = 10, random_state=0),
                              n_estimators=50,
                              random_state=0)
ada_model.fit(X_train, y_train)

evaluate(ada_model, X_train, X_val, y_train, y_val)
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

```python
df = df_model[['Model_Name','No Outliers Encoded','Transformed Variables',]].melt('Model_Name', var_name='Steps',  value_name='F1 Micro')
fig=plt.figure(figsize=(10,100))
g = sns.catplot(x='Steps', y="F1 Micro", hue='Model_Name', data=df, kind='point')
xticks=plt.xticks(rotation=70)
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
search.best_params_
```

```python
model_pipeline = Pipeline([('Standardization', MinMaxScaler()),
                           ('Classifier', RandomForestClassifier(bootstrap=True,
                                                   max_depth=20,
                                                   max_features='sqrt',
                                                   min_impurity_decrease=0,
                                                   n_estimators=200,
                                                   warm_start=True))])

model_pipeline.fit(X_train, y_train)
evaluate(model_pipeline, X_train, X_val, y_train, y_val)
```

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

## Gradient Boosting


### Feature Importance

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=70,
                                             learning_rate=0.2,
                                             max_features=None,
                                             subsample=0.8))
])

grad_boost.fit(X_train, y_train)
evaluate(grad_boost, X_train, X_val, y_train, y_val)

feature_importance=pd.DataFrame([data_transformed.drop(columns='Income').columns,grad_boost['Classifier'].feature_importances_])
feature_importance=feature_importance.T
feature_importance.rename(columns={0:'Feature',1:'Importance'},inplace=True)
feature_importance = feature_importance[feature_importance['Importance'] > 0.001]

##Feature Importance Graph
plt.figure(figsize=(10,10))
sns.barplot(x='Feature',y='Importance',data=feature_importance, order=feature_importance.sort_values('Importance', ascending=False).Feature)
ax_ticks=plt.xticks(rotation='vertical')
len(feature_importance)
```

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=70,
                                             learning_rate=0.2,
                                             max_features=32,
                                             subsample=0.8))
])

grad_boost.fit(X_train, y_train)
evaluate(grad_boost, X_train, X_val, y_train, y_val)
```

### Max_features

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=70,
                                             learning_rate=0.2,
                                             subsample=0.8))
])

parameters = {
    'Classifier__max_features' : range(10, len(data_transformed.columns) + 1, 2)
}

search = GridSearchCV(grad_boost, parameters, scoring=f1_scorer,
                     verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

### Max Depth

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             learning_rate=0.2,
                                             max_features=30,
                                             subsample=0.8))
])

f1_scorer = make_scorer(f1_score, average='micro')

parameters = {
    'Classifier__n_estimators' : range(200, 311, 20)
}

search = GridSearchCV(grad_boost, parameters, scoring=f1_scorer,
                     verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

```python
search_n_est = pd.DataFrame(search.cv_results_)[['param_Classifier__n_estimators', 'mean_test_score']]
```

### Max_depth and min_samples_split

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=220,
                                             learning_rate=0.2,
                                             max_features=30,
                                             subsample=0.8))
])

parameters = {
    'Classifier__max_depth' : range(5,16,2),
    'Classifier__min_samples_split' : range(50, 250, 20)
}

search = GridSearchCV(grad_boost, parameters, scoring=f1_scorer,
                     verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

```python
search_max_depth_min_samples_split = pd.DataFrame(search.cv_results_)[['param_Classifier__min_samples_split','param_Classifier__max_depth',  'mean_test_score']]
```

### min_samples_leaf

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=220,
                                             min_samples_split=190,
                                              max_depth=5,
                                             learning_rate=0.2,
                                             max_features=30,
                                             subsample=0.8))
])

parameters = {
    'Classifier__min_samples_leaf' : range(10, 101, 10)
}

search = GridSearchCV(grad_boost, parameters, scoring=f1_scorer,
                     verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

```python
search_min_samples_leaf = pd.DataFrame(search.cv_results_)[['param_Classifier__min_samples_leaf', 'mean_test_score']]
search_min_samples_leaf
```

```python
search_max_feat = pd.DataFrame(search.cv_results_)[['param_Classifier__max_features', 'mean_test_score']]
```

### Subsample

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=220,
                                             min_samples_split=190,
                                              min_samples_leaf=50,
                                              max_depth=5,
                                              max_features=30,
                                             learning_rate=0.2))
])

parameters = {
    'Classifier__subsample' : np.linspace(0.6, 0.9, num=6) 
}

search = GridSearchCV(grad_boost, parameters, scoring=f1_scorer,
                     verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

```python
search_subsample = pd.DataFrame(search.cv_results_)[['param_Classifier__subsample', 'mean_test_score']]
```

### Learning and Number of Estimators


#### Originals

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=4400,
                                             min_samples_split=190,
                                              max_depth=5,
                                              min_samples_leaf=50,
                                             learning_rate=0.01,
                                             max_features=30,
                                             subsample=0.9))
])

grad_boost.fit(X_train, y_train)
evaluate(grad_boost, X_train, X_val, y_train, y_val)
```

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=140,
                                             min_samples_split=150,
                                              max_depth=7,
                                              min_samples_leaf=20,
                                             learning_rate=0.1,
                                             max_features=33,
                                             subsample=0.9))
])

grad_boost.fit(X_train, y_train)
evaluate(grad_boost, X_train, X_val, y_train, y_val)
```

```python
export_results(grad_boost, 19, test_data_transformed)
```

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=1400,
                                             min_samples_split=150,
                                              max_depth=7,
                                              min_samples_leaf=20,
                                             learning_rate=0.01,
                                             max_features=33,
                                             subsample=0.9))
])

grad_boost.fit(X_train, y_train)
evaluate(grad_boost, X_train, X_val, y_train, y_val)
```

### Other parameters

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=140,
                                             min_samples_split=150,
                                              min_samples_leaf=20,
                                              max_depth=7,
                                              max_features=33,
                                              subsample=0.9,
                                             learning_rate=0.1))
])

parameters = {
    'Classifier__tol' : [0.1, 0.01, 0.001, 0.0001],
    'Classifier__n_iter_no_change' : range(0, 71, 10)
}

search = GridSearchCV(grad_boost, parameters, scoring=f1_scorer,
                     verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=140,
                                             min_samples_split=150,
                                              min_samples_leaf=20,
                                              max_depth=7,
                                              max_features=33,
                                              subsample=0.9,
                                              n_iter_no_change=50,
                                              tol=0.01,
                                             learning_rate=0.1))
])

parameters = {
    'Classifier__init': ['zero', LogisticRegression(), None]
}

search = GridSearchCV(grad_boost, parameters, scoring=f1_scorer,
                     verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_params_, search.best_score_
```

```python
grad_boost = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', GradientBoostingClassifier(random_state=0,
                                             n_estimators=140,
                                             min_samples_split=150,
                                              min_samples_leaf=20,
                                              max_depth=7,
                                              max_features=33,
                                              subsample=0.9,
                                              n_iter_no_change=50,
                                              tol=0.01,
                                              init='zero',
                                             learning_rate=0.1))
])

grad_boost.fit(X_train, y_train)
evaluate(grad_boost, X_train, X_val, y_train, y_val)
```

```python
grad_boost = Pipeline([('Scaler', MinMaxScaler()),
                       ('Classifier',GradientBoostingClassifier(random_state=10, 
                                                                learning_rate=0.1, 
                                                                n_estimators=140, 
                                                                max_depth=7, 
                                                                min_samples_leaf=60,
                                                                max_features=30,
                                                                subsample=0.8))
                                                                ])

grad_boost.fit(X_train, y_train)
evaluate(grad_boost, X_train, X_val, y_train, y_val)
```

## Hist-Gradient Boosting


### Default Parameters

```python
hist_grad = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Classifier', HistGradientBoostingClassifier(random_state=0))
])

hist_grad.fit(X_train, y_train)
evaluate(hist_grad, X_train, X_val, y_train, y_val)
```

```python
search.best_params_, search.best_score_
```

```python
optimized_randForest = RandomForestClassifier(random_state=0,
                                             max_depth=15,
                                             max_features='sqrt',
                                             n_estimators=60,
                                              min_samples_leaf=2,
                                             warm_start=True)
parameters = {
    'bootstrap' : [True, False]
}

search = GridSearchCV(optimized_randForest, parameters, scoring=f1_scorer, verbose=10)
search.fit(X_train, y_train)
search.best_score_, search.best_params_
```

```python
optimized_randForest = BaggingClassifier(RandomForestClassifier(random_state=0,
                                                                 max_depth=30,
                                                                 max_features='sqrt',
                                                                 n_estimators=60,
                                                                 bootstrap=False,
                                                                 min_samples_leaf=2,
                                                                 warm_start=True),
                                         random_state=0,
                                         bootstrap=False,
                                         n_estimators=10
                                        )
optimized_randForest.fit(X_train, y_train)
evaluate(optimized_randForest, X_train, X_val, y_train, y_val)
```

```python
ensembler = VotingClassifier([
        ('Grad Bag', grad_bag),
        ('HG Bag', hg_bag),
        ('Grad Boosting', grad_boost),
        ('Hist Grad', hist_grad),
        ('Random Forest', optimized_randForest)
    ],
    voting='hard', flatten_transform=False
)
ensembler.fit(X_train, y_train)
evaluate(ensembler, X_train, X_val, y_train, y_val)
```

```python
hg_bag = BaggingClassifier(base_estimator=HistGradientBoostingClassifier(random_state=0,
                                                                        learning_rate=0.25,
                                                                        max_iter=30,
                                                                        warm_start=True,
                                                                        max_depth=20),
                           random_state=0,
                           n_estimators = 10, 
                           bootstrap=False)

hg_bag.fit(X_train, y_train)
evaluate(hg_bag, X_train, X_val, y_train, y_val)
```

```python
export_results(hg_bag, 30, test_data_transformed)
```

```python
grad_bag = BaggingClassifier(base_estimator=GradientBoostingClassifier(random_state=0,
                                                                        learning_rate=0.2,
                                                                        warm_start=True,
                                                                        max_depth=4),
                           random_state=0,
                           n_estimators = 20, 
                           bootstrap=False)

grad_bag.fit(X_train, y_train)
evaluate(grad_bag, X_train, X_val, y_train, y_val)
```

```python
etc = ExtraTreesClassifier(random_state=0, )
etc.fit(X_train, y_train)
evaluate(etc, X_train, X_val, y_train, y_val)
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

## Decision Tree

```python
optimized_dt = DecisionTreeClassifier(random_state=0,
                                     max_depth=7,
                                     min_samples_leaf=9,
                                     min_samples_split=3,
                                     max_features=None)

parameters = {
    'ccp_alpha': [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4] 
}

search = GridSearchCV(optimized_dt, parameters, scoring=f1_scorer, verbose=10)
search.fit(X_train, y_train)
```

```python
search.best_score_, search.best_params_
```

```python
optimized_dt = DecisionTreeClassifier(random_state=0,
                                     max_depth=10,
                                     min_samples_leaf=9,
                                     min_samples_split=3,
                                     max_features=None)
optimized_dt.fit(X_train, y_train)
evaluate(optimized_dt, X_train, X_val, y_train, y_val)
```

```python

```

## Multi Layer Perceptron


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
