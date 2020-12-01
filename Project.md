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
    display_name: Python [conda env:dm_env] *
    language: python
    name: conda-env-dm_env-py
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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from datetime import datetime
```

## Importing Data

```python
data = pd.read_excel('data/Train.xlsx')
```

```python
data.head()
```

```python
# Backup the data in case we make significant changes we need to revert
backup = data.copy()
```

```python
data = backup.copy()
```

### Pandas Profiling
To get a quick overview of the dataset, we used the pandas-profiling library to generate an automatic report. This, however, does not and will not replace an actual, in-depth, analysis.

The ProfileReport shows us the basic statistics and the general distribution for each variable, as well as a Correlation Matrix and Scatterplots for comparison between multiple attributes.

```python
# Generate a profile
# profile = ProfileReport(
#    data, 
#    title='Newland Citizens Report',
#    correlations={
#        "pearson": {"calculate": True},
#        "spearman": {"calculate": True},
#        "kendall": {"calculate": False},
#        "phi_k": {"calculate": False},
#        "cramers": {"calculate": False},
#    }
# )


# Export this profile to a file
# profile.to_file('reports/citizen_profiling.html')
```

## Missing Values

We considered that a value o '?' in a column represents a missing value.

For better treatment, we replaced them with numpy.NaN.

```python
data[col_with_missing] = data[col_with_missing].replace('?', np.NaN)
```

```python
data.loc[:,col_with_missing].isnull().mean()
```

## Variables to transform
* Age(from birthday)
* Marital Status(simplify maybe)
* employment_sector_simplified
* Education Level(The PostGraduation Paradox)

```python
# data['Birthday'].astype('datetime64[ns]')
# data['Birthday'] = data['Birthday'].apply(lambda x: x.replace(',', ' '))
# data['Birthday'].apply(lambda x: datetime.strptime(x, " %B %d %Y").date())
data['Age'] = data['Birthday'].apply(lambda x: datetime.strptime(x[-4:], "%Y").date()).astype('datetime64[ns]')
```

```python
data['Age'] = data['Age'].apply(lambda x: 2048 - x.year)
```

## Possible variables

* is_group_X(multiple binaries)
* has_children
* years_of_edu_per_age
* Education Level(The PostGraduation Paradox)


```python
data['years_of_edu_per_age'] = data['Years of Education'] / data['Age']
```

```python
data['is_group_a'] = data['Ticket Price'] + data['Money Received']
data['is_group_a'] = data['is_group_a'].apply(lambda x: 1 if x == 0 else 0)
data['is_group_b'] = data['Money Received'].apply(lambda x: 1 if x > 0 else 0)
data['is_group_c'] = data['Ticket Price'].apply(lambda x: 1 if x > 0 else 0)
data[['is_group_a', 'is_group_b', 'is_group_c']]
```
