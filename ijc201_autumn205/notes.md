# IJC201 - Week 3 Notes - 16th October 2025

## df.to_dict()

```python
df.to_dict(orient=...)
```

### The main `orient` options and what they do:

| Orient      | Structure Returned                            | Description                                                                               | Example Output                                                              |
| :---------- | :-------------------------------------------- | :---------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| `'dict'`    | dict of dicts                                 | Default. Each **column** becomes a **key**, with values as a nested dict `{index: value}` | `{ 'col1': {0: 10, 1: 20}, 'col2': {0: 30, 1: 40} }`                        |
| `'list'`    | dict of lists                                 | Each **column** is a key; values are lists (column values).                               | `{ 'col1': [10, 20], 'col2': [30, 40] }`                                    |
| `'series'`  | dict of Series                                | Same as `'dict'`, but each value is a **pandas Series** instead of dict.                  | `{ 'col1': pd.Series({0:10,1:20}), 'col2': pd.Series({0:30,1:40}) }`        |
| `'split'`   | dict with keys `['index', 'columns', 'data']` | Breaks DataFrame into labeled parts — good for reconstruction.                            | `{ 'index': [0,1], 'columns': ['col1','col2'], 'data': [[10,30],[20,40]] }` |
| `'records'` | list of dicts                                 | Each **row** becomes a dict (best for JSON).                                              | `[{'col1':10,'col2':30}, {'col1':20,'col2':40}]`                            |
| `'index'`   | dict of dicts                                 | Each **row index** becomes a key; values are dicts of column/value pairs.                 | `{ 0: {'col1':10,'col2':30}, 1: {'col1':20,'col2':40} }`                    |

---

### Example in Action

Say we have:

```python
import pandas as pd
df = pd.DataFrame({
    'col1': [10, 20],
    'col2': [30, 40]
})
```

```python
|index|col1|col2|
|---|---|---|
|0|10|30|
|1|20|40|
```

Then:

```python
df.to_dict('dict')
# {'col1': {0: 10, 1: 20}, 'col2': {0: 30, 1: 40}}

df.to_dict('list')
# {'col1': [10, 20], 'col2': [30, 40]}

df.to_dict('series')
# {'col1': Series([10, 20]), 'col2': Series([30, 40])}

df.to_dict('split')
# {'index': [0, 1], 'columns': ['col1', 'col2'], 'data': [[10, 30], [20, 40]]}

df.to_dict('records')
# [{'col1': 10, 'col2': 30}, {'col1': 20, 'col2': 40}]

df.to_dict('index')
# {0: {'col1': 10, 'col2': 30}, 1: {'col1': 20, 'col2': 40}}
```

---

**Quick summary:**

* **Row-oriented:** `'records'`, `'index'`
* **Column-oriented:** `'dict'`, `'list'`, `'series'`
* **Full structure:** `'split'` (useful for saving/reconstructing DataFrames)

---

### Example Data

```python
import pandas as pd

# Small sample dataset
planets_df = pd.DataFrame({
    'method': ['Radial Velocity', 'Radial Velocity', 'Transit'],
    'year': [2006, 2008, 2011],
    'orbital_period': [269.3, 874.8, 3.5],
    'mass': [7.1, 2.3, 0.02],
    'distance': [77.4, 56.2, 420.3]
})
```

---

## Tree Building: Directly from dataframe vs converting to dictionary first. 
### 1. Building the Tree **Directly from the DataFrame**

```python
planets_tree_df = {}

for _, row in planets_df.iterrows():  # iterrows() gives each row as a Series
    method = row['method']
    year = row['year']
    planet_details = {
        'orbital_period': row['orbital_period'],
        'mass': row['mass'],
        'distance': row['distance']
    }

    if method not in planets_tree_df:
        planets_tree_df[method] = {}
    if year not in planets_tree_df[method]:
        planets_tree_df[method][year] = []
    planets_tree_df[method][year].append(planet_details)

print(planets_tree_df)
```

**Works fine**, but:

* `iterrows()` is **slow** for large DataFrames.
* Each `row` is a `pandas.Series`, not a normal dictionary — accessing fields repeatedly adds overhead.
* You stay dependent on pandas until you’re done.

---

### 🪄 2. Building the Tree **After Converting to List of Dictionaries**

```python
planets_data = planets_df.to_dict(orient='records')
planets_tree_dict = {}

for planet in planets_data:  # each planet is a pure Python dict
    method = planet['method']
    year = planet['year']
    planet_details = {
        'orbital_period': planet['orbital_period'],
        'mass': planet['mass'],
        'distance': planet['distance']
    }

    if method not in planets_tree_dict:
        planets_tree_dict[method] = {}
    if year not in planets_tree_dict[method]:
        planets_tree_dict[method][year] = []
    planets_tree_dict[method][year].append(planet_details)

print(planets_tree_dict)
```

**Advantages:**

* Each iteration is pure Python — no pandas overhead.
* Clean, consistent access to values via dictionary keys.
* Easier to reuse with non-pandas data (e.g., from JSON or APIs).
* More efficient for nested data manipulation.

---

### Conceptual Difference

| Aspect      | DataFrame Iteration            | List of Dictionaries                      |
| ----------- | ------------------------------ | ----------------------------------------- |
| Type        | pandas Series                  | Python dict                               |
| Performance | Slower (`iterrows()` overhead) | Faster (native Python loop)               |
| Readability | Slightly verbose               | Clean and intuitive                       |
| Flexibility | Tied to pandas                 | Works anywhere in Python                  |
| Best for    | Small, quick operations        | Building hierarchical / nested structures |

---

### Summary

* When you’re building **hierarchical data structures** like trees or graphs, it’s better to convert your DataFrame to a **list of dictionaries** first. 
* It turns your data into **Python-native objects** that can be nested, grouped, and accessed quickly.

---


