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

