
### 🧭 **Python & Pandas Iteration Summary (with Data Type + Output Examples)**

| Method                          | Scope               | Suitable Data Type(s)                                   | Data Type Example                          | Returns              | Speed              | Recommended | Example                                            | Output Example                                           |
| ------------------------------- | ------------------- | ------------------------------------------------------- | ------------------------------------------ | -------------------- | ------------------ | ----------- | -------------------------------------------------- | -------------------------------------------------------- |
| `for x in iterable`             | General             | `list`, `tuple`, `set`, `str`, `range`, `numpy.ndarray` | `nums = [1, 2, 3]`                         | element              | ✅ fast             | yes         | `for x in nums: print(x)`                          | `1\n2\n3`                                                |
| `for i in range(n)`             | Index-based         | `range`                                                 | `range(5)` → `[0,1,2,3,4]`                 | integer index        | ✅ fast             | yes         | `for i in range(5): print(i)`                      | `0\n1\n2\n3\n4`                                          |
| `enumerate()`                   | Indexed iteration   | `list`, `tuple`, `set`, `str`                           | `features = ['age','height']`              | `(index, value)`     | ✅ fast             | yes         | `for i, f in enumerate(features, 1): print(i, f)`  | `1 age\n2 height`                                        |
| `zip()`                         | Parallel iteration  | multiple iterables                                      | `names=['A','B'], scores=[85,90]`          | tuple                | ✅ fast             | yes         | `for n, s in zip(names, scores): print(n, s)`      | `A 85\nB 90`                                             |
| `dict.items()`                  | Dictionary          | `dict`                                                  | `person = {'name':'Ana', 'age':25}`        | `(key, value)`       | ✅ fast             | yes         | `for k, v in person.items(): print(k, v)`          | `name Ana\nage 25`                                       |
| `dict.keys()` / `dict.values()` | Dictionary          | `dict`                                                  | same as above                              | key / value          | ✅ fast             | yes         | `for k in person.keys(): print(k)`                 | `name\nage`                                              |
| **List comprehension**          | Build lists         | `list`, `range`, `tuple`                                | `nums = [1,2,3,4]`                         | new list             | ⚡ very fast        | yes         | `[x*x for x in nums]`                              | `[1, 4, 9, 16]`                                          |
| **Set / Dict comprehension**    | Build sets/dicts    | `list`, `range`                                         | `nums = [1,2,3]`                           | set / dict           | ⚡ very fast        | yes         | `{x: x*x for x in nums}`                           | `{1:1, 2:4, 3:9}`                                        |
| **Generator expression**        | Lazy iteration      | any iterable                                            | `nums = [1,2,3]`                           | generator            | ⚡ memory efficient | yes         | `(x*x for x in nums)`                              | `<generator object ...>` (use `list()` to see `[1,4,9]`) |
| `df.iterrows()`                 | Pandas row-wise     | `pd.DataFrame`                                          | `df = pd.DataFrame({'a':[1,2],'b':[3,4]})` | `(index, Series)`    | ❌ slow             | rarely      | `for i, row in df.iterrows(): print(row['a'])`     | `1\n2`                                                   |
| `df.itertuples()`               | Pandas row-wise     | `pd.DataFrame`                                          | same as above                              | namedtuple           | ⚡ fast             | yes         | `for row in df.itertuples(): print(row.a, row.b)`  | `1 3\n2 4`                                               |
| `df.items()`                    | Pandas column-wise  | `pd.DataFrame`                                          | same as above                              | `(col_name, Series)` | ✅ good             | yes         | `for col, s in df.items(): print(col, s.tolist())` | `a [1, 2]\nb [3, 4]`                                     |
| `df.apply()`                    | Pandas row/col-wise | `pd.DataFrame`, `pd.Series`                             | same as above                              | Series / DataFrame   | ⚠️ medium          | sometimes   | `df.apply(lambda r: r['a']+r['b'], axis=1)`        | `0 4\n1 6` *(Series)*                                    |
| `Series.map()`                  | Element-wise (1D)   | `pd.Series`                                             | `s = pd.Series([1,2,3])`                   | Series               | ⚡ fast             | yes         | `s.map(lambda x: x*2)`                             | `0 2\n1 4\n2 6`                                          |
| `df.applymap()`                 | Element-wise (2D)   | `pd.DataFrame`                                          | same as above                              | DataFrame            | ⚠️ medium          | sometimes   | `df.applymap(lambda x: x*2)`                       | `a [2, 4]\nb [6, 8]`                                     |
| **Vectorized operations**       | Whole columns       | `pd.Series`, `pd.DataFrame`, `numpy.ndarray`            | same as above                              | Series / DataFrame   | 🚀 fastest         | always      | `df['c'] = df['a'] + df['b']`                      | `a b c\n1 3 4\n2 4 6`                                    |
| `NumPy iteration`               | Array iteration     | `numpy.ndarray`                                         | `arr = np.array([1,2,3])`                  | element              | ⚡ fast             | yes         | `for x in arr: print(x)`                           | `1\n2\n3`                                                |

---

### ⚡ **Quick Takeaways**

* 🧱 **Pure Python** → Use `enumerate`, `zip`, and comprehensions.
* 🧮 **Pandas** → Avoid row loops (`iterrows`); prefer **vectorized** ops or **`itertuples()`**.
* 💨 **Performance order**:
  `vectorized > itertuples > apply > iterrows`.
* 🧠 **Readability order**:
  `comprehension > zip/enumerate > traditional for`.

