
### 🧭 **Python & Pandas Iteration Summary**

| Method                   | Scope               | Returns          | Speed              | Recommended | Example                                           |
| ------------------------ | ------------------- | ---------------- | ------------------ | ----------- | ------------------------------------------------- |
| `for x in iterable`      | any iterable        | value            | ✅ fast             | yes         | `for x in [1,2,3]: print(x)`                      |
| `for i in range(n)`      | integer range       | int              | ✅ fast             | yes         | `for i in range(5): print(i)`                     |
| `enumerate()`            | any iterable        | (index, value)   | ✅ fast             | yes         | `for i, f in enumerate(features, 1): print(i, f)` |
| `zip()`                  | multiple iterables  | tuple            | ✅ fast             | yes         | `for a, b in zip(x, y): print(a, b)`              |
| `dict.items()`           | dict                | (key, value)     | ✅ fast             | yes         | `for k, v in mydict.items(): print(k, v)`         |
| **List comprehension**   | any iterable        | list             | ⚡ very fast        | yes         | `[x*x for x in range(5)]`                         |
| **Generator expression** | any iterable        | generator        | ⚡ memory efficient | yes         | `(x*x for x in range(5))`                         |
| `df.iterrows()`          | pandas rows         | (index, Series)  | ❌ slow             | rarely      | `for i, row in df.iterrows(): print(row['a'])`    |
| `df.itertuples()`        | pandas rows         | namedtuple       | ⚡ fast             | yes         | `for row in df.itertuples(): print(row.a)`        |
| `df.items()`             | pandas columns      | (col, Series)    | ✅ good             | yes         | `for col, s in df.items(): print(col, s.mean())`  |
| `df.apply()`             | rows/columns        | Series/DataFrame | ⚠️ medium          | sometimes   | `df.apply(lambda r: r['a']+r['b'], axis=1)`       |
| `Series.map()`           | single column       | Series           | ⚡ fast             | yes         | `df['a'] = df['a'].map(lambda x: x*2)`            |
| `df.applymap()`          | all cells           | DataFrame        | ⚠️ medium          | sometimes   | `df = df.applymap(lambda x: x*2)`                 |
| **Vectorized ops**       | pandas/numpy arrays | Series/DataFrame | 🚀 fastest         | always      | `df['c'] = df['a'] + df['b']`                     |
| `NumPy iteration`        | np array            | value            | ⚡ fast             | yes         | `for x in df['a'].to_numpy(): print(x)`           |

---

✅ **Tips to remember**

* In **pure Python**, use `enumerate()`, `zip()`, and comprehensions.
* In **pandas**, prefer **vectorized** operations whenever possible.
* `itertuples()` is the **best option** when you truly need row-wise iteration.
* `iterrows()` is **intuitive but slow** (creates a Series per row).

---

Would you like me to make a **cheat sheet image** (like a compact visual table) for easier reference when coding?
