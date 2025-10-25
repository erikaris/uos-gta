# Dataset Inspection Functions in R (Base + Tidyverse)

A quick reference for inspecting datasets in R, including both **base R** and **tidyverse** approaches.

---

## 1. Preview the Data

| Function | Description | Example |
|----------|-------------|---------|
| `head()` | Show the first 6 rows (default) | `head(iris)` |
| `tail()` | Show the last 6 rows | `tail(iris)` |
| `View()` | Open a spreadsheet-style viewer | `View(iris)` |
| `dplyr::glimpse()` | Compact view of structure | `dplyr::glimpse(iris)` |

---

## 2. Dataset Dimensions and Structure

| Function | Description | Example |
|----------|-------------|---------|
| `dim()` | Returns number of rows and columns | `dim(iris)` |
| `nrow()` / `ncol()` | Number of rows / columns | `nrow(iris)` / `ncol(iris)` |
| `names()` / `colnames()` | Get column names | `names(iris)` |
| `str()` | Display structure and types of columns | `str(iris)` |
| `class()` | Get object class | `class(iris)` |
| `typeof()` | Get internal type | `typeof(iris)` |


*class vs type:*

1. They describe the same thing, but at different levels.
2. `typeof()`: related to the object's storage &rarr; how R saves it in memory. 
3. `class()`: related to the object's behavior &rarr; how R behaves with it
4. example:
   
   a. `typeof(x)` → "double": R stores these numbers as double-precision floats in memory.
   b. `class(x)` → "numeric": R treats it as something you can do math with.
6. analogy: You have a box.
7. 
   a. `typeof()` tells you what the box is made of &rarr; e.g., wood, metal, plastic.
   b. `class()` tells you what the box is used for &rarr; e.g., lunchbox, toolbox, gift box.
   
---

## 3. Summary Statistics

| Function | Description | Example |
|----------|-------------|---------|
| `summary()` | Summary statistics for each column | `summary(iris)` |
| `skimr::skim()` | Detailed summary with distributions and missing values | `skimr::skim(iris)` |

---

## 4. Inspect Column Types and Missing Values

| Function | Description | Example |
|----------|-------------|---------|
| `sapply()` | Apply a function to each column | `sapply(iris, class)` |
| `is.na()` | Detect missing values | `sum(is.na(iris))` |
| `anyNA()` | Check if any missing values exist | `anyNA(iris)` |

---

## 5. Explore Unique Values and Factor Levels

| Function | Description | Example |
|----------|-------------|---------|
| `unique()` | Get unique values of a column | `unique(iris$Species)` |
| `table()` | Count frequency of values | `table(iris$Species)` |
| `levels()` | Get factor levels | `levels(iris$Species)` |

---

## 6. Quick Visual Inspection

| Function | Description | Example |
|----------|-------------|---------|
| `plot()` | Quick scatterplots or histograms | `plot(iris$Sepal.Length)` |
| `pairs()` | Matrix of scatterplots | `pairs(iris[1:4])` |
| `hist()` | Histogram of a numeric column | `hist(iris$Sepal.Length)` |
| `dplyr::count()` | Count occurrences in a column | `iris %>% dplyr::count(Species)` |
| `dplyr::glimpse()` | Compact structure view | `iris %>% dplyr::glimpse()` |

---

## 7. Tidyverse Workflow Example

```r
library(dplyr)
library(skimr)

# Load dataset
data(iris)

# Quick overview
iris %>% glimpse()
iris %>% head()
iris %>% tail()
iris %>% summary()
iris %>% skim()

# Check column types and missing values
sapply(iris, class)
anyNA(iris)

# Frequency counts of factors
iris %>% count(Species)

# Select specific columns
iris %>% select(Sepal.Length, Species) %>% head()
