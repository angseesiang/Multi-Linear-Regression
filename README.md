# Multi Linear Regression - House Price Prediction

This project demonstrates a **Multiple Linear Regression** workflow for predicting house prices using selected property features. It covers basic exploratory data analysis, missing value handling, outlier treatment using the Interquartile Range method, categorical encoding, model training, prediction, and evaluation using the R² score.

## Project Overview

The goal of this project is to build a regression model that predicts `SalePrice` based on multiple house-related input features such as lot size, frontage, masonry veneer area, floor area, garage information, and living area.

The project uses a house price dataset and applies a supervised machine learning workflow with Python libraries including NumPy, pandas, Seaborn, Matplotlib, and Scikit-learn.

## Technologies Used

- Python
- NumPy
- pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Linear Regression
- Train-Test Split
- One-Hot Encoding
- Interquartile Range (IQR)
- R² Score

## Dataset

The dataset used in this project is:

```text
House Price.csv
```

It contains 1,460 house records with the following columns:

```text
Id, LotFrontage, LotArea, Alley, MasVnrArea, 1stFlrSF, 2ndFlrSF,
GrLivArea, GarageType, GarageArea, SalePrice
```

The target variable is:

```text
SalePrice
```

## Project Files

```text
.
├── Multi Linear Regression.ipynb
├── House Price.csv
└── README.md
```

## Workflow

### 1. Import Libraries

The notebook imports the main Python libraries required for data analysis, visualization, and machine learning:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Load Dataset

The dataset is loaded using pandas:

```python
df = pd.read_csv('House Price.csv')
```

> Note: If your uploaded dataset has a different name such as `House Price(3).csv`, rename it to `House Price.csv` or update the notebook path before running the project.

### 3. Data Inspection

The project checks:

- Number of records
- First few rows
- Missing values
- Percentage of missing values
- Unique values per column
- Data types

### 4. Missing Value Handling

The notebook applies different strategies for missing data:

- Drops `Alley` because it has a high percentage of missing values
- Fills `GarageType` using the most frequent category
- Fills `LotFrontage` using the mean
- Fills `MasVnrArea` using the median

### 5. Outlier Handling with IQR

The project defines an IQR-based outlier removal function:

```python
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
```

### 6. Encoding Categorical Data

The categorical `GarageType` feature is converted into numerical columns using one-hot encoding:

```python
df = pd.get_dummies(df, columns=['GarageType'], drop_first=True)
```

### 7. Data Visualization

The notebook uses Seaborn and Matplotlib to visualize relationships between features and the target variable.

Example visualization:

```python
sns.heatmap(df.corr(), annot=False)
```

### 8. Train-Test Split

The dataset is split into training and testing sets:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)
```

### 9. Train Multiple Linear Regression Model

The model is trained using Scikit-learn's `LinearRegression`:

```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm = lm.fit(x_train, y_train)
```

### 10. Make Predictions

Predictions are generated on the test set:

```python
y_pred = lm.predict(x_test)
```

### 11. Evaluate Model

The model is evaluated using the R² score:

```python
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
```

The notebook reports an R² score of approximately:

```text
0.7253
```

This means the model explains about 72.5% of the variance in the test target values.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Multi Linear Regression.ipynb
```

Then run the cells from top to bottom.

## Running in VS Code

You can also run this project in Visual Studio Code:

1. Open the project folder in VS Code.
2. Install the Python and Jupyter extensions.
3. Select your Python interpreter.
4. Open the notebook file.
5. Run each cell sequentially.

## Key Concepts Practiced

- Data loading with pandas
- Exploratory data analysis
- Missing value treatment
- IQR-based outlier handling
- One-hot encoding
- Correlation heatmap visualization
- Multiple linear regression
- Train-test split
- Model prediction
- R² score evaluation

## Possible Improvements

- Apply feature scaling where appropriate
- Compare Linear Regression with other regression models
- Use cross-validation
- Add more feature engineering
- Remove or transform highly skewed variables
- Evaluate with additional metrics such as MAE, MSE, and RMSE
- Save the trained model using `pickle` or `joblib`

## Conclusion

This project provides a beginner-friendly implementation of **Multiple Linear Regression** for house price prediction. It demonstrates the full machine learning workflow from data cleaning and preprocessing to model training, prediction, and evaluation.
