#  TCS Stock Price Prediction

A **Machine Learning regression project** that predicts the **Adjusted Closing Price of TCS stock** using historical stock market data and engineered time-series features.

The project includes data cleaning, exploratory data analysis, feature engineering, outlier treatment, feature scaling, multiple regression models, hyperparameter tuning, and model evaluation.

---

##  Project Objective

The main objective of this project is to use historical TCS stock market data to build and compare different machine learning regression models for predicting the **Adjusted Closing Price**.

The project also demonstrates how time-dependent stock data can be transformed into meaningful features such as lag values, moving averages, percentage changes, and volatility.

---

## Dataset

The project uses historical **TCS stock market data**.

The original dataset contains:

* **4,763 rows**
* **7 columns**
* Data ranging from **August 2002 to September 2021**

### Original Features

| Column    | Description                  |
| --------- | ---------------------------- |
| Date      | Trading date                 |
| Open      | Opening stock price          |
| High      | Highest price during the day |
| Low       | Lowest price during the day  |
| Close     | Closing stock price          |
| Adj Close | Adjusted closing price       |
| Volume    | Trading volume               |

---

##  Exploratory Data Analysis

The project performs exploratory analysis to understand the stock data and identify patterns.

### EDA Performed

* Dataset shape and structure
* Data types
* Descriptive statistics
* Missing-value analysis
* Duplicate-value analysis
* Skewness analysis
* Closing-price trend analysis
* Trading-volume trend analysis
* Closing-price distribution
* Outlier detection using box plots

The closing price and trading volume were visualized over time to understand historical stock behavior.

---

##  Data Preprocessing

Several preprocessing steps were performed before model training.

### 1. Date Conversion

The `Date` column was converted into datetime format and the dataset was sorted chronologically.

### 2. Missing Values

Missing values were handled using forward filling.

### 3. Invalid Zero Values

Zero values in the `Open`, `High`, `Low`, `Close`, and `Volume` columns were treated as missing values and forward-filled.

### 4. Duplicate Check

Duplicate records were checked during the data-cleaning stage.

### 5. Outlier Treatment

Outliers were detected using the **IQR (Interquartile Range)** method and values outside the lower and upper bounds were capped.

### 6. Feature Scaling

`MinMaxScaler` was used to transform numerical variables to a normalized range.

---

##  Feature Engineering

Additional features were created from the historical stock data.

### Engineered Features

* `Close_Lag1` — Previous day's closing price
* `MA7` — 7-day moving average
* `MA21` — 21-day moving average
* `Pct_Change` — Percentage change between Open and Close
* `Daily_Return` — Daily percentage return
* `Volatility` — 7-day rolling standard deviation of daily returns
* `DayOfWeek` — Day of the week extracted from the date

These features were created to provide the machine learning models with additional information about recent price behavior and market movement.

---

##  Target Variable

The target variable used for prediction is:

```text
Adj Close
```

The following features were used for model training:

```text
Open
High
Low
Volume
Close_Lag1
MA7
MA21
Pct_Change
Volatility
DayOfWeek
```

`Close` and `Daily_Return` were excluded from the modeling feature set because of target-information leakage concerns identified during the project.

---

##  Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

Because the data is time-dependent, shuffling was disabled:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
```

This preserves the chronological order of the observations.

---

#  Machine Learning Models

Multiple regression algorithms were trained and compared.

### Models Used

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Decision Tree Regressor
6. Random Forest Regressor
7. Gradient Boosting Regressor
8. Support Vector Regression (SVR)
9. K-Nearest Neighbors (KNN)

The project uses `Pipeline` with `StandardScaler` and the respective regression model. Hyperparameters were optimized using `GridSearchCV`.

---

##  Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used for models with hyperparameter grids.

Examples of tuned parameters include:

* Ridge → `alpha`
* Lasso → `alpha`
* ElasticNet → `alpha`, `l1_ratio`
* Decision Tree → `max_depth`
* Random Forest → `n_estimators`, `max_depth`
* Gradient Boosting → `n_estimators`, `learning_rate`
* SVR → `C`, `kernel`
* KNN → `n_neighbors`

The best-performing parameter configuration for each model was recorded.

---

#  Model Evaluation

The models were evaluated using three regression metrics:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

### RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

### R² Score

Measures how well the model explains the variation in the target variable.

---

## 📊 Model Results

The results obtained in the notebook were:

| Model                 |        MAE |       RMSE |         R² |
| --------------------- | ---------: | ---------: | ---------: |
| **Linear Regression** | **0.0437** | **0.0499** | **0.8970** |
| Ridge                 |     0.0441 |     0.0504 |     0.8952 |
| ElasticNet            |     0.0476 |     0.0535 |     0.8815 |
| Lasso                 |     0.0786 |     0.0822 |     0.7205 |
| Decision Tree         |     0.3142 |     0.3497 |    -4.0532 |
| Random Forest         |     0.3265 |     0.3608 |    -4.3798 |
| Gradient Boosting     |     0.3267 |     0.3611 |    -4.3880 |
| KNN                   |     0.3316 |     0.3645 |    -4.4902 |
| SVR                   |     0.5466 |     0.5839 |   -13.0919 |

Based on the notebook's reported test-set results, **Linear Regression achieved the lowest RMSE and MAE and the highest R² score** among the evaluated models.

---

## Best Performing Model

### Linear Regression

The Linear Regression model achieved:

* **MAE:** 0.0437
* **RMSE:** 0.0499
* **R²:** 0.8970

Based on these reported evaluation results, Linear Regression performed best among the models tested in this project.

---

##  Key Learnings

Through this project, I gained practical experience in:

* Stock market data analysis
* Exploratory Data Analysis
* Data cleaning
* Missing-value treatment
* Outlier detection and treatment
* Time-series feature engineering
* Moving averages
* Lag features
* Volatility calculation
* Feature scaling
* Regression modeling
* Hyperparameter tuning
* Model comparison
* Regression evaluation metrics
* Avoiding target leakage

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

##  Project Structure

```text
TCS-Stock-Price-Prediction/
│
├── stock_price_prediction.ipynb
├── stock_price_modelling.ipynb
├── stock_price_cleaned.csv
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/TCS-Stock-Price-Prediction.git
```

### 2. Navigate to the Project

```bash
cd TCS-Stock-Price-Prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks

Open the notebooks using Jupyter:

```bash
jupyter notebook
```

Run:

```text
stock_price_prediction.ipynb
```

first to perform preprocessing and generate the cleaned dataset.

Then run:

```text
stock_price_modelling.ipynb
```

to train and evaluate the machine learning models.

---

##  Important Note

This project is intended for **educational and portfolio purposes**.

Stock prices are influenced by many factors, and historical data alone cannot reliably predict future market movements. The model predictions should **not be considered financial advice or investment recommendations**.

---

##  Author

**Sindhuja Goud**

B.Tech — Computer Science & Engineering

---

 If you found this project useful, consider giving the repository a star!
