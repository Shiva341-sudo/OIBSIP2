# 🚗 Car Selling Price Prediction & Regression Diagnostics

An end-to-end Machine Learning regression pipeline designed to predict the resale price of used cars based on their historical and physical attributes. This project implements data feature engineering, runs a baseline **Linear Regression** model backed by **5-Fold Cross-Validation**, and uses a **Random Forest Regressor** to determine feature hierarchies.


## 📌 Project Architecture

This project is built as a complete analytical pipeline broken down into distinct data science workflows:

### 1. Feature Engineering & Preprocessing
Raw automotive data requires strategic structuring before it can be used in linear models:
* **Temporal Transformation:** Converts a standard `Year` attribute into a dynamic `Car_Age` feature (`current_year - Year`) to extract linear depreciation trends.
* **Dimensionality Reduction:** Drops uninformative or overly complex textual identifiers like `Car_Name` to prevent overfitting.
* **One-Hot Encoding:** Transforms categorical constraints (`Fuel_Type`, `Selling_type`, and `Transmission`) into numeric dummy variables using `pd.get_dummies(drop_first=True)` to avoid the dummy variable trap.

### 2. Exploratory Data Analysis (EDA)
Understanding pricing mechanics through data visualizations:
* **Target Distribution:** Uses a kernel density estimation (`kde=True`) histogram to evaluate skewness in `Selling_Price`.
* **Multicollinearity Screening:** Generates an annotated feature correlation heatmap (`coolwarm` palette) to track relationships between independent features.
* **Scatter Pattern Tracking:** Maps correlations directly by plotting `Present_Price` vs. `Selling_Price` and `Car_Age` vs. `Selling_Price`.

### 3. Predictive Modeling & Evaluation
* **Supervised Learning Split:** Isolates features and targets into an 80/20 train-test split using fixed random states for experimental reproducibility.
* **Baseline Engine:** Trains a `LinearRegression` model to establish a linear pricing baseline.
* **Cross-Validation Robustness:** Protects the model against split bias by executing a `cross_val_score` pipeline running a **5-Fold Cross-Validation ($cv=5$)** optimizing for the $R^2$ coefficient.

### 4. Advanced Regression Diagnostics
Unlike classification tasks, regression requires rigorous diagnostic tracking to verify model assumptions:
* **Actual vs. Predicted Evaluation:** A scatter plot confirming whether the predicted prices line up linearly against true validation metrics.
* **Residual Error Analysis:** Plots residuals ($y_{test} - y_{pred}$) against predictions to check for homoscedasticity (even variance of error terms).
* **Algorithmic Interpretability:** Employs an independent `RandomForestRegressor` to calculate structural **Feature Importances**, ranking which vehicle attributes dictate market valuation most heavily.

---

## 🛠️ Technical Stack & Frameworks

| Category | Tools / Libraries Used |
| :--- | :--- |
| **Language** | Python 3.x |
| **Data Architecture** | `pandas`, `numpy` |
| **Visual Diagnostic Suites** | `seaborn`, `matplotlib` |
| **Machine Learning Core** | `scikit-learn` |

---

## 📊 Evaluation Metrics Glossary

The execution panel evaluates performance using industry-standard regression metrics:

* **Mean Absolute Error (MAE):** Represents the average absolute difference between the actual car prices and the model's predictions.
* **Mean Squared Error (MSE):** Penalizes larger pricing errors heavily by squaring the prediction deviations.
* **Root Mean Squared Error (RMSE):** Translates error metrics directly back into the target variable's unit currency for easy real-world interpretation.
* **$R^2$ Score (Coefficient of Determination):** Indicates the percentage of variance in vehicle pricing that our independent features successfully explain.

