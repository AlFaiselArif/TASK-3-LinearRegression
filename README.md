# House Price Prediction using Linear Regression

# Objective
To build and evaluate a **Linear Regression model** that predicts the price of a house based on its physical and location-based features. This is part of the AI & ML Internship Task 3.

# Dataset
The dataset includes information on houses such as:
- Area (in sq. ft.)
- Number of Bedrooms
- Number of Bathrooms
- Location indicators
- Price (target variable)

-  Total Records: ~980 before cleaning
-  Format: CSV (Comma-Separated Values)
-  Dataset:'C:/Users/alfai/Downloads/Housing.csv'
  
# Tools Used
- Python 3
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

# Workflow
# Import & Preprocess Dataset
- Loaded dataset into a pandas DataFrame
- Removed rows/columns with missing values
- Cleaned and selected important numeric features

#Feature Selection
Selected key numerical features for regression:
- `area`
- `bedrooms`
- `bathrooms`
Target:
- `price`

#Train-Test Split
- 80% Training Data
- 20% Testing Data
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
 Metric    Value        
 
 MAE      1265275.6699454375  
 MSE      2750040479309.0503
 R²        0.45

#Visualizations
🔸 Correlation Heatmap
Used to analyze which features most affect price.

🔸 Actual vs Predicted Price
plt.scatter(X_test['area'], y_test, color='blue', label='Actual')
plt.scatter(X_test['area'], y_pred, color='red', label='Predicted')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (₹)")
plt.title("Actual vs Predicted House Price")
plt.legend()
plt.show()
      

