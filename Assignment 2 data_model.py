import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk    


life = pd.read_csv('Life Expectancy Data.csv')
print(life.info())
#region -- Data Cleaning --
# we can see that there are some null values in the dataset, we'll handle them later by imputation technique, 
# We need to see the correlation between the numerical columns to identify significant variables for regression training.
# Compute correlation matrix for numerical columns
numerical_cols = life.select_dtypes(include=np.number).columns
corr_matrix = life[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Columns')
plt.show()
# as we can see that there are some strong correlations between certain variables, such as "under-five deaths" and "infant deaths","GDP" and "percentage expenditure", "Schooling" and "Income composition of resources", etc. 
# Those highly related variables can lead to multicollinearity issues in regression models, so we might consider dropping one of the correlated variables.
# Some column names have leading/trailing spaces, we can remove them for easier referencing
life.columns = life.columns.str.strip()

print(life.columns)


#further data preprocessing by treating null values and encoding categorical variables
life.isnull().sum()*100/len(life) # percentage of nulls in each column

# Adult mortality
life[life['Adult Mortality'].isna()] #ten rows with null Adult Mortality, all from 2013, we can impute them with the mean value in 2013

life['Adult Mortality'].fillna(life[life['Year']==2013]['Adult Mortality'].mean(), inplace=True) #there's only 10 rows so any null handling technique will not affect the overall data distribution

# Alcohol
life[life['Alcohol'].isna()] # 194 rows with null Alcohol, we can impute them with the mean value also
# observe the alcohol distribution
sns.histplot(life['Alcohol'], kde=True)
plt.title('Alcohol Distribution')
plt.show()
# it seems that the Alcohol distribution is right-skewed, so using median might be a better choice than mean
life['Alcohol'].fillna(life['Alcohol'].median(), inplace=True)

# Hepatitis B
life[life['Hepatitis B'].isna()] # 553 rows with null Hepatitis B, 18% of the data, so we should be careful when handling them, let's check if there's any pattern
life.groupby('Year')['Hepatitis B'].mean() 
# there is a trend of increasing Hepatitis B vaccination coverage over the years, interpolating missing values might be a good approach to preserve the trend
life['Hepatitis B'] = life['Hepatitis B'].interpolate(method='linear')

# BMI, Polio, Diphtheria and thinness
life[life['BMI'].isna()] # 34 rows with null BMI
life[life['Polio'].isna()] # 19 rows with null Polio
life[life['Diphtheria'].isna()] # 19 rows with null Diphtheria
life[life['thinness  1-19 years'].isna()] # 34 rows with null thinness
life[life['thinness 5-9 years'].isna()] # 34 rows with null thinness
# all of them have relatively small number of nulls, any null handling technique will not affect the overall data distribution, we can use mean imputation
life['BMI'].fillna(life['BMI'].mean(), inplace=True)
life['Polio'].fillna(life['Polio'].mean(), inplace=True)
life['Diphtheria'].fillna(life['Diphtheria'].mean(), inplace=True)
life['thinness  1-19 years'].fillna(life['thinness  1-19 years'].mean(), inplace=True)
life['thinness 5-9 years'].fillna(life['thinness 5-9 years'].mean(), inplace=True)

# Total expenditure
life[life['Total expenditure'].isna()] # 226 rows with null Total expenditure values
# observe the Total expenditure distribution
sns.histplot(life['Total expenditure'], kde=True)
plt.title('Total expenditure Distribution')
plt.show()
# it seems like a normal distribution, so using mean imputation is fine
life['Total expenditure'].fillna(life['Total expenditure'].mean(), inplace=True)

# GDP 
life[life['GDP'].isna()] # 448 rows with null GDP values
# observe the GDP distribution over the years
sns.lineplot(x='Year', y='GDP', data=life)
plt.title('GDP over Years')
plt.show()
# GDP shows an increasing trend over the years (despite some fluctuations in the later years), we can use interpolation to fill in the missing values
life['GDP'] = life['GDP'].interpolate(method='linear')
# check the GDP distribution after interpolation
sns.histplot(life['GDP'], kde=True)
plt.title('GDP Distribution after Interpolation')
plt.show()
# the distribution is not affected much by using linear interpolation, so we can proceed with this approach

# Population
life[life['Population'].isna()] # 652 rows with null Population values
# observe the Population distribution 
sns.histplot(life['Population'], kde=True)
plt.title('Population Distribution')
plt.show()
# a highly right-skewed distribution, so using median imputation might be a better choice
life['Population'].fillna(life['Population'].median(), inplace=True)

# Schooling
life[life['Schooling'].isna()] # 163 rows with null Schooling values
# observe the Schooling distribution 
sns.histplot(life['Schooling'], kde=True)
plt.title('Schooling Distribution')
plt.show()
# a roughly normal distribution, so using mean imputation is fine
life['Schooling'].fillna(life['Schooling'].mean(), inplace=True)

# Income composition of resources
life[life['Income composition of resources'].isna()] # 160 rows with null Income
# observe the Income composition of resources distribution
sns.histplot(life['Income composition of resources'], kde=True)
plt.title('Income composition of resources Distribution')
plt.show()
# roughly normal distribution, having outliers and some right-skewness, so median imputation might be a better choice
life['Income composition of resources'].fillna(life['Income composition of resources'].median(), inplace=True)

# Life expectancy (target variable)
life[life['Life expectancy'].isna()] # only 10 rows with null Life expectancy values, we'd rather drop them to avoid any potential bias
life.dropna(subset=['Life expectancy'], inplace=True)
#endregion

# Check again if there are any remaining nulls
print(life.isnull().sum())
# Now we can proceed to model training
#region -- Data preparation for training --
X = life.drop(columns=['Life expectancy'])
y = life['Life expectancy']
# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)  # drop_first=True to avoid dummy variable trap
print(X.info()) # verify no nulls and all numerical types       
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalized the data before fitting the model
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#endregion

#region -- Model Training and Evaluation --

# Train a Linear Regression Model
from sklearn.linear_model import LinearRegression
lmodel = LinearRegression()
lmodel.fit(X_train_scaled, y_train)
# Make predictions on the test set
y_pred = lmodel.predict(X_test_scaled)
# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error    
l_mse = mean_squared_error(y_test, y_pred)
l_rmse = root_mean_squared_error(y_test,y_pred)
l_r2 = r2_score(y_test, y_pred) 

# Analysis of model performance
print()
print('Linear Regression Model Results')
print(f'R^2 Score: {l_r2:.5f}') #R^2 = 0.95581 = 95.58%
# This suggest that 95.58% of the variation in Life Expectency value (y) can be explained by the predictors (X) used in the model.
print(f'Mean Squared Error: {l_mse:.3f}') # MSE = 3.822
print(f'Root Mean Squared Error: {l_rmse:.3f}') #RMSE = 1.955
# This suggest that, on average, the prediction by the Linear Regression model deviate from the true Life Expectancy value by 1.96 years (age)
# Considering that the values of Life Expectancy range from 40s to 80s, MSE = 3.82 and RMSE = 1.96 are small enough to indicate that the model performs well
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()
# As seem from the graph, the data points generally follows the best fit line.
# This suggest that the model performs well.
print()

# Train a Rigde regression model with different alpha values
from sklearn.linear_model import Ridge
alphas_values =[0.001, 0.01, 0.1, 1.0, 10.0,100.0]
ridge_results = [] #List to store the result
# Train the Rigde regression model using different alpha
for alpha in alphas_values:
    rmodel = Ridge(alpha=alpha)
    rmodel.fit(X_train_scaled, y_train)
    # Make predictions on the test set
    y_ridge_pred = rmodel.predict(X_test_scaled)
    # Evaluate the model
    r_mse = mean_squared_error(y_test, y_ridge_pred)
    r_rmse = root_mean_squared_error(y_test,y_ridge_pred)
    r_r2 = r2_score(y_test, y_ridge_pred)
    # Saves the result into the list
    ridge_results.append({
        'alpha': alpha,
        'r2': r_r2,
        'mse': r_mse,
        'rmse': r_rmse
    })
# Plot a graph to compare different alphas
alphas = [result['alpha'] for result in ridge_results]
r2_scores = [result['r2'] for result in ridge_results]
plt.semilogx(alphas, r2_scores, 'bo-')
plt.xlabel('Alpha (log scale)')
plt.ylabel('R^2 Score')
plt.title('Ridge Regression: Alpha vs R²')
plt.show()
print('Ridge Regression Results')
print(f'The best alpha is {alphas[r2_scores.index(max(r2_scores))]}', f'with R^2 = {max(r2_scores):.5f}')
# The graph shows that when alpha = 10^0 = 1 has the highest r^2 value, hence the best alpha
best_ridge_results = ridge_results[alphas_values.index(1.0)]
print(f'R^2 Score: {best_ridge_results["r2"]:.5f}') #R^2 = 0.95669 = 95.67%
# This suggest that 95.58% of the variation in Life Expectency value (y) can be explained by the predictors (X) used in the model.
print(f'Mean Squared Error: {best_ridge_results["mse"]:.3f}') # MSE = 3.746
print(f'Root Mean Squared Error: {best_ridge_results["rmse"]:.3f}') #RMSE = 1.936
# This suggest that, on average, the prediction by the Ridge Regression model deviate from the true Life Expectancy value by 1.94 years (age)
print()

# Implement k-fold Cross-Validation
from sklearn.model_selection import KFold, cross_val_score
# Define a 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=40)

#Results of k-fold CV for Linear Regression
l_cvscores = cross_val_score(lmodel, X_train_scaled, y_train, cv=kfold, scoring='r2')
l_msescores = -cross_val_score(lmodel, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_error')
l_rmsescores = -cross_val_score(lmodel, X_train_scaled, y_train, cv=kfold, scoring='neg_root_mean_squared_error')

print("k-fold (5-fold) Cross-Validation for Linear Regression")
print("Average R^2:", np.mean(l_cvscores)) #R^2 = 0.95592 = 95.59%
# This suggest that 95.59% of the variation in Life Expectency value (y) can be explained by the predictors (X) used in the model.
print("Average MSE:", np.mean(l_msescores)) #MSE = 4.0304
print("Average RMSE:", np.mean(l_rmsescores)) #RMSE = 2.0017
# This suggest that, on average, the prediction by the Ridge Regression model deviate from the true Life Expectancy value by 2.00 years (age)

#Results of k-fold CV for Ridge Regression for different alphas
k_ridge_results = []
for alpha in alphas_values:
    rmodel = Ridge(alpha=alpha)
    rmodel.fit(X_train_scaled, y_train)
    # Evaluate the model
    # Cross-validation R^2
    r_cvscores = cross_val_score(rmodel, X_train_scaled, y_train, cv=kfold, scoring='r2')
    # Cross-validation MSE
    r_msescores = -cross_val_score(rmodel, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_error')
    # Cross-validation RMSE
    r_rmsescores = -cross_val_score(rmodel, X_train_scaled, y_train, cv=kfold, scoring='neg_root_mean_squared_error')
    # Saves the result into the list
    k_ridge_results.append({
        'alpha': alpha,
        'mean_r2': np.mean(r_cvscores),
        'mean_mse': np.mean(r_msescores),
        'mean_rmse': np.mean( r_rmsescores)
    })
# Plot a graph to compare different alphas
k_alphas = [result['alpha'] for result in k_ridge_results]
k_r2_scores = [result['mean_r2'] for result in k_ridge_results]
plt.semilogx(k_alphas, k_r2_scores, 'bo-')
plt.xlabel('Alpha (log scale)')
plt.ylabel('R^2 Score')
plt.title('Ridge Regression: Alpha vs R²')
plt.show()
print("\nk-fold (5-fold) Cross-Validation for Ridge Regression")
print(f'The best alpha is {k_alphas[k_r2_scores.index(max(k_r2_scores))]}', f'with R^2 = {max(k_r2_scores):.5f}')
bestk_ridge_results = k_ridge_results[k_alphas.index(1.0)]
print(f'R^2 Score: {bestk_ridge_results["mean_r2"]:.5f}') #R^2 = 0.95632 = 95.63%
# This suggest that 95.63% of the variation in Life Expectency value (y) can be explained by the predictors (X) used in the model.
print(f'Mean Squared Error: {bestk_ridge_results["mean_mse"]:.3f}') # MSE = 3.998
print(f'Root Mean Squared Error: {bestk_ridge_results["mean_rmse"]:.3f}') #RMSE = 1.992
# This suggest that, on average, the prediction by the Ridge Regression model deviate from the true Life Expectancy value by 1.99 years (age)

#endregion

#region -- Analysis and Comparision --
# Get variable names
variable_names = X.columns.tolist()

# Get coefficients from Linear Regression
linear_coefficients = lmodel.coef_
linear_intercept = lmodel.intercept_

print("\n=== Linear Regression Coefficients ===")
print(f"Intercept: {linear_intercept:.4f}")

# Create a DataFrame for better visualization
coef_df_linear = pd.DataFrame({
    'Feature': variable_names,
    'Coefficient': linear_coefficients,
    'Abs_Coefficient': np.abs(linear_coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features (Linear Regression):")
# Keep only features that are not dummies
coef_df_linear = coef_df_linear[~coef_df_linear['Feature'].str.contains('Country_')]
coef_df_linear = coef_df_linear[~coef_df_linear['Feature'].str.contains('Status_')]
print(coef_df_linear.head(10))

# Get coefficients from Ridge Regression
best_ridge_alpha = 1.0
ridge_model = Ridge(alpha=best_ridge_alpha)
ridge_model.fit(X_train_scaled, y_train)

# Extract coefficients from the Ridge step
ridge_coefficients = ridge_model.coef_
ridge_intercept = ridge_model.intercept_

print("\n=== Ridge Regression Coefficients ===")
print(f"Intercept: {ridge_intercept:.4f}")

coef_df_ridge = pd.DataFrame({
    'Feature': variable_names,
    'Coefficient': ridge_coefficients,
    'Abs_Coefficient': np.abs(ridge_coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features (Ridge Regression):")
# Keep only features that are not dummies
coef_df_ridge = coef_df_ridge[~coef_df_ridge['Feature'].str.contains('Country_')]
coef_df_ridge = coef_df_ridge[~coef_df_ridge['Feature'].str.contains('Status_')]
print(coef_df_ridge.head(10))
#endregion
