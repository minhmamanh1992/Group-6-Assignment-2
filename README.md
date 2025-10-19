# Group-6-Assignment-2
# Exploring Regression Techniques to Analyze and Predict Life Expectancy

## 1. Project Overview

This project explores the factors influencing global life expectancy using regression analysis. The primary goal is to build a predictive model based on 15 years of data (2000-2015) to identify the most significant determinants of longevity.

The insights from this model can help countries identify key areas for policy intervention to improve the life expectancy of their populations.

---

## 2. Dataset

* **Source:** World Health Organization (WHO) Global Health Observatory (GHO) and the United Nations (UN).
* **Scope:** Data from 193 countries, spanning the years 2000 to 2015.
* **Features:** The final dataset includes 22 columns (20 predictors, 1 year, 1 target variable) categorized into:
    * **Immunization Factors:** (e.g., `Hepatitis B`, `Polio`, `Diphtheria`)
    * **Mortality Factors:** (e.g., `Adult Mortality`, `infant deaths`)
    * **Economic Factors:** (e.g., `GDP`, `Total expenditure`)
    * **Social Factors:** (e.g., `Schooling`, `Income composition of resources`)
* **Target Variable:** `Life expectancy`

---

## 3. Installation & Requirements

To run the analysis, you will need Python 3.x and the following libraries. You can install them using pip:

pip install pandas numpy scikit-learn matplotlib seaborn
python your_script_name.py
## 4. How to Run the Code
Clone this repository to your local machine.

Ensure you have the dataset (Life Expectancy Data.csv) in the same directory.

Run the main Python script from your terminal:
python model_training.py
This script will perform the following steps:

Load and clean the dataset (as detailed in Section 5).

Train the regression model(s).

Evaluate the model's performance.

Output the key findings, including model coefficients and feature importance.

## 5. Data Cleaning & Preprocessing
A thorough data cleaning process was essential to prepare the dataset for modeling.

Column Name Cleaning: Removed leading/trailing whitespace from all column names for easier referencing (e.g., ' thinness 1-19 years' became 'thinness 1-19 years').

Handling Multicollinearity:

A correlation matrix revealed strong positive correlations between several predictor variables.

Examples: (infant deaths & under-five deaths), (GDP & percentage expenditure), and (Schooling & Income composition of resources).

Action: To prevent multicollinearity issues in the regression model, one variable from each highly correlated pair was dropped. We chose to drop:

percentage expenditure

Income composition of resources

infant deaths

Missing Value Imputation:

A tailored imputation strategy was used for each feature based on its distribution and the percentage of missing data.

Median Imputation (for skewed data):

Alcohol: Right-skewed distribution.

Population: Highly right-skewed distribution.

Mean Imputation (for normal data or few nulls):

Adult Mortality: Only 10 nulls, all from 2013. Imputed with the mean of that year.

Total expenditure: Roughly normal distribution.

Schooling: Roughly normal distribution.

BMI, Polio, Diphtheria, thinness 1-19 years, thinness 5-9 years: All had a very small number of missing values (<2%), so mean imputation was used.

Linear Interpolation (for temporal data):

Hepatitis B: Showed a clear increasing trend over the years. Interpolation was used to preserve this pattern.

GDP: Also showed a general increasing trend, making interpolation a suitable method.

Cleaning the Target Variable:

The target variable, Life expectancy, had 10 missing values.

Action: These rows were dropped from the dataset. Imputing the target variable would introduce bias, and the model should only be trained and evaluated on real, observed data.

