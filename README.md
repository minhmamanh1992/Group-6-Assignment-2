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

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python your_script_name.py
