#!/usr/bin/env python
# coding: utf-8

# In[ ]:


AUTHOR -REETU KHATRI


# # About the sales predictions 
Sales prediction refers to the process of forecasting or estimating future sales volumes or revenues for a particular product, service, or business. It involves analyzing historical sales data, identifying patterns and trends, and using that information to make predictions about future sales performance.

Sales predictions are crucial for businesses as they help in various decision-making processes such as production planning, inventory management, marketing strategy development, resource allocation, and financial forecasting. By accurately predicting sales, businesses can optimize their operations, identify potential growth opportunities, and make informed decisions to drive revenue and profitability.

To perform sales prediction, various techniques can be used, including time series analysis, regression analysis, machine learning algorithms, and data mining. These methods leverage historical sales data, along with relevant variables such as marketing spend, seasonality, economic indicators, customer demographics, and competitor data, to create models that can forecast future sales performance with a certain level of accuracy.
# # importing libraries

# In[ ]:





# In[71]:


# importing required libraray
import numpy as np   # linear algebra
import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

from scipy import stats 
#from pandas_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
#from autoviz.classify_method import data_cleaning_suggestions ,data_suggestions




# # pandas_profiling
# 
# from pandas_profiling import ProfileReport
# 
# Like pandas df. describe() function, that is so handy, pandas-profiling delivers an extended analysis of a DataFrame while alllowing the data analysis to be exported in different formats such as html and json. The package outputs a simple and digested analysis of a dataset, including time-series and text.
# 
# Pandas profiling is an open-source Python module with which we can quickly do an exploratory data analysis, it also generates interactive reports in web format. pandas profiling helps in visualizing and understanding the distribution of each variable. It generates a report with all the information easily available.

# # variance_inflation_factor
# 
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# 
# How do you interpret variance inflation factors?
# Variance Inflation Factor (VIF)
# In general terms,
# VIF equal to 1 = variables are not correlated.
# VIF between 1 and 5 = variables are moderately correlated.
# VIF greater than 5 = variables are highly correlated2.
# 
# variance_inflation_factor. The variance inflation factor is a measure for the increase of the variance of the parameter estimates if an additional variable, given by exog_idx is added to the linear regression. It is a measure for multicollinearity of the design matrix, exog.
# 

# # AutoViz
# 
# Automatically Visualize any dataset, any size with a single line of code.
# It detects missing values, identifies rare categories, finds infinite values, detects mixed data types, and so much more. This will help you tremendously speed

# # What is Regression, What Does It Do?
# 
# Regression
# 
# In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a dependent variable (often called the 'outcome' or 'response' variable, or a 'label' in machine learning parlance) and one or more independent variables (often called 'predictors', 'covariates', 'explanatory variables' or 'features'). The most common form of regression analysis is linear regression, in which one finds the line (or a more complex linear combination) that most closely fits the data according to a specific mathematical criterion. If you don't know, today I will tell you in detail about regression, one of the most important types of data analysis, and how regression analysis works. for ex.(rain is dependent variable ,and weather or clouds are independent variable )
# 
# 
# # What Does Regression Do?
# 
# We can say that regression analysis is used to estimate the value of the dependent variable depending on the independent variables.
# 
# Linear Regression
# Linear regression is used to model the relationship between two variables and estimate the value of a response by using a line-of-best-fit. best-fit-line show us linear relationship between varibales .
# 
# Regression analysis formula:
# 
# Formula Y = MX + b  , or  Y= MC+ b
# 
# Y is the dependent variable of the regression equation. M is the slope of the regression equation. X is the dependent variable of the regression equation. b is the constant of the equation.
# 
# If we want to examine it on an example. Advertising data sales (in thousands of units) for a particular product advertising budgets (in thousands of dollars) for TV, radio, and newspaper media

# # Downloading the dataset

# In[12]:


df=pd.read_csv(r"C:\Users\Suman\Downloads\Advertising.csv")
df


# In[13]:


df=pd.read_csv("Advertising.csv")


# In[7]:


df


# In[8]:


df.head()


# # EDA

# In[9]:


# find the shape of the data
df.shape


# In[45]:


df.drop('Unnamed: 0', axis = 1, inplace = True)


# In[46]:


# finding the info of dataset 
df.info()


# In[16]:


# finding the statistical measures of dataset
df.describe()


# In[27]:


# checking for null values
df.isnull().sum()


# In[48]:


#from autoviz.classify_method import data_cleaning_suggestions ,data_suggestions


# In[47]:


#data_cleaning_suggestions(df)


# # Plotting the sales variable vs each element seperately

# In[ ]:


# To check the relationship between two variables we can use scatter plots


# In[28]:


import plotly.graph_objects as go


# In[29]:


# creating an empty figure
fig = go.Figure()
fig


# In[35]:


fig.add_trace(go.Scatter(x=df['TV'], y=df['Sales'], mode='markers', marker=dict(color='red')))
## Here we update these values under function attributes such as title,xaxis_title and yaxis_title
fig.update_layout(title='TV ads Sales', xaxis_title='TV', yaxis_title='Sales')
# Display the figure
fig.show()


# 
coloraxis
            :class:`plotly.graph_objects.layout.Coloraxis` instance
            or dict with compatible properties
        colorscale
            :class:`plotly.graph_objects.layout.Colorscale`
            instance or dict with compatible properties
        colorway
            Sets the default trace colors.
#  #  TV

# In[56]:


# Let's try to estimate the advertising fees spent on TV ads based on product sales.

X = df[["TV"]]
y = df[["Sales"]]

# Model

reg_model = LinearRegression().fit(X, y)

# constant (b - bias)
b = reg_model.intercept_[0]

# coefficient of TV (M)
M = reg_model.coef_[0][0]

print("Linear regression parameters at : b = {0}, M = {1}".format(b, M))

reg_model.intercept
The intercept (sometimes called the “constant”) in a regression model represents the mean value of the response variable when all of the predictor variables in the model are equal to zero
# In[17]:


# plotting the dataset for relationship between all columns of ads


plt.figure(figsize=(7,5))
plt.plot(X, label='X', color='green')
plt.plot(y, label='y', color='red')
plt.title("TV ads on sales")
plt.legend()
plt.show


# In[18]:


df.columns


# 
The most common use of regression analysis is to predict future opportunities and threats. For this example, let's estimate what the sales would be if there were 200 units of spending on TV ads.
# In[37]:


reg_model.intercept_[0] + reg_model.coef_[0][0] * 200


# In[38]:


reg_model.intercept_[0] + reg_model.coef_[0][0]


# In[39]:


# Visualization of the Model
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'g', 's': 9},
                ci=False, color="r")
g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Sales")
g.set_xlabel("TV")
plt.xlim(-10, 310) # Get or set the x limits of the current axes.
plt.ylim(bottom=0)  # Get or set the y-limits of the current axes.
plt.show()


# # MSE (mean squared error)
Evaluating Forecast Success

MSE

In statistics, the mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. MSE is a risk function, corresponding to the expected value of the squared error loss. The MSE is a measure of the quality of an estimator. As it is derived from the square of Euclidean distance, it is always a positive value that decreases as the error approaches zero.


# In[40]:


# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)


# # RMSD- root-mean-square deviation (RMSD)
RMSE

The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. The RMSD represents the square root of the second sample moment of the differences between predicted values and observed values or the quadratic mean of these differences. RMSD is always non-negative, and a value of 0 (almost never achieved in practice) would indicate a perfect fit to the data. 
It measures the average difference between values predicted by a model and the actual values. It provides an estimation of how well the model is able to predict the target value (accuracy)


A value of 0 means that the predicted values perfectly match the actual values, but you'll never see that in practice. Low RMSE values indicate that the model fits the data well and has more precise predictions.

Conversely, higher values suggest more error and less precise predictions

# In[41]:


# RMSE
np.sqrt(mean_squared_error(y, y_pred))


# #  Radio

# In[43]:


fig.add_trace(go.Scatter(x=df['Radio'], y=df['Sales'], mode='markers', marker=dict(color='yellow')))
## Here we update these values under function attributes such as title,xaxis_title and yaxis_title
fig.update_layout(title='Radio Sales', xaxis_title='Radio', yaxis_title='Sales')
# Display the figure
fig.show()


# In[44]:


sns.lmplot(data=df,x='Radio',y="Sales")


# # Newspaper

# In[49]:


fig.add_trace(go.Scatter(x=df['Newspaper'], y=df['Sales'], mode='markers', marker=dict(color='blue')))
## Here we update these values under function attributes such as title,xaxis_title and yaxis_title
fig.update_layout(title='Newspaper Sales', xaxis_title='Newspaper', yaxis_title='Sales')
# Display the figure
fig.show()


# In[36]:



heat_maps=df[['TV', 'Radio', 'Newspaper', 'Sales']]


heat_maps = heat_maps.corr()

plt.figure(figsize=(10,5))
sns.set_context('notebook',font_scale=1)
sns.heatmap(heat_maps, annot=True,cmap='turbo_r');

Heat map
Using a heatmap to visualise a confusion matrix, time-series movements, temperature changes, correlation matrix and SHAP interaction values. Heatmaps can bring your data to life. Versatile and eye-catching. There are many situations where they can highlight important relationships in your data.
# In[ ]:


# splitting the data into attributes(X) and target variable(Y)


# In[50]:


X = df.drop('Sales', axis=1)


# In[51]:


X


# In[52]:


Y = df[['Sales']]
Y


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.20, random_state = 0)


# In[55]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=20)


# In[ ]:


#Importing Linear Regression Library


# In[57]:


from sklearn.linear_model import LinearRegression

# creating a Linear regression object
lr = LinearRegression()
lr


# In[ ]:


Model Training


# In[58]:


lr.fit(X_train, Y_train)


# In[59]:


#predictions

Y_predict = lr.predict(X_test)


# In[60]:


Y_predict


# # Model Evaluation

# In[61]:


from sklearn import metrics


# In[62]:


print('Mean Absolute Error: ',metrics.mean_absolute_error(Y_predict,Y_test))


# In[63]:


print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(Y_predict,Y_test)))


# In[64]:


print('R-Squared: ',metrics.r2_score(Y_predict,Y_test))


# In[67]:


act_predict=pd.DataFrame({
    'Actual':Y_test.values.flatten(),
    'Predict':Y_predict.flatten()
})
act_predict.head(20)


# In[68]:


sns.lmplot(data=act_predict,x='Actual',y="Predict")


Differences Between Correlation and Regression


Correlation and regression are often confused with each other because correlation can often lead to regression. However, there is an important difference between them.

The difference between these two statistical measures is that correlation measures the degree of relationship between two variables (x and y), whereas regression measures how one variable affects the other.

The regression determines how x causes y to change and how the results will change if x and y are changed. With correlation, x and y are variables that can be exchanged and get the same result. Correlation is a single statistic or data point while regression is the entire equation with all data points represented by a line. While correlation shows the relationship between two variables, regression allows us to see how one affects the other. Data represented by regression, when one changes, the other does not always change in the same direction, creating a cause and effect. With correlation, the variables move together.
Linear Regression Model: The code utilizes a linear regression model to predict sales based on advertising data. Linear regression assumes a linear relationship between the independent variables (TV, radio, newspaper) and the dependent variable (sales).

Training and Testing Data: The dataset is split into training and testing sets using an 80:20 ratio, respectively. The training set is used to train the model, while the testing set is used to evaluate its performance.

Model Evaluation: The performance of the model is evaluated using two metrics: Mean Squared Error (MSE) and R-squared (R2). MSE measures the average squared difference between the predicted and actual sales values. R2 represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

Prediction and Results: The model makes predictions on the testing data using the trained model. The predicted sales values (y_pred) are then compared to the actual sales values (y_test). The MSE and R2 metrics are calculated to assess the accuracy and goodness of fit of the model.

Interpretation: The lower the MSE, the better the model's predictive accuracy. R2 ranges from 0 to 1, with 1 indicating a perfect fit. Therefore, a higher R2 value suggests that a larger proportion of the sales variation can be explained by the advertising features.