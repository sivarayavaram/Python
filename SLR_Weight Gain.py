import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
import statsmodels as stm
import statsmodels.formula.api as smf
import statsmodels.api as sm
############## Linear Regression - Weight gain #################

cal_cons = pd.read_csv(".\calories_consumed.csv")
cal_cons.columns # 'Weight gained (grams)', 'Calories Consumed'

cal_cons.columns="Weight","Calories" # Renaming column names
cal_cons = cal_cons.iloc[:,[1,0]] # Rearranging columns

plt.hist(cal_cons.Calories)
plt.boxplot(cal_cons.Calories,0,"rs",0) # Right skewed

plt.hist(cal_cons.Weight)
plt.boxplot(cal_cons.Weight) # Right skewed

plt.plot(cal_cons.Weight,cal_cons.Calories,"bo");
plt.xlabel("Weight");
plt.ylabel("Calories")

cal_cons.Calories.corr(cal_cons.Weight) # # correlation value between X and Y
np.corrcoef(cal_cons.Calories,cal_cons.Weight)

# For preparing linear regression model we need to import the statsmodels.formula.api
model=smf.ols("Weight~Calories",data=cal_cons).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary() # adjusted R2 = 0.888

model.conf_int(0.05) # 95% confidence interval
model.conf_int(0.01) # 99% confidence interval
model.conf_int(0.1) # 90% confidence interval

pred = model.predict(cal_cons.iloc[:,0]) # Predicted values of Weight gained using the model

# Visualization of regresion line over the scatter plot of Weight and Calories consumed
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt

plt.scatter(x=cal_cons['Calories'],y=cal_cons['Weight'],color='red');
plt.plot(cal_cons['Calories'],pred,color='black');
plt.xlabel('Calories');
plt.ylabel('Weight')

pred.corr(cal_cons.Calories) # 1.0

# LOG transformation
model2 = smf.ols('Weight~np.log(Calories)',data=cal_cons).fit()
model2.params
model2.summary() # Adj R2 = 79.2%

# Exponential transformation
model3 = smf.ols('np.log(Weight)~Calories',data=cal_cons).fit()
model3.params
model3.summary() # Adj R2 = 86.7

# SQRT transformation
model4 = smf.ols('Weight~np.sqrt(Calories)',data=cal_cons).fit()
model4.params
model4.summary() # Adj R2 = 84.5

# Result
## Applying transformation is decreasing Multiple R Squared Value. 
# So model doesnot need further transformation, Multiple R-squared:  0.888



