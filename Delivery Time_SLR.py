############## Linear Regression - Delivery Time #################
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
import statsmodels as stm
import statsmodels.formula.api as smf
import statsmodels.api as sm

d_t = pd.read_csv(".\delivery_time.csv")
d_t.columns # 'Delivery Time', 'Sorting Time'

d_t.columns="dt","st" # Renaming column names
d_t = d_t.iloc[:,[1,0]] # Rearranging columns

plt.hist(d_t.dt)
plt.boxplot(d_t.dt,0,"rs",0)

plt.hist(d_t.st)
plt.boxplot(d_t.st)

plt.plot(d_t.st,d_t.dt,"bo");
plt.xlabel("Sorting Time");
plt.ylabel("Delivery Time")

d_t.st.corr(d_t.dt) # # correlation value between X and Y
np.corrcoef(d_t.st,d_t.dt)

# For preparing linear regression model we need to import the statsmodels.formula.api
model_dt=smf.ols("dt~st",data=d_t).fit()

# For getting coefficients of the varibles used in equation
model_dt.params

# P-values for the variables and R-squared value for prepared model
model_dt.summary() # Adj R2 --> 66.6%

model_dt.conf_int(0.05) # 95% confidence interval
model_dt.conf_int(0.01) # 99% confidence interval
model_dt.conf_int(0.1) # 90% confidence interval

pred_dt = model_dt.predict(pd.DataFrame(d_t['st'])) # Predicted values of Weight gained using the model

# Visualization of regresion line over the scatter plot of st and dt
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt

plt.scatter(x=d_t['st'],y=d_t['dt'],color='red');
plt.plot(d_t['st'],pred_dt,color='black');
plt.xlabel('Sorting Time');
plt.ylabel('Delivery Time')

pred_dt.corr(d_t.st) # 0.99

# Transforming variables for accuracy
model2 = smf.ols('dt~np.log(st)',data=d_t).fit()
model2.params
model2.summary() # Adj R2 = 67.9%

# Exponential transformation
model3 = smf.ols('np.log(dt)~st',data=d_t).fit()
model3.params
model3.summary() # Adj R2 = 69.6%

# SQRT transformation
model4 = smf.ols('dt~np.sqrt(st)',data=d_t).fit()
model4.params
model4.summary() # Adj R2 = 68.0%

# So model3 is giving more accuracy
pred_log = model3.predict(pd.DataFrame(d_t['st']))
pred_log
pred3=np.exp(pred_log)  # as we have used log(dt) in preparing model so we need to convert it back
pred3
pred3.corr(d_t.dt)
plt.scatter(x=d_t['st'],y=d_t['dt'],color='green');plt.plot(d_t.st,np.exp(pred_log),color='red');
plt.xlabel('Sorting Time');plt.ylabel('Delivery Time')

import statsmodels.api as sm
sm.graphics.influence_plot(model3,criterion='cooks',size=48)
# index 20 is showing high influence so we can exclude that entire row

# to drop rows 
dt_new = d_t.drop([20,8],axis = 'rows')

# Exponential transformation after deleting 20th observation
model5 = smf.ols('np.log(dt)~st',data=dt_new).fit()
model5.params
model5.summary() # Adj R2 = 83.4%


pred_log1 = model5.predict(pd.DataFrame(d_t['st']))
pred3_1=np.exp(pred_log1)  # as we have used log(dt) in preparing model so we need to convert it back
pred3_1
pred3_1.corr(d_t.dt)
plt.scatter(x=d_t['st'],y=d_t['dt'],color='green');plt.plot(d_t.st,np.exp(pred_log1),color='red');
plt.xlabel('Sorting Time');plt.ylabel('Delivery Time')

# Result: Exponential transformation is giving more accuracy.
