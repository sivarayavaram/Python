########### Multipe Linear regression - 50_Startups still needs to complete as per above program logic #################

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
import statsmodels as stm
import statsmodels.formula.api as smf
import statsmodels.api as sm


"""
Before proceeding with building a Linear Regression model there are certain assumptions which should be true…
1)LINEARITY
2)HOMOSCEDASTICITY
3)MULTIVARIATE NORMALITY
4)INDEPENDENCE OF ERRORS
5)LACK OF MULTICOLLINEARITY
"""
# 50_Startups
dataset = pd.read_csv(r".\50_Startups.csv")
dataset.columns
dataset.dtypes

dataset.columns="RD","Admin","MS","State","Profit" # Renaming column names
"""
If we take a look at our Dataset we can clearly see that State is a String type variable and like we have discussed,
We cannot feed String type variables into our Machine Learning model as it can only work with numbers.
To overcome this problem we use the Label Encoder object and create Dummy Variables using the OneHotEncoder object…
Lets say that if we had only 2 states New York and California namely in our dataset then 
our OneHotEncoder will be of 2 columns only… Similarly for n different states it would have n columns and each 
state would be represented by a series of 0s and 1s wherein all columns would be 0
except for the column for that particular state.
For ex:-
If A,B,C are 3 states then A=100,B=010,C=001

I think now you might be getting my point as to how the OneHotEncoder works…
Encoding categorical data

Avoiding the Dummy Variable Trap
The Linear Regression equation would look like —> y=b(0)+b(1)x(1)+b(2)x(2)+b(3)x(3)+b(4)D(1)+b(5)D(2)+b(6)D(3)…b(n+3)D(m-1)
Here D(1)…D(m-1) are the m dummy variable’s which we had defined earlier in LabelEncoder and OneHotEncoder
Well if you are sharp enough you might have noticed that the even though there are m dummy variables we have excluded the last dummy variable D(m)
The reason to that is a concept called Dummy Variable Trap in Machine Learning…and to avoid that we must always exclude the last Dummy Variable
If you are more interested then feel free to research a bit on Dummy Variable Trap!!
"""
# Dropping the 1st column out of the Dataset which contains one of the OneHotEncoded values…

dataset = pd.get_dummies(dataset, columns=['State'])
dataset = dataset.iloc[:,:-1] # deleting/drop the last column
#df.drop(df.columns[[-1,]], axis=1, inplace=True)
#If you want to delete/drop the last two columns, replace [-1,] by [-1, -2]

# Correlation matrix 
cor = dataset.corr()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(dataset)

y = dataset['Profit'] #Only dependent colum
x = dataset.drop(['Profit'],axis = 1) #Other columns

dataset.columns
import statsmodels.formula.api as smf # for regression model
#ml1 = smf.ols('y~x',data=dataset).fit()
ml1 = smf.ols('Profit~RD+Admin+MS+State_California+State_Florida',data=dataset).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary() # R2 = 94.5

# p-values for MS,State_California, State_Florida are more than 0.05 

# preparing model based only on MS
ml_ms=smf.ols('Profit~MS',data = dataset).fit()  
ml_ms.summary() 
# p-value <0.05 .. It is significant 

# preparing model based only on State_California
ml_State_California=smf.ols('Profit~State_California',data = dataset).fit()  
ml_State_California.summary() 
# p-value >0.05 .. It is not significant 

# preparing model based only on State_Florida
ml_State_Florida=smf.ols('Profit~State_Florida',data = dataset).fit()  
ml_State_Florida.summary() 
# p-value >0.05 .. It is not significant 

# Preparing model based only on MS, State_California & State_Florida
ml_MS_Cal_Flo=smf.ols('Profit~MS+State_California+State_Florida',data = dataset).fit()  
ml_MS_Cal_Flo.summary() 

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

import statsmodels.stats.outliers_influence
infl = ml1.get_influence()
sm_fr = infl.summary_frame()
sm_fr.cooks_d.describe()
sm_fr1 = sm_fr.nlargest(3,['cooks_d']) # top n by column "cooks_d"

# Studentized Residuals = Residual/standard deviation of residuals
dataset_new=dataset.drop(dataset.index[[49,48]],axis=0)

# Preparing model                  
ml_new = smf.ols('Profit~RD+Admin+MS+State_California+State_Florida',data = dataset_new).fit()    
ml_new.summary() # 0.958

# calculating VIF's values of independent variables
rsq_rd = smf.ols('RD~Admin+MS+State_California+State_Florida',data=dataset_new).fit().rsquared  
vif_rd = 1/(1-rsq_rd) 

rsq_Admin = smf.ols('Admin~RD+MS+State_California+State_Florida',data=dataset_new).fit().rsquared  
vif_Admin = 1/(1-rsq_Admin) 

rsq_MS = smf.ols('MS~RD+Admin+State_California+State_Florida',data=dataset_new).fit().rsquared  
vif_MS = 1/(1-rsq_MS) 

rsq_sc = smf.ols('State_California~RD+Admin+MS+State_Florida',data=dataset_new).fit().rsquared  
vif_sc = 1/(1-rsq_sc)

rsq_sf = smf.ols('State_Florida~RD+Admin+MS+State_California',data=dataset_new).fit().rsquared  
vif_sf = 1/(1-rsq_sf)

# Storing vif values in a data frame
d1 = {'Variables':['RD','Admin','MS','State_California','State_California'],'VIF':[vif_rd,vif_Admin,vif_MS,vif_sc,vif_sf]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Removing State columns since their P-value is more 
ml_ = smf.ols('Profit~RD+Admin+MS',data = dataset_new).fit()    
ml_.summary() # 0.96

import statsmodels.api as sm
# added variable plot for the model
sm.graphics.plot_partregress_grid(ml_)

# Deleting Admin snce it has negative correlation compared to MS
data_n = dataset_new.drop(["Admin",'State_California', 'State_Florida'],axis=1) # to drop/delete columns

ml_f = smf.ols('Profit~RD+MS',data = data_n).fit()    
ml_f.summary() # 0.95.9

# Predicted values of Profit 
price_pred = ml_f.predict(data_n[['RD', 'MS']])
price_pred



# Residual values for MPG
# residual = true_val - pred_val
price_resid = ml_f.resid
price_resid

# Converting Series to a DataFrame
comp_lr_1 = pd.DataFrame({'Predicted':price_pred, 'Residuals': price_resid})

# Adding Series to an existing Data Frame
comp_lr_2 = pd.concat([data_n, price_pred.rename('Predicted'),price_resid.rename('Residuals')], axis=1)

import statsmodels.api as sm
# added variable plot for the model
sm.graphics.plot_partregress_grid(ml_f)

######## Normality plot for residuals ######
# histogram
plt.hist(ml_f.resid_pearson) # Checking the standardized residuals are normally distributed
plt.show()
# QQ plot for residuals 
import pylab          
import scipy.stats as st
# Checking Residuals are normally distributed
st.probplot(ml_f.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######
# Residuals VS Fitted Values 
plt.scatter(price_pred,ml_f.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



















































