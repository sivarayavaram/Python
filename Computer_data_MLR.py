import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
import statsmodels as stm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
########### Multipe Linear regression - Computer data #################

# Computer data MLP
comp = pd.read_csv(".\Computer_Data.csv")
comp.columns

comp.drop("Unnamed: 0",inplace = True,axis=1)

des = comp.describe() # Saving to a dataset
comp.info() # To check if there are any NULL values in the dataset

# to get the unique values/ to delete the duplicate rows
comp.drop_duplicates(keep=False, inplace=True)

comp.columns

## Step 3: Visualizing the data

plt.figure(figsize=(18,9))
plt.subplot(1,2,1)
plt.title('Computer Price Distribution Plot')
sns.distplot(comp.price) # distribution plot

plt.subplot(1,2,2)
plt.title('Car Price Spread')
sns.boxplot(y=comp.price)
plt.show()

print(comp.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))
"""
Inference :
The plot seemed to be right-skewed, meaning that the most prices in the dataset are low(Below 2800).
(85% of the prices are below 2,800, whereas the remaining 15% are between 2,800 and 5,400.)
"""
## Step 3.1 : Visualising Categorical Data
# cd # multi # premium

plt.figure(figsize=(18, 9))

plt.subplot(1,3,1)
plt1 = comp.cd.value_counts().plot('bar')
plt.title('CD Histogram')
plt1.set(xlabel = 'CD', ylabel='Frequency of company')

plt.subplot(1,3,2)
plt1 = comp.multi.value_counts().plot('bar')
plt.title('Multi Histogram')
plt1.set(xlabel = 'Multi Type', ylabel='Frequency of fuel type')

plt.subplot(1,3,3)
plt1 = comp.premium.value_counts().plot('bar')
plt.title('Premium Histogram')
plt1.set(xlabel = 'Premium Type', ylabel='Frequency of Car type')

plt.show()

## Step 3.2 : Visualising numerical data
def pp(x,y,z):
    sns.pairplot(comp, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')
    plt.show()

sns.pairplot(comp,vars = ['speed', 'hd', 'ram','screen', 'ads','trend'],kind='scatter')

# OR
cor =  comp.corr(method='pearson')

## Step 6 : Dummy Variables

comp_lr = pd.get_dummies(comp, columns=['cd'])
comp_lr = comp_lr.iloc[:,:-1] # deleting/drop the last column

comp_lr = pd.get_dummies(comp_lr, columns=['multi'])
comp_lr = comp_lr.iloc[:,:-1] # deleting/drop the last column

comp_lr = pd.get_dummies(comp_lr, columns=['premium'])
comp_lr = comp_lr.iloc[:,:-1] # deleting/drop the last column

comp_lr.shape
comp_lr.columns
ml1 = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_no+multi_no+premium_no',data=comp_lr).fit() # regression model
ml1.summary() # 77.5
ml1.params

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

import statsmodels.stats.outliers_influence
"""
The resultant DataFrame contains six variables in addition to the DFBETAS. These are:

cooks_d         : Cook’s Distance defined in Influence.cooks_distance
standard_resid  : Standardized residuals defined in Influence.resid_studentized_internal
hat_diag        : The diagonal of the projection, or hat, matrix defined in Influence.hat_matrix_diag
dffits_internal : DFFITS statistics using internally Studentized residuals defined in Influence.dffits_internal
dffits          : DFFITS statistics using externally Studentized residuals defined in Influence.dffits
student_resid   : Externally Studentized residuals defined in Influence.resid_studentized_external
"""
infl = ml1.get_influence()
sm_fr = infl.summary_frame()
sm_fr.cooks_d.describe()
sm_fr1 = sm_fr.nlargest(3,['cooks_d']) # top n by column "cooks_d"

comp_lr=comp_lr.drop(comp_lr.index[[1700,1440]],axis=0)

ml_new = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_no+multi_no+premium_no',data=comp_lr).fit() # regression model
ml_new.summary() # 77.5

# Predicted values of MPG 
price_pred = ml_new.predict(comp_lr[['speed','hd','ram','screen','ads','trend','cd_no','multi_no','premium_no']])
price_pred

# calculating VIF's values of independent variables
rsq_sp = smf.ols('speed~hd+ram+screen+ads+trend+cd_no+multi_no+premium_no',data=comp_lr).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

rsq_rm = smf.ols('ram~speed+hd+screen+ads+trend+cd_no+multi_no+premium_no',data=comp_lr).fit().rsquared  
vif_rm = 1/(1-rsq_rm) 

rsq_hd = smf.ols('hd~speed+ram+screen+ads+trend+cd_no+multi_no+premium_no',data=comp_lr).fit().rsquared  
vif_hd = 1/(1-rsq_hd) 

rsq_scr = smf.ols('screen~hd+ram+speed+ads+trend+cd_no+multi_no+premium_no',data=comp_lr).fit().rsquared  
vif_scr = 1/(1-rsq_scr) 

rsq_ads = smf.ols('ads~hd+ram+screen+speed+trend+cd_no+multi_no+premium_no',data=comp_lr).fit().rsquared  
vif_ads = 1/(1-rsq_ads) 

rsq_trd = smf.ols('trend~hd+ram+screen+speed+ads+cd_no+multi_no+premium_no',data=comp_lr).fit().rsquared  
vif_trd = 1/(1-rsq_trd) 

rsq_cd = smf.ols('cd_no~hd+ram+screen+speed+trend+ads+multi_no+premium_no',data=comp_lr).fit().rsquared  
vif_cd = 1/(1-rsq_cd) 

rsq_mul = smf.ols('multi_no~hd+ram+screen+speed+trend+ads+cd_no+premium_no',data=comp_lr).fit().rsquared  
vif_mul = 1/(1-rsq_mul) 

rsq_prem = smf.ols('premium_no~hd+ram+screen+speed+trend+ads+multi_no+cd_no',data=comp_lr).fit().rsquared  
vif_prem = 1/(1-rsq_prem) 

# Storing vif values in a data frame
d1 = {'Variables':['speed','ram','hd','screen','ads','trend','cd_no','multi_no','premium_no'],
      'VIF':[vif_sp,vif_rm,vif_hd,vif_scr,vif_ads,vif_trd,vif_cd,vif_mul,vif_prem]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# As hd is having higher VIF value, we can exclude this in prediction model, 
#  since accuracy is going down we are not deleting for now

# Residual values for MPG
# residual = true_val - pred_val
price_resid = ml_new.resid
price_resid

# Converting Series to a DataFrame
comp_lr_1 = pd.DataFrame({'Predicted':price_pred, 'Residuals': price_resid})

# Adding Series to an existing Data Frame
comp_lr_2 = pd.concat([comp_lr, price_pred.rename('Predicted'),price_resid.rename('Residuals')], axis=1)

import statsmodels.api as sm
# added variable plot for the model
sm.graphics.plot_partregress_grid(ml_new)

######## Normality plot for residuals ######
# histogram
plt.hist(ml_new.resid_pearson) # Checking the standardized residuals are normally distributed
plt.show()
# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(ml_new.resid_pearson, dist="norm", plot=pylab)

############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(price_pred,ml_new.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train,test  = train_test_split(comp_lr,test_size = 0.2) # 20% size

# preparing the model on train data 
model_train = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_no+multi_no+premium_no',data=train).fit()

# train_data prediction
train_pred = model_train.predict(train)

# train residual values 
train_resid  = train_pred - train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) # 278.83

# prediction on test data set 
test_pred = model_train.predict(test)

# test residual values 
test_resid  = test_pred - test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) # 268.74




















