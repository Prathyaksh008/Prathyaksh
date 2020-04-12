#linear regression
#By Prathyaksh N P
#20171CSE0529


import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
data= pd.read_csv('state_wise_data.csv')  
x= data.iloc[:, :4].values  
y= data.iloc[:, 2].values  
from sklearn.preprocessing import LabelEncoder 
labelencoder_x= LabelEncoder()  
x[:, 2]= labelencoder_x.fit_transform(x[:,2])  

x = x[:, 1:]  
from sklearn.model_selection import train_test_split  
x_1, x_2, y_1, y_2= train_test_split(x, y, test_size= 0.4, random_state=0)  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_1, y_1)  
y_pred= regressor.predict(x_2) 
print('Train Score: ', regressor.score(x_1, y_1))  
print('Test Score: ', regressor.score(x_2, y_2))  