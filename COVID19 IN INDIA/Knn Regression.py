#linear regression
#By Prathyaksh N P
#20171CSE0529



import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
  

data= pd.read_csv('state_wise_data.csv')  
  
 
x= data.iloc[:, [2,3]].values  
y= data.iloc[:, 2].values  
  

x_1, x_2, y_1, y_2= train_test_split(x, y, test_size= 0.25, random_state=0)  
print(x_1, x_2, y_1, y_2)
  

st_x= StandardScaler()    
x_1= st_x.fit_transform(x_1)    
print(x_1)
x_2= st_x.transform(x_2)  
print(x_2)


classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
print(classifier.fit(x_1, y_1))


y_pred= classifier.predict(x_2) 
print(y_pred)

