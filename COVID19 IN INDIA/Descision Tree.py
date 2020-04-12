#linear regression
#By Prathyaksh N P
#20171CSE0529



import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier 
  
#importing datasets  
data= pd.read_csv('state_wise_data.csv')  
  
#Extracting Independent and dependent Variable  
x= data.iloc[:, [1,2]].values  
y= data.iloc[:, 4].values  
  

 
x_1, x_2, y_1, y_2= train_test_split(x, y, test_size= 0.25, random_state=0)  
  

   
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_1)    
x_test= st_x.transform(x_2)

 
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_1)  

y_pred= classifier.predict(x_test)  
print(y_pred)