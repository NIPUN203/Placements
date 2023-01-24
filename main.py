import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

card_data=pd.read_csv('C:/Users/nipun/OneDrive/Desktop/ML_Projects/credit_card_data.csv')

# print(card_data.head())
# print(card_data.tail()) 

card_data.info()
# print(card_data.isnull().sum())

# print(card_data['Class'].value_counts())

X=card_data.drop(columns=['Class'], axis=1)
Y=card_data['Class']
print(X)
print(Y)

valid_transac=card_data[card_data.Class == 0]
fraud_transac=card_data[card_data.Class == 1]

# print(valid_transac)
# print(fraud_transac)

# print(valid_transac['Amount'].describe())
# print(fraud_transac['Amount'].describe())

# print(card_data.groupby('Class').mean())

bal_valid_trans=valid_transac.sample(n=492)
# print(bal_valid_trans)

# sns.histplot(card_data.Class)
# plt.show()


temp=card_data.drop(columns=['Time','Amount','Class'], axis=1)

# fig,ax=plt.subplots(ncols=4,nrows=7,figsize=(20,30))
# i=0
# ax=ax.flatten()

# for col in temp.columns:
#     sns.histplot(temp[col], ax=ax[i])
#     i+=1
# plt.tight_layout(pad=0.5,w_pad=0.5, h_pad=5)
#  plt.show()

# corr=card_data.corr()
# plt.figure(figsize=(50,100))
# sns.heatmap(corr,annot=True)
# plt.show()

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_scale=scale.fit_transform(X)

#splitting the data 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale, Y, test_size=.25, stratify=Y, random_state=42)


l_model=LogisticRegression()

l_model.fit(x_train,y_train)

train_predict=l_model.predict(x_train)
test_predict=l_model.predict(x_test)

# print("Trining data prediction")
# train_accuracy=accuracy_score(train_predict,y_train)
# print(train_accuracy)

# print("Testing data prediction")
# test_accuracy=accuracy_score(test_predict,y_test)
# print(test_accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test,test_predict))