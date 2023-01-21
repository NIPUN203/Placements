import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

card_data=pd.read_csv('C:/Users/nipun/OneDrive/Desktop/Ml-Project/credit_card_data.csv')

print(card_data.head())
print(card_data.tail())