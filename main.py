import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("loan_data_set.csv")
data.drop(columns=['Loan_ID'], inplace=True)
data = data.dropna()
LS = pd.get_dummies(data['Loan_Status'], drop_first=True)
Gndr = pd.get_dummies(data['Gender'], drop_first=True)
Mrrd = pd.get_dummies(data['Married'], drop_first=True)
EduStat = pd.get_dummies(data['Education'], drop_first=True)
Self = pd.get_dummies(data['Self_Employed'], drop_first=True)
Prop = pd.get_dummies(data['Property_Area'])
Deps = pd.get_dummies(data['Dependents'])
data = data.drop(
    columns=['Loan_Status', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents'], axis=1)
data = pd.concat([LS, data, Gndr, Mrrd, EduStat, Self, Prop, Deps], axis=1)
y = data['Y']
x = data.drop(columns=['Y'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)
pred = model.predict(x_test)


#print(classification_report(y_test, pred))
#print('===================================')
#print(confusion_matrix(y_test, pred))


