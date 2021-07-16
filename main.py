import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
data = pd.read_csv("credit_train.csv")
data.drop(columns=['Loan ID', 'Customer ID', 'Months since last delinquent', 'Years in current job', 'Tax Liens'], inplace=True)
data = data.dropna()
LS = pd.get_dummies(data['Loan Status'], drop_first=True)
Ownr = pd.get_dummies(data['Home Ownership'])
Prp = pd.get_dummies(data['Purpose'])
Trm = pd.get_dummies(data['Term'], drop_first=True)
data = data.drop(columns=['Loan Status', 'Home Ownership', 'Purpose', 'Term'], axis=1)
data = pd.concat([LS, data, Ownr, Prp, Trm], axis=1)
y = data['Fully Paid']
x = data.drop(columns=['Fully Paid'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model = LogisticRegressionCV(max_iter=200, Cs=50)
model.fit(x_train, y_train)
pred = model.predict(x_test)


#print(classification_report(y_test, pred))
#print('===================================')
#print(confusion_matrix(y_test, pred))


