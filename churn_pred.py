import pandas as pd

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.head())
print(df.columns.tolist())
print(df.isnull().sum())
print(df.shape)
print(df.dtypes)

# cleaning

df= df.drop(columns=['customerID'])

df['TotalCharges'] =pd.to_numeric(df['TotalCharges'] , errors= 'coerce')
df['Churn'] = df['Churn'].map({ 'Yes':1 , 'No' :0})
df = df.dropna()
print(df.shape)
print(df['Churn'].unique())
print(df['Churn'].value_counts())

print(df['TotalCharges'].dtype)

# Encoding

df = pd.get_dummies(df, drop_first= True)

print(df.shape)
print(df.columns.tolist())

# pipeline :

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df.drop(columns=['Churn'])
Y= df['Churn']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y , test_size=0.2 , random_state=42)

# used pipeline:
Pipeline=Pipeline([
   ('scalar' , StandardScaler()),
   ('model', LogisticRegression(class_weight='balanced'))
])

Pipeline.fit(X_train,Y_train)
Y_pred = Pipeline.predict(X_test)

print('accuracy:',accuracy_score(Y_test, Y_pred))
print("\nClassification report", classification_report(Y_test,Y_pred))