import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("newsample.csv")
print(df)
print(df.head())

print(df.tail())
print(df.shape)
print(df.info())
print(df.describe())
print(df[df['PlacedOrNot']==1])
print(df.isnull().sum())
df=df.dropna()
print(df)
print(df.isnull().sum())
print(df['Gender'].unique() )
df['Gender']=df['Gender'].map({'Male':1,'Female':0})
print(df.head())
print(df['Stream'].unique())
df['Stream']=df['Stream'].map({'Electronics And Communication':1,'Computer Science':2,'Information Technology':3,'Mechanical':4,'Electrical':5,'Civil':6})
print(df.head())
print(df.corr())
df.drop(df.columns[[0,1,2,5]],axis=1,inplace=True)
print(df)

import matplotlib.pyplot as plt
print(df.plot())
df.plot(kind='scatter',x='PlacedOrNot',y='CGPA')
print(plt.show())
df["PlacedOrNot"].plot(kind='hist')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop('PlacedOrNot', axis=1)
y = df['PlacedOrNot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


rf=RandomForestClassifier()
rf.fit(X_train,y_train)




y_pred5 = rf.predict(X_test)


from sklearn.metrics import accuracy_score



score5=accuracy_score(y_test,y_pred5)


print(score5)

final_data = pd.DataFrame({'Models':['RF'],
            'ACC':[score5*100]})

print(final_data)
pickle.dump(rf, open("model.pkl", "wb"))
