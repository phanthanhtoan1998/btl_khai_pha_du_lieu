import pandas as pd
from sklearn import linear_model, neighbors 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
data =pd.read_csv("./btl.csv")

x=data.drop("LUNG_CANCER",axis=1)
y=data["LUNG_CANCER"].values 
for col in data.columns:
    missing_data = data[col].isna().sum()
    missing_percent = missing_data/len(data) * 100
    print(f"Column {col}:has { missing_percent} %")




imputer=SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(data)
data1 = imputer.transform(data)
# print(data1)

df = pd.DataFrame( data1, columns=['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC' 'DISEASE','FATIGUE','ALLERGY','WHEEZING','ALCOHOL' 'CONSUMING','COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN','LUNG_CANCER'
])
export_csv = df.to_csv ("2.csv", index = None, header=True) # here you have to write path, where result file will be stored



df.head
X=df.drop("LUNG_CANCER",axis=1)
Y=df["LUNG_CANCER"].values 

model = KMeans(n_clusters=3).fit(X)
print("test")

print (metrics.confusion_matrix(Y,model.predict(X)))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)




# x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2)
clf.fit(x_train, y_train)
kq1 = clf.predict(x_test)
print("Accuracy_Random:", clf.score(x_test, y_test))
print("confusion_matrix: ")
print(metrics.confusion_matrix(y_test, kq1))


