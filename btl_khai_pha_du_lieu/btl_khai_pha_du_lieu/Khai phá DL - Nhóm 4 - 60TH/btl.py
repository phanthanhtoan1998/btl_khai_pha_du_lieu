from calendar import day_abbr
import pandas as pd
from sklearn import datasets, linear_model, neighbors 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
data =pd.read_csv("./dulieu1.csv")




# x=data.drop("LUNG_CANCER",axis=1)
# y=data["LUNG_CANCER"].values 
x= data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# các cột có missing_data 
for col in data.columns:
    missing_data = data[col].isna().sum()
    missing_percent = missing_data/len(data) * 100
    print(f"Column {col}:has { missing_percent} %")


from sklearn.impute import SimpleImputer
#  sử lý missing_data

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(x[:, 2:5]) 

x[:, 2:5] = imputer.transform(x[:, 2:5])

print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#  Mã hoá các dữ liệu danh mục 
cx=ColumnTransformer(transformers=[('GENDER',OneHotEncoder(),[0])],remainder="passthrough")
x=cx.fit_transform(x)
print(x)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)

from sklearn.preprocessing import StandardScaler 
# co dãn dữ liệu  

sc=StandardScaler()
x[:,2:3]=sc.fit_transform(x[:,2:3])
print(x)

# export_csv = df.to_csv ("2.csv", index = None, header=True) # here you have to write path, where result file will be stored



# df.head
# X=df.drop("LUNG_CANCER",axis=1)
# Y=df["LUNG_CANCER"].values 

# model = KMeans(n_clusters=3).fit(X)
# print("test")

# print (metrics.confusion_matrix(Y,model.predict(X)))

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)




# # x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)
# from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(max_depth=2)
# clf.fit(x_train, y_train)
# kq1 = clf.predict(x_test)
# print("Accuracy_Random:", clf.score(x_test, y_test))
# print("confusion_matrix: ")
# print(metrics.confusion_matrix(y_test, kq1))


