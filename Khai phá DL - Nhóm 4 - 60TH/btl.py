from calendar import day_abbr
import pandas as pd
from PIL import ImageTk
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


# # roi rac hoa du lieu
# data["AGE"]=pd.cut(data['AGE'] , bins=3, labels=[0, 1, 2])
# print("data sau khi roi rac ")
# print(data)

# x=data.drop("LUNG_CANCER",axis=1)
# y=data["LUNG_CANCER"].values
print("dulieugoc")
print(data)


# x['AGE'] = pd.cut(x['AGE'] , bins=3, labels=[0, 1, 2])
x= data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# các cột có missing_data 
for col in data.columns:
    missing_data = data[col].isna().sum()
    missing_percent = missing_data/len(data) * 100
    print(f"Column {col}:has { missing_percent} %")

from sklearn.impute import SimpleImputer
#  sử lý missing_data
print(x)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(x[:, 2:5])

x[:, 2:5] = imputer.transform(x[:, 2:5])

print("xulymissingdata")
print(x)



print(x[2:3,2:3])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Mã hoá các dữ liệu danh mục
cx=ColumnTransformer(transformers=[('GENDER',OneHotEncoder(),[0])],remainder="passthrough")
x = cx.fit_transform(x)
print(x[2])

from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.preprocessing import StandardScaler 
# co dãn dữ liệu
sc=StandardScaler()
x[:,2:3]=sc.fit_transform(x[:,2:3])
print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)




from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2).fit(x)
y_pred_kmeans = kmeans.predict(x)

print("confusion_matrix Kmeans: ")
print(metrics.confusion_matrix(y, y_pred_kmeans))

print(len(x_test[1]))
# x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2)
clf.fit(x_train, y_train)
kq1 = clf.predict(x_test)
print("Accuracy_Random:", clf.score(x_test, y_test))
print("confusion_matrix: ")
print(metrics.confusion_matrix(y_test, kq1))






from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter

# Khởi tạo giao diện
windown = Tk()
windown.geometry("800x600")
windown.title("Dự đoán tỉ lệ ung thư phổi")
windown.configure(bg='Aqua')
# Tạo lable, input
from PIL import Image
img =Image.open('.//1.jpg')
bg = ImageTk.PhotoImage(img)



# Add image
label = Label(windown, image=bg)
label.place(x = 0,y = 0)

lable1 = tkinter.Label(windown, width=30, text="Giới tính", bg="#00ff00", fg="black", font=("Arial,21"))
lable1.grid(column=1, row=1, padx=60)

lable2 = tkinter.Label(windown, width=30, text="tuổi", bg="#00ff00", fg="black", font=("Arial,21"))
lable2.grid(column=1, row=2, padx=60)

lable3 = tkinter.Label(windown, width=30, text="Thói quen hút thuốc", bg="#00ff00", fg="black", font=("Arial,21"))
lable3.grid(column=1, row=3, padx=60)

lable4 = tkinter.Label(windown, width=30, text="Màu sắc ngón tay", bg="#00ff00", fg="black", font=("Arial,21"))
lable4.grid(column=1, row=4, padx=60)

lable5 = tkinter.Label(windown, width=30, text="Trạng thái lo âu", bg="#00ff00", fg="black", font=("Arial,21"))
lable5.grid(column=1, row=5, padx=60)

lable6 = tkinter.Label(windown, width=30, text="Áp lực cuộc sống", bg="#00ff00", fg="black", font=("Arial,21"))
lable6.grid(column=1, row=6, padx=60)

lable7 = tkinter.Label(windown, width=30, text="Tình trạng mãn tính", bg="#00ff00", fg="black",
                       font=("Arial,21"))
lable7.grid(column=1, row=7, padx=60)

lable8 = tkinter.Label(windown, width=30, text="Sự mệt mỏi", bg="#00ff00", fg="black", font=("Arial,21"))
lable8.grid(column=1, row=8, padx=60)

lable9 = tkinter.Label(windown, width=30, text="Dị ứng", bg="#00ff00", fg="black", font=("Arial,21"))
lable9.grid(column=1, row=9, padx=60)

lable10 = tkinter.Label(windown, width=30, text="Hơi thở khò khè", bg="#00ff00", fg="black",font=("Arial,21"))
lable10.grid(column=1, row=10, padx=60)

lable11 = tkinter.Label(windown, width=30, text="Mức độ tiêu thụ rượu", bg="#00ff00", fg="black",font=("Arial,21"))
lable11.grid(column=1, row=11, padx=60)

lable12 = tkinter.Label(windown, width=30, text="Tần suất ho", bg="#00ff00", fg="black",font=("Arial,21"))
lable12.grid(column=1, row=12,padx =60)
lable13 = tkinter.Label(windown, width=30, text="Tần suất thở gấp", bg="#00ff00", fg="black",font=("Arial,21"))
lable13.grid(column=1, row=13, padx=60)
lable14 = tkinter.Label(windown, width=30, text="Sự khó khăn khi nuốt", bg="#00ff00", fg="black",font=("Arial,21"))
lable14.grid(column=1, row=14, padx=60)
lable15 = tkinter.Label(windown, width=30, text="Sự tức ngực", bg="#00ff00", fg="black",font=("Arial,21"))
lable15.grid(column=1, row=15, padx=60)

#
# txtlable1 = Entry(windown, width=23)
# txtlable1.grid(column=3, row=1, pady=3)
txtlable1 = Combobox(windown, width=20)
txtlable1.grid(column=3, row=1, pady=3)
txtlable1['value'] = ("Nam", "Nữ")
txtlable1.current(0)
txtlable2 = Entry(windown, width=23)
txtlable2.grid(column=3, row=2, pady=3)

txtlable3 = Entry(windown, width=23)
txtlable3.grid(column=3, row=3, pady=3)

txtlable4 = Entry(windown, width=23)
txtlable4.grid(column=3, row=4, pady=3)

txtlable5 = Entry(windown, width=23)
txtlable5.grid(column=3, row=5, pady=3)

txtlable6 = Entry(windown, width=23)
txtlable6.grid(column=3, row=6, pady=3)

txtlable7 = Entry(windown, width=23)
txtlable7.grid(column=3, row=7, pady=3)

txtlable8 = Entry(windown, width=23)
txtlable8.grid(column=3, row=8, pady=3)

txtlable9 = Entry(windown, width=23)
txtlable9.grid(column=3, row=9, pady=3)

txtlable10 = Entry(windown, width=23)
txtlable10.grid(column=3, row=10, pady=3)

txtlable11 = Entry(windown, width=23)
txtlable11.grid(column=3, row=11, pady=3)

txtlable12 = Entry(windown, width=23)
txtlable12.grid(column=3, row=12, pady=3)
txtlable13 = Entry(windown, width=23)
txtlable13.grid(column=3, row=13, pady=3)

txtlable14 =Entry(windown, width=23)
txtlable14.grid(column=3, row=14, pady=3)

txtlable15 = Entry(windown, width=23)
txtlable15.grid(column=3, row=15, pady=3)

lable0=''




def btn():

    lable1 = txtlable1.get()
    lable2 = txtlable2.get()
    lable3 = txtlable3.get()
    lable4 = txtlable4.get()
    lable5 = txtlable5.get()
    lable6 = txtlable6.get()
    lable7 = txtlable7.get()
    lable8 = txtlable8.get()
    lable9 = txtlable9.get()
    lable10 = txtlable10.get()
    lable11 = txtlable11.get()
    lable12 = txtlable12.get()
    lable13 = txtlable13.get()
    lable14 = txtlable14.get()
    lable15 = txtlable15.get()

    if (lable1 == "" or lable2 == "" or lable3 == "" or lable4 == "" or lable5 == ""
            or lable6 == "" or lable7 == "" or lable8 == "" or lable9 == "" or lable10 == ""
            or lable11 == ""or lable12 == ""or lable13 == ""or lable14 == ""or lable15 == ""):
        messagebox.showwarning("ALert!", "Điền đầy đủ dữ liệu!")
    # elif(type(lable1) != int or float or type(lable2) != int or float
    #       or type(lable3) != int or float or type(lable4) != int or float or type(lable5) != int or float or lable6 != "" or type(lable7) != int or float)or type(lable8) != int or float or (lable9) != "" or type(lable10) != int or float or type(lable11) != int or float or type(lable12) != int or float or type(lable13) != int or float or type(lable14) != int or float or type(lable15) != int or float :
    #     messagebox.showwarning("ALert!", "Dữ liệu phải là số!")

    else:
        if (lable1 == "Nam"):
            lable1 = 1
            lable0=0
        else:
            lable1 = 0
            lable0 = 1
        datatest = np.array(
            [[float(lable0), float(lable1), float(lable2), float(lable3), float(lable4), float(lable5)
                 , float(lable6), float(lable7), float(lable8),
              float(lable9), float(lable10)
                 , float(lable11), float(lable12), float(lable13),
              float(lable14), float(lable15)
              ]])
        print(datatest)
        result = kmeans.predict(datatest)
        if (result == 1):
            result = "bị ung thư"
        else:
            result = "Không"
        lableR = tkinter.Label(windown, width=20, text="kmeans:" + result, bg="#fff", fg="black", font=("Arial,21"))
        lableR.grid(column=3, row=22, pady=10)

        km = clf.predict(datatest)
        print("km")
        print(km)
        if (km == 1):
            km = "bị ùng thư"
        else:
            km = "Không "
        lableR = tkinter.Label(windown, width=40, text="RandomForestClassifier:" + km, bg="#fff", fg="black", font=("Arial,21"))
        lableR.grid(column=3, row=24, pady=10)
    return


# def exit():
#     exit()
#     return


btnTest = Button(windown, width=20, text="Test", command=btn)
btnTest.grid(column=1, row=20)
btnExit = Button(windown, width=20, text="Exit", command=exit)
btnExit.grid(column=1, row=23)

windown.mainloop()