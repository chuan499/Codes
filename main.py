# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import  pandas as pd
import numpy as np
import  matplotlib
from  sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv("iris.data",header = None)
    x,y = data[np.arange(4)],data[4]
    y = LabelEncoder().fit_transform(y)
    model = RandomForestClassifier(n_estimators=20,criterion="entropy",max_depth=5,min_samples_split=7,min_samples_leaf=3,random_state=2020)
    model.fit(x,y)
    y_pred = model.predict(x)

    result = y_pred == y
    print(result)
    print(pd.value_counts(result))
    print("正确率：",np.mean(result))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
