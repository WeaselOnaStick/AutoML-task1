import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y)


# Сохранение данных в папках "train" и "test"
np.savetxt("train/x_train.csv", x_train, delimiter=",")
np.savetxt("train/y_train.csv", y_train, delimiter=",")
np.savetxt("train/x_test.csv",  x_test, delimiter=",")
np.savetxt("train/y_test.csv",  y_test, delimiter=",")