import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()

# Загрузка данных из файла
x_train = np.loadtxt("train/x_train.csv", delimiter=",")
x_test  = np.loadtxt("train/x_test.csv",  delimiter=",")

# Стандартизация данных
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std  = scaler.transform(x_test)

# Сохранение предобработанных данных

np.savetxt("train/x_train_std.csv", x_train_std, delimiter=",")
np.savetxt("train/x_test_std.csv",  x_test_std, delimiter=",")