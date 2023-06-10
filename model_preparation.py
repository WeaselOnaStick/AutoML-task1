import numpy as np
import pandas as pd
from sklearn import tree
import pickle

# Загрузка данных из файла
x_train = np.loadtxt("train/x_train_std.csv", delimiter=",")
y_train = np.loadtxt("train/y_train.csv", delimiter=",")


# Создание модели
model = tree.DecisionTreeClassifier()

# Обучение модели
model.fit(x_train,y_train)

# Сохранение модели
with open("model/model.pickle", "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    #np.save("model/model.npy", model)