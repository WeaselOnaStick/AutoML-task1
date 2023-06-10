import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# Загрузка данных из файла
x_test  = np.loadtxt("train/x_test_std.csv",  delimiter=",")
y_test  = np.loadtxt("train/y_test.csv",  delimiter=",")

# Загрузка модели
with open("model/model.pickle", "rb") as f:
    model = pickle.load(f)
    #model = np.load("model/model.npy")

# Получение предсказаний
y_pred = model.predict(x_test)

# Сохранение предсказаний
with open("test/score.txt", "w") as f:
    f.write(f"MODEL SCORE: {(accuracy_score(y_test,y_pred))}")