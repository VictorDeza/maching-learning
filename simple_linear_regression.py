# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando dataset
dataset = pd.read_csv('D:/Dataset/02 Simple Linear Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Separando data para entrenamiento y test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# Ajustando la regresion lineal simple con la data de entrenamiento
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Prediciendo los salarios de la data de prueba
Y_pred = regressor.predict(X_test)

# Visualizando los resultados de entrenamientos
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Conjunto de Entrenamiento)')
plt.xlabel('Años de Experiecia')
plt.ylabel('Salario')
plt.show()


# Visualizando los resultados de pruebas
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Conjunto de Pruebas)')
plt.xlabel('Años de Experiecia')
plt.ylabel('Salario')
plt.show()
