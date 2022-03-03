import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as PF

dataset = pd.read_csv('file.csv')

X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[1]].values

X_train, X_train, y_train, y_test = tts(X,y)

polyModel = PF(degree=5)
linearModel = LR()
polynomialFeatures = polyModel.fit_transform(X.reshape(-1,1))
linearModel.fit(polynomialFeatures,y)
y_pred = linearModel.predict(polynomialFeatures)
plt.scatter(X,y)
plt.plot(X,y_pred)
plt.title("Polynomial Regression")
plt.xlabel("Variable")
plt.ylabel("Value")
plt.show()