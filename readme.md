# Ploynomial Regression
## _The Last Markdown Editor, Ever_



Polynomial regression, like linear regression, uses the relationship between the variables x and y to find the best way to draw a line through the data points.
### Libraries:
- sklearn
- pandas
- matplotlib

## Imports

```python
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
import matplotlib as plt
from sklearn.preprocessing import PolynomialFeatures as PF
```

## Read File

```python
dataset = pd.read_csv('file.csv')
```
reading values from rows and columns
```python
X = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[1]].values
```

```python
X_train, X_train, y_train, y_test = tts(X,y)
```
```python
polyModel = PF(degree=5)
linearModel = LR()
polynomialFeatures = polyModel.fit_transform(X.reshape(-1,1))
linearModel.fit(polynomialFeatures,y)
y_pred = linearModel.predict(polynomialFeatures)
```

ploting points on graph
```python
plt.scatter(X,y)
plt.plot(X,y_pred)
plt.title("Polynomial Regression")
plt.xlabel("Variable")
plt.ylabel("Value")
plt.show()
```
