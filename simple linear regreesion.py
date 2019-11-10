import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


#Creating instance of Linear Regression and fit the  data
#h(x) = theta(0) + theta(1)*x1  known as hypothesis function
model = LinearRegression().fit(x,y)

#Finding the root_square value or square_error which find the optimal value
r_sq = model.score(x, y)
print("Coefficient of determination : " , r_sq)

#values of theta(0) scalar value
print("Intercept : " , model.intercept_)

#value of theta(1) array
print("Slop : " , model.coef_)

#predicting output based on the trained model
y_pred = model.predict(x)
print("Predicted Response : ", y_pred, sep='\n')

x_new = np.arange(5).reshape((-1, 1))
y_new = model.predict(x_new)
print(y_new)