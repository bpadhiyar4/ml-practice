import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#reading csv and store in data
data = pd.read_csv('dataset/india_population.csv', index_col=0)

#Converting dataframes into an 1-D arrays
x = np.array(data.index).reshape((-1, 1))
y = np.array(data.values).reshape((-1, 1))

#selecting model
model = LinearRegression()

#Splitting some data for the training and test.
x_trains, x_test, y_trains, y_test = train_test_split(x, y, random_state =1)

#fit the model on training set
model.fit(x_trains, y_trains)

#predict output based on model
y_predic = model.predict(x_test)

#Value of squared_error or root_sqaure which define accuracy of model
print(np.sqrt(metrics.mean_squared_error(y_test, y_predic)))

#potting data into a graph
plt.plot(x,y)
plt.scatter(x_test, y_predic, edgecolors='r')
plt.xlabel('Year')
plt.ylabel('Population in Billion')
plt.title('Population Prediction')
plt.show()