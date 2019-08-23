# Recurrent Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def analyse(start_date):
    # Importing the training set
    dataset_total = pd.read_json('C:/Users/User/Desktop/Emporium/Emporium/web/NABIL_Data.json')
    training_set = dataset_total.iloc[:, 4:5].values

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)


    regressor = load_model("C:/Users/User/Desktop/Emporium/Emporium/web/Model.h5")

    close_data = dataset_total['Closing Price']
    data_list = dataset_total.values.tolist()  # convert dataframe into a list
    dataset_of_month = []
    #data of the month selected by the user
    for row in data_list:
        time = str(row[0])
        if (time[:7]) == start_date:
            dataset_of_month.append(row)
    
    #change in the stock price in the month selected by the user
    difference_lsit = []
    for i in range(len(dataset_of_month)-1):
        difference_lsit.append(dataset_of_month[i+1][4] - dataset_of_month[i][4])  
    #training set prepared by adding values of difference list to current day
    dataset_test = []
    dataset_test.append(data_list[-1][4])
    print(dataset_test)

    for i in range(len(difference_lsit)):
        dataset_test.append(dataset_test[i]+ difference_lsit[i])
    dataset_test = dataset_test[1:]
    dataset_test = pd.DataFrame(np.array(dataset_test))   


    close_data = pd.concat((dataset_total['Closing Price'], dataset_test[0]), axis = 0)
    inputs = close_data[len(close_data) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []

    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)   
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    analysed_stock_price = regressor.predict(X_test)
    analysed_stock_price = sc.inverse_transform(analysed_stock_price)

    analysed_stock_price = np.concatenate([training_set[-1:],analysed_stock_price])



    #prediction


    dataset_test = dataset_total[:-60]
    #dataset_test = pd.DataFrame(np.array(dataset_test), columns = ("Closing Price","High","Low","Close","Volume"))
    
    close_data = dataset_total['Closing Price'][len(dataset_total)-60:]
    inputs = close_data.values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    X_test.append(inputs)
    X_test.append(inputs)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test[1:])
    for i in range(19):
         X_test[1] = np.concatenate((X_test[1][1:], predicted_stock_price), axis = 0)
         predicted_stock_price = regressor.predict(X_test[1:])
    X_test = sc.inverse_transform(X_test[1])

    predicted_stock_price = X_test[40:]

    plt.figure(figsize=(12,7))
    # Visualising the results
    plt.plot(training_set[-60:], color = 'green', label = 'Real stock data')

    x_data = []
    for i in range(59,59+len(analysed_stock_price)):
        x_data.append(i)
    x_data = np.array(x_data)
    plt.plot(x_data, analysed_stock_price, color = 'blue', label = 'User given criteria')

    x_data = []
    for i in range(59,79):
        x_data.append(i)
    x_data = np.array(x_data)
    plt.plot(x_data, predicted_stock_price, color = 'red', label = 'Predicted data')
    #plt.plot(x_data, recent_month_stock_price, color = 'green', label = 'Recent driven')


    font = {'family': 'times',
        'color':  'black',
        'weight': 'normal',
        'size': 24,
        }
    plt.title(start_date+'\n ', loc='left', fontdict=font)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    #plt.axis('off')
    imgpath = '/static/image/graph.png'
    if os.path.exists(imgpath):
        os.remove(imgpath)
    plt.savefig('/static/image/graph.png')