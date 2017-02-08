#Stock Prices prediction app
#Date 5th Feb 2017



import csv #our file is csv
import numpy as np #numpy for calculations
from sklearn.svm import SVR #for  building predictive model Support Vector Regression
import matplotlib.pyplot as plt #for plotting graphs

#download data from google finance
dates=[]
prices=[]

def get_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)

        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return

#SVM- Support Vector Machine is a linear separator
#what it dows is if we have classified data available and an unclassified is presented at input then it tries to classify it
#by %%%   |  ooo
#   %%% % | o oo
#   %%%   |  ooo
#now it depends on where the input lies and the smallest distance is takenup and the input is classified ...so here it is a classification problem

#but what we have is a regression problem
#so we use SVR for it

def predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    svr_lin = SVR(kernel='linear',C=1e3)
    svc_poly= SVR(kernel='poly',C=1e3,degree=2)
    svc_rbf= SVR(kernel= 'rbf',C=1e3,gamma=0.1)

    svr_lin.fit(dates,prices)
    svc_poly.fit(dates,prices)
    svc_rbf.fit(dates,prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svc_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svc_poly.predict(dates), color='blue', label='Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    return svc_rbf.predict((x)[0]),svr_lin.predict((x)[0]),svc_poly.predict((x)[0])

get_data('aapl.csv')
predicted_prices = predict_prices(dates,prices,29)

print(predicted_prices)


