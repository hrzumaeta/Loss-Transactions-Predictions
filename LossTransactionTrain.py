# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:42:45 2017

@author: hzumaeta
"""

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score, roc_curve, auc
from sklearn.externals import joblib
from scipy.stats import randint as sp_randint
from matplotlib import pyplot as plt
from time import time
from operator import itemgetter

import pandas as pd
import numpy as np
import pyodbc

def reader():
    #The reader function allows the user to bring in data from a database using basic sql commands
    #There is CTE to deal with a log table of unstructured data and then binning within sql itself
    #to keep data transformation within the python script to a minimum
    server = ''
    database = ''
    username = ''
    password = ''
    driver = ''
    query = '''

    with returns as (
    select
    dl.logkey as producttype, --productid
    count(dl.id) as 'numberoftimesreturned'
    from deliverylog dl
    inner join orderrequest or on or.orderid=dl.key
    where
    dl.tablename='TransactionStatus'
    and dl.columnname='Status'
    and dl.changevalue='Returned'
    group by dl.logkey
    )

    select
    p.orderid,
    p.producttypename, --what type, fruit, clothing, tagged by seller
    p.producttypeid,
    p.case when revenue-cost < 0 then 1 else 0 end as 'losstransaction',
    p.customerid,
    case
    when c.annualprofit between 0 and 1000 then 1
    when c.annualprofit between 1000 and 20000 then 2
    when c.annualprofit between 2000 and 30000 then 3
    when c.annualprofit between 3000 and 40000 then 4
    when c.annualprofit between 4000 and 50000 then 5
    when c.annualprofit between 5000 and 60000 then 6
    when c.annualprofit between 6000 and 70000 then 7
    when c.annualprofit between 7000 and 80000 then 8
    when c.annualprofit between 9000 and 10000 then 9
    when c.annualprofit > 1000000 then 10
    else null end as 'customertier', --10 tiers based on market research of customer operating profit
    p.salelocationid,
    p.revenue,
    p.cost,
    p.revenue-p.cost as 'margin',
    (p.revenue-p.cost)/p.revenue as 'marginpercent',
    p.shipstate,
    p.storestate,
    cast (p.transactioncomplete as date) as 'transactiondate',
    p.commodityweight,
    isnull(r.numberoftimesreturned,0) as timesreturned
    from producttable p
    left join returns on r.producttypeid=p.producttypeid --left join here because we want products types that had returns or no returns
    inner join customertable c on p.customerid=c.customerid
    where p.revenue-p.cost between -2000 AND 100000 --cleaning for faulty manually inputted user data'''

    cnxn = pyodbc.connect('DRIVER='+driver+';PORT=1433;SERVER='+server+';PORT=1443;DATABASE='+database+';UID='+username+';PWD='+ password)
    d_frame = pd.read_sql_query (query, cnxn)
    return d_frame;


def split_dataset(dataset, train_percentage):
    #The pertinent variables were determined through exploratory data analysis, correlationa analysis and PCA#
    #Those steps were not shown here, neither was controlling for multicollinearity
    #Or normalizing the dataset
    #Or taking care of the class imbalance problem (most transactions aren't losses)
    #Or optimizing for training and prediction time by using a parallel computing frame work like Dask
    #The main goal here is just to show how to simply build a classification model from database tables with a bit of data wrangling
    #And then exposing this model through a REST API

    #Here we define which variables we will choose to train the model
    df=dataset
    include = ['customertier', 'timesreturned', 'shipstate', 'storestate','losstransaction']
    df_=df.filter(items=include)

    #This part turns categorical columns (if any) into dummy columns
    categoricals = []

    for col, col_type in df_.dtypes.iteritems(): #iteritems() provides a key value pair, col is key, col_type is value
        if col_type == 'O':
            categoricals.append(col)
        else:
              df_[col].fillna(0, inplace=True)

    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
    dependent_variable = 'losstransaction'
    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(x,y,train_size=train_percentage)
    return train_x, test_x, train_y, test_y


def main():

    #Read the data in
    df = reader()
    #split the dataset
    train_x, test_x, train_y, test_y = split_dataset(df, .7)

    #Build the random forest classifier, and fit it on the training data
    clf = rf(n_estimators=30).fit(train_x, train_y)
    #Use the model to predict on the test data
    predictions=clf.predict(test_x)

    #For the first five observations, print the actual and predicted values of the test data
    for i in range(0, 5):
           print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

    #Various classification metrics, such as accuracy, a confusion matrix for false and true positives and negatives, and roc score
    print ("Train Accuracy :: ", accuracy_score(train_y, clf.predict(train_x)))
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print ("Confusion matrix :: ", confusion_matrix(test_y, predictions))
    print ("ROC AUC :: ", roc_auc_score(test_y, predictions))

    fpr, tpr, thr = roc_curve(test_y, predictions) #false positive, true positive, threshold
    # Different way of calculating the AUC, helps with the plot
    roc_auc = auc(fpr, tpr)

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    #Pickling the model for later use
    joblib.dump(clf, 'lmmodel.pkl')

if __name__ == "__main__":
    main()
