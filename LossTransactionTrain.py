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
from time import time
from operator import itemgetter

import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import numpy as np

def reader():
    server = ''
    database = ''
    username = ''
    password = ''
    driver = '{ODBC Driver 13 for SQL Server}'
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
    df=dataset
    include = ['customertier', 'timesreturned', 'shipstate', 'storestate','losstransaction']
    df_=df.filter(items=include)

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

    df = reader()
    train_x, test_x, train_y, test_y = split_dataset(df, .7)

    clf = rf(n_estimators=30).fit(train_x, train_y)
    #print("Train Accuracy :: ",clf.score(train_x,train_y))
    #print("Test Accuracy :: ",clf.score(test_x,test_y))
    predictions=clf.predict(test_x)

    for i in range(0, 5):
           print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

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

    joblib.dump(clf, 'lmmodel.pkl')

    clf = joblib.load('lmmodel.pkl')

    global model_columns
    model_columns = list(train_x.columns)
    joblib.dump(model_columns, 'lmmodel_columns.pkl')
    #######
    def report(grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")
    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(2, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(train_x, train_y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)

if __name__ == "__main__":
    main()
