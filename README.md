# LossTransaction
Sample code on fake data to predict whether a store that is purchasing products from a customer will face total losses on their transactions or not.

No real data included, but rather the process of cleaning messy database data and binning categories to form predictive value.

Scikit learn random forest classifier is used, as well as various scikit learn performance metrics to measure the predictive value of the models.

The end product is dumped into a pickle file and that pickle file is read by the Flask API.py file.

Since there is no real data, there is also no pickle file for the meantime.

The flask api has two endpoints "Predict" and "Shutdown". On startup the API reads the model pickle and its columns, if it doesn't exist throws an error. 

From there the endpoints are decorated with swagger (through flasgger) for documentation purposes.
