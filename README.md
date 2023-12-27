# Credit Card Fraud Detection with Tensorflow
Repository for code (mostly jupyter notebook) for data loading, cleaning, preprocessing and engineering 
of 6 million credit card transactions to train a fraud detection 
autoencoder in tensorflow.

This project was undertaken as a data-scientist track finance data-challenge in the 
Correlation-One Data Skills 4 All program. 

The full notebook is outlined as follows:

### Exploratory Data Analysis and Pre-Processing

This was the initial step in readying a train-test-validation split ready
dataset to train an anomaly detection model to best find fraudulent transactions.
The data covered the years between 2016-2019 inclusive, and had just over 
6 million transactions for 2000 customers with multiple credit cards between
them in 3 separate datasets. The 3 datasets contained a mixture of Datetime data,
categorical data, and quantitative data.

Involved in the EDA process:
- Merging the three datasets on their customerID/creditcardID
- Cleaning data columns of null/na values
- Transforming string columns of transaction amounts to floats
- Splitting the date/time of transactions into YR/M/D/HR:Min:Sec columns


### Data Engineering and Feature Selection
The next step after merging and cleaning these datasets into one main dataset
was creating engineered data columns that would capture any relevant structure
to better sus out fraudulent transactions. The Datetime data was used in conjunction
with the credit card file data for each customer to establish features that may
indicate with fraudulent transactions such as:
- Time between transactions and credit card expirations
- Time between transactions and last PIN change date
- Time between transactions and account activation dates
- Time between transactions and customer retirement age


Categorical features were made with dummy variables on the
card brand, as well as types of errors (if any) present
in the transactions made with each credit card. Additional
qualitative features were engineered from various transaction
meta-data that might better indicate a fraudulent transaction
such as whether customer location and merchant location matched
at the city, ZIP, and state level. Additional categorical
features were engineered such as transaction type
(credit, debit, pre-paid), and if the transaction was
online, swipe, or chip processed.


Quantitative features were made to reflect different ratios
describing each customer and their credit cards such as
debt-income ratio, credit-debt ratio, credit-income ratio,
etc. This condenses the range of these floats down 
several magnitudes, which results in better performance
in the ensuing model fit.

There are 55 features for each transaction in the final data
set. 