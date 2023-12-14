import pandas as pd
import parquet
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DD = "/mnt/g/WSL/downloaded_ml_data/"

#cctrans = pd.read_parquet(f"{DD}credit_card_dataset_fraud_detect/credit_card_transaction_data_de.parquet")
#cccards = pd.read_parquet(f"{DD}credit_card_dataset_fraud_detect/sd254_cards_de.parquet")
#ccusers = pd.read_parquet(f"{DD}credit_card_dataset_fraud_detect/credit_card_users_de.parquet")

cctrans = pd.read_csv("/mnt/g/WSL/downloaded_ml_data/credit_card_dataset_fraud_detect/credit_card_transactions-ibm_v2.csv")
cccards = pd.read_csv("/mnt/g/WSL/downloaded_ml_data/credit_card_dataset_fraud_detect/sd254_cards.csv")
ccusers = pd.read_csv("/mnt/g/WSL/downloaded_ml_data/credit_card_dataset_fraud_detect/sd254_users.csv")

if 'Person' in ccusers.columns:
    ccusers['User'] = ccusers['Person']
if "Person" in cccards.columns:
    cccards['User'] = cccards['Person']


cctrans.loc[cctrans['Is Fraud?']=='Yes']
cctrans['Fraud'] = cctrans['Is Fraud?'].apply(lambda x:1 if x=='Yes' else 0).astype('bool')
assert cctrans.loc[cctrans['Is Fraud?']=='Yes'].shape[0] == cctrans['Fraud'].sum()
hr = cctrans.copy()['Time'].apply(lambda x: x.split(":")[0])
min = cctrans.copy()['Time'].apply(lambda x: x.split(":")[1])

# print(hr)
cctrans['Hour'] = hr
cctrans['Minute'] = min
cctrans['Hour_decimal'] = hr.astype("float") + (min.astype("float")/60.)
cctrans['Amount'] = cctrans['Amount'].str.strip("$")

ccusers.loc[:,'User'] = ccusers.index

onlyfrauds = cctrans.loc[cctrans.Fraud==1]
print(onlyfrauds.shape)
onlyfrauds.sort_values("Hour_decimal", inplace=True)
onlyfraudsbymerchant = onlyfrauds.groupby('Merchant State')['Fraud'].count().reset_index()
# print(onlyfraudsbymerchant)

cctrans['CARD INDEX'] = cctrans['Card']

cctrans_aug = cctrans.merge(cccards, on=['User','CARD INDEX'])

cctrans_aug = cctrans_aug.merge(ccusers, on='User')
print(cctrans_aug.shape)

#### stripping dollar signs and un-needed strings from values
cctrans_aug['Amount'] = cctrans_aug['Amount'].str.strip("$").astype("float32")
cctrans_aug['Per Capita Income - Zipcode'] = cctrans_aug['Per Capita Income - Zipcode'].str.strip('$').astype('float32')
cctrans_aug['Total Debt'] = cctrans_aug['Total Debt'].str.strip("$").astype('float32')
cctrans_aug['Credit Limit'] = cctrans_aug['Credit Limit'].str.strip("$").astype('float32')
cctrans_aug['Yearly Income - Person'] = cctrans_aug['Yearly Income - Person'].str.strip("$").astype('float32')

cctrans_aug['Same City'] = cctrans_aug['Merchant City'] == cctrans_aug['City']
cctrans_aug['Same City'] = cctrans_aug['Same City'].astype('int32')

cctrans_aug['Same State'] = cctrans_aug['Merchant State'] == cctrans_aug['State']
cctrans_aug['Same State'] = cctrans_aug['Same State'].astype('int32')

cctrans_aug = cctrans_aug.join(pd.get_dummies(cctrans_aug['Card Brand']).astype('int32'))

time_bw_open_trans = pd.to_datetime(cctrans_aug[['Year','Month','Day']])- pd.to_datetime(cctrans_aug['Acct Open Date'],format='%m/%Y')
cctrans_aug['Time Since Opening'] = time_bw_open_trans.dt.days/365.25

time_bw_pin_trans = pd.to_datetime(cctrans_aug[['Year','Month','Day']]) - pd.to_datetime(cctrans_aug['Year PIN last Changed'],format='%Y')
cctrans_aug['Time Since PIN Change'] = time_bw_pin_trans.dt.days/365.25

cctrans_aug['Time Till Retire'] = cctrans_aug['Retirement Age'] - cctrans_aug['Current Age']
print(cctrans_aug['Time Till Retire'].min(), cctrans_aug['Time Till Retire'].median(), cctrans_aug['Time Till Retire'].max())

cctrans_aug['Zipcode Income Ratio'] = cctrans_aug['Per Capita Income - Zipcode']/cctrans_aug['Yearly Income - Person']
print(cctrans_aug['Zipcode Income Ratio'].min(), cctrans_aug['Zipcode Income Ratio'].median(), cctrans_aug['Zipcode Income Ratio'].max())

cctrans_aug['Debt Income Ratio'] = cctrans_aug['Total Debt'] / cctrans_aug['Yearly Income - Person']
print(cctrans_aug['Debt Income Ratio'].min(), cctrans_aug['Debt Income Ratio'].median(), cctrans_aug['Debt Income Ratio'].max())

time_till_retire = pd.to_datetime(cctrans_aug['Expires'], format='%m/%Y') - pd.to_datetime(cctrans_aug[['Year','Month','Day']])
cctrans_aug['Time Till Expire'] = time_till_retire.dt.days/365.25
print(cctrans_aug['Time Till Expire'].min(), cctrans_aug['Time Till Expire'].median(), cctrans_aug['Time Till Expire'].max())

card_type_dummies = pd.get_dummies(cctrans_aug[['Card Type']])
cctrans_aug = cctrans_aug.join(card_type_dummies)

cctrans_aug['Errors?'] = cctrans_aug['Errors?'].apply(lambda x: 'No Errors' if str(x)=='None' else x)

error_type_dummies = pd.get_dummies(cctrans_aug['Errors?'])
cctrans_aug = cctrans_aug.join(error_type_dummies)

for cc in error_type_dummies.columns:
    cctrans_aug[cc] = cctrans_aug[cc].astype('int32')

cctrans_aug['Has Chip'] = cctrans_aug['Has Chip'].apply(lambda x: 1 if x=='Yes' else 0)
chip_type_dummies = pd.get_dummies(cctrans_aug['Use Chip'])
cctrans_aug = cctrans_aug.join(chip_type_dummies)

cctrans_aug['Fraud'] = cctrans_aug['Fraud'].astype('int32')

cctrans_aug.to_feather("/mnt/g/WSL/downloaded_ml_data/credit_card_dataset_fraud_detect/credit_card_transactions_augmented.feather")

print("DONE.")

