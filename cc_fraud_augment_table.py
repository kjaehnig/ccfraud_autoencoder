import pandas as pd
import parquet
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DD = "/mnt/g/WSL/downloaded_ml_data/"

cctrans = pd.read_parquet(f"{DD}credit_card_dataset_fraud_detect/credit_card_transaction_data_de.parquet")
cccards = pd.read_parquet(f"{DD}credit_card_dataset_fraud_detect/sd254_cards_de.parquet")
ccusers = pd.read_parquet(f"{DD}credit_card_dataset_fraud_detect/credit_card_users_de.parquet")

# cctrans.loc[cctrans['Is Fraud?']=='Yes']
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

ccuser_merge_dict = {
    'User':[],
    'Current Age':[],
    'Retirement Age':[],
    'Birth Year':[],
    'Birth Month':[],
    'City':[],
    'State':[],
    'Per Capita Income - Zipcode':[],
    'Yearly Income - Person':[],
    'Total Debt':[],
    'FICO Score':[],
}

cccard_merge_dict = {
    'User':[],
    'CARD INDEX':[],
    'Expires':[],
    'CVV':[],
    'Has Chip':[],
    'Cards Issued':[],
    'Credit Limit':[],
    'Acct Open Date':[],
    'Year PIN last Changed':[],
    'Card on Dark Web':[],
}

# subsamp = 10000
for ii,row in tqdm(cctrans.iterrows(), total=cctrans.shape[0]):
    user_row = ccusers.loc[ccusers['User']==row['User']]
    cc_row = cccards.loc[
                    (cccards['User'].values==row['User']) &
                    (cccards['CARD INDEX'].values==row['Card'])
                    ]
    for ucol in list(ccuser_merge_dict.keys()):
        ccuser_merge_dict[ucol].append(user_row[ucol].squeeze())
    for ccol in list(cccard_merge_dict.keys()):
        cccard_merge_dict[ccol].append(cc_row[ccol].squeeze())


cctrans['User'] = cctrans['User'].astype('int64')
cctrans['CARD INDEX'] = cctrans['Card']
cccard_merge_df = pd.DataFrame(cccard_merge_dict, columns=cccard_merge_dict.keys())
ccuser_merge_df = pd.DataFrame(ccuser_merge_dict, columns=ccuser_merge_dict.keys())

print(cccard_merge_df)

cctrans1 = cctrans.merge(cccard_merge_df, on=['User', 'CARD INDEX'])
cctrans2 = cctrans1.merge(ccuser_merge_df, on=['User'])
cctrans2.to_feather("/mnt/g/WSL/downloaded_ml_data/credit_card_dataset_fraud_detect/credit_card_transactions_augmented.feather")


