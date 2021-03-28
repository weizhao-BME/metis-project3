#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file includes the process of data cleaning.

@author: Wei Zhao @ Metis, 01/28/2021
"""
#%%
import numpy as np
from util import *
import warnings
#%%
warnings.filterwarnings('ignore')
#%%
def reduce_feats():
    """
    Function to recude the number of features by selecting certain features

    Returns
    -------
    df_new : datafrome
        data frame with reduced feature.

    """
    q = """
    SELECT * from ma_refinance
    """
    df = do(q)

    feats = ['derived_msa-md', 'county_code', 'conforming_loan_limit',
             'derived_loan_product_type', 'derived_race', 'derived_sex',
             'lien_status', 'open-end_line_of_credit',
             'business_or_commercial_purpose', 'hoepa_status',
             'interest_only_payment', 'balloon_payment',
             'occupancy_type', 'total_units', 'applicant_race-1',
             'applicant_sex','applicant_age_above_62',
             'co-applicant_age_above_62',
             'loan_term', 'loan_amount', 'property_value',
             'loan_to_value_ratio', 'income',
             'debt_to_income_ratio', 'applicant_age', 'co-applicant_age',
             'action_taken', 'denial_reason-1',
             'denial_reason-2', 'denial_reason-3', 'denial_reason-4']

    df_new = df[feats]
    # sql does not identidy dash '-', so replace '-' with underline '_'
    columns = df_new.columns.str.replace('-', '_')
    df_new.columns = columns
    return df_new
#---------------------------------------------------------------------------
def drop_rows(df):
    """
    Function to drop all rows based on selected features.

    Parameters
    ----------
    df : data frame
        data frame before drop.

    Returns
    -------
    df : data frame
        data frame after drop.

    """
    df.dropna(subset=['income', 'debt_to_income_ratio'], axis=0, inplace=True)
    idx = df[df['income'] <= 0].index
    df = df.drop(idx, inplace=False)
    # exclude n/a
    idx = df[df['applicant_age'] == '8888'].index
    df = df.drop(idx, inplace=False)
    # exclude n/a
    idx = df[df['co_applicant_age'] == '8888'].index
    df = df.drop(idx, inplace=False)
    # exclude exempt
    idx = df[df['debt_to_income_ratio'] == 'Exempt'].index
    df = df.drop(idx, inplace=False)
    # exclude n/a
    idx = df[df['applicant_race_1'] == 7].index
    df = df.drop(idx, inplace=False)
    # exclude exempt
    idx = df[df['denial_reason_1'] == 1111].index
    df = df.drop(idx, inplace=False)
    # exclude county code n/a
    idx = df[np.isnan(df['county_code'])].index
    df = df.drop(idx, inplace=False)
    # exclude None propery value
    idx = df[df['property_value'] == 'Exempt'].index
    df = df.drop(idx, inplace=False)

    df.dropna(subset=['loan_term', 'loan_amount', 'property_value',
                      'loan_to_value_ratio', 'income',
                      'debt_to_income_ratio', 'applicant_age',
                      'co_applicant_age', 'conforming_loan_limit',
                      'applicant_race_1'
                      ], axis=0, inplace=True)

    return df
#---------------------------------------------------------------------------
def col_txt_split(t):
    """
    Function to split colum txt
    """
    tt = t.split(':')[-1].split(' ')[0]

    return tt
#---------------------------------------------------------------------------
def clean_col_txt(df):
    """
    Function to clean column texts.

    Parameters
    ----------
    df : data frame
        data frame before clean texts.

    Returns
    -------
    df : data frame
        data frame after clean texts.

    """
    df['derived_loan_product_type']= df['derived_loan_product_type'].apply(col_txt_split)
    df['co_applicant_age_above_62'] = df['co_applicant_age_above_62'].fillna('N/A')
    return df
#---------------------------------------------------------------------------
def clean_col_num(df):
    """
    Function to clean column numbers.

    Parameters
    ----------
    df : data frame
        data frame before clean numbers.

    Returns
    -------
    df : data frame
        data frame after clean numbers.

    """
    # 1 and 2 are approved; 3 is denied
    df['action_taken'][df['action_taken'] == 2] = 1
    df['action_taken'][df['action_taken'] == 3] = 0
    df['action_taken'] = df['action_taken'].astype(np.int64)
    return df
#---------------------------------------------------------------------------
def col_num2text(df):
    """
    Funcito to convert categorical values represented by numbers
    to text to facilitate ohe

    Parameters
    ----------
    df : pandas data frame
        df befor conversion.

    Returns
    -------
    df : pandas data frame
        df after conversion.

    """
    df[['derived_msa_md', 'county_code',
        'hoepa_status', 'interest_only_payment', 'balloon_payment',
        'occupancy_type', 'applicant_sex',
        'lien_status', 'open_end_line_of_credit',
        'business_or_commercial_purpose'
        ]] = df[['derived_msa_md', 'county_code',
        'hoepa_status', 'interest_only_payment', 'balloon_payment',
        'occupancy_type', 'applicant_sex',
        'lien_status', 'open_end_line_of_credit',
        'business_or_commercial_purpose'
        ]].astype(str)
    return df
#---------------------------------------------------------------------------
def col_numericalize(df):
    """
    Function to convert column-wise str to num.

    Parameters
    ----------
    df : data frame
        string column.

    Returns
    -------
    df : data frame
        numerical column.

    """
    # covnert range to a representative ratio using range average '<20%'--> (0+20)/2
    # '>60' --> (60+100)/2
    df['debt_to_income_ratio'][df['debt_to_income_ratio'] == '<20%'] = '20'
    df['debt_to_income_ratio'][df['debt_to_income_ratio'] == '20%-<30%'] = '25'
    df['debt_to_income_ratio'][df['debt_to_income_ratio'] == '30%-<36%'] = '33'
    df['debt_to_income_ratio'][df['debt_to_income_ratio'] == '50%-60%'] = '55'
    df['debt_to_income_ratio'][df['debt_to_income_ratio'] == '>60%'] = '60'
    df['debt_to_income_ratio'] = df['debt_to_income_ratio'].astype(np.int64)
    # do the same for age, but for age >74, set 80. For age < 25, set 20
    df['applicant_age'][df['applicant_age']=='55-64']=60
    df['applicant_age'][df['applicant_age']=='45-54']=50
    df['applicant_age'][df['applicant_age']=='65-74']=70
    df['applicant_age'][df['applicant_age']=='35-44']=40
    df['applicant_age'][df['applicant_age']=='>74']=80
    df['applicant_age'][df['applicant_age']=='25-34']=30
    df['applicant_age'][df['applicant_age']=='<25']=20
    #
    df['co_applicant_age'][df['co_applicant_age'] == '9999'] = 0
    df['co_applicant_age'][df['co_applicant_age'] == '35-44'] = 40
    df['co_applicant_age'][df['co_applicant_age'] == '45-54'] = 50
    df['co_applicant_age'][df['co_applicant_age'] == '65-74'] = 70
    df['co_applicant_age'][df['co_applicant_age'] == '55-64'] = 60
    df['co_applicant_age'][df['co_applicant_age'] == '>74'] = 80
    df['co_applicant_age'][df['co_applicant_age'] == '25-34'] = 30
    df['co_applicant_age'][df['co_applicant_age'] == '<25'] = 20
    #
    df['loan_to_value_ratio'] = df['loan_to_value_ratio'].astype(np.float64)
    df['property_value'] = df['property_value'].astype(np.int64)
    df['loan_term'] = df['loan_term'].astype(np.int64)
    df['county_code'] = df['county_code'].astype(np.int64)
    return df
#---------------------------------------------------------------------------
def total_age(df):
    """
    Function to add a feature that considers
    the total age of applicants and their co-applicants

    Returns
    -------
    df : pandas series
        Total age.

    """

    mask_co = df['co_applicant_age'] != 0
    mask = df['co_applicant_age'] == 0

    t_age_co = (df['applicant_age'][mask_co]
                + df['co_applicant_age'][mask_co])

    t_age = df['applicant_age'][mask]

    total_age = df['applicant_age'].copy()
    total_age = np.zeros_like(df['applicant_age'])

    total_age[mask_co] = t_age_co
    total_age[mask] = t_age
    df.insert(loc=df.shape[1]-7, column='total_age', value=total_age)
    return df
#---------------------------------------------------------------------------
def group_race(df):
    df['applicant_race_1'][df['applicant_race_1'] == 6] = 'No info.'
    df['applicant_race_1'][df['applicant_race_1'] == 5] = 'White'
    df['applicant_race_1'][df['applicant_race_1'] == 3] = 'Black'
    df['applicant_race_1'][(df['applicant_race_1'] == 2) |
                           (df['applicant_race_1'] == 21) |
                           (df['applicant_race_1'] == 22) |
                           (df['applicant_race_1'] == 23) |
                           (df['applicant_race_1'] == 24) |
                           (df['applicant_race_1'] == 25) |
                           (df['applicant_race_1'] == 26) |
                           (df['applicant_race_1'] == 27)
                           ] = 'Asian'
    df['applicant_race_1'][(df['applicant_race_1'] == 4) |
                           (df['applicant_race_1'] == 41) |
                           (df['applicant_race_1'] == 42) |
                           (df['applicant_race_1'] == 43) |
                           (df['applicant_race_1'] == 44) |
                           (df['applicant_race_1'] == 1)
                           ] = 'Others'
    return df
#---------------------------------------------------------------------------
def data_cleaning(df):
    """
    Function to run all of the
    relavant functions in data_cleaning.py

    Parameters
    ----------
    df : data frame
        raw data.

    Returns
    -------
    df : data frame
        clean data frame.

    """
    df = reduce_feats()
    df = drop_rows(df)
    df = clean_col_num(df)
    df = clean_col_txt(df)
    df = col_numericalize(df)
    df = total_age(df)
    df = group_race(df)
    df = col_num2text(df)
    return df

#%%
def main(df):
    """
    main function returs a clean data frame

    Returns
    -------
    df_new : pandas data frame
        clean data.

    """
    df_new = data_cleaning(df)
    return df_new

if __name__ == "__main__":
    df = main(df)
