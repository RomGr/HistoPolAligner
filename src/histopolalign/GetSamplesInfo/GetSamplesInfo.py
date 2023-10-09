import pandas as pd
import math
import os
from datetime import date
import numpy as np

def return_samples_info():
    df = pd.read_excel(os.path.join(os.getcwd().split('notebooks')[0], 'data', 'samples_paper.xlsx'))
    print('The samples were obtained from ' + str(len(df)) + ' different samples.')
    
    df1 = df.loc[df['Kept'] == 'BF, AF']
    df_final = pd.concat([df1, df], axis = 0)
    print('We had a total of ' + str(len(df_final)) + ' measurements.' + '\n')
    
    df_final = df_final.drop(['Unnamed: 0'], axis = 1)
    
    print('Proportion Wowen = ' + str(100 * sum(df_final['Sex'] == 'W')/len(df_final)) + '\n')
    print('Mean Age = ' + str(np.mean(df_final['Date of Birth'])) + '\n')
    tumor_types, tumor_types_proportions = get_tumor_type(df_final)
    print('Tumor types:')
    print(tumor_types)
    print('\n')
    print('Tumor types proportion:')
    print(tumor_types_proportions)
    print('\n')
    return df_final


def get_tumor_type(df_final):
    tumor_types = {'meningioma': 0, 'GBM': 0, 'astro': 0, 'oligo': 0, 'carcinoma': 0, 'papilloma': 0}
    tumor_types_W = {'meningioma': 0, 'GBM': 0, 'astro': 0, 'oligo': 0, 'carcinoma': 0, 'papilloma': 0}
    tumor_types_age = {'meningioma': 0, 'GBM': 0, 'astro': 0, 'oligo': 0, 'carcinoma': 0, 'papilloma': 0}


    for idx, row in df_final.iterrows():
        type_tumor = row['Histology Tumor']
        
        found = False
        for key, val in tumor_types.items():
            if key.lower() in type_tumor.lower():
                found = True
                tumor_types[key] = val + 1
                
                if row['Sex'] == 'W':
                    tumor_types_W[key] = tumor_types_W[key] + 1
                else:
                    pass
                
                tumor_types_age[key] = tumor_types_age[key] + row['Date of Birth']
        if found:
            pass
        else:
            raise ValueError
            
    tumor_types_proportions = {}
    for key, val in tumor_types.items():
        try:
            tumor_types_proportions[key] = val / len(df_final) * 100
            tumor_types_W[key] = (tumor_types_W[key] / val) * 100
            tumor_types_age[key] = (tumor_types_age[key] / val)
        except:
            pass
        
    return tumor_types, tumor_types_proportions

def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def get_kept_samples(df):
    current_sample_nr = None
    current_sex = None
    current_tumor_type = None
    current_age = None

    for idx, row in df.iterrows():
        try:
            if not math.isnan(row['Sample Nr']):
                current_sample_nr = row['Sample Nr']
            else:
                df.at[idx, 'Sample Nr'] = current_sample_nr
        except:
            pass
            
        if type(row['Sex']) == str:
            current_sex = row['Sex']
        else:
            df.at[idx, 'Sex'] = current_sex
            
        if type(row['Histology Tumor']) == str:
            current_tumor_type = row['Histology Tumor']
        else:
            df.at[idx, 'Histology Tumor'] = current_tumor_type
            
        if type(row['Date of Birth']) == pd._libs.tslibs.timestamps.Timestamp:
            current_age = calculate_age(row['Date of Birth'])
            df.at[idx, 'Date of Birth'] = current_age
        else:
            df.at[idx, 'Date of Birth'] = current_age

    df = df[~df['Kept'].isna()]
    return df