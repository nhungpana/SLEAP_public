import pandas as pd
import numpy as np

from collections import defaultdict         # Find duplicated column names and their indexes

def remove_extra_id(df,del_ID_list, ID_col="nsrrid"):
    """Remove rows for subjects that are not present in the dataframe."""
    df.reset_index(drop=True, inplace=True)
    del_idx = []

    # Looking for the row indexes of unwanted subject and put in a list
    for i in range(len(df[ID_col])):
        if df[ID_col][i] in del_ID_list:
            del_idx.append(i)
    
    # If the list is not empty, drop unwanted rows
    if len(del_idx) > 0:
        df.drop(del_idx,inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def process_missing(df, feature_ref, feature_fill, ref_value = 0, fill_value = 0):
    """ In the questionnaires, some main questions have follow-up sub-questions.
        If a subject answers 0 = No or 8 = Don't know to a main question, they are allowed to skip the related sub-questions, which results in missing data.
        However, we can fill in the missing sub-question values based on the main question's answer.
    
        Example:    if have ever snore = 0 then missing data for loudness=0,
                    if have ever snore = 8 then missing data for loudness=8.
                    the dataframe must have feature_ref column """

    # if df[feature_ref].apply(lambda x: isinstance(x, (int, float))).all():
    for i in range(len(df[feature_fill])):                        
        if df[feature_ref][i] == ref_value and pd.isna(df[feature_fill][i]):
            df[feature_fill][i] = fill_value                                        
        elif df[feature_ref][i] == 8 and pd.isna(df[feature_fill][i]):
            df[feature_fill][i] = 8                                        
    return df

def find_duplicated_cols(df):
    # Find duplicated column names and their indexes
    name_to_indexes = defaultdict(list)
    for idx, col in enumerate(df.columns):
        name_to_indexes[col].append(idx)

    # Filter to show only duplicates
    duplicates = {name: idxs for name, idxs in name_to_indexes.items() if len(idxs) > 1}
    return duplicates

def unnecessary_cols_idx(list_duplicates):
    cols_idx = []
    for name, indexes in list_duplicates.items():
        print(f"Duplicated column name: '{name}' at indexes {indexes}")
        cols_idx.extend(indexes[1:])
    return cols_idx

def AHI_convert(col):
    col = pd.cut(col,
                [-1,0,5,15,30,np.inf],
                labels = ["NaN", "Normal", "Mild", "Moderate", "Severe"],
                right=False)
    return col

def cat_identify(df):
    """Identify categorical columns"""
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    return cat_cols

def st_cat2str(df):
    """Cast the column to the correct type to be displayed with streamlit"""
    for col in df.select_dtypes(include="object").columns:
        print(col)
        df[col] = df[col].astype(str)
    return df













