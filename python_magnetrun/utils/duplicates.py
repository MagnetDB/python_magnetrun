import numpy as np

import pandas as pd



def find_duplicates(df: pd.DataFrame, name: str, key: str, strict: bool = False):
    """Find duplicates key in dataframe and eventually drop them

    :param df: _description_
    :type df: pd.DataFrame
    :param name: _description_
    :type name: str
    :param key: _description_
    :type key: str
    :param strict: _description_, defaults to False
    :type strict: bool, optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: _type_
    """

    dups = pd.Index(df[key].to_numpy()).duplicated()
    unique, counts = np.unique(dups, return_counts=True)
    #print(f'duplicated df[t]: {dict(zip(unique, counts))}')
    dups_dict = dict(zip(unique, counts))
    # print(f'dups_dict df[t]: {dups_dict}')
    if np.True_ in dups_dict:
        dups_index = np.where(dups == np.True_)
        drop_index = [i.item() for i in dups_index]
        print(f'duplicated index found in {name} at rows={drop_index}')
        for i in drop_index:
            print(f'rows={i}/{i-1}:\n{df.loc[i-1:i].head()}')
        if strict:
            raise RuntimeError(f"found duplicates time in {name}")
        else:
            print("remove duplicates")
            df.drop(df.index[drop_index], inplace=True)
            return df
        
    return df
