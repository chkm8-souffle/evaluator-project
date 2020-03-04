import pandas as pd
import numpy as np

def clean_ga_data(ga_df):
    ga_df['Page'].replace(regex=True,inplace=True,to_replace='^.*\?fbclid.*$',value=r'')
    ga_df=ga_df.groupby(['Page']).sum()
    ga_df=ga_df.reset_index()
    ga_df['Page'] = 'https://website' + ga_df['Page'].astype(str)
    ga_df=ga_df.rename(columns={'Page':'url'})
    return ga_df


    



