import sklearn
from sklearn.preprocessing import LabelEncoder


# Encode categorical variables based on the column list
def encode_categorical(df_train, df_test, column_list):
    for f in column_list:
        print(f'colums {f} is being encoded')
        lbl = LabelEncoder()
        lbl.fit(list(df_train[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_test[f] = lbl.transform(list(df_test[f].values))
    
    return[df_train, df_test]

# Parse string column based on the characted and position of the element
def parse_str_col(df, str_col, parse_by, n_element):
    parsed_col = df[str_col].str.split(parse_by).str.get(n_element)
    
    return parsed_col
    
# Create basic stats variables based on the group category as well as nominator for proportions 
def create_stats_features(df, nominator, groupby):
    df[f'{nominator}_by_{groupby}_std'] = df.groupby(groupby)[nominator].transform('std')
    df[f'{nominator}_by_{groupby}_mean'] = df.groupby(groupby)[nominator].transform('mean')
    df[f'{nominator}_by_{groupby}_median'] = df.groupby(groupby)[nominator].transform('median')
    df[f'{nominator}_by_{groupby}_max'] = df.groupby(groupby)[nominator].transform('max')
    df[f'{nominator}_by_{groupby}_min'] = df.groupby(groupby)[nominator].transform('min')
    
    return df