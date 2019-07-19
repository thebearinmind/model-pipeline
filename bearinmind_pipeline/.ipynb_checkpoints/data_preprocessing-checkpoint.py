import sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class dataPreprocessing:

    def __init__(self):
        self.path = path
        
    # Reduce memory needed for the data frame by changing int and float formats
    def reduce_mem(df):
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



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
    def parse_str_col(df, str_col_list, parse_by, n_element):
        for str_col in str_col_list:
            df[str_col] = df[str_col].str.split(parse_by).str.get(n_element)
        return df

    # Create basic stats variables based on the group category as well as nominator for proportions 
    def create_stats_features(df, nominator, groupby):
        grp = '_'.join(groupby)
        
        df[f'{nominator}_by_{grp}_std'] = df[nominator]/df.groupby(groupby)[nominator].transform('std')
        df[f'{nominator}_by_{grp}_mean'] = df[nominator]/df.groupby(groupby)[nominator].transform('mean')
        df[f'{nominator}_by_{grp}_median'] = df[nominator]/df.groupby(groupby)[nominator].transform('median')
        df[f'{nominator}_by_{grp}_max'] = df[nominator]/df.groupby(groupby)[nominator].transform('max')
        df[f'{nominator}_by_{grp}_min'] = df[nominator]/df.groupby(groupby)[nominator].transform('min')

        return df