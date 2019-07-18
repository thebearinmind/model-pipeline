import pandas as pd

class resultsBlender: 

    def __init__(self):
        self.path_df1 = path_df1
        self.path_df2 = path_df2
    
    def blend_results(self, _id, target, proport1 = 0.6):
        df1 = pd.read_csv(self.path_df1)
        df2 = pd.read_csv(self.path_df2)
        target1 = target + '_x'
        target2 = target + '_y'
        df1 = df1.merge(df2, on = 'id')

        dif = (df1[target1] - df1[target2]).abs().mean()
        print(f'the mean absolute difference between 2 submissions is: {dif}')

        df1[target] = (proport*df1[target1] + (1-proport)*df1[target2])

        df = df1[[_id, target]]

        return df