import pandas as pd
import numpy as np
from haversine import haversine

class GPSTrajectoriesModel:

    FEATURES = [
        'latitude_mean', 'latitude_std', 'longitude_mean', 'longitude_std', 'time_diff_mean', 'time_diff_median',
        'time_diff_std', 'velocity_0', 'velocity_10', 'velocity_100',  'velocity_105', 'velocity_110', 'velocity_115',
        'velocity_120',  'velocity_125', 'velocity_130', 'velocity_135', 'velocity_140','velocity_145', 'velocity_15',
        'velocity_150', 'velocity_155', 'velocity_160', 'velocity_165', 'velocity_170', 'velocity_175', 'velocity_180',
        'velocity_185','velocity_190','velocity_20', 'velocity_25','velocity_30', 'velocity_35', 'velocity_40',
        'velocity_45', 'velocity_5','velocity_50', 'velocity_55', 'velocity_60', 'velocity_65', 'velocity_70',
        'velocity_75', 'velocity_80', 'velocity_85', 'velocity_90', 'velocity_95'
        ]

    @staticmethod
    def extract_features(data: dict) -> pd.DataFrame:

        df_raw = pd.DataFrame().from_dict(data)
        features = GPSTrajectoriesModel.extraction_logics(df_raw)
        df_features = pd.DataFrame().from_dict(features, orient='index').T
        return df_features[GPSTrajectoriesModel.FEATURES].values.tolist()

    @staticmethod
    def extraction_logics(df: pd.DataFrame) -> dict:
        """
        Extracts features from a DataFrame representing a gps trajectory
        :param df:
        :return:
        """
        features = {}

        # Calculate the speed
        df = GPSTrajectoriesModel.calc_speed(df)

        # meta
        features['segment'] = df['segment'].iloc[0]

        # location
        features['latitude_mean'] = df['latitude'].mean()
        features['longitude_mean'] = df['longitude'].mean()
        features['latitude_std'] = df['latitude'].std()
        features['longitude_std'] = df['longitude'].std()

        # velocity
        hist, _ = np.histogram(df['speed'], bins=range(0, 200, 5), normed=True)
        for i in range(0, int(195 / 5)):
            features['velocity_%d' % (i * 5)] = hist[i]

        # sample freq
        features['time_diff_mean'] = df['time_diff'].mean()
        features['time_diff_median'] = df['time_diff'].median()
        features['time_diff_std'] = df['time_diff'].std()

        return features

    @staticmethod
    def calc_speed(df):
        """
        Calculates the speed from GPS trajectories DataFrame

        :param df: Dataframe containing "latitude", "longitude" and "sample_time" columns
        :return: 
        """
        df = df.copy()
        df.loc[:, 'sample_time'] = df['sample_time'].apply(pd.Timestamp)

        dfs = []
        for segment, df_segment in df.groupby('segment'):
            df_segment['lat_lag'] = df_segment['latitude'].shift()
            df_segment['lon_lag'] = df_segment['longitude'].shift()
            df_segment['time_diff'] = df_segment['sample_time'].diff()
            dfs.append(df_segment)

        df = pd.concat(dfs).dropna()

        df['time_diff'] = df['time_diff'].apply(lambda x: x.seconds) / (60 * 60)
        df['dist'] = df.apply(lambda row: haversine((row['latitude'], row['longitude']),
                                                                  (row['lat_lag'], row['lon_lag'])), axis=1)
        df['speed'] = df['dist'] / (df['time_diff'])
        df = df[df['time_diff'] > 1 / (60 * 600)]

        return df

    @staticmethod
    def parse_results(probs, classes):
        return [classes[np.argmax(np.array(x))] for x in probs]

    def __init__(self):
        self.classes = ['airplane', 'bike', 'boat', 'bus', 'car', 'motorcycle', 'run', 'subway',  'taxi']