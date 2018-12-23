# [START setup]
import datetime
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from google.cloud import storage
import xgboost as xgb

BUCKET_ID = 'gps-trajectories-meetup'

dataset_file_name = 'df_all.csv.gz'

# Bucket holding the dataset
bucket = storage.Client().bucket(BUCKET_ID)

# Path to the data inside the public bucket
data_dir = 'data/'

# Download the data
blob = bucket.blob(''.join([data_dir, dataset_file_name]))
blob.download_to_filename(dataset_file_name)

features = [
    'latitude_mean', 'latitude_std', 'longitude_mean', 'longitude_std',
 'time_diff_mean', 'time_diff_median', 'time_diff_std', 'velocity_0', 'velocity_10', 'velocity_100',
  'velocity_105', 'velocity_110',
  'velocity_115', 'velocity_120',  'velocity_125', 'velocity_130', 'velocity_135', 'velocity_140',
 'velocity_145', 'velocity_15', 'velocity_150', 'velocity_155', 'velocity_160', 'velocity_165',
 'velocity_170', 'velocity_175', 'velocity_180', 'velocity_185','velocity_190','velocity_20', 'velocity_25',
 'velocity_30', 'velocity_35', 'velocity_40', 'velocity_45', 'velocity_5',
 'velocity_50', 'velocity_55', 'velocity_60', 'velocity_65', 'velocity_70', 'velocity_75', 'velocity_80',
 'velocity_85', 'velocity_90', 'velocity_95'
]
label = 'mode'
# Load the training census dataset

df_all = pd.read_csv(dataset_file_name)
print (df_all.columns)

le = LabelEncoder().fit(df_all['mode'])
print("encoded classes: " ,le.classes_)

#
# create training labels list

# Filter rare classes
ser_count = df_all.groupby(label)[['segment']].count()
modes = ser_count[ ser_count > 1000 ].index.tolist()
df_all = df_all[df_all[label].isin(modes)]

# Train test split
segments_all = pd.Series(df_all['segment'].unique())
segments_train = segments_all.sample(frac=0.7).tolist()
segments_test = segments_all[~segments_all.isin(segments_train)].tolist()

segments_all = pd.Series(df_all['segment'].unique())
segments_train = segments_all.sample(frac=0.7).tolist()
segments_test = segments_all[~segments_all.isin(segments_train)].tolist()

print("test segments: " , len(segments_test))
print("train segments: " , len(segments_train))
print("total segments: " , len(segments_all))


X_train = df_all[df_all['segment'].isin(segments_train)][features]
y_train = le.transform(df_all[df_all['segment'].isin(segments_train)][label])

X_test = df_all[df_all['segment'].isin(segments_test)][features]
y_test = le.transform(df_all[df_all['segment'].isin(segments_test)][label])


# load data into DMatrix object
dtrain = xgb.DMatrix(X_train, y_train)
dtest =  xgb.DMatrix(X_test, y_test)

params = {'max_depth':12,'eta':0.25,'min_child_weight':80,
             'silent':0, 'subsample':.8,'colsample_bytree':.4,
             'objective':'multi:softprob', 'eval_metric':'merror',
         'num_class':len(modes)}

# train model
bst = xgb.train(params, dtrain, 2000000,
                verbose_eval=10, early_stopping_rounds=10,
               evals=[(dtrain,'train'), (dtest,'eval')] )

# ---------------------------------------
# 2. Export and save the model to GCS
# ---------------------------------------
# [START export-to-gcs]
# Export the model to a file
model = 'model.bst'
bst.save_model(model)

# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_ID)
blob = bucket.blob('{}/model/{}'.format(
    datetime.datetime.now().strftime('gps_traj_%Y%m%d_%H%M%S'),
    model))
blob.upload_from_filename(model)
# [END export-to-gcs]