import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

import joblib
import numpy as np

df = pd.read_csv('test.csv')
ids = df['id']
df= df.drop(columns=['id'])

#only working on time_spent_alone column  so we create a df_temp so not to change original df as of now 

# Create a working copy
df_temp = df.copy()

# Step 1a: Temporarily fill predictors with mean
df_temp['Going_outside_temp'] = df_temp['Going_outside'].fillna(df_temp['Going_outside'].mean())
df_temp['Social_event_attendance_temp'] = df_temp['Social_event_attendance'].fillna(df_temp['Social_event_attendance'].mean())

# Step 1b: KNN Impute Time_spent_Alone
imputer = KNNImputer(n_neighbors=10)
subset = df_temp[['Time_spent_Alone', 'Going_outside_temp', 'Social_event_attendance_temp']]
imputed = imputer.fit_transform(subset)

# Step 1c: Update Time_spent_Alone only
df_temp['Time_spent_Alone'] = imputed[:, 0]

# Drop temp columns
df_temp.drop(['Going_outside_temp', 'Social_event_attendance_temp'], axis=1, inplace=True)







#filling up going_alone 
# Step 2a: Fill missing in Social_event_attendance (only for imputation use)
df_temp['Social_event_attendance_temp'] = df_temp['Social_event_attendance'].fillna(df_temp['Social_event_attendance'].mean())

# Step 2b: KNN Impute Going_outside
subset2 = df_temp[['Going_outside', 'Time_spent_Alone', 'Social_event_attendance_temp']]
imputed2 = KNNImputer(n_neighbors=10).fit_transform(subset2)

# Step 2c: Update Going_outside
df_temp['Going_outside'] = imputed2[:, 0]
df_temp.drop('Social_event_attendance_temp', axis=1, inplace=True)






#filling up social event attendance 
# Step 3: Final KNN Imputation for Social_event_attendance
subset3 = df_temp[['Social_event_attendance', 'Time_spent_Alone', 'Going_outside']]
imputed3 = KNNImputer(n_neighbors=10).fit_transform(subset3)

df_temp['Social_event_attendance'] = imputed3[:, 0]




#print(df_temp[['Time_spent_Alone', 'Going_outside', 'Social_event_attendance']].isna().sum()) #check if converted properly




#change the column of original df
df[['Time_spent_Alone', 'Going_outside', 'Social_event_attendance']] = df_temp[['Time_spent_Alone', 'Going_outside', 'Social_event_attendance']]






# #time for friend_circle_size

# Select relevant columns
cols = ['Friends_circle_size', 'Time_spent_Alone', 'Going_outside', 'Social_event_attendance']
df_temp = df[cols].copy()

# Initialize KNN imputer
imputer = KNNImputer(n_neighbors=10)

# Apply imputation
df_imputed = imputer.fit_transform(df_temp)

# Replace only the 'Friends_circle_size' column in original df
df['Friends_circle_size'] = df_imputed[:, 0]


#print(df.corr(numeric_only=True)['Post_frequency'].sort_values(ascending=False))







#the last numerical column that is post frequency
from sklearn.impute import KNNImputer

# Select relevant columns
cols = ['Post_frequency', 'Time_spent_Alone', 'Going_outside', 'Social_event_attendance']
df_temp = df[cols].copy()

# Initialize KNN imputer
imputer = KNNImputer(n_neighbors=10)

# Apply imputation
df_imputed = imputer.fit_transform(df_temp)

# Replace only the 'Post_frequency' column in original df
df['Post_frequency'] = df_imputed[:, 0]




#time to impute the final 2 yes/no columns stage_fear and Drained after socialsing


# Step 1: Map Yes/No to 1/0, keep NaN as is
binary_cols = ['Stage_fear', 'Drained_after_socializing']
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else np.nan)

# Step 2: Select all numeric columns (since now everything is imputed)
df_numeric = df.select_dtypes(include=[np.number])

# Step 3: Apply KNN imputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df_numeric)

# Step 4: Replace only the binary columns
df['Stage_fear'] = df_imputed[:, df_numeric.columns.get_loc('Stage_fear')].round().astype(int)
df['Drained_after_socializing'] = df_imputed[:, df_numeric.columns.get_loc('Drained_after_socializing')].round().astype(int)



print(df.head(10))
print(df.info())
print(df.isna().sum())




# Load Majority Voting Ensemble
majority_model = joblib.load("final_majority_ensemble.pkl")

# Make prediction
preds_ensemble = majority_model.predict(df)

# Convert binary predictions to labels
labels_ensemble = ['Introvert' if p == 0 else 'Extrovert' for p in preds_ensemble]

# Create submission dataframe
submission2 = pd.DataFrame({
    'id': ids,
    'Personality': labels_ensemble
})

# Save
submission2.to_csv("allmodelsmix.csv", index=False)