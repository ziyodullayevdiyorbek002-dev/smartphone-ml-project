import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# dataset load
df = pd.read_csv("smartphone_usage.csv")

# drop columns
if 'transaction_id' in df.columns:
    df.drop('transaction_id', axis=1, inplace=True)

if 'user_id' in df.columns:
    df.drop('user_id', axis=1, inplace=True)

# missing values
mode_value = df['addiction_level'].mode()[0]
df['addiction_level'] = df['addiction_level'].fillna(mode_value)

# encoding
cat_cols = ['gender', 'stress_level', 'academic_work_impact', 'addiction_level']

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# feature engineering
df['entertainment_hours'] = df['social_media_hours'] + df['gaming_hours']
df['sleep_deficit'] = 8 - df['sleep_hours']
df['weekend_usage_ratio'] = df['weekend_screen_time'] / df['daily_screen_time_hours']

# features and target
X = df.drop('addicted_label', axis=1)
y = df['addicted_label']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# train model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# save model
pickle.dump(dt_model, open('model.pkl', 'wb'))

# save scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model and scaler saved!")