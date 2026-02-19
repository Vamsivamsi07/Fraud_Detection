import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pandas', 'scikit-learn'])

import pandas as pd, pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

print("Loading data...")
df = pd.read_csv('fraud.csv')
print(f"Data loaded: {len(df)} rows")

print("Encoding features...")
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

X = df[['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
y = df['isFraud']

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_scaled, y)

print("Saving model files...")
pickle.dump(model, open('svm_model.pkl','wb'))
pickle.dump(le, open('encoder.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))

print("✓ Training complete! Model files saved.")
