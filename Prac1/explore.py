import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data reading and adding columns
train = pd.read_csv(r'/home/dell/Documents/medical/dataset/archive (1)/mitbih_train.csv')
test = pd.read_csv(r'/home/dell/Documents/medical/dataset/archive (1)/mitbih_test.csv')

new_column_names = [f"column {i}" for i in range(1, 188)] + ["label"]

train.columns = new_column_names
test.columns = new_column_names

# Data processing
X_train = train.iloc[:, :-1]  
y_train = train.iloc[:, -1]  
X_test = test.iloc[:, :-1]  
y_test = test.iloc[:, -1]  

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

smote = SMOTE(sampling_strategy={4.0: 20000, 2.0: 20000}, random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

adasyn = ADASYN(sampling_strategy={3.0: 10000, 1.0: 15000}, random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_smote, y_smote)

undersampler = RandomUnderSampler(sampling_strategy={0.0: 20000}, random_state=42)
X_undersampled, y_undersampled = undersampler.fit_resample(X_adasyn, y_adasyn)

tomek = TomekLinks()
X_final, y_final = tomek.fit_resample(X_undersampled, y_undersampled)

print("New Class Distribution:", Counter(y_final))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Model initialization
rf_model = RandomForestClassifier(
    n_estimators=100,  
    max_depth=None, 
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1 
)

# Train the model
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(rf_model, "random_forest_ecg.pkl")