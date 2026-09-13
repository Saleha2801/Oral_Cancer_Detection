import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

train_csv = "NDU-UFES/histopathological img dataset/ndbufes_TaskII_parsed_folders.csv"
test_csv = "NDU-UFES/histopathological img dataset/ndbufes_TaskII_parsed_test.csv"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

feature_cols = [c for c in train_df.columns if c not in ['path', 'TaskII', 'folder', 'label_number']]
print(f"Number of clinical features: {len(feature_cols)}")

X_train = train_df[feature_cols]
# Map OSCC -> 0 (Cancer), Leukoplakia -> 1 (Non-cancer / Dysplasia) to match ResNet18 label map
# Or 1: Cancer, 0: Non-cancer? Let's check consistency:
# In Oral_Cancer (1).ipynb: LABEL_MAP = {'cancer': 0, 'non_cancer': 1}
# Let's save both probability for Cancer and Non-cancer explicitly
y_train = (train_df['TaskII'] != 'OSCC').astype(int) # 0 for OSCC (Cancer), 1 for Leukoplakia
X_test = test_df[feature_cols]
y_test = (test_df['TaskII'] != 'OSCC').astype(int)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test) # prob for class 0 (Cancer) is y_prob[:, 0]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob[:, 1])

print(f"Clinical Model Accuracy: {acc:.4f}")
print(f"Clinical Model ROC-AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Cancer (OSCC)', 'Non-cancer (Leukoplakia)']))

artifact = {
    "model": rf,
    "feature_cols": feature_cols,
    "class_names": ["Cancer (OSCC)", "Non-cancer (Leukoplakia)"],
    "label_map": {"cancer": 0, "non_cancer": 1},
    "test_accuracy": acc,
    "test_roc_auc": auc,
    "feature_importances": dict(zip(feature_cols, rf.feature_importances_)),
}

output_path = "models/clinical_model.joblib"
joblib.dump(artifact, output_path)
print(f"Saved clinical model artifact to {output_path}")
