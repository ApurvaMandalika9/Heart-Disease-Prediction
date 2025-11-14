#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, json, time, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[2]:


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "data/heart.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ### Loading Data 

# In[3]:


df = pd.read_csv(DATA_PATH)
print(df.shape)
df.head()


# In[5]:


df.info()


# ### Basic Sanity Checks

# In[6]:


# Missing values
df.isnull().sum().sort_values(ascending=False)


# In[7]:


# describe
df.describe(include='all').T


# ### Target Balance

# In[8]:


assert 'target' in df.columns, "Expected 'target' column in dataset."
ax = df['target'].value_counts().sort_index().plot(kind='bar')
ax.set_title("Target distribution (0 = no disease, 1 = disease)")
ax.set_xlabel("target"); ax.set_ylabel("count")
plt.show()

df['target'].value_counts(normalize=True)


# ### Numeric and Categorical

# In[9]:


num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]


# ### EDA

# In[10]:


# Histograms for numeric features
df[num_cols].hist(bins=20, figsize=(10,6))
plt.tight_layout(); plt.show()


# In[11]:


# Correlation heatmap for numeric features
plt.figure(figsize=(7,5))
sns.heatmap(df[num_cols + ["target"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation (numeric features)")
plt.show()


# In[12]:


# Numeric vs target (boxplots)

fig, axes = plt.subplots(1, len(num_cols), figsize=(4*len(num_cols), 4), sharey=False)
for i, c in enumerate(num_cols):
    sns.boxplot(data=df, x="target", y=c, ax=axes[i])
    axes[i].set_title(c)
plt.tight_layout(); plt.show()


# In[13]:


# Categorical vs target (stacked bars)
for c in cat_cols:
    ct = pd.crosstab(df[c], df['target'], normalize='index')
    ct.plot(kind='bar', stacked=True, title=f"{c} vs target", figsize=(5,3))
    plt.legend(title="target"); plt.ylabel("proportion"); plt.show()


# ### Train - Test Split

# In[14]:


X = df.drop(columns=["target"])
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

X_train.shape, X_test.shape


# ### Preprocessing 

# In[15]:


num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)


# ### Baseline & model comparison (CV on train)

# In[32]:


models = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "RF": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
    #"XGB": XGBClassifier(
    #    n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
    #    reg_lambda=1.0, eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE
    #)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = []
for name, mdl in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", mdl)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results.append((name, scores.mean(), scores.std()))
    print(f"{name}: ROC-AUC {scores.mean():.4f} ± {scores.std():.4f}")

pd.DataFrame(cv_results, columns=["model","roc_auc_mean","roc_auc_std"]).sort_values("roc_auc_mean", ascending=False)


# In[33]:


best_model = Pipeline([("pre", preprocessor), ("clf", models["RF"])])

# refit on full train explicitly
best_model.fit(X_train, y_train)


# ### Fit best model & evaluate on test

# In[34]:


# Predict proba on test
y_proba = best_model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)  # PR-AUC
print(f"Test ROC-AUC: {roc:.4f} | Test PR-AUC: {ap:.4f}")


# ### Threshold selection (maximize F1 on validation split from train)

# In[35]:


# Create a small validation split from train to pick threshold
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
)

best_model.fit(X_tr, y_tr)
val_proba = best_model.predict_proba(X_val)[:,1]

prec, rec, thr = precision_recall_curve(y_val, val_proba)
f1s = 2*prec*rec/(prec+rec+1e-9)
best_idx = np.nanargmax(f1s)
BEST_THRESHOLD = float(thr[best_idx-1]) if best_idx > 0 and best_idx < len(thr) else 0.5
BEST_THRESHOLD


# In[36]:


# Lock the model by refitting on full train
best_model.fit(X_train, y_train)

# Evaluate on test with chosen threshold
test_proba = best_model.predict_proba(X_test)[:,1]
test_pred = (test_proba >= BEST_THRESHOLD).astype(int)

print("Threshold:", BEST_THRESHOLD)
print("Accuracy :", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall   :", recall_score(y_test, test_pred))
print("F1       :", f1_score(y_test, test_pred))
print("ROC-AUC  :", roc_auc_score(y_test, test_proba))
print("\nClassification report:\n", classification_report(y_test, test_pred, digits=3))


# ### Curves & confusion matrix

# In[37]:


# ROC
fpr, tpr, _ = roc_curve(y_test, test_proba)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_test, test_proba):.3f}")
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.show()

# PR
p, r, _ = precision_recall_curve(y_test, test_proba)
plt.figure(figsize=(5,4))
plt.plot(r, p, label=f"PR-AUC={average_precision_score(y_test, test_proba):.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend(); plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test)"); plt.xlabel("Predicted"); plt.ylabel("True"); plt.show()


# ### Feature importance

# In[38]:


# Getting feature names after preprocessing to map importances
ohe = best_model.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
num_feature_names = np.array(num_cols)
feature_names = np.concatenate([num_feature_names, cat_feature_names])

clf = best_model.named_steps["clf"]
importances = None

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
elif hasattr(clf, "coef_"):
    # For linear models with OHE+scaler, coefficients can be inspected
    coef = clf.coef_[0] if clf.__class__.__name__ == "LogisticRegression" else clf.coef_.ravel()
    importances = np.abs(coef)
else:
    print("Model does not expose feature importances / coefficients.")

if importances is not None:
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(20)
    plt.figure(figsize=(6,6))
    sns.barplot(data=imp_df, y="feature", x="importance")
    plt.title("Top Feature Importances")
    plt.tight_layout(); plt.show()


# ### Saving model, schema and metadata

# In[40]:


MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.joblib")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

# Save full pipeline (preprocessor + model)
joblib.dump(best_model, MODEL_PATH)

# Save expected columns & chosen threshold
meta = {
    "expected_feature_order": num_cols + cat_cols,
    "threshold": BEST_THRESHOLD,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "random_state": RANDOM_STATE,
    "model_type": best_model.named_steps["clf"].__class__.__name__
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

MODEL_PATH, META_PATH


# ### Sanity check: single-record inference

# In[41]:


loaded = joblib.load(MODEL_PATH)
sample = X_test.iloc[[0]].copy()
proba = loaded.predict_proba(sample)[0,1]
pred  = int(proba >= meta["threshold"])
print("Proba:", round(float(proba), 4), "Pred:", pred, "True:", int(y_test.iloc[0]))
sample

