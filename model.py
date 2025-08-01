import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib

# ──────────────────────────────────────────────────────
# Load and prepare data
# ──────────────────────────────────────────────────────
df_tree = pd.read_csv("final_cleaned_data.csv")
df_scaled = pd.read_csv("final_cleaned_data.csv")

X_raw = df_scaled.drop(columns=['Personality'])
y = df_scaled['Personality']

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_raw)
df_scaled = pd.DataFrame(X_scaled_array, columns=X_raw.columns)
df_scaled['Personality'] = y

X_tree = df_tree.drop(columns=['Personality'])
y_tree = df_tree['Personality']
X_scaled = df_scaled.drop(columns=['Personality'])
y_scaled = df_scaled['Personality']

# ──────────────────────────────────────────────────────
# Define finely tuned models
# ──────────────────────────────────────────────────────
tuned_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    max_depth=4,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False
)

tuned_lgb = LGBMClassifier(
    n_estimators=1306,
    learning_rate=0.06004829153091318,
    num_leaves=138,
    max_depth=9,
    min_child_samples=32,
    subsample=0.6598821793573304,
    colsample_bytree=0.6249338558888039,
    reg_alpha=5.640389745765319,
    reg_lambda=0.10574026292815943,
    random_state=42
)

# ──────────────────────────────────────────────────────
# Stack A: Tree-based models (use X_tree)
# ──────────────────────────────────────────────────────
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42)

base_learners_A = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42)),
    ('lgb', tuned_lgb),
    ('tuned_xgb', tuned_xgb)
]

stacked_model_A = StackingClassifier(
    estimators=base_learners_A,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

stacked_model_A.fit(X_train_A, y_train_A)
y_pred_A = stacked_model_A.predict(X_test_A)
print(f"Stacked Model A Accuracy: {accuracy_score(y_test_A, y_pred_A):.4f}")

# ──────────────────────────────────────────────────────
# Stack B: Mixed models (use X_scaled)
# ──────────────────────────────────────────────────────
base_learners_B = [
    ('catboost', CatBoostClassifier(verbose=0, random_state=42)),
    ('extratrees', ExtraTreesClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('tuned_xgb', tuned_xgb)
]

stacked_model_B = StackingClassifier(
    estimators=base_learners_B,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=False,
    n_jobs=-1
)

stacked_model_B.fit(X_scaled, y_scaled)
y_pred_B = stacked_model_B.predict(X_scaled)
print("Accuracy of Stacked Model B:", accuracy_score(y_scaled, y_pred_B))

# ──────────────────────────────────────────────────────
# Stack C: KNN, MLP, GB (use X_scaled)
# ──────────────────────────────────────────────────────
base_learners_C = [
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('mlp', MLPClassifier(max_iter=500, random_state=42)),
    ('tuned_xgb', tuned_xgb)
]

stacked_model_C = StackingClassifier(
    estimators=base_learners_C,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=False,
    n_jobs=-1
)

stacked_model_C.fit(X_scaled, y_scaled)
y_pred_C = stacked_model_C.predict(X_scaled)
print("Accuracy of Stacked Model C:", accuracy_score(y_scaled, y_pred_C))

# ──────────────────────────────────────────────────────
# Final Voting Ensemble
# ──────────────────────────────────────────────────────
voting_ensemble = VotingClassifier(
    estimators=[
        ('stacked_A', stacked_model_A),
        ('stacked_B', stacked_model_B),
        ('stacked_C', stacked_model_C)
    ],
    voting='hard',
    weights=[2, 1.5, 1.5],  # Give A more influence
    n_jobs=-1
)

voting_ensemble.fit(X_tree, y_tree)
y_pred_ensemble = voting_ensemble.predict(X_tree)
print("Accuracy of Majority Voting Ensemble:", accuracy_score(y_tree, y_pred_ensemble))

# Save final model
joblib.dump(voting_ensemble, "final_majority_ensemble.pkl")
