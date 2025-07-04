# ========================================================================
# BANK‑MARKETING TERM‑DEPOSIT – END‑TO‑END PIPELINE (CRISP‑ML(Q))
# ========================================================================
# Phase 1.a – BUSINESS UNDERSTANDING
# ------------------------------------------------------------------------
#  • Business Problem:  Identify the right customers to target to accept offers (like Term Deposit) 
#  • Objective:         Flag 'Subscribers' & 'Non subscribers' with probabilities
#  • Success metrics:   – ML:       ≥ 0.85 Accuracy
#                       
#
# Phase 1.b – DATA UNDERSTANDING
# ------------------------------------------------------------------------
#  • Source:  `bank-additional.csv`
#  • Size:    4119 rows × 21 columns
#  • Target:  `Subscriber`  {“yes”, “no”}
#  • Features: 10 numeric + 10 categorical (age, job, marital, …, month, …)
# ========================================================================

# --------------------  REQUIRED LIBS  -----------------------------------
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import joblib , pickle, dtale

from sqlalchemy import create_engine
from urllib.parse import quote_plus

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV , StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from feature_engine.outliers import Winsorizer


# --------------------  PHASE 2 – DATA COLLECTION & LOADING  -------------
DATA_PATH = "https://raw.githubusercontent.com/Phani-ISB/ML-Ops_Bank/refs/heads/main/bank-additional.csv"    # Using dataset uploaded into github
bank_df   = pd.read_csv(DATA_PATH, sep=',', na_values=['unknown'])

# MySQL data

user     = "root"
password = quote_plus("Manasa@123")
host     = "127.0.0.1"
port     = 3306
db       = "bank_db"

engine = create_engine(
    f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}",
    pool_pre_ping=True,
    connect_args={"ssl": {"ssl_disabled": True}} 
)
bank_df.to_sql('raw_campaign', con=engine, if_exists='replace', index=False)

# --------------------  QUICK EDA SNAPSHOT -------------------------------
d = dtale.show(bank_df, ignore_duplicate=True)
d.open_browser()

# --------------------  PHASE 3 – DATA PREPARATION  ----------------------
# ---- 3.1  Target recoding
bank_df['y'] = bank_df['y'].map({'no': 'Non-subscriber', 'yes': 'Subscriber'})

# ---- 3.2  Feature / target split
X = bank_df.drop(columns=['y'])
y = bank_df['y'].dropna()

# ---- 3.3  Column groups
numeric_feats     = X.select_dtypes(exclude=['object']).columns.tolist()
numeric_feats     = [col for col in numeric_feats
                     if col not in ('pdays','previous')]  # Based on EDA, dropping columns with no variation
categorical_feats = X.select_dtypes(include=['object']).columns.tolist()


# ---- 3.4  Pre‑processing pipelines
num_pipe = Pipeline([
    ('impute',    SimpleImputer(strategy='mean')),
    ('winsorize', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),
    ('scale',     StandardScaler())
])

cat_pipe = Pipeline([
        ('encode', OneHotEncoder(drop= None, handle_unknown='ignore', sparse_output=False))
])

preprocess = ColumnTransformer([
    ('num', num_pipe, numeric_feats),
    ('cat', cat_pipe, categorical_feats)
])

# ---- 3.5  Feature selection
selector  = SelectKBest(score_func=f_classif, k=15)

full_pipe = Pipeline([
    ('prep', preprocess),
    ('fs',   selector)
])

X_selected = full_pipe.fit_transform(X, y)

# Saving Joblib file containing the pipeline
joblib.dump(full_pipe, 'bank_pipeline.joblib')


# --------------------  PHASE 4 – DATA SPLIT & BALANCING -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.20, random_state=42, stratify=y
)


# --------------------  PHASE 5 – MODEL BUILDING & TUNING  ---------------


# Compute balanced weights (optional baseline)
classes_ = np.unique(y_train)
cw = compute_class_weight("balanced", classes=classes_, y=y_train)
baseline_wts = dict(zip(classes_, cw))

# Build weight grid that uses string labels
k_values = [3, 5, 7, 9, 12, 15]
weight_grid = [
    {"Non-subscriber": 1, "Subscriber": k}
    for k in k_values
]

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(class_weight=None, solver="liblinear",
                               max_iter=1000))
])

param_grid = {"clf__class_weight": weight_grid, "clf__C": [0.1, 1, 10]}

prec = make_scorer(precision_score, pos_label="Subscriber")

grid = GridSearchCV(pipe, param_grid, scoring=prec,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    n_jobs=-1, verbose=2)

grid.fit(X_train, y_train)
print("Best weights:", grid.best_params_["clf__class_weight"])
print("Best C:", grid.best_params_["clf__C"])

best_model = grid.best_estimator_

#Saving model into pickle file
pickle.dump(best_model, open('bank_model.pkl', 'wb'))

# --------------------  PHASE 6 – EVALUATION -----------------------------
y_pred       = best_model.predict(X_test)
y_proba_pos  = best_model.predict_proba(X_test)[:, 1]   # column for “Subscriber”

# Classification report
print("===== Classification report =====")
print(classification_report(y_test, y_pred, target_names=["Non‑subscriber", "Subscriber"]))

# Confusion Matrix display
cm           = confusion_matrix(y_test, y_pred, labels=['Subscriber','Non-subscriber'])
ConfusionMatrixDisplay(cm,display_labels=['Sub.', 'Non-Sub.']).plot()
plt.title('Bank‑Marketing – Confusion Matrix')
plt.show()

