import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
import json
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib

import lightgbm as lgb
import xgboost as xgb

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("=" * 78)
print("  Universal 38-Plant Disease Classification — Advanced Ensemble & Threshold")
print("=" * 78)

# ==============================================================================
# 1. تحميل السمات المحفوظة لكافة الـ 38 فئة
# ==============================================================================
cache_file = "all_plants_features_cache.npz"
if not os.path.exists(cache_file):
    print(f"[!] خطأ: ملف السمات الشامل ({cache_file}) غير موجود. يرجى تشغيل train_all_plants_ml.py أولاً.")
    sys.exit(1)

print(f"[*] تحميل سمات الـ 38 فئة من: {cache_file}...")
cached = np.load(cache_file)
X = cached["X"]
y = cached["y"]
paths = cached["paths"]

print(f"[+] تم التحميل بنجاح! إجمالي العينات: {X.shape[0]} | عدد الميزات: {X.shape[1]}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
all_labels_dict = {str(i): name for i, name in enumerate(le.classes_)}

# تقسيم البيانات (70% تدريب، 15% تحقق، 15% اختبار نهائي)
idx = np.arange(len(X))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_encoded)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y_encoded[temp_idx])

X_train, y_train, paths_train = X[train_idx], y_encoded[train_idx], paths[train_idx]
X_val, y_val, paths_val       = X[val_idx], y_encoded[val_idx], paths[val_idx]
X_test, y_test, paths_test    = X[test_idx], y_encoded[test_idx], paths[test_idx]

print(f"  * عينات التدريب (Train): {len(X_train)} (70%)")
print(f"  * عينات التحقق (Val):    {len(X_val)} (15%)")
print(f"  * عينات الاختبار (Test): {len(X_test)} (15%)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ==============================================================================
# 2. بناء وتدريب النماذج الفردية للـ 38 فئة
# ==============================================================================
print("\n[*] إعداد النماذج الخمسة المتنوعة للـ Universal Ensemble...")

# 1. LightGBM
m_lgb = lgb.LGBMClassifier(
    n_estimators=160,
    learning_rate=0.07,
    max_depth=6,
    num_leaves=35,
    subsample=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1
)

# 2. Random Forest
m_rf = RandomForestClassifier(
    n_estimators=180,
    max_depth=22,
    random_state=SEED,
    n_jobs=-1
)

# 3. Extra Trees (مقاومة عالية للـ Overfitting والتنوع)
m_et = ExtraTreesClassifier(
    n_estimators=160,
    max_depth=22,
    random_state=SEED,
    n_jobs=-1
)

# 4. XGBoost
m_xgb = xgb.XGBClassifier(
    n_estimators=130,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    eval_metric="mlogloss",
    n_jobs=-1
)

# 5. Multi-Layer Perceptron (Dense Neural Net)
m_mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    alpha=1e-4,
    max_iter=200,
    early_stopping=True,
    random_state=SEED
)

print("[*] تجميع النماذج الخمسة في Soft-Voting Ensemble مع أوزان معايرة...")
universal_ensemble = VotingClassifier(
    estimators=[
        ('lgb', m_lgb),
        ('rf',  m_rf),
        ('et',  m_et),
        ('xgb', m_xgb),
        ('mlp', m_mlp)
    ],
    voting='soft',
    weights=[3, 2, 2, 2, 1]
)

t0 = time.time()
print("[*] جاري تدريب الـ Universal Ensemble بالكامل عبر الأنوية المتعددة...")
universal_ensemble.fit(X_train_s, y_train)
print(f"[+] اكتمل تدريب الـ Ensemble للـ 38 فئة بنجاح في {time.time() - t0:.1f} ثانية!")

# ==============================================================================
# 3. مقارنة دقة النماذج على مجموعة التحقق (Validation)
# ==============================================================================
print("\n" + "=" * 70)
print("  مقارنة الأداء على مجموعة التحقق لكافة النباتات الـ 38 (Validation)")
print("=" * 70)

val_results = {}
individual_models = {
    "MLP (Neural Net)": universal_ensemble.named_estimators_['mlp'],
    "Random Forest": universal_ensemble.named_estimators_['rf'],
    "Extra Trees": universal_ensemble.named_estimators_['et'],
    "XGBoost": universal_ensemble.named_estimators_['xgb'],
    "LightGBM": universal_ensemble.named_estimators_['lgb'],
    "🔥 Universal 38-Plant Ensemble": universal_ensemble
}

for name, clf in individual_models.items():
    preds = clf.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    print(f"  -> {name:<32} | Val Acc: {acc*100:6.2f}% | Macro-F1: {f1:0.4f}")

# ==============================================================================
# 4. تحليل عتبة الثقة (Threshold Tuning) على مجموعة الاختبار المستقلة
# ==============================================================================
print("\n" + "=" * 70)
print("  تحليل عتبة الأمان (Threshold Safety Guard) على مجموعة الاختبار الـ 38")
print("=" * 70)

test_probs = universal_ensemble.predict_proba(X_test_s)
test_preds = np.argmax(test_probs, axis=1)
max_confidences = np.max(test_probs, axis=1)

thresholds = [0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
th_data = []

for th in thresholds:
    accepted_mask = (max_confidences >= th)
    num_acc = np.sum(accepted_mask)
    coverage = (num_acc / len(y_test)) * 100
    
    if num_acc > 0:
        acc_th = accuracy_score(y_test[accepted_mask], test_preds[accepted_mask]) * 100
    else:
        acc_th = 100.0

    rejected = len(y_test) - num_acc
    th_data.append({
        "Threshold": f">={th*100:.0f}%",
        "Accuracy (%)": round(acc_th, 2),
        "Accepted Leaves": num_acc,
        "Coverage (%)": round(coverage, 2),
        "Uncertain / Rejected": rejected
    })

df_th = pd.DataFrame(th_data)
print(df_th.to_string(index=False))

# رسم وحفظ منحنى الأمان الشامل
plt.figure(figsize=(9, 5))
plt.plot(thresholds, [r["Accuracy (%)"] for r in th_data], marker='o', color='darkgreen', linewidth=2.5, label="Test Accuracy (%)")
plt.plot(thresholds, [r["Coverage (%)"] for r in th_data], marker='s', color='navy', linestyle='--', linewidth=2, label="Leaf Coverage (%)")
plt.title("Universal 38-Plant Model: Threshold vs Accuracy & Coverage", fontsize=12)
plt.xlabel("Confidence Threshold", fontsize=10)
plt.ylabel("Percentage (%)", fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("universal_threshold_curve.png", dpi=200)
plt.close()
print("\n[+] تم حفظ منحنى الأمان الشامل في: universal_threshold_curve.png")

# ==============================================================================
# 5. حفظ وتصدير النموذج الشامل والملحقات
# ==============================================================================
out_dir = "saved_models"
os.makedirs(out_dir, exist_ok=True)

model_path = os.path.join(out_dir, "all_plants_ensemble_model.pkl")
scaler_path = os.path.join(out_dir, "all_plants_ensemble_scaler.pkl")
labels_path = os.path.join(out_dir, "all_plants_ensemble_labels.json")

joblib.dump(universal_ensemble, model_path)
joblib.dump(scaler, scaler_path)

with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(all_labels_dict, f, indent=2)

print(f"\n[+] تم حفظ نموذج الـ Ensemble الشامل في: {model_path}")
print(f"[+] تم حفظ معالج التقييس في:           {scaler_path}")
print(f"[+] تم حفظ قاموس الفئات الـ 38 في:     {labels_path}")

print("\n" + "=" * 78)
print("  اكتمل تدريب نموذج الـ Ensemble الشامل لجميع النباتات بنجاح!")
print("=" * 78)

