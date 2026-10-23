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

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib

import lightgbm as lgb
import xgboost as xgb

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("=" * 75)
print("  Plant Disease Diagnosis — Production Ensemble & Threshold Pipeline")
print("=" * 75)

# ==============================================================================
# 1. تحميل السمات المحفوظة مسبقاً (أو استخراجها إن لم تكن متوفرة)
# ==============================================================================
cache_file = "features_cache.npz"
if not os.path.exists(cache_file):
    print("[!] ملف السمات غير موجود، جاري توليده...")
    from fashion_inspired_tomato_ml import *
else:
    print(f"[*] تحميل السمات المستخرجة من: {cache_file}...")
    cached = np.load(cache_file)
    X = cached["X"]
    y = cached["y"]
    paths = cached["paths"]

print(f"[+] حجم البيانات: {X.shape[0]} عينة | عدد الميزات: {X.shape[1]}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
clean_names = [cls.replace("Tomato___", "") for cls in le.classes_]

# تقسيم البيانات (70% تدريب، 15% تحقق، 15% اختبار نهائي)
idx = np.arange(len(X))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_encoded)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y_encoded[temp_idx])

X_train, y_train, paths_train = X[train_idx], y_encoded[train_idx], paths[train_idx]
X_val, y_val, paths_val       = X[val_idx], y_encoded[val_idx], paths[val_idx]
X_test, y_test, paths_test    = X[test_idx], y_encoded[test_idx], paths[test_idx]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ==============================================================================
# 2. بناء النماذج الأساسية المتنوعة للـ Ensemble
# ==============================================================================
print("\n[*] إعداد النماذج الفردية المتنوعة لتكوين الـ Ensemble...")

m_lgb = lgb.LGBMClassifier(
    n_estimators=140,
    learning_rate=0.07,
    max_depth=5,
    num_leaves=28,
    subsample=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1
)

m_rf = RandomForestClassifier(
    n_estimators=160,
    max_depth=22,
    random_state=SEED,
    n_jobs=-1
)

m_xgb = xgb.XGBClassifier(
    n_estimators=120,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    eval_metric="mlogloss",
    n_jobs=-1
)

m_svm = SVC(
    C=10.0,
    gamma='scale',
    probability=True,
    random_state=SEED
)

# بناء مجمع التصويت الاحتمالي (Soft Voting Ensemble)
# نعطي وزناً أكبر للنموذجين الأقوى (LightGBM و Random Forest)
print("[*] دمج النماذج في Soft-Voting Ensemble مع أوزان معايرة...")
ensemble_model = VotingClassifier(
    estimators=[
        ('lgb', m_lgb),
        ('rf',  m_rf),
        ('xgb', m_xgb),
        ('svm', m_svm)
    ],
    voting='soft',
    weights=[3, 2, 2, 1]
)

t0 = time.time()
print("[*] جاري تدريب الـ Ensemble بالكامل...")
ensemble_model.fit(X_train_s, y_train)
print(f"[+] اكتمل تدريب الـ Ensemble بنجاح في {time.time() - t0:.1f} ثانية!")

# ==============================================================================
# 3. مقارنة دقة النماذج الفردية مقابل الـ Ensemble على مجموعة التحقق
# ==============================================================================
print("\n" + "=" * 65)
print("  مقارنة الأداء: النماذج الفردية مقابل النموذج التجميعي (Validation)")
print("=" * 65)

individual_models = {
    "SVM (RBF)": ensemble_model.named_estimators_['svm'],
    "XGBoost": ensemble_model.named_estimators_['xgb'],
    "Random Forest": ensemble_model.named_estimators_['rf'],
    "LightGBM": ensemble_model.named_estimators_['lgb'],
    "Soft-Voting Ensemble": ensemble_model
}

for name, clf in individual_models.items():
    preds = clf.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    print(f"  -> {name:<26} | Validation Acc: {acc*100:6.2f}% | Macro-F1: {f1:0.4f}")

# ==============================================================================
# 4. دراسة تأثير عتبة الثقة (Threshold Tuning & Safety Guard)
# ==============================================================================
print("\n" + "=" * 65)
print("  تحليل تأثير عتبة الثقة (Confidence Threshold Analysis)")
print("=" * 65)

# حساب التوزيع الاحتمالي للـ Ensemble على مجموعة الاختبار المستقلة (Test Set)
test_probs = ensemble_model.predict_proba(X_test_s)
test_preds = np.argmax(test_probs, axis=1)
max_confidences = np.max(test_probs, axis=1)

thresholds = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
th_results = []

for th in thresholds:
    accepted_mask = (max_confidences >= th)
    num_accepted = np.sum(accepted_mask)
    coverage = (num_accepted / len(y_test)) * 100
    
    if num_accepted > 0:
        acc_th = accuracy_score(y_test[accepted_mask], test_preds[accepted_mask]) * 100
    else:
        acc_th = 100.0

    rejected = len(y_test) - num_accepted
    th_results.append({
        "Threshold": f">={th*100:.0f}%",
        "Accuracy (%)": round(acc_th, 2),
        "Accepted Leaves": num_accepted,
        "Coverage (%)": round(coverage, 2),
        "Uncertain / Rejected": rejected
    })

df_th = pd.DataFrame(th_results)
print(df_th.to_string(index=False))

# رسم منحنى التبادل بين الدقة والتغطية (Trade-off Curve)
plt.figure(figsize=(9, 5))
plt.plot(thresholds, [r["Accuracy (%)"] for r in th_results], marker='o', color='green', linewidth=2.5, label="Accuracy (%)")
plt.plot(thresholds, [r["Coverage (%)"] for r in th_results], marker='s', color='blue', linestyle='--', linewidth=2, label="Coverage (%)")
plt.title("Threshold vs Accuracy & Coverage (Decision Safety Curve)", fontsize=13)
plt.xlabel("Confidence Threshold", fontsize=11)
plt.ylabel("Percentage (%)", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("threshold_tradeoff_curve.png", dpi=200)
plt.close()
print("\n[+] تم حفظ منحنى عتبة القرار في: threshold_tradeoff_curve.png")

# ==============================================================================
# 5. حفظ وتصدير نموذج الـ Ensemble المتطور
# ==============================================================================
out_dir = "saved_models"
os.makedirs(out_dir, exist_ok=True)

model_path = os.path.join(out_dir, "ensemble_model.pkl")
scaler_path = os.path.join(out_dir, "ensemble_scaler.pkl")
labels_path = os.path.join(out_dir, "ensemble_labels.json")

joblib.dump(ensemble_model, model_path)
joblib.dump(scaler, scaler_path)

with open(labels_path, "w", encoding="utf-8") as f:
    json.dump({str(i): name for i, name in enumerate(le.classes_)}, f, indent=2)

print(f"\n[+] تم حفظ نموذج الـ Ensemble في: {model_path}")
print(f"[+] تم حفظ معالج التقييس في:   {scaler_path}")
print(f"[+] تم حفظ قاموس الفئات في:     {labels_path}")

print("\n" + "=" * 75)
print("  اكتمل بناء الـ Ensemble ودمج نظام الـ Threshold بنجاح تام!")
print("=" * 75)
