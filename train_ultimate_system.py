import os
import sys

# ضمان ترميز UTF-8 في ويندوز
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
import concurrent.futures

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib

import lightgbm as lgb
import xgboost as xgb

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("  Plant Disease Diagnosis — The Ultimate Classical ML Production System")
print("  (Multi-Model Benchmark, Weighted Ensemble, Threshold Guard & Morphology)")
print("=" * 80)

DATA_DIR = "raw/color"
CACHE_FILE = "ultimate_features_cache.npz"

# ==============================================================================
# 1. هندسة السمات البصرية الفائقة وعالية السرعة (HSV + LAB + GLCM + LBP + Hu)
# ==============================================================================
def extract_ultimate_features(image_path, target_size=(128, 128)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. مدرج ألوان HSV (512 ميزة)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_hsv, hist_hsv)

        # 2. فضاء الألوان LAB (6 ميزات: متوسط وانحراف الألوان لرصد موت الأنسجة)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        mean_lab, std_lab = cv2.meanStdDev(lab)
        color_stats = np.hstack([mean_lab.flatten(), std_lab.flatten()])

        # 3. مصفوفة الملمس GLCM المحسنة والفائقة السرعة (مستويات 64، متينة جداً ومناعية)
        gray_64 = (gray // 4).astype(np.uint8)
        glcm = graycomatrix(gray_64, distances=[1], angles=[0, np.pi/2], levels=64, symmetric=True, normed=True)
        f_glcm = [
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'dissimilarity').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'correlation').mean()
        ]

        # 4. ملمس السطح المجهري LBP (18 ميزة بنمط موحد)
        lbp = local_binary_pattern(gray, 16, 2, method="uniform")
        (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 19), range=(0, 18))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)

        # 5. عزوم الشكل الهندسية Hu Moments (7 ميزات للتمييز بين شكل ورقة العنب والذرة والطماطم)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        for i in range(len(hu)):
            hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i]) + 1e-10)

        # دمج 548 ميزة فيزيائية مركّزة
        return np.hstack([hist_hsv.flatten(), color_stats, f_glcm, hist_lbp, hu])
    except Exception:
        return None

# ==============================================================================
# 2. تجميع عينة ضخمة ومتوازنة إحصائياً (150 صورة لكل فئة = ~5,700 صورة)
# ==============================================================================
all_classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"[*] فحص الفئات: تم العثور على {len(all_classes)} فئة تغطي 14 محصولاً زراعياً.")

SAMPLES_PER_CLASS = 150  # عينة ضخمة تضمن التوازن وتمنع طغيان الفئات الكبرى

if os.path.exists(CACHE_FILE):
    try:
        print(f"[*] فحص ملف السمات ({CACHE_FILE})...")
        cached = np.load(CACHE_FILE)
        X = cached["X"]
        y = cached["y"]
        paths = cached["paths"]
        print(f"[+] تم التحميل بنجاح! حجم البيانات: {X.shape[0]} عينة | عدد السمات: {X.shape[1]}")
    except Exception:
        print("[!] الكاش تالف أو غير مكتمل، جاري إعادة الاستخراج من الصفر...")
        os.remove(CACHE_FILE)
        cached = None
else:
    cached = None

if cached is None:
    print(f"[*] جاري تجميع عينة متوازنة ({SAMPLES_PER_CLASS} صورة لكل فئة كحد أقصى)...")
    tasks = []
    for cls in all_classes:
        folder = os.path.join(DATA_DIR, cls)
        img_names = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp'))]
        selected = img_names[:SAMPLES_PER_CLASS]
        for name in selected:
            tasks.append((os.path.join(folder, name), cls))

    print(f"[*] بدء الاستخراج المتوازي للسمات لـ {len(tasks):,} صورة عبر كافة الأنوية...")
    t0 = time.time()

    def worker(item):
        p, c = item
        f = extract_ultimate_features(p)
        return f, c, p

    X_list, y_list, paths_list = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, (f, c, p) in enumerate(executor.map(worker, tasks), 1):
            if f is not None:
                X_list.append(f)
                y_list.append(c)
                paths_list.append(p)
            if idx % 1000 == 0 or idx == len(tasks):
                print(f"  -> معالجة {idx:,}/{len(tasks):,} صورة ({idx/len(tasks)*100:.1f}%)...")

    X = np.array(X_list)
    y = np.array(y_list)
    paths = np.array(paths_list)
    
    np.savez_compressed(CACHE_FILE, X=X, y=y, paths=paths)
    print(f"[+] اكتمل الاستخراج لجميع العينات بنجاح في {time.time() - t0:.1f} ثانية! الحجم: {X.shape}")

# ==============================================================================
# 3. التقسيم الطبقي ومنع تسريب البيانات (70% Train, 15% Val, 15% Test)
# ==============================================================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels_dict = {str(i): name for i, name in enumerate(le.classes_)}

idx = np.arange(len(X))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_encoded)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y_encoded[temp_idx])

X_train, y_train, paths_train = X[train_idx], y_encoded[train_idx], paths[train_idx]
X_val, y_val, paths_val       = X[val_idx], y_encoded[val_idx], paths[val_idx]
X_test, y_test, paths_test    = X[test_idx], y_encoded[test_idx], paths[test_idx]

print(f"\n[*] تقسيم البيانات الطبقي:")
print(f"  * عينات التدريب (Train):   {len(X_train):,} (70%)")
print(f"  * عينات التحقق (Val):      {len(X_val):,} (15%)")
print(f"  * عينات الاختبار الأعمى:   {len(X_test):,} (15%)")

# تقييس رياضي حصري على بيانات التدريب
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ==============================================================================
# 4. مقارنة معيارية شاملة: خط الأساس + 4 نماذج متقدمة
# ==============================================================================
print("\n" + "=" * 70)
print("  المقارنة المعيارية على مجموعة التحقق (Validation Benchmark)")
print("=" * 70)

models_pool = {
    "1. Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    
    "2. Logistic Regression": LogisticRegression(max_iter=500, random_state=SEED),
    
    "3. Random Forest (180 Trees)": RandomForestClassifier(
        n_estimators=180,
        max_depth=22,
        random_state=SEED,
        n_jobs=-1
    ),
    
    "4. Extra Trees (160 Trees)": ExtraTreesClassifier(
        n_estimators=160,
        max_depth=22,
        random_state=SEED,
        n_jobs=-1
    ),
    
    "5. XGBoost Classifier": xgb.XGBClassifier(
        n_estimators=130,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="mlogloss",
        n_jobs=-1
    ),
    
    "6. LightGBM Classifier": lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.07,
        max_depth=6,
        num_leaves=35,
        subsample=0.8,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1
    )
}

val_records = []
fitted_models = {}

for name, clf in models_pool.items():
    t_m = time.time()
    clf.fit(X_train_s, y_train)
    t_elapsed = time.time() - t_m
    
    val_pred = clf.predict(X_val_s)
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average="macro")
    
    fitted_models[name] = clf
    val_records.append({
        "Model": name,
        "Val Accuracy (%)": round(acc * 100, 2),
        "Val Macro-F1": round(f1, 4),
        "Training Time (s)": round(t_elapsed, 2)
    })
    print(f"  -> {name:<28} | Val Acc: {acc*100:6.2f}% | Macro-F1: {f1:0.4f} | Time: {t_elapsed:5.2f}s")

df_benchmark = pd.DataFrame(val_records).sort_values(by="Val Macro-F1", ascending=False)
print("\n" + "=" * 65)
print("  جدول مقارنة النماذج:")
print(df_benchmark.to_string(index=False))

# ==============================================================================
# 5. بناء نموذج الـ Ensemble الموزون المثالي (Weighted Soft-Voting)
# ==============================================================================
print("\n" + "=" * 70)
print("  دمج النماذج المتفوقة في The Ultimate Soft-Voting Ensemble")
print("=" * 70)

class SoftVotingEnsemble:
    """مجمع تصويت احتمالي موزون فائق السرعة والدقة"""
    def __init__(self, models, weights, classes):
        self.models = models
        self.weights = weights
        self.classes_ = classes
        
    def predict_proba(self, X):
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.models)) / sum(self.weights)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# دمج أفضل النماذج الأربعة: LightGBM (وزن 3) + ExtraTrees (وزن 2) + RF (وزن 2) + XGBoost (وزن 2)
selected_models = [
    fitted_models["6. LightGBM Classifier"],
    fitted_models["4. Extra Trees (160 Trees)"],
    fitted_models["3. Random Forest (180 Trees)"],
    fitted_models["5. XGBoost Classifier"]
]
selected_weights = [3, 2, 2, 2]

ultimate_ensemble = SoftVotingEnsemble(selected_models, selected_weights, le.classes_)

val_ens_pred = ultimate_ensemble.predict(X_val_s)
ens_val_acc = accuracy_score(y_val, val_ens_pred)
ens_val_f1 = f1_score(y_val, val_ens_pred, average="macro")
print(f"  🏆 دقة التحقق لنموذج الـ Ensemble الموزون: {ens_val_acc * 100:.2f}% | Macro-F1: {ens_val_f1:.4f}")

# ==============================================================================
# 6. التقييم النهائي المستقل على مجموعة الاختبار (Test Set Evaluation)
# ==============================================================================
print("\n" + "=" * 70)
print("  التقييم المستقل النهائي على مجموعة الاختبار الأعمى (Test Set)")
print("=" * 70)

test_probs = ultimate_ensemble.predict_proba(X_test_s)
test_preds = np.argmax(test_probs, axis=1)
raw_test_acc = accuracy_score(y_test, test_preds)
raw_test_f1 = f1_score(y_test, test_preds, average="macro")

print(f"  * الدقة الأساسية لمجموعة الاختبار (Overall Test Acc): {raw_test_acc * 100:.2f}%")
print(f"  * مقياس الأداء الكلي (Overall Macro-F1):           {raw_test_f1:.4f}")

# ==============================================================================
# 7. دراسة عتبة القرار (Threshold Guard & Safety Curve)
# ==============================================================================
print("\n" + "=" * 70)
print("  تحليل تأثير عتبة الثقة (Confidence Threshold Analysis)")
print("=" * 70)

max_confs = np.max(test_probs, axis=1)
threshold_levels = [0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
th_rows = []

for th in threshold_levels:
    mask = (max_confs >= th)
    acc_count = np.sum(mask)
    coverage = (acc_count / len(y_test)) * 100
    
    if acc_count > 0:
        cur_acc = accuracy_score(y_test[mask], test_preds[mask]) * 100
    else:
        cur_acc = 100.0
    
    th_rows.append({
        "Threshold": f">={th*100:.0f}%",
        "Accuracy (%)": round(cur_acc, 2),
        "Coverage (%)": round(coverage, 2),
        "Accepted Leaves": acc_count,
        "Rejected (Uncertain)": len(y_test) - acc_count
    })

df_th_res = pd.DataFrame(th_rows)
print(df_th_res.to_string(index=False))

# رسم وحفظ منحنى الأمان الشامل
plt.figure(figsize=(9, 5))
plt.plot(threshold_levels, [r["Accuracy (%)"] for r in th_rows], marker='o', color='forestgreen', linewidth=2.5, label="Accuracy (%)")
plt.plot(threshold_levels, [r["Coverage (%)"] for r in th_rows], marker='s', color='navy', linestyle='--', linewidth=2, label="Coverage (%)")
plt.title("Ultimate Plant Model: Decision Safety & Threshold Curve", fontsize=12)
plt.xlabel("Confidence Threshold", fontsize=10)
plt.ylabel("Percentage (%)", fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("ultimate_threshold_curve.png", dpi=200)
plt.close()
print("  [+] تم حفظ منحنى عتبة القرار في: ultimate_threshold_curve.png")

# رسم مصفوفة الالتباس المعيارية
cm = confusion_matrix(y_test, test_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 10))
clean_classes = [c.replace("___", " -> ") for c in le.classes_]
sns.heatmap(cm_norm, cmap='Blues', annot=False, xticklabels=False, yticklabels=clean_classes)
plt.title(f"Ultimate Ensemble Confusion Matrix (Acc: {raw_test_acc*100:.2f}%)", fontsize=12)
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.tight_layout()
plt.savefig("ultimate_confusion_matrix.png", dpi=200)
plt.close()
print("  [+] تم حفظ مصفوفة الالتباس في: ultimate_confusion_matrix.png")

# ==============================================================================
# 8. حفظ وتصدير النظام النهائي المتكامل
# ==============================================================================
out_dir = "saved_models"
os.makedirs(out_dir, exist_ok=True)

joblib.dump(ultimate_ensemble, os.path.join(out_dir, "ultimate_ensemble_model.pkl"))
joblib.dump(scaler, os.path.join(out_dir, "ultimate_scaler.pkl"))
with open(os.path.join(out_dir, "ultimate_labels.json"), "w", encoding="utf-8") as f:
    json.dump(labels_dict, f, indent=2)

print("\n[+] تم تصدير النموذج التجميعي الفائق وقاموس الفئات في: saved_models/ultimate_ensemble_model.pkl")
print("=" * 80)
print("  اكتمل بناء وتدريب النظام الشامل المثالي بنجاح تام!")
print("=" * 80)
