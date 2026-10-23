import os
import sys

# ضمان ترميز UTF-8 في موجه أوامر ويندوز
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib
import lightgbm as lgb

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("=" * 88)
print("  PlantVillage Grand Master Pipeline — Full 54,305 Images | All 38 Classes")
print("  (Illumination Invariance, Morphological Hu, Balanced GBDT Ensemble & Threshold Guard)")
print("=" * 88)

DATA_DIR = "raw/color"
CACHE_FILE = "grand_54k_features_cache.npz"
CHECKPOINT_FILE = "grand_54k_checkpoint.npz"

# ==============================================================================
# المرحلة 1: فحص وتحليل كافة البيانات (54,305 صورة | 38 فئة) وحساب أوزان التوازن
# ==============================================================================
print("\n" + "-" * 75)
print("[المرحلة 1]: حصر وتحليل توزيع الـ 54,305 صورة بالكامل مع حساب الأوزان العادلة")
print("-" * 75)

all_classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
all_image_tasks = []
class_counts = {}

for cls in all_classes:
    cls_folder = os.path.join(DATA_DIR, cls)
    valid_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp'))]
    class_counts[cls] = len(valid_files)
    for fname in valid_files:
        all_image_tasks.append((os.path.join(cls_folder, fname), cls))

total_dataset_size = len(all_image_tasks)
print(f"[*] إجمالي الصور الفعلية المفحوصة: {total_dataset_size:,} صورة.")
print(f"[*] إجمالي الفئات: 38 فئة عبر 14 محصولاً زراعياً.")

max_class_name = max(class_counts, key=class_counts.get)
min_class_name = min(class_counts, key=class_counts.get)
print(f"[*] أكبر فئة: {max_class_name} ({class_counts[max_class_name]:,} صورة)")
print(f"[*] أصغر فئة: {min_class_name} ({class_counts[min_class_name]:,} صورة)")
print(f"[*] نسبة التفاوت: {class_counts[max_class_name]/class_counts[min_class_name]:.2f}x (سيتم تفعيل class_weight='balanced' لإعطاء وزن مضاعف للفئات النادرة)")

# ==============================================================================
# المرحلة 2: هندسة السمات البصرية الفائقة ومعالجة الإضاءة (557 ميزة)
# ==============================================================================
print("\n" + "-" * 75)
print("[المرحلة 2]: استخراج السمات المتقدمة (CLAHE + HSV + LAB + GLCM + LBP + Hu Moments + Laplacian)")
print("-" * 75)

clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def extract_grand_features(image_path, target_size=(128, 128)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. تسوية الإضاءة والظلال التكيفية (CLAHE)
        gray_eq = clahe_filter.apply(gray)

        # 2. فضاء الألوان HSV (512 ميزة) لرصد تشبع الكلوروفيل والاصفرار
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_hsv, hist_hsv)

        # 3. عزوم وإحصائيات الألوان في HSV و LAB (10 ميزات)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        color_moments = [
            hsv[:,:,0].mean(), hsv[:,:,0].std(),
            hsv[:,:,1].mean(), hsv[:,:,1].std(),
            hsv[:,:,2].mean(), hsv[:,:,2].std(),
            lab[:,:,1].mean(), lab[:,:,1].std(),
            lab[:,:,2].mean(), lab[:,:,2].std()
        ]

        # 4. مصفوفة الملمس GLCM المحسنة (64 مستوى) لرصد التآكل النسيجي
        gray_64 = (gray_eq // 4).astype(np.uint8)
        glcm = graycomatrix(gray_64, distances=[1], angles=[0, np.pi/2], levels=64, symmetric=True, normed=True)
        f_glcm = [
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'dissimilarity').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'correlation').mean()
        ]

        # 5. نسيج السطح المجهري LBP (18 ميزة بنمط منتظم) لرصد جراثيم الفطريات
        lbp = local_binary_pattern(gray_eq, 16, 2, method="uniform")
        (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 19), range=(0, 18))
        hist_lbp = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-7)

        # 6. عزوم الشكل الهندسية Hu Moments (7 ميزات لتمييز شكل أوراق المحاصيل)
        moments = cv2.moments(gray_eq)
        hu = cv2.HuMoments(moments).flatten()
        for i in range(len(hu)):
            hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i]) + 1e-10)

        # 7. حدة حواف البقع (Laplacian Variance)
        lap_var = [cv2.Laplacian(gray_eq, cv2.CV_64F).var()]

        # إجمالي 554 ميزة فيزيائية مركّزة
        return np.hstack([hist_hsv.flatten(), color_moments, f_glcm, hist_lbp, hu, lap_var])
    except Exception:
        return None

# فحص أو استخراج سمات الـ 54,305 صورة
if os.path.exists(CACHE_FILE):
    print(f"[*] تم العثور على الكاش الكامل ({CACHE_FILE})، جاري التحميل فوراً...")
    cached = np.load(CACHE_FILE)
    X = cached["X"]
    y = cached["y"]
    paths = cached["paths"]
    print(f"[+] تم تحميل مصفوفة السمات للـ 54,305 صورة بالكامل! الأبعاد: {X.shape}")
else:
    print(f"[*] بدء الاستخراج المتوازي للسمات لـ {len(all_image_tasks):,} صورة عبر 4 مسارات...")
    t_start = time.time()

    def process_task(task):
        p, c = task
        feat = extract_grand_features(p)
        return feat, c, p

    X_list, y_list, paths_list = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, (f, c, p) in enumerate(executor.map(process_task, all_image_tasks), 1):
            if f is not None:
                X_list.append(f)
                y_list.append(c)
                paths_list.append(p)
            
            # تقرير مرحلي وحفظ دوري كل 5000 صورة
            if idx % 5000 == 0 or idx == len(all_image_tasks):
                elapsed = time.time() - t_start
                rate = idx / elapsed
                rem_mins = (len(all_image_tasks) - idx) / rate / 60
                print(f"  -> معالجة {idx:,}/{len(all_image_tasks):,} صورة ({idx/len(all_image_tasks)*100:.1f}%) | المتبقي: {rem_mins:.1f} دقيقة")
                np.savez_compressed(CHECKPOINT_FILE, X=np.array(X_list), y=np.array(y_list), paths=np.array(paths_list))

    X = np.array(X_list)
    y = np.array(y_list)
    paths = np.array(paths_list)
    np.savez_compressed(CACHE_FILE, X=X, y=y, paths=paths)
    print(f"[+] اكتمل استخراج سمات الـ 54,305 صورة بنجاح في {(time.time()-t_start)/60:.1f} دقيقة!")

# ==============================================================================
# المرحلة 3: التقسيم الطبقي الصارم (70% Train, 15% Val, 15% Test)
# ==============================================================================
print("\n" + "-" * 75)
print("[المرحلة 3]: تقسيم البيانات الطبقي لـ 54 ألف صورة ومنع تسريب البيانات")
print("-" * 75)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels_dict = {str(i): name for i, name in enumerate(le.classes_)}

idx = np.arange(len(X))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_encoded)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y_encoded[temp_idx])

X_train, y_train, paths_train = X[train_idx], y_encoded[train_idx], paths[train_idx]
X_val, y_val, paths_val       = X[val_idx], y_encoded[val_idx], paths[val_idx]
X_test, y_test, paths_test    = X[test_idx], y_encoded[test_idx], paths[test_idx]

print(f"  * عينات مجموعة التدريب (Training Set):   {len(X_train):,} صورة (70%)")
print(f"  * عينات مجموعة التحقق (Validation Set): {len(X_val):,} صورة (15%)")
print(f"  * عينات مجموعة الاختبار (Test Set):       {len(X_test):,} صورة (15%)")

# تقييس رياضي حصري على بيانات التدريب
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ==============================================================================
# المرحلة 4: خط الأساس الإحصائي + مقارنة النماذج على مجموعة التحقق
# ==============================================================================
print("\n" + "-" * 75)
print("[المرحلة 4]: المقارنة المعيارية للنماذج على مجموعة التحقق (Validation Benchmark)")
print("-" * 75)

candidate_models = {
    "1. Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    
    "2. Logistic Regression (Balanced)": LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=SEED
    ),
    
    "3. Random Forest (150 Trees, Balanced)": RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    ),
    
    "4. Extra Trees (150 Trees, Balanced)": ExtraTreesClassifier(
        n_estimators=150,
        max_depth=25,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    ),
    
    "5. LightGBM (300 Trees, Balanced)": lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=SEED,
        verbose=-1,
        n_jobs=-1
    )
}

benchmark_results = []
trained_estimators = {}

for name, clf in candidate_models.items():
    t_start_model = time.time()
    clf.fit(X_train_s, y_train)
    t_train = time.time() - t_start_model
    
    val_preds = clf.predict(X_val_s)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    
    trained_estimators[name] = clf
    benchmark_results.append({
        "Model": name,
        "Val Accuracy (%)": round(val_acc * 100, 2),
        "Val Macro-F1": round(val_f1, 4),
        "Training Time (s)": round(t_train, 2)
    })
    print(f"  -> {name:<36} | Val Acc: {val_acc*100:6.2f}% | Macro-F1: {val_f1:0.4f} | Time: {t_train:5.2f}s")

df_benchmark = pd.DataFrame(benchmark_results).sort_values(by="Val Macro-F1", ascending=False)
print("\n" + "=" * 70)
print("  جدول مقارنة النماذج على الـ 54,305 صورة:")
print(df_benchmark.to_string(index=False))

# ==============================================================================
# المرحلة 5: بناء نموذج Soft-Voting Ensemble الموزون المثالي
# ==============================================================================
print("\n" + "-" * 75)
print("[المرحلة 5]: دمج النماذج المتفوقة في Grand Soft-Voting Ensemble")
print("-" * 75)

class GrandVotingEnsemble:
    def __init__(self, models, weights, classes):
        self.models = models
        self.weights = weights
        self.classes_ = classes
        
    def predict_proba(self, X):
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.models)) / sum(self.weights)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# دمج: LightGBM (وزن 3) + ExtraTrees (وزن 2) + Random Forest (وزن 2)
grand_ensemble = GrandVotingEnsemble(
    models=[
        trained_estimators["5. LightGBM (300 Trees, Balanced)"],
        trained_estimators["4. Extra Trees (150 Trees, Balanced)"],
        trained_estimators["3. Random Forest (150 Trees, Balanced)"]
    ],
    weights=[3, 2, 2],
    classes=le.classes_
)

ens_val_preds = grand_ensemble.predict(X_val_s)
ens_val_acc = accuracy_score(y_val, ens_val_preds)
ens_val_f1 = f1_score(y_val, ens_val_preds, average="macro")
print(f"  🏆 دقة التحقق لنموذج الـ Ensemble لـ 54 ألف صورة: {ens_val_acc * 100:.2f}% | Macro-F1: {ens_val_f1:.4f}")

# ==============================================================================
# المرحلة 6: التقييم النهائي المستقل على مجموعة الاختبار (Test Set Evaluation)
# ==============================================================================
print("\n" + "-" * 75)
print("[المرحلة 6]: التقييم المستقل النهائي على مجموعة الاختبار الأعمى (8,146 صورة)")
print("-" * 75)

test_probs = grand_ensemble.predict_proba(X_test_s)
test_preds = np.argmax(test_probs, axis=1)
raw_test_acc = accuracy_score(y_test, test_preds)
raw_test_f1 = f1_score(y_test, test_preds, average="macro")

print(f"  * إجمالي صور الاختبار الأعمى المستقلة: {len(y_test):,} صورة.")
print(f"  * دقة الاختبار الأساسية للبيانات الكاملة:  {raw_test_acc * 100:.2f}%")
print(f"  * مقياس الأداء الكلي (Overall Macro-F1):   {raw_test_f1:.4f}")

# ==============================================================================
# المرحلة 7: دراسة عتبة القرار (Threshold Guard) ومصفوفة الالتباس
# ==============================================================================
print("\n" + "-" * 75)
print("[المرحلة 7]: تحليل عتبة الأمان (Threshold Guard) ورسم المنحنيات")
print("-" * 75)

max_confs = np.max(test_probs, axis=1)
threshold_levels = [0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
th_rows = []

for th in threshold_levels:
    mask = (max_confs >= th)
    acc_count = np.sum(mask)
    coverage = (acc_count / len(y_test)) * 100
    cur_acc = accuracy_score(y_test[mask], test_preds[mask]) * 100 if acc_count > 0 else 100.0
    
    th_rows.append({
        "Threshold": f">={th*100:.0f}%",
        "Accuracy (%)": round(cur_acc, 2),
        "Coverage (%)": round(coverage, 2),
        "Accepted Leaves": acc_count,
        "Rejected (Uncertain)": len(y_test) - acc_count
    })

df_th = pd.DataFrame(th_rows)
print(df_th.to_string(index=False))

# رسم وحفظ منحنى الأمان الشامل
plt.figure(figsize=(9, 5))
plt.plot(threshold_levels, [r["Accuracy (%)"] for r in th_rows], marker='o', color='forestgreen', linewidth=2.5, label="Test Accuracy (%)")
plt.plot(threshold_levels, [r["Coverage (%)"] for r in th_rows], marker='s', color='navy', linestyle='--', linewidth=2, label="Leaf Coverage (%)")
plt.title("PlantVillage 54k Grand Model: Threshold Safety Curve", fontsize=12)
plt.xlabel("Confidence Threshold", fontsize=10)
plt.ylabel("Percentage (%)", fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("grand_54k_threshold_curve.png", dpi=200)
plt.close()
print("  [+] تم حفظ منحنى عتبة القرار في: grand_54k_threshold_curve.png")

# رسم مصفوفة الالتباس المعيارية
cm = confusion_matrix(y_test, test_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 12))
clean_classes = [c.replace("___", " -> ") for c in le.classes_]
sns.heatmap(cm_norm, cmap='Blues', annot=False, xticklabels=False, yticklabels=clean_classes)
plt.title(f"Grand 54k Ensemble Confusion Matrix (Test Acc: {raw_test_acc*100:.2f}%)", fontsize=13)
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.tight_layout()
plt.savefig("grand_54k_confusion_matrix.png", dpi=200)
plt.close()
print("  [+] تم حفظ مصفوفة الالتباس المعيارية في: grand_54k_confusion_matrix.png")

# ==============================================================================
# المرحلة 8: حفظ وتصدير النظام الفائق للإنتاج
# ==============================================================================
print("\n" + "-" * 75)
print("[المرحلة 8]: حفظ وتصدير النموذج النهائي وقاموس الفئات")
print("-" * 75)

out_dir = "saved_models"
os.makedirs(out_dir, exist_ok=True)

joblib.dump(grand_ensemble, os.path.join(out_dir, "grand_54k_ensemble_model.pkl"))
joblib.dump(scaler, os.path.join(out_dir, "grand_54k_scaler.pkl"))
with open(os.path.join(out_dir, "grand_54k_labels.json"), "w", encoding="utf-8") as f:
    json.dump(labels_dict, f, indent=2)

print("\n[+] تم تصدير نموذج الـ 54,305 صورة في: saved_models/grand_54k_ensemble_model.pkl")
print("=" * 88)
print("  اكتمل تنفيذ المشروع الأكبر للـ 54,305 صورة بكافة مراحله الثمانية بنجاح تام!")
print("=" * 88)

