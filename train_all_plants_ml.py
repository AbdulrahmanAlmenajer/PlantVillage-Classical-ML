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
import concurrent.futures

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib
import lightgbm as lgb

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("=" * 75)
print("  PlantVillage — Universal Plant Disease Classification (All 38 Classes)")
print("=" * 75)

# ==============================================================================
# 1. قراءة وفحص جميع فئات النباتات الـ 38
# ==============================================================================
data_dir = "raw/color"
all_classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f"[*] تم العثور على {len(all_classes)} فئة تشمل 14 محصولاً زراعياً مختلفاً:")

class_distribution = {}
for c in all_classes:
    cnt = len(os.listdir(os.path.join(data_dir, c)))
    class_distribution[c] = cnt

total_dataset_images = sum(class_distribution.values())
print(f"[*] إجمالي الصور في قاعدة البيانات بالكامل: {total_dataset_images:,} صورة.\n")

# ==============================================================================
# 2. دوال استخراج السمات البصرية الفائقة
# ==============================================================================
def extract_single_image_features(image_path, target_size=(128, 128)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. ألوان HSV (الاصفرار والبقع)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_hsv, hist_hsv)
        mean_bgr = cv2.mean(img)[:3]
        f_color = np.hstack([hist_hsv.flatten(), mean_bgr])

        # 2. ملمس GLCM
        glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
        f_glcm = [
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'dissimilarity').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'correlation').mean()
        ]

        # 3. أنماط LBP
        lbp = local_binary_pattern(gray, 24, 3, method="uniform")
        (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)

        return np.hstack([f_color, f_glcm, hist_lbp])
    except Exception:
        return None

# ==============================================================================
# 3. التحميل المتوازي والتخزين المؤقت (Parallel Extraction & Caching)
# ==============================================================================
cache_file = "all_plants_features_cache.npz"
SAMPLES_PER_CLASS = 100  # 100 صورة لكل فئة = 3800 صورة ممثلة ومتوازنة للـ 38 فئة

if os.path.exists(cache_file):
    print(f"[*] تم العثور على ملف الكاش العام ({cache_file})، جاري التحميل فوراً...")
    cached = np.load(cache_file)
    X = cached["X"]
    y = cached["y"]
    paths = cached["paths"]
    print(f"[+] تم تحميل السمات بنجاح! الأبعاد: {X.shape}")
else:
    print(f"[*] جاري استخراج السمات لـ 38 فئة بمعدل ({SAMPLES_PER_CLASS} صورة لكل فئة)...")
    tasks = []
    for cls in all_classes:
        folder = os.path.join(data_dir, cls)
        imgs = os.listdir(folder)[:SAMPLES_PER_CLASS]
        for img_name in imgs:
            full_p = os.path.join(folder, img_name)
            tasks.append((full_p, cls))

    print(f"[*] بدء المعالجة المتوازية لـ {len(tasks)} صورة عبر أنوية المعالج...")
    t0 = time.time()
    
    # دالة مساعدة للمعالجة
    def worker(item):
        p, c = item
        f = extract_single_image_features(p)
        return f, c, p

    X_list, y_list, paths_list = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, tasks))

    for f, c, p in results:
        if f is not None:
            X_list.append(f)
            y_list.append(c)
            paths_list.append(p)

    X = np.array(X_list)
    y = np.array(y_list)
    paths = np.array(paths_list)
    print(f"[+] اكتمل الاستخراج لجميع النباتات في {time.time() - t0:.1f} ثانية! الحجم: {X.shape}")
    np.savez_compressed(cache_file, X=X, y=y, paths=paths)

# ==============================================================================
# 4. ترميز الفئات والتقسيم الطبقي (70% Train, 15% Val, 15% Test)
# ==============================================================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

idx = np.arange(len(X))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_encoded)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y_encoded[temp_idx])

X_train, y_train = X[train_idx], y_encoded[train_idx]
X_val, y_val     = X[val_idx], y_encoded[val_idx]
X_test, y_test   = X[test_idx], y_encoded[test_idx]

print(f"\n[+] التقسيم لجميع النباتات:")
print(f"  * عينات التدريب:   {len(X_train)} (70%)")
print(f"  * عينات التحقق:    {len(X_val)} (15%)")
print(f"  * عينات الاختبار:  {len(X_test)} (15%)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ==============================================================================
# 5. تدريب النموذج الشامل (LightGBM على 38 فئة)
# ==============================================================================
print("\n[*] تدريب نموذج LightGBM الشامل على 38 فئة نباتية...")
t_train = time.time()
model_all = lgb.LGBMClassifier(
    n_estimators=180,
    learning_rate=0.07,
    max_depth=6,
    num_leaves=35,
    subsample=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1
)
model_all.fit(X_train_s, y_train)
print(f"[+] اكتمل تدريب النموذج في {time.time() - t_train:.1f} ثانية!")

# التحقق (Validation)
val_preds = model_all.predict(X_val_s)
val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds, average="macro")
print(f"  * دقة التحقق العامة (Validation Accuracy): {val_acc*100:.2f}%")
print(f"  * مقياس Macro-F1 العام للـ 38 فئة: {val_f1:.4f}")

# الاختبار النهائي (Unseen Test Set)
test_preds = model_all.predict(X_test_s)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average="macro")
print(f"\n" + "=" * 60)
print(f"  [النتيجة النهائية للاختبار على كافة النباتات (Test Set)]")
print(f"  => Final Accuracy for all 38 Classes: {test_acc * 100:.2f}%")
print(f"  => Final Macro-F1 across all 38 Classes: {test_f1:.4f}")
print("=" * 60)

# ==============================================================================
# 6. حفظ النموذج الشامل وقاموس الفئات الـ 38
# ==============================================================================
out_dir = "saved_models"
os.makedirs(out_dir, exist_ok=True)

# قاموس الفئات
all_labels_dict = {str(i): name for i, name in enumerate(le.classes_)}
with open(os.path.join(out_dir, "all_plants_labels.json"), "w", encoding="utf-8") as f:
    json.dump(all_labels_dict, f, indent=2)

joblib.dump(model_all, os.path.join(out_dir, "all_plants_model.pkl"))
joblib.dump(scaler, os.path.join(out_dir, "all_plants_scaler.pkl"))
print("\n[+] تم حفظ النموذج الشامل وملف all_plants_labels.json بنجاح في saved_models/!")

