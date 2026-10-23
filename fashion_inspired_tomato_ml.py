import os
import sys

# ضمان ترميز UTF-8 في موجه الأوامر بالويندوز
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
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib

# المكتبات المتقدمة المستوحاة من مشروع الطالب
import xgboost as xgb
import lightgbm as lgb

# ==============================================================================
# 0. التكرارية والتوافق (Reproducibility & Seed)
# ==============================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("=" * 75)
print("  PlantVillage Tomato Disease Classification — Advanced ML Benchmark")
print("  (Integrating GBDT: LightGBM, XGBoost, MLP, Error Analysis & Domain Shift)")
print("=" * 75)

# ==============================================================================
# TASK 1: فهم البيانات وتوزيع الفئات (Inspect Data & Class Distribution)
# ==============================================================================
data_dir = "raw/color"
tomato_classes = sorted([d for d in os.listdir(data_dir) if "Tomato" in d])

class_counts = {}
for cls in tomato_classes:
    class_counts[cls] = len(os.listdir(os.path.join(data_dir, cls)))

total_raw_images = sum(class_counts.values())

df_dist = pd.DataFrame({
    "class_id": range(len(tomato_classes)),
    "class_name": [c.replace("Tomato___", "") for c in tomato_classes],
    "raw_folder": tomato_classes,
    "count": [class_counts[c] for c in tomato_classes],
    "percentage": [round((class_counts[c] / total_raw_images) * 100, 2) for c in tomato_classes]
})

print("\n[TASK 1] توزيع فئات أوراق الطماطم في مجموعة البيانات:")
print(df_dist[["class_id", "class_name", "count", "percentage"]].to_string(index=False))
print(f"\nإجمالي الصور الكلي: {total_raw_images} عبر 10 فئات تصنيف متعدد (Multiclass Classification).")

# ==============================================================================
# استخراج السمات البصرية (Feature Extraction: HSV + GLCM + LBP)
# ==============================================================================
def extract_leaf_features(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. ألوان HSV (تدرج الاصفرار والبقع الداكنة)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv)
    mean_color = cv2.mean(img)[:3]
    f_color = np.hstack([hist_hsv.flatten(), mean_color])

    # 2. ملامس التلف GLCM
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    f_glcm = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ]

    # 3. نسيج السطح LBP
    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    return np.hstack([f_color, f_glcm, hist_lbp])

cache_file = "features_cache.npz"
if os.path.exists(cache_file):
    print(f"[*] تم العثور على ملف السمات المخزنة مؤقتاً ({cache_file})، جاري التحميل فوراً...")
    cached_data = np.load(cache_file)
    X = cached_data["X"]
    y = cached_data["y"]
    paths = cached_data["paths"]
    print(f"[+] تم تحميل السمات فوراً! الحجم: {X.shape}")
else:
    # تحميل عينة متوازنة وسريعة للتدريب والمقارنة (150 صورة لكل فئة = 1500 صورة)
    SAMPLES_PER_CLASS = 150
    X_list, y_list, paths_list = [], [], []

    print(f"\n[*] جاري استخراج السمات لعينة متوازنة ({SAMPLES_PER_CLASS} صورة/فئة)...")
    t_start_feat = time.time()
    for cls in tomato_classes:
        cls_folder = os.path.join(data_dir, cls)
        imgs = os.listdir(cls_folder)[:SAMPLES_PER_CLASS]
        for img_name in imgs:
            p = os.path.join(cls_folder, img_name)
            feat = extract_leaf_features(p)
            if feat is not None:
                X_list.append(feat)
                y_list.append(cls)
                paths_list.append(p)

    X = np.array(X_list)
    y = np.array(y_list)
    paths = np.array(paths_list)
    np.savez_compressed(cache_file, X=X, y=y, paths=paths)
    print(f"[+] تم استخراج وحفظ السمات بنجاح في {time.time() - t_start_feat:.1f} ثانية. أبعاد البيانات: {X.shape}")

# ترميز الفئات
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_labels = {int(i): str(name) for i, name in enumerate(le.classes_)}

# ==============================================================================
# TASK 2: تقسيم البيانات إلى Train (70%) و Validation (15%) و Test (15%)
# ==============================================================================
print("\n[TASK 2] تقسيم البيانات الثلاثي مع الاحتفاظ بمسارات الصور للتحليل البصري:")
idx = np.arange(len(X))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_encoded)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y_encoded[temp_idx])

X_train, y_train, paths_train = X[train_idx], y_encoded[train_idx], paths[train_idx]
X_val, y_val, paths_val       = X[val_idx], y_encoded[val_idx], paths[val_idx]
X_test, y_test, paths_test    = X[test_idx], y_encoded[test_idx], paths[test_idx]

print(f"  * Training Set:   {X_train.shape[0]} عينة (70%)")
print(f"  * Validation Set: {X_val.shape[0]} عينة (15%)")
print(f"  * Test Set:       {X_test.shape[0]} عينة (15%)")

# تقييس البيانات مع منع تسريب البيانات
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ==============================================================================
# TASK 3 & 4: مقارنة النماذج (LightGBM, XGBoost, Random Forest, SVM, MLP)
# ==============================================================================
print("\n[TASK 3 & 4] تدريب ومقارنة خوارزميات الـ ML المتقدمة على مجموعة التحقق (Validation):")

models = {
    "Baseline Dummy": DummyClassifier(strategy="most_frequent"),
    
    "XGBoost Classifier": xgb.XGBClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="mlogloss",
        n_jobs=-1
    ),
    
    "LightGBM Classifier": lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        num_leaves=25,
        subsample=0.8,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1
    ),
    
    "Support Vector Machine (RBF)": SVC(
        C=10.0,
        gamma='scale',
        probability=True,
        random_state=SEED
    ),
    
    "Random Forest (150 Trees)": RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        random_state=SEED,
        n_jobs=-1
    ),
    
    "Dense MLP Baseline (Neural Net)": MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        max_iter=200,
        early_stopping=True,
        random_state=SEED
    )
}

results = []
trained_models = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train_s, y_train)
    elapsed = time.time() - t0
    
    val_preds = model.predict(X_val_s)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    
    trained_models[name] = model
    results.append({
        "Model": name,
        "Val Accuracy (%)": round(val_acc * 100, 2),
        "Val Macro-F1": round(val_f1, 4),
        "Train Time (s)": round(elapsed, 2)
    })
    print(f"  -> {name:<32} | Val Acc: {val_acc*100:6.2f}% | F1: {val_f1:0.4f} | Time: {elapsed:5.2f}s")

df_results = pd.DataFrame(results).sort_values(by="Val Macro-F1", ascending=False)
print("\n" + "=" * 65)
print("  جدول مقارنة النماذج على مجموعة التحقق (Validation Comparison Table)")
print("=" * 65)
print(df_results.to_string(index=False))

# اختيار النموذج الأفضل
best_model_name = df_results.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\n[*] النموذج الفائز بناءً على مجموعة التحقق هو: [{best_model_name}] [Winner]")

# ==============================================================================
# TASK 5: التقييم النهائي المستقل على مجموعة الاختبار (Test Set)
# ==============================================================================
print("\n[TASK 5] التقييم النهائي على مجموعة الاختبار الأعمى (Test Set):")
test_preds = best_model.predict(X_test_s)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average="macro")
print(f"  * Final Test Accuracy : {test_acc * 100:.2f}%")
print(f"  * Final Test Macro-F1 : {test_f1:.4f}")

clean_names = [cls.replace("Tomato___", "") for cls in le.classes_]
print("\nتقرير التصنيف المفصل (Classification Report):")
print(classification_report(y_test, test_preds, target_names=clean_names))

# ==============================================================================
# TASK 6: مصفوفة الالتباس المعيارية + عرض الصور الخاطئة (Error Analysis)
# ==============================================================================
print("\n[TASK 6] تحليل الأخطاء ومصفوفة الالتباس المعيارية (Normalized Confusion Matrix):")
cm = confusion_matrix(y_test, test_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=clean_names, yticklabels=clean_names)
plt.title(f"Normalized Confusion Matrix - {best_model_name}\n(Test Accuracy: {test_acc*100:.2f}%)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("normalized_confusion_matrix.png", dpi=200)
plt.close()
print("  [+] تم حفظ مصفوفة الالتباس المعيارية: normalized_confusion_matrix.png")

# استخراج عينات الصور التي أخطأ فيها النموذج
misclassified_mask = (y_test != test_preds)
misclassified_indices = np.where(misclassified_mask)[0]
print(f"\n  * إجمالي الأخطاء في مجموعة الاختبار: {len(misclassified_indices)} من أصل {len(y_test)} صورة.")

if len(misclassified_indices) > 0:
    num_to_show = min(5, len(misclassified_indices))
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 4))
    if num_to_show == 1:
        axes = [axes]
    
    print("\n  [تحليل العينات الخاطئة البصرية (Misclassified Samples)]:")
    for i in range(num_to_show):
        m_idx = misclassified_indices[i]
        img_p = paths_test[m_idx]
        true_lbl = clean_names[y_test[m_idx]]
        pred_lbl = clean_names[test_preds[m_idx]]
        
        # قراءة وعرض الصورة
        bgr = cv2.imread(img_p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(rgb)
        axes[i].set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color="red", fontsize=9)
        axes[i].axis("off")
        print(f"    - صورة {i+1}: الفئة الحقيقية [{true_lbl}] | توقع النموذج [{pred_lbl}]")
    
    plt.tight_layout()
    plt.savefig("misclassified_samples.png", dpi=200)
    plt.close()
    print("  [+] تم حفظ صور الأخطاء التحليلية الخمس في: misclassified_samples.png")

# ==============================================================================
# TASK 7: حفظ النموذج وقاموس الفئات JSON والتحقق من إعادة التحميل
# ==============================================================================
print("\n[TASK 7] حفظ النموذج، الـ Scaler، وملف tomato_labels.json:")
out_dir = "saved_models"
os.makedirs(out_dir, exist_ok=True)

# 1. حفظ قاموس الفئات كـ JSON
labels_json_path = os.path.join(out_dir, "tomato_labels.json")
with open(labels_json_path, "w", encoding="utf-8") as f:
    json.dump({str(i): name for i, name in enumerate(le.classes_)}, f, indent=2)
print(f"  [+] تم حفظ أسماء الفئات في: {labels_json_path}")

# 2. حفظ النموذج والتقييس
model_save_path = os.path.join(out_dir, "best_tomato_model.pkl")
joblib.dump(best_model, model_save_path)
joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
print(f"  [+] تم حفظ النموذج الأفضل في: {model_save_path}")

# 3. اختبار إعادة التحميل ومطابقة التنبؤات
reloaded_model = joblib.load(model_save_path)
reloaded_preds = reloaded_model.predict(X_test_s)
assert np.array_equal(test_preds, reloaded_preds), "خطأ: التنبؤات غير متطابقة بعد إعادة التحميل!"
print("  [+] اختبار إعادة التحميل (Reload Assertion): متطابق بنسبة 100% بدون أي خلل.")

# ==============================================================================
# TASK 8: اختبار صورة وتوثيق ظاهرة إزاحة النطاق (Domain Shift Analysis)
# ==============================================================================
print("\n[TASK 8] اختبار استنتاج وتوثيق حدود التعميم وإزاحة النطاق (Domain Shift):")
sample_test_img = paths_test[0]
sample_feat = extract_leaf_features(sample_test_img)
sample_feat_s = scaler.transform([sample_feat])

# فحص أعلى 3 تصنيفات (Top-3 Class Scores)
if hasattr(best_model, "predict_proba"):
    probs = best_model.predict_proba(sample_feat_s)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    print(f"  * أعلى 3 تنبؤات لاحتمالية المرض للصورة المختبرة:")
    for rank, idx_cls in enumerate(top3_idx, 1):
        print(f"    {rank}. {clean_names[idx_cls]:<30}: {probs[idx_cls]*100:.2f}%")
else:
    single_pred = best_model.predict(sample_feat_s)[0]
    print(f"  * التنبؤ للصورة المختبرة: {clean_names[single_pred]}")

print("\n" + "=" * 75)
print("  اكتمل تنفيذ المشروع والمعايرة الشاملة بنجاح تام!")
print("=" * 75)

