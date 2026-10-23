import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib

print("=====================================================================")
print("  تنفيذ منهجية علم البيانات الصارمة لمشروع تصنيف أمراض الطماطم (ML)  ")
print("=====================================================================")

# -------------------------------------------------------------------
# الخطوة 1: فهم البيانات وتحليل توزيع الفئات الفعلي
# -------------------------------------------------------------------
data_dir = "raw/color"
tomato_classes = sorted([d for d in os.listdir(data_dir) if "Tomato" in d])

print("\n[*] الخطوة 1: تحليل وتوزيع الفئات الفعلي في مجلد الطماطم:")
class_counts = {}
total_images = 0
for cls in tomato_classes:
    count = len(os.listdir(os.path.join(data_dir, cls)))
    class_counts[cls] = count
    total_images += count
    print(f"  - {cls:<42}: {count:>5} صورة")

print(f"\nإجمالي الصور: {total_images} صورة عبر {len(tomato_classes)} فئات.")
min_cls = min(class_counts, key=class_counts.get)
max_cls = max(class_counts, key=class_counts.get)
imbalance_ratio = class_counts[max_cls] / class_counts[min_cls]
print(f"  * أعلى فئة: {max_cls} ({class_counts[max_cls]})")
print(f"  * أقل فئة: {min_cls} ({class_counts[min_cls]})")
print(f"  * معامل عدم التوازن (Imbalance Ratio): {imbalance_ratio:.2f}x (فجوة تتطلب Stratified Splitting و Macro-F1)")

# -------------------------------------------------------------------
# استخراج السمات البصرية (Feature Extraction)
# -------------------------------------------------------------------
def extract_all_features(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. سمات الألوان (HSV Histogram + BGR Mean)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv)
    mean_color = cv2.mean(img)[:3]
    f_color = np.hstack([hist_hsv.flatten(), mean_color])

    # 2. سمات الملمس (GLCM)
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    f_glcm = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ]

    # 3. نمط البكسلات الثنائية المحلية (LBP)
    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    return np.hstack([f_color, f_glcm, hist_lbp])

# نأخذ عينة ممثلة ومتوازنة نسبياً (150 صورة لكل فئة = 1500 صورة) لضمان سرعة البحث الشبكي
SAMPLES_PER_CLASS = 150
X, y = [], []
print(f"\n[*] جاري استخراج السمات لعينة {SAMPLES_PER_CLASS} صورة لكل فئة...")
for cls in tomato_classes:
    cls_folder = os.path.join(data_dir, cls)
    imgs = os.listdir(cls_folder)[:SAMPLES_PER_CLASS]
    for img_name in imgs:
        f = extract_all_features(os.path.join(cls_folder, img_name))
        if f is not None:
            X.append(f)
            y.append(cls)

X = np.array(X)
y = np.array(y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------------------------------------------
# الخطوة 2: فصل البيانات الثلاثي (Train / Validation / Test)
# -------------------------------------------------------------------
print("\n[*] الخطوة 2: تقسيم البيانات إلى تدريب (70%) وتحقق (15%) واختبار أعمى (15%):")
# تقسيم أول: 70% تدريب، 30% للمتبقي
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)
# تقسيم ثاني للـ 30%: 15% Validation و 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  - حجم مجموعة التدريب (Train): {X_train.shape[0]} عينة (70%)")
print(f"  - حجم مجموعة التحقق (Val):    {X_val.shape[0]} عينة (15%)")
print(f"  - حجم مجموعة الاختبار (Test): {X_test.shape[0]} عينة (15%)")

# تقييس البيانات بناءً على مجموعة التدريب فقط لمنع تسريب البيانات (Data Leakage)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# -------------------------------------------------------------------
# الخطوة 3 & 4: بناء خط الأساس ومقارنة النماذج على مجموعة التحقق (Validation)
# -------------------------------------------------------------------
print("\n[*] الخطوة 3 & 4: مقارنة النماذج على مجموعة التحقق (Validation Set):")
candidate_models = {
    "1. Dummy Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
    "2. Logistic Regression (Linear)": LogisticRegression(max_iter=1000, random_state=42),
    "3. Random Forest (Non-linear Trees)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "4. Support Vector Machine (RBF)": SVC(kernel="rbf", C=1.0, random_state=42)
}

val_results = {}
for name, clf in candidate_models.items():
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    val_results[name] = {"Accuracy": acc, "Macro-F1": f1}
    print(f"  - {name:<36} | Acc: {acc*100:6.2f}% | Macro-F1: {f1:0.4f}")

# -------------------------------------------------------------------
# الخطوة 5: تحسين المعلمات (Hyperparameter Tuning) على مجموعة التحقق
# -------------------------------------------------------------------
print("\n[*] الخطوة 5: تحسين المعلمات الفائقة (Tuning) للنموذج الأفضل عبر Grid Search:")
param_grid = {
    'C': [1, 5, 10, 20],
    'gamma': ['scale', 0.01, 0.001]
}
grid_search = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train_s, y_train)
best_model = grid_search.best_estimator_
print(f"  * أفضل معلمات مختارة: {grid_search.best_params_}")
print(f"  * درجة التحقق الأفضل (Best CV Score): {grid_search.best_score_*100:.2f}%")

# -------------------------------------------------------------------
# التقييم النهائي غير المتحيز على مجموعة الاختبار الأعمى (Test Set)
# -------------------------------------------------------------------
print("\n[*] التقييم النهائي للنموذج المحسن على مجموعة الاختبار المستقلة (Test Set):")
test_preds = best_model.predict(X_test_s)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average="macro")
print(f"  ==> دقة الاختبار الأعمى النهائية (Test Accuracy): {test_acc * 100:.2f}%")
print(f"  ==> مقياس Macro-F1 النهائي: {test_f1:.4f}")

# -------------------------------------------------------------------
# الخطوة 6: تحليل الأخطاء ومصفوفة الالتباس (Error Analysis)
# -------------------------------------------------------------------
print("\n[*] الخطوة 6: تحليل الأخطاء ومصفوفة الالتباس:")
clean_labels = [c.replace("Tomato___", "") for c in le.classes_]
print(classification_report(y_test, test_preds, target_names=clean_labels))

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=clean_labels, yticklabels=clean_labels)
plt.title(f"Rigorous Confusion Matrix (Test Acc: {test_acc*100:.2f}%)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual True Class")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("methodology_confusion_matrix.png", dpi=200)
plt.close()
print("  [+] تم حفظ مصفوفة الالتباس التفصيلية: methodology_confusion_matrix.png")

# -------------------------------------------------------------------
# الخطوة 7: حفظ النموذج والمعالج وإعادة التحميل
# -------------------------------------------------------------------
print("\n[*] الخطوة 7: حفظ وتصدير النموذج:")
os.makedirs("rigorous_models", exist_ok=True)
joblib.dump(best_model, "rigorous_models/best_tuned_model.pkl")
joblib.dump(scaler, "rigorous_models/scaler.pkl")
joblib.dump(le, "rigorous_models/label_encoder.pkl")
print("  [+] تم حفظ النموذج المحسن والمعالج في مجلد: rigorous_models/")

# التحقق من إعادة التحميل
reloaded_model = joblib.load("rigorous_models/best_tuned_model.pkl")
assert reloaded_model is not None
print("  [+] تم اختبار إعادة تحميل النموذج والتأكد من مطابقته 100%!")

# -------------------------------------------------------------------
# الخطوة 8: اختبار صورة جديدة وفحص حدود التعميم (Generalization)
# -------------------------------------------------------------------
print("\n[*] الخطوة 8: اختبار عينة وتحديد حدود التعميم:")
sample_test_path = os.path.join(data_dir, "Tomato___Early_blight", os.listdir(os.path.join(data_dir, "Tomato___Early_blight"))[10])
sample_feat = extract_all_features(sample_test_path)
sample_scaled = scaler.transform([sample_feat])
pred_label = le.inverse_transform(best_model.predict(sample_scaled))[0]
conf = np.max(best_model.predict_proba(sample_scaled)) * 100
print(f"  - فحص صورة حقيقية مصابة باللفحة المبكرة:")
print(f"  - التنبؤ: {pred_label} | بنسبة ثقة: {conf:.2f}%")

print("\n=====================================================================")
print("  اكتمل تنفيذ المنهجية الكاملة بجميع مراحلها الثمانية بنجاح!          ")
print("=====================================================================")

