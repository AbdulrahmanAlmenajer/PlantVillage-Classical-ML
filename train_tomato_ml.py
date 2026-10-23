import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ========================================================
# 1. دوال استخراج السمات البصرية (Feature Extraction)
# ========================================================

def extract_color_features(image, bins=(8, 8, 8)):
    """
    استخراج توزيع الألوان في فضاء HSV وفضاء LAB
    حساس جداً لاصفرار وبقع الأوراق ومراحل التلف
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv)
    
    # متوسط وانحراف الألوان في قنوات BGR
    mean_color = cv2.mean(image)[:3]
    
    return np.hstack([hist_hsv.flatten(), mean_color])

def extract_texture_glcm(gray_image):
    """
    استخراج ميزات الملمس والخشونة من مصفوفة التواجد الرمادي (GLCM)
    """
    glcm = graycomatrix(gray_image, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    return [contrast, dissimilarity, homogeneity, energy, correlation]

def extract_texture_lbp(gray_image, num_points=24, radius=3):
    """
    استخراج نمط البكسلات الثنائية المحلية (LBP) لتمييز سطح الورقة الناعم عن البقع المتقرحة
    """
    lbp = local_binary_pattern(gray_image, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_hu_moments(gray_image):
    """
    عزوم هو (Hu Moments) لوصف الشكل الهندسي وتوزيع الكتل في الصورة
    """
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    # تحويل لوغاريتمي لتفادي القيم المتناهية الصغر
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-10)
    return hu_moments

def extract_all_features(image_path, target_size=(128, 128)):
    """
    دمج كافة السمات في متجهة رقمية موحدة
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f_color = extract_color_features(img)
    f_glcm = extract_texture_glcm(gray)
    f_lbp = extract_texture_lbp(gray)
    f_hu = extract_hu_moments(gray)

    return np.hstack([f_color, f_glcm, f_lbp, f_hu])

# ========================================================
# 2. تحديد البيانات وفئات الطماطم
# ========================================================

def load_tomato_dataset(data_dir="raw/color", max_samples_per_class=300):
    print(f"[*] فحص مجلد البيانات: {data_dir}")
    
    # فئات الطماطم المتاحة
    all_tomato_classes = sorted([
        d for d in os.listdir(data_dir) 
        if "Tomato" in d and os.path.isdir(os.path.join(data_dir, d))
    ])
    
    print(f"[*] تم العثور على {len(all_tomato_classes)} فئات للطماطم:")
    for i, cls in enumerate(all_tomato_classes, 1):
        print(f"  {i}. {cls}")

    X = []
    y = []

    start_time = time.time()
    for label in all_tomato_classes:
        class_dir = os.path.join(data_dir, label)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        selected_files = image_files[:max_samples_per_class]
        
        print(f" -> معالجة {len(selected_files)} صورة من فئة [{label}]...")
        for img_file in selected_files:
            img_path = os.path.join(class_dir, img_file)
            feat = extract_all_features(img_path)
            if feat is not None:
                X.append(feat)
                y.append(label)

    elapsed = time.time() - start_time
    print(f"\n[+] اكتمل استخراج السمات في {elapsed:.1f} ثانية!")
    return np.array(X), np.array(y), all_tomato_classes

# ========================================================
# 3. التدريب والتقييم وحفظ النموذج
# ========================================================

def train_and_evaluate(X, y):
    print("\n" + "="*50)
    print("تجهيز البيانات والتدريب (Training & Evaluation)")
    print("="*50)
    print(f"حجم العينات الإجمالي: {X.shape[0]}")
    print(f"عدد الميزات المستخرجة لكل صورة: {X.shape[1]}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # تقسيم البيانات (80% تدريب، 20% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    # تقييس الميزات (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # تدريب SVM
    print("\n[1] تدريب نموذج Support Vector Machine (SVM RBF Kernel)...")
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f" => دقة SVM (Accuracy): {acc_svm * 100:.2f}%\n")

    # تدريب Random Forest
    print("[2] تدريب نموذج Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f" => دقة Random Forest: {acc_rf * 100:.2f}%\n")

    # اختيار النموذج الأفضل
    best_model, best_name, best_pred = (svm_model, "SVM", y_pred_svm) if acc_svm >= acc_rf else (rf_model, "Random Forest", y_pred_rf)
    print(f"[*] النموذج الأفضل هو: {best_name}")
    print("\nتقرير التصنيف المفصل:")
    # تبسيط أسماء الفئات للعرض
    clean_names = [cls.replace("Tomato___", "") for cls in le.classes_]
    print(classification_report(y_test, best_pred, target_names=clean_names))

    # رسم مصفوفة الارتباك (Confusion Matrix)
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clean_names, yticklabels=clean_names)
    plt.title(f"Confusion Matrix - {best_name} (Acc: {max(acc_svm, acc_rf)*100:.2f}%)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[+] تم حفظ مصفوفة الارتباك كصورة: {cm_path}")

    # حفظ النموذج والمعالج
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_model, "saved_models/tomato_best_model.pkl")
    joblib.dump(scaler, "saved_models/scaler.pkl")
    joblib.dump(le, "saved_models/label_encoder.pkl")
    print("[+] تم تصدير النموذج والـ Scaler بنجاح في مجلد: saved_models/")

    return best_model, scaler, le

# ========================================================
# 4. نقطة الانطلاق الرئيسية (Main)
# ========================================================

if __name__ == "__main__":
    data_path = "raw/color"
    if not os.path.exists(data_path):
        data_path = "color"
    
    # تشغيل التدريب على عينة كافية وسريعة (200 صورة لكل فئة = 2000 صورة إجمالاً)
    X, y, classes = load_tomato_dataset(data_dir=data_path, max_samples_per_class=200)
    train_and_evaluate(X, y)

