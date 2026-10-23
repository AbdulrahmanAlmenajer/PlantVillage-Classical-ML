import os
import sys

# ضمان ترميز UTF-8 في ويندوز
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import json
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ========================================================
# تعريف فئات الـ Ensemble لإلغاء التسلسل (Unpickling)
# ========================================================
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

class SoftVotingEnsemble:
    def __init__(self, models, weights, classes):
        self.models = models
        self.weights = weights
        self.classes_ = classes
        
    def predict_proba(self, X):
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.models)) / sum(self.weights)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class HierarchicalPlantEngine:
    def __init__(self, crop_router, crop_classes, crop_to_indices, crop_specialists, all_classes):
        self.crop_router = crop_router
        self.crop_classes = list(crop_classes)
        self.crop_to_indices = crop_to_indices
        self.crop_specialists = crop_specialists
        self.classes_ = list(all_classes)
        self.n_classes = len(all_classes)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        final_probs = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        crop_probs = self.crop_router.predict_proba(X)
        for crop_idx, crop_name in enumerate(self.crop_classes):
            target_indices = self.crop_to_indices[crop_name]
            p_crop = crop_probs[:, crop_idx:crop_idx+1]
            specialist = self.crop_specialists[crop_name]
            if specialist is None:
                final_probs[:, target_indices[0]] = p_crop.ravel()
            else:
                cond_probs = specialist.predict_proba(X)
                final_probs[:, target_indices] = p_crop * cond_probs
        return final_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# ========================================================
# استخراج السمات المتقدمة المطابقة للنموذج الأكبر (554 ميزة)
# ========================================================
clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def extract_leaf_features(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. تسوية الإضاءة والظلال (CLAHE)
    gray_eq = clahe_filter.apply(gray)

    # 2. فضاء الألوان HSV (512 ميزة)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv)

    # 3. إحصائيات الألوان في HSV و LAB (10 ميزات)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    color_moments = [
        hsv[:,:,0].mean(), hsv[:,:,0].std(),
        hsv[:,:,1].mean(), hsv[:,:,1].std(),
        hsv[:,:,2].mean(), hsv[:,:,2].std(),
        lab[:,:,1].mean(), lab[:,:,1].std(),
        lab[:,:,2].mean(), lab[:,:,2].std()
    ]

    # 4. مصفوفة الملمس GLCM (64 مستوى)
    gray_64 = (gray_eq // 4).astype(np.uint8)
    glcm = graycomatrix(gray_64, distances=[1], angles=[0, np.pi/2], levels=64, symmetric=True, normed=True)
    f_glcm = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ]

    # 5. ملمس LBP المجهري (18 ميزة)
    lbp = local_binary_pattern(gray_eq, 16, 2, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 19), range=(0, 18))
    hist_lbp = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-7)

    # 6. عزوم الشكل الهندسية Hu Moments (7 ميزات)
    moments = cv2.moments(gray_eq)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(len(hu)):
        hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i]) + 1e-10)

    # 7. حدة حواف البقع (Laplacian Variance)
    lap_var = [cv2.Laplacian(gray_eq, cv2.CV_64F).var()]

    return np.hstack([hist_hsv.flatten(), color_moments, f_glcm, hist_lbp, hu, lap_var])

# ========================================================
# التنبؤ الذكي مع بوابة الأمان (Threshold Guard)
# ========================================================
def predict_leaf(image_path, model_type="grand_54k", threshold=0.50):
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at path: {image_path}")
        return None

    # تحديد النموذج المناسب
    if model_type in ["hierarchical", "hier", "fast"] and os.path.exists("saved_models/hierarchical_calibrated_engine.pkl"):
        model_path = "saved_models/hierarchical_calibrated_engine.pkl"
        scaler_path = "saved_models/grand_54k_scaler.pkl"
        labels_path = "saved_models/grand_54k_labels.json"
        engine_title = "Hierarchical Calibrated Engine (14 Crops | 38 Classes | Fast 17MB)"
    elif model_type in ["grand", "grand_54k", "all", "54k"] and os.path.exists("saved_models/grand_54k_ensemble_model.pkl"):
        model_path = "saved_models/grand_54k_ensemble_model.pkl"
        scaler_path = "saved_models/grand_54k_scaler.pkl"
        labels_path = "saved_models/grand_54k_labels.json"
        engine_title = "PlantVillage Grand Master Ensemble (Full 54,305 Images | 38 Classes)"
    elif os.path.exists("saved_models/hierarchical_calibrated_engine.pkl"):
        model_path = "saved_models/hierarchical_calibrated_engine.pkl"
        scaler_path = "saved_models/grand_54k_scaler.pkl"
        labels_path = "saved_models/grand_54k_labels.json"
        engine_title = "Hierarchical Calibrated Engine (14 Crops | 38 Classes | Fast 17MB)"
    elif os.path.exists("saved_models/ultimate_ensemble_model.pkl"):
        model_path = "saved_models/ultimate_ensemble_model.pkl"
        scaler_path = "saved_models/ultimate_scaler.pkl"
        labels_path = "saved_models/ultimate_labels.json"
        engine_title = "The Ultimate Balanced Ensemble (38 Classes)"
    elif os.path.exists("saved_models/ensemble_model.pkl"):
        model_path = "saved_models/ensemble_model.pkl"
        scaler_path = "saved_models/ensemble_scaler.pkl"
        labels_path = "saved_models/ensemble_labels.json"
        engine_title = "Tomato Soft-Voting Ensemble (10 Classes)"
    else:
        model_path = "saved_models/best_tomato_model.pkl"
        scaler_path = "saved_models/scaler.pkl"
        labels_path = "saved_models/tomato_labels.json"
        engine_title = "LightGBM Tomato Model"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)

    # استخراج وتقييس الميزات
    features = extract_leaf_features(image_path)
    if features is None:
        print("[!] Error: Could not read or process image.")
        return None

    # ضبط أبعاد الميزات بما يطابق الـ Scaler المحفوظ
    if features.shape[0] != scaler.mean_.shape[0]:
        # إذا كان هناك اختلاف طفيف في الميزات القديمة، نقوم بملاءمة الطول
        features = features[:scaler.mean_.shape[0]]

    features_scaled = scaler.transform([features])

    # التنبؤ وحساب الاحتماليات
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx] * 100
        top3_indices = np.argsort(probs)[::-1][:3]
    else:
        pred_idx = model.predict(features_scaled)[0]
        confidence = 100.0
        top3_indices = [pred_idx]
        probs = [1.0]

    raw_label = labels_dict[str(pred_idx)]
    
    if "___" in raw_label:
        crop_name, disease_name = raw_label.split("___", 1)
    else:
        crop_name, disease_name = raw_label, "Unknown"

    disease_name_clean = disease_name.replace("_", " ")
    is_healthy = "healthy" in disease_name.lower()

    # تطبيق عتبة الأمان (Threshold Safety Guard)
    passed_threshold = (confidence / 100.0) >= threshold

    print("\n" + "=" * 75)
    print(f"  Plant Health Diagnosis Report [{engine_title}]")
    print("=" * 75)
    print(f"[*] Image File    : {os.path.basename(image_path)}")
    print(f"[*] Crop / Plant  : {crop_name}")
    print(f"[*] Confidence    : {confidence:.2f}%")
    print(f"[*] Safety Thresh : {threshold*100:.0f}%")

    if passed_threshold:
        print(f"[*] Decision Gate : [PASSED] - High Confidence Diagnosis")
        if is_healthy:
            print(f"[*] Health Status : [HEALTHY] - Plant leaf shows no disease symptoms.")
        else:
            print(f"[*] Health Status : [INFECTED / DISEASED]")
            print(f"[*] Diagnosis     : {disease_name_clean}")
    else:
        print(f"[*] Decision Gate : [REJECTED / UNCERTAIN] - Confidence below {threshold*100:.0f}% threshold!")
        print(f"[*] Health Status : [UNCERTAIN ADVISORY]")
        print(f"[*] Most Likely   : {disease_name_clean} (Caution: low confidence)")
        print(f"[*] Advisory Note : Disease symptoms are ambiguous (early stage or poor lighting).")
        print(f"                    Please re-take photo with clearer focus and uniform lighting")
        print(f"                    or consult an agricultural extension specialist before spraying.")

    print("\n[+] Top 3 Predicted Probabilities:")
    for rank, idx_c in enumerate(top3_indices, 1):
        c_label = labels_dict[str(idx_c)].replace("___", " -> ")
        print(f"    {rank}. {c_label:<48}: {probs[idx_c]*100:6.2f}%")

    print("=" * 75 + "\n")
    return raw_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Diagnosis using Grand Master ML Pipeline")
    parser.add_argument("image_path", nargs="?", default="images.jfif", help="Path to input leaf image")
    parser.add_argument("--model", choices=["hierarchical", "fast", "hier", "grand", "grand_54k", "all", "ultimate", "ensemble", "tomato"], default="hierarchical", help="Model type to use")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold (0.0 to 1.0)")
    args = parser.parse_args()

    target_img = args.image_path
    if not os.path.exists(target_img):
        sample_folder = "raw/color/Tomato___healthy"
        if os.path.exists(sample_folder) and os.listdir(sample_folder):
            target_img = os.path.join(sample_folder, os.listdir(sample_folder)[0])

    predict_leaf(target_img, model_type=args.model, threshold=args.threshold)
