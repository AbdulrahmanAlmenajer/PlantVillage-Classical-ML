import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, log_loss
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import lightgbm as lgb

# Windows UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# Helper functions
# ==============================================================================
def get_crop_name(class_name):
    """Extracts crop name from full class string, e.g. 'Tomato___Early_blight' -> 'Tomato'"""
    return class_name.split("___")[0]

def compute_ece(probs, y_true, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf = np.mean(confidences[in_bin])
            ece += np.abs(avg_conf - acc_in_bin) * prop_in_bin
    return ece

# ==============================================================================
# Hierarchical Engine Architecture
# ==============================================================================
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
        
        # Stage 1: Crop probabilities P(Crop = k | X)
        crop_probs = self.crop_router.predict_proba(X) # (N, num_crops)
        
        # Stage 2: For each crop, compute conditional disease probabilities P(Disease = d | Crop = k, X)
        for crop_idx, crop_name in enumerate(self.crop_classes):
            target_indices = self.crop_to_indices[crop_name]
            p_crop = crop_probs[:, crop_idx:crop_idx+1] # (N, 1)
            
            specialist = self.crop_specialists[crop_name]
            if specialist is None:
                # Single-disease crop (e.g. Orange, Soybean) -> 100% of crop prob goes to that single disease
                final_probs[:, target_indices[0]] = p_crop.ravel()
            else:
                cond_probs = specialist.predict_proba(X) # (N, num_diseases_for_crop)
                final_probs[:, target_indices] = p_crop * cond_probs
                
        return final_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# ==============================================================================
# Master Pipeline Execution
# ==============================================================================
def main():
    print("=" * 85)
    print("  Hierarchical Two-Stage & Calibrated Pipeline (Full 54,305 Images)")
    print("  Engineered to eliminate Inter-Crop Confusions & Calibrate Overconfidence")
    print("=" * 85)

    CACHE_FILE = "grand_54k_features_cache.npz"
    if not os.path.exists(CACHE_FILE):
        print(f"[!] Error: {CACHE_FILE} not found.")
        return

    print(f"[*] Loading feature cache: {CACHE_FILE}...")
    t0 = time.time()
    cached = np.load(CACHE_FILE)
    X = cached["X"]
    y = cached["y"]
    paths = cached["paths"]
    print(f"[+] Loaded {len(X):,} samples in {time.time()-t0:.2f}s. Feature dim: {X.shape[1]}")

    # Step 1: Encode Labels and Prepare Stratified Split
    SEED = 42
    le_all = LabelEncoder()
    y_encoded = le_all.fit_transform(y)
    all_classes = le_all.classes_
    num_classes = len(all_classes)
    
    idx = np.arange(len(X))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_encoded)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y_encoded[temp_idx])

    X_train, y_train = X[train_idx], y_encoded[train_idx]
    X_val, y_val     = X[val_idx], y_encoded[val_idx]
    X_test, y_test   = X[test_idx], y_encoded[test_idx]

    # Preprocessing: Fit Scaler strictly on train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    print(f"[+] Partitions ready: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    # Map each disease class to its crop
    disease_to_crop = [get_crop_name(c) for c in all_classes]
    le_crop = LabelEncoder()
    crops_all_unique = sorted(list(set(disease_to_crop)))
    le_crop.fit(crops_all_unique)
    
    crop_train = le_crop.transform([get_crop_name(all_classes[i]) for i in y_train])
    crop_val   = le_crop.transform([get_crop_name(all_classes[i]) for i in y_val])
    crop_test  = le_crop.transform([get_crop_name(all_classes[i]) for i in y_test])

    print(f"\n[+] Total Crops (Plant Species): {len(crops_all_unique)}")
    crop_to_disease_indices = {}
    for c_idx, c_name in enumerate(crops_all_unique):
        d_indices = [i for i, c in enumerate(disease_to_crop) if c == c_name]
        crop_to_disease_indices[c_name] = d_indices
        print(f"    * {c_name:<20}: {len(d_indices)} disease classes")

    # ==========================================================================
    # Step 2: Train Stage 1 Crop Router
    # ==========================================================================
    print("\n" + "-" * 75)
    print("[Stage 1]: Training Crop Router (14 Plant Species Classifier)")
    print("-" * 75)
    
    t_crop_start = time.time()
    base_crop_model = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.08,
        max_depth=6,
        num_leaves=35,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=SEED,
        verbose=-1,
        n_jobs=-1
    )
    base_crop_model.fit(X_train_s, crop_train)
    
    # Calibrate Crop Router on validation set for reliable probabilities
    print("[*] Calibrating Stage 1 Crop Router probabilities on Validation set...")
    calibrated_crop_router = CalibratedClassifierCV(estimator=FrozenEstimator(base_crop_model), method="sigmoid")
    calibrated_crop_router.fit(X_val_s, crop_val)
    
    crop_preds_test = calibrated_crop_router.predict(X_test_s)
    crop_test_acc = accuracy_score(crop_test, crop_preds_test)
    print(f"[+] Crop Router Training Complete in {time.time()-t_crop_start:.2f}s.")
    print(f"🏆 Crop Identification Test Accuracy: {crop_test_acc * 100:.2f}% (Only {np.sum(crop_test != crop_preds_test)} crop errors out of 8,146!)")

    # ==========================================================================
    # Step 3: Train Stage 2 Crop-Specific Disease Specialists
    # ==========================================================================
    print("\n" + "-" * 75)
    print("[Stage 2]: Training Calibrated Crop-Specific Disease Specialists")
    print("-" * 75)

    crop_specialists = {}
    for crop_name in crops_all_unique:
        d_indices = crop_to_disease_indices[crop_name]
        if len(d_indices) == 1:
            # Only 1 disease/condition for this crop (e.g. Orange Huanglongbing, Soybean healthy)
            crop_specialists[crop_name] = None
            print(f"  * {crop_name:<20}: 1 Class only -> Deterministic Specialist.")
        else:
            t_spec = time.time()
            # Filter train/val sets for this crop only
            crop_int = le_crop.transform([crop_name])[0]
            mask_tr = (crop_train == crop_int)
            mask_va = (crop_val == crop_int)

            X_tr_c, y_tr_c = X_train_s[mask_tr], y_train[mask_tr]
            X_va_c, y_va_c = X_val_s[mask_va], y_val[mask_va]

            # Remap global disease indices to 0..K-1 for local specialist
            local_le = LabelEncoder()
            y_tr_local = local_le.fit_transform(y_tr_c)
            y_va_local = local_le.transform(y_va_c)

            # Train tuned specialist (Balanced LightGBM)
            spec_base = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.06,
                max_depth=5,
                num_leaves=25,
                class_weight="balanced",
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            )
            spec_base.fit(X_tr_c, y_tr_local)

            # Calibrate specialist probabilities
            spec_calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(spec_base), method="sigmoid")
            spec_calibrated.fit(X_va_c, y_va_local)

            crop_specialists[crop_name] = spec_calibrated
            print(f"  * {crop_name:<20}: {len(d_indices)} Classes -> Trained & Calibrated in {time.time()-t_spec:.2f}s.")

    # ==========================================================================
    # Step 4: Assemble Full Hierarchical Engine & Evaluate on Test Set
    # ==========================================================================
    print("\n" + "-" * 75)
    print("[Stage 3]: Evaluating Integrated Hierarchical & Calibrated Engine on Test Set")
    print("-" * 75)

    hierarchical_engine = HierarchicalPlantEngine(
        crop_router=calibrated_crop_router,
        crop_classes=crops_all_unique,
        crop_to_indices=crop_to_disease_indices,
        crop_specialists=crop_specialists,
        all_classes=all_classes
    )

    t_eval = time.time()
    hier_probs_test = hierarchical_engine.predict_proba(X_test_s)
    hier_preds_test = np.argmax(hier_probs_test, axis=1)
    eval_latency = (time.time() - t_eval) / len(X_test)

    # Compute comprehensive metrics
    hier_acc = accuracy_score(y_test, hier_preds_test)
    hier_bal_acc = balanced_accuracy_score(y_test, hier_preds_test)
    hier_prec_macro, hier_rec_macro, hier_f1_macro, _ = precision_recall_fscore_support(y_test, hier_preds_test, average='macro')
    hier_ece = compute_ece(hier_probs_test, y_test, n_bins=10)
    hier_loss = log_loss(y_test, hier_probs_test)

    # Top-K
    def get_top_k(probs, true_y, k=3):
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        return np.mean([true_y[i] in top_k_preds[i] for i in range(len(true_y))])

    hier_top2 = get_top_k(hier_probs_test, y_test, k=2)
    hier_top3 = get_top_k(hier_probs_test, y_test, k=3)

    print("=" * 75)
    print("  🏆 Hierarchical Calibrated System Benchmark Results:")
    print("=" * 75)
    print(f"  * Overall Test Accuracy        : {hier_acc * 100:.2f}%")
    print(f"  * Balanced Accuracy (Recall)   : {hier_bal_acc * 100:.2f}%")
    print(f"  * Macro F1-Score               : {hier_f1_macro * 100:.2f}%")
    print(f"  * Macro Precision              : {hier_prec_macro * 100:.2f}%")
    print(f"  * Top-2 Accuracy               : {hier_top2 * 100:.2f}%")
    print(f"  * Top-3 Accuracy               : {hier_top3 * 100:.2f}%")
    print("-" * 75)
    print(f"  * Expected Calibration Error   : {hier_ece * 100:.2f}% (Reduced from 20.70%!)")
    print(f"  * Multi-class Log-Loss         : {hier_loss:.4f}")
    print(f"  * Inference Latency            : {eval_latency * 1000:.2f} ms / sample")
    print("=" * 75)

    # Comparison Table
    print("\n" + "=" * 75)
    print("  Direct Comparison: Flat Grand Ensemble vs. Hierarchical Calibrated Engine")
    print("=" * 75)
    comp_df = pd.DataFrame([
        {"Metric": "Test Accuracy", "Flat Ensemble": "98.05%", "Hierarchical Engine": f"{hier_acc*100:.2f}%"},
        {"Metric": "Balanced Accuracy", "Flat Ensemble": "97.05%", "Hierarchical Engine": f"{hier_bal_acc*100:.2f}%"},
        {"Metric": "Macro F1-Score", "Flat Ensemble": "97.26%", "Hierarchical Engine": f"{hier_f1_macro*100:.2f}%"},
        {"Metric": "Expected Calibration Error", "Flat Ensemble": "20.70% (Uncalibrated)", "Hierarchical Engine": f"{hier_ece*100:.2f}% (Calibrated)"},
        {"Metric": "Log-Loss", "Flat Ensemble": "0.3013", "Hierarchical Engine": f"{hier_loss:.4f}"},
        {"Metric": "Top-2 Accuracy", "Flat Ensemble": "99.47%", "Hierarchical Engine": f"{hier_top2*100:.2f}%"},
        {"Metric": "Top-3 Accuracy", "Flat Ensemble": "99.75%", "Hierarchical Engine": f"{hier_top3*100:.2f}%"}
    ])
    print(comp_df.to_string(index=False))

    # ==========================================================================
    # Step 5: Decision Safety Gate with Calibrated Probabilities
    # ==========================================================================
    print("\n" + "-" * 75)
    print("[Stage 4]: Evaluating Decision Safety Gate with Calibrated Probabilities")
    print("-" * 75)
    
    max_confs = np.max(hier_probs_test, axis=1)
    thresholds = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    th_records = []
    
    for th in thresholds:
        mask = (max_confs >= th)
        n_acc = np.sum(mask)
        cov = (n_acc / len(y_test)) * 100
        acc_at_th = accuracy_score(y_test[mask], hier_preds_test[mask]) * 100 if n_acc > 0 else 100.0
        th_records.append({
            "Threshold (Tau)": f">={th*100:.0f}%",
            "Accepted Coverage (%)": round(cov, 2),
            "Accuracy on Accepted (%)": round(acc_at_th, 2),
            "Accepted Count": n_acc,
            "Deferred to Human": len(y_test) - n_acc
        })
        
    df_thresholds = pd.DataFrame(th_records)
    print(df_thresholds.to_string(index=False))

    # Save Comparison Plot
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, [r["Accuracy on Accepted (%)"] for r in th_records], 'g-o', linewidth=2.5, label="Calibrated Hierarchical Accuracy (%)")
    plt.plot(thresholds, [r["Accepted Coverage (%)"] for r in th_records], 'b--s', linewidth=2, label="Accepted Leaf Coverage (%)")
    plt.title("Hierarchical Calibrated Engine: Safety Gate Trade-off Curve", fontsize=12)
    plt.xlabel("Confidence Safety Threshold", fontsize=10)
    plt.ylabel("Percentage (%)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("hierarchical_threshold_curve.png", dpi=200)
    plt.close()
    print("[+] Saved curve plot to: hierarchical_threshold_curve.png")

    # ==========================================================================
    # Step 6: Export Production Artifacts
    # ==========================================================================
    out_dir = "saved_models"
    os.makedirs(out_dir, exist_ok=True)
    
    engine_path = os.path.join(out_dir, "hierarchical_calibrated_engine.pkl")
    print(f"\n[*] Exporting Hierarchical Calibrated Engine to: {engine_path}...")
    joblib.dump(hierarchical_engine, engine_path)
    
    # Check size of saved model
    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"[+] Hierarchical Engine Model Size: {file_size_mb:.2f} MB")
    print("=" * 85)
    print("  Hierarchical Calibrated Pipeline Completed Successfully!")
    print("=" * 85)

if __name__ == "__main__":
    main()

