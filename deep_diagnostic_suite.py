import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    log_loss, brier_score_loss, confusion_matrix, classification_report
)

# UTF-8 for Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Re-declare GrandVotingEnsemble for unpickling
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

def top_k_accuracy(probs, y_true, k=3):
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    match = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return np.mean(match)

def compute_ece(probs, y_true, n_bins=10):
    """Computes Expected Calibration Error (ECE)"""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_data.append({
                "bin": f"{bin_lower:.1f}-{bin_upper:.1f}",
                "confidence": avg_confidence_in_bin,
                "accuracy": accuracy_in_bin,
                "count": np.sum(in_bin)
            })
    return ece, bin_data

def main():
    print("=" * 80)
    print("  Deep Diagnostic Suite: Comprehensive Multi-Metric Audit (Full 54,305 Images)")
    print("=" * 80)

    # 1. Load Data & Partitions
    CACHE_FILE = "grand_54k_features_cache.npz"
    if not os.path.exists(CACHE_FILE):
        print(f"[!] Error: {CACHE_FILE} not found.")
        return

    print(f"[*] Loading feature cache: {CACHE_FILE}...")
    cached = np.load(CACHE_FILE)
    X = cached["X"]
    y = cached["y"]
    paths = cached["paths"]
    print(f"[+] Loaded {len(X):,} samples with {X.shape[1]} features.")

    # 2. Strict Partitioning (Exact replica of training split)
    SEED = 42
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    idx = np.arange(len(X))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_encoded)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y_encoded[temp_idx])

    X_test = X[test_idx]
    y_test = y_encoded[test_idx]
    paths_test = paths[test_idx]
    print(f"[+] Isolated Unseen Test Partition: {len(X_test):,} samples.")

    # 3. Load Production Scaler and Model
    SCALER_FILE = "saved_models/grand_54k_scaler.pkl"
    MODEL_FILE = "saved_models/grand_54k_ensemble_model.pkl"
    
    print(f"[*] Loading scaler: {SCALER_FILE}...")
    scaler = joblib.load(SCALER_FILE)
    X_test_s = scaler.transform(X_test)

    print(f"[*] Loading ensemble model: {MODEL_FILE}...")
    t0 = time.time()
    model = joblib.load(MODEL_FILE)
    print(f"[+] Model loaded in {time.time()-t0:.2f} seconds.")

    # 4. Generate Predictions & Probabilities
    print("\n[*] Generating posterior probabilities across 38 classes...")
    t0 = time.time()
    probs = model.predict_proba(X_test_s)
    preds = np.argmax(probs, axis=1)
    inference_time = (time.time() - t0) / len(X_test)
    print(f"[+] Prediction complete. Average Latency: {inference_time*1000:.2f} ms/sample.")

    # =========================================================================
    # 5. Global Multi-Metric Computation
    # =========================================================================
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
    
    top1 = acc
    top2 = top_k_accuracy(probs, y_test, k=2)
    top3 = top_k_accuracy(probs, y_test, k=3)
    top5 = top_k_accuracy(probs, y_test, k=5)
    
    ce_loss = log_loss(y_test, probs)
    ece, bin_data = compute_ece(probs, y_test, n_bins=10)

    # Multi-class Brier score
    y_onehot = np.eye(len(le.classes_))[y_test]
    brier = np.mean(np.sum((probs - y_onehot)**2, axis=1))

    print("\n" + "=" * 70)
    print("  [1] Global Performance Metrics Matrix:")
    print("=" * 70)
    print(f"  * Overall Accuracy         : {acc*100:6.2f}%")
    print(f"  * Balanced Accuracy (Recall): {bal_acc*100:6.2f}%")
    print(f"  * Macro Precision           : {prec_macro*100:6.2f}%")
    print(f"  * Macro Recall              : {rec_macro*100:6.2f}%")
    print(f"  * Macro F1-Score            : {f1_macro*100:6.2f}%")
    print(f"  * Weighted F1-Score         : {f1_weighted*100:6.2f}%")
    print("-" * 70)
    print(f"  * Top-1 Accuracy            : {top1*100:6.2f}%")
    print(f"  * Top-2 Accuracy            : {top2*100:6.2f}%")
    print(f"  * Top-3 Accuracy            : {top3*100:6.2f}%")
    print(f"  * Top-5 Accuracy            : {top5*100:6.2f}%")
    print("-" * 70)
    print(f"  * Multi-class Cross-Entropy : {ce_loss:0.4f}")
    print(f"  * Multi-class Brier Score   : {brier:0.4f}")
    print(f"  * Expected Calibration Error: {ece*100:0.2f}%")
    print("=" * 70)

    # =========================================================================
    # 6. Slice-Based / Per-Class Breakdown
    # =========================================================================
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(y_test, preds, average=None)
    
    df_classes = pd.DataFrame({
        "Class": le.classes_,
        "Support": s_per,
        "Precision": p_per * 100,
        "Recall": r_per * 100,
        "F1": f1_per * 100
    })
    
    # Worst 5 by Recall (False Negatives)
    worst_recall = df_classes.sort_values(by="Recall", ascending=True).head(5)
    print("\n" + "=" * 70)
    print("  [2] Worst 5 Classes by Recall (Under-Diagnosed / High False Negatives):")
    print("=" * 70)
    for _, row in worst_recall.iterrows():
        print(f"  - {row['Class']:<45} | Recall: {row['Recall']:5.2f}% | Prec: {row['Precision']:5.2f}% | N={row['Support']}")

    # Worst 5 by Precision (False Positives)
    worst_prec = df_classes.sort_values(by="Precision", ascending=True).head(5)
    print("\n" + "=" * 70)
    print("  [3] Worst 5 Classes by Precision (Over-Diagnosed / High False Positives):")
    print("=" * 70)
    for _, row in worst_prec.iterrows():
        print(f"  - {row['Class']:<45} | Prec: {row['Precision']:5.2f}% | Recall: {row['Recall']:5.2f}% | N={row['Support']}")

    # =========================================================================
    # 7. Crop-Level vs Disease-Level Error Breakdown
    # =========================================================================
    errors_mask = (y_test != preds)
    total_errors = np.sum(errors_mask)
    
    # Extract crop names
    def get_crop(cls_name):
        return cls_name.split("___")[0]
        
    crops_all = [get_crop(c) for c in le.classes_]
    true_crops = np.array([crops_all[i] for i in y_test])
    pred_crops = np.array([crops_all[i] for i in preds])
    
    intra_crop_errors = 0 # Correct crop, wrong disease
    inter_crop_errors = 0 # Wrong crop entirely
    
    for i in np.where(errors_mask)[0]:
        if true_crops[i] == pred_crops[i]:
            intra_crop_errors += 1
        else:
            inter_crop_errors += 1

    print("\n" + "=" * 70)
    print("  [4] Error Structural Distribution (Crop vs. Disease):")
    print("=" * 70)
    print(f"  * Total Errors in Test Set  : {total_errors:,} out of {len(y_test):,} ({total_errors/len(y_test)*100:.2f}%)")
    print(f"  * Intra-Crop Errors (Same Crop, Wrong Disease): {intra_crop_errors:,} ({intra_crop_errors/total_errors*100:.1f}%)")
    print(f"  * Inter-Crop Errors (Different Crop Confused) : {inter_crop_errors:,} ({inter_crop_errors/total_errors*100:.1f}%)")
    print("-" * 70)

    # =========================================================================
    # 8. Top 10 Most Confused Disease Pairs
    # =========================================================================
    confused_pairs = {}
    for i in np.where(errors_mask)[0]:
        true_name = le.classes_[y_test[i]]
        pred_name = le.classes_[preds[i]]
        pair = (true_name, pred_name)
        confused_pairs[pair] = confused_pairs.get(pair, 0) + 1

    sorted_confusions = sorted(confused_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n" + "=" * 70)
    print("  [5] Top 10 Most Confused Disease Pairs (True -> Predicted):")
    print("=" * 70)
    for (t, p), count in sorted_confusions:
        print(f"  [{count:2d} times] True: {t:<38} -> Pred: {p}")

    # =========================================================================
    # 9. Calibration Analysis & Diagnostics Plots
    # =========================================================================
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Reliability / Calibration Diagram
    plt.subplot(1, 2, 1)
    bin_confs = [b["confidence"] for b in bin_data]
    bin_accs = [b["accuracy"] for b in bin_data]
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.plot(bin_confs, bin_accs, 's-', color='crimson', linewidth=2, label=f"Model (ECE={ece*100:.2f}%)")
    plt.title("Reliability Diagram (Confidence vs. Accuracy)", fontsize=11)
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Fraction of Correct Diagnoses")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: Top-K Accuracy Growth
    plt.subplot(1, 2, 2)
    k_vals = [1, 2, 3, 5]
    k_accs = [top1*100, top2*100, top3*100, top5*100]
    bars = plt.bar([str(k) for k in k_vals], k_accs, color=['#2b5c8f', '#3581b8', '#56c2d6', '#2ecc71'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval - 3.5, f"{yval:.2f}%", ha='center', va='bottom', color='white', fontweight='bold')
    plt.title("Top-K Accuracy Progression", fontsize=11)
    plt.xlabel("Top-K Choices Included")
    plt.ylabel("Accuracy (%)")
    plt.ylim(90, 101)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("calibration_and_diagnostics.png", dpi=200)
    plt.close()
    print("\n[+] Saved diagnostic and calibration plots to: calibration_and_diagnostics.png")

    # 10. Save JSON Diagnostic Report
    report = {
        "global_metrics": {
            "test_accuracy": round(acc * 100, 2),
            "balanced_accuracy": round(bal_acc * 100, 2),
            "macro_precision": round(prec_macro * 100, 2),
            "macro_recall": round(rec_macro * 100, 2),
            "macro_f1": round(f1_macro * 100, 2),
            "weighted_f1": round(f1_weighted * 100, 2),
            "top1_accuracy": round(top1 * 100, 2),
            "top2_accuracy": round(top2 * 100, 2),
            "top3_accuracy": round(top3 * 100, 2),
            "top5_accuracy": round(top5 * 100, 2),
            "log_loss": round(ce_loss, 4),
            "brier_score": round(brier, 4),
            "ece_percentage": round(ece * 100, 2),
            "total_test_samples": len(y_test),
            "total_errors": int(total_errors),
            "intra_crop_errors": int(intra_crop_errors),
            "inter_crop_errors": int(inter_crop_errors)
        },
        "worst_recall_classes": worst_recall.to_dict(orient="records"),
        "worst_precision_classes": worst_prec.to_dict(orient="records"),
        "top_confusions": [{"true": t, "pred": p, "count": int(c)} for (t, p), c in sorted_confusions]
    }

    with open("deep_diagnostic_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[+] Saved complete JSON audit to: deep_diagnostic_report.json")
    print("=" * 80)

if __name__ == "__main__":
    main()

