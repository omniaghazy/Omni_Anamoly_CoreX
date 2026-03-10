import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# --- Settings ---
dataset = 'RobotArm'
result_path = 'result/test_score.pkl'
label_path = f'processed/{dataset}_test_label.pkl'

def plot_omnia_results():
    if not os.path.exists(result_path) or not os.path.exists(label_path):
        print(f"❌ Error: Files not found!")
        return

    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    with open(result_path, 'rb') as f:
        scores = pickle.load(f)

    labels = np.array(labels).flatten()
    scores = np.array(scores).flatten()

    # Sync lengths
    min_len = min(len(labels), len(scores))
    labels = labels[-min_len:]
    scores = scores[-min_len:]
    
    # حساب الـ Threshold والـ Predictions
    # ملحوظة: في OmniAnomaly الحقيقي بنستخدم الـ POT، بس هنا بنعمل تقريب للمعاينة
    threshold = np.mean(scores) + 3 * np.std(scores)
    predictions = (scores > threshold).astype(int)
    
    # حساب الـ Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    print(f"--- [coreX Metrics] Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} ---")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # الرسمة الأولى: السكور مع الـ Ground Truth
    ax1.plot(scores, color='royalblue', label='Anomaly Score', linewidth=0.8)
    ax1.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold (Mean+3std)')
    ax1.fill_between(range(min_len), 0, np.max(scores), where=(labels > 0), 
                    color='crimson', alpha=0.3, label='Actual Anomaly (Ground Truth)')
    ax1.set_title(f'OmniAnomaly Performance on {dataset}')
    ax1.legend(loc='upper right')

    # الرسمة التانية: مقارنة الـ Ground Truth بالـ Prediction (RCA Analysis)
    # هنا هنشوف الموديل غلط في إيه بالظبط
    ax2.step(range(min_len), labels, color='crimson', label='Ground Truth', where='post', alpha=0.7)
    ax2.step(range(min_len), predictions, color='forestgreen', label='Model Prediction', where='post', alpha=0.7)
    ax2.set_title('Ground Truth vs Model Prediction')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Anomaly'])
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{dataset}_final_report.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_omnia_results()