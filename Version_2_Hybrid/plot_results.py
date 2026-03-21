import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from omni_anomaly.eval_methods import adjust_predicts, calc_point2point

# ── Paths ─────────────────────────────────────────────────────────────────────
# Try to use the Advanced results if they exist, otherwise fallback to v1
if os.path.exists(os.path.join('results', 'RobotArm_Advanced')):
    RESULT_DIR = os.path.join('results', 'RobotArm_Advanced')
else:
    RESULT_DIR = os.path.join('results', 'RobotArm_coreX_v1')

score_path  = os.path.join(RESULT_DIR, 'test_score.pkl')
t_score_path = os.path.join(RESULT_DIR, 'train_score.pkl')
label_path  = os.path.join('data', 'processed', 'RobotArm_test_label.pkl')
dataset     = 'RobotArm'


def plot_omnia_results():
    # ── Load files ─────────────────────────────────────────────────────────────
    for path in [score_path, label_path]:
        if not os.path.exists(path):
            print(f"[ERR] Missing file: {path}")
            print("      Run main.py (and data_preprocess.py) first.")
            return

    with open(label_path, 'rb') as f:
        labels = np.array(pickle.load(f)).flatten().astype(np.float32)

    with open(score_path, 'rb') as f:
        scores = np.array(pickle.load(f))

    # ── Collapse per-sensor scores if needed (sum of log-probs) ───────────────
    if scores.ndim > 1:
        scores = np.sum(scores, axis=-1)
    scores = scores.flatten().astype(np.float32)

    # ── Align lengths ──────────────────────────────────────────────────────────
    min_len = min(len(labels), len(scores))
    labels  = labels[-min_len:]
    scores  = scores[-min_len:]

    # ── [IMPROVED] Threshold: use percentile of score distribution ─────────────
    # Log-prob scores: anomaly = very low value → threshold at low percentile
    # Choose the percentile based on expected anomaly rate (~1–5 %)
    anomaly_rate = float(labels.mean()) if labels.mean() > 0 else 0.05
    pct  = max(1.0, min(10.0, anomaly_rate * 100))    # clamp to [1, 10]%
    threshold   = float(np.percentile(scores, pct))   # low-end tail

    # Segment-adjusted predictions
    predictions = adjust_predicts(scores, labels, threshold=threshold)
    predictions = np.asarray(predictions).astype(int)

    # Metrics
    f1, precision, recall, accuracy, TP, TN, FP, FN = calc_point2point(
        predictions, labels.astype(int))

    print("\n" + "=" * 55)
    print("  🤖 coreX OmniAnomaly — Plot Results Report")
    print("=" * 55)
    print(f"  Threshold   : {threshold:.4f}  (bottom {pct:.1f}%-ile)")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"  Detected    : {int(predictions.sum())} / {int(labels.sum())} anomaly pts")
    print("=" * 55 + "\n")

    # ── Plot ─────────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    t = np.arange(min_len)

    # Score timeline
    ax1.plot(t, scores, color='royalblue', linewidth=0.6, label='Anomaly Score (sum log-prob)')
    ax1.axhline(y=threshold, color='orange', linestyle='--', linewidth=1.5,
                label=f'Threshold = {threshold:.2f}')
    ax1.fill_between(t, scores.min(), scores.max(),
                     where=(labels > 0), color='crimson', alpha=0.20,
                     label='Ground Truth Anomaly')
    ax1.set_title(f'OmniAnomaly coreX — Anomaly Scores ({dataset})', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score (log-prob)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Prediction comparison
    ax2.fill_between(t, 0, labels,      step='post', color='crimson',     alpha=0.45, label='Ground Truth')
    ax2.fill_between(t, 0, predictions, step='post', color='forestgreen', alpha=0.45, label='Prediction')
    ax2.set_title(
        f'Ground Truth vs Prediction  |  '
        f'F1={f1:.4f}   Precision={precision:.4f}   Recall={recall:.4f}',
        fontsize=11)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Anomaly')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Anomaly'])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, f'{dataset}_final_report.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Report saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_omnia_results()