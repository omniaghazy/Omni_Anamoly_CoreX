# -*- coding: utf-8 -*-
import numpy as np
from omni_anomaly.spot import SPOT


def calc_point2point(predict, actual):
    """
    Strict point-to-point F1, Precision, Recall, Accuracy + confusion matrix.
    """
    predict = np.asarray(predict, dtype=np.int32)
    actual  = np.asarray(actual,  dtype=np.int32)

    TP = int(np.sum((predict == 1) & (actual == 1)))
    TN = int(np.sum((predict == 0) & (actual == 0)))
    FP = int(np.sum((predict == 1) & (actual == 0)))
    FN = int(np.sum((predict == 0) & (actual == 1)))

    eps = np.finfo(float).eps
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (TP + TN) / (len(actual) + eps)

    return f1, precision, recall, accuracy, TP, TN, FP, FN


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Segment-level adjustment: if the model detects ANY point inside an anomaly
    segment, the entire segment is credited as detected.

    NOTE on score direction:
        OmniAnomaly scores are *sum of log-probs* — higher = more normal.
        Therefore anomaly  ⟺  score < threshold  (score is BELOW the threshold).
    """
    score = np.asarray(score).flatten()
    label = np.asarray(label).flatten()

    # ── [FIX] Align lengths (take the tail of the longer array) ──────────────
    if len(score) != len(label):
        min_len = min(len(score), len(label))
        label = label[-min_len:]
        score = score[-min_len:]
        if pred is not None:
            pred = np.asarray(pred).flatten()[-min_len:]

    # Binary predictions
    if pred is None:
        # Anomaly = score falls BELOW threshold (log-prob convention)
        predict = (score < threshold).astype(np.int32)
    else:
        predict = np.asarray(pred).flatten().astype(np.int32)

    actual = (label > 0.1).astype(np.int32)

    # Segment boundaries
    change = np.diff(np.concatenate([[0], actual, [0]]))
    starts = np.where(change ==  1)[0]
    ends   = np.where(change == -1)[0]

    latency = 0
    anomaly_count = len(starts)

    for start, end in zip(starts, ends):
        if np.any(predict[start:end]):
            first_detection = int(np.where(predict[start:end] == 1)[0][0])
            latency += first_detection
            predict[start:end] = 1   # credit the whole segment

    if calc_latency:
        avg_latency = latency / (anomaly_count + 1e-6)
        return predict.astype(bool), avg_latency

    return predict.astype(bool)


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Compute adjusted metrics for a sequence of scores.
    Returns a list: [f1, precision, recall, accuracy, TP, TN, FP, FN, (latency)]
    """
    label = np.asarray(label).flatten()

    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=False)
        latency = None

    metrics = list(calc_point2point(predict, label))

    if calc_latency:
        metrics.append(latency)

    return metrics


def bf_search(score, label, start, end=None, step_num=500, display_freq=50, verbose=True):
    """
    Brute-force threshold search to maximise adjusted F1-Score.

    Using more steps (500 default) gives finer granularity for log-prob scores.
    The search correctly tests BOTH directions by scanning from very negative
    (only extreme outliers flagged) to less negative (more flagged).
    """
    score = np.asarray(score).flatten()
    label = np.asarray(label).flatten()

    if end is None:
        end = start
        step_num = 1

    thresholds = np.linspace(start, end, step_num)

    best_metrics  = (-1.0, -1.0, -1.0, -1.0)
    best_threshold = 0.0
    best_latency   = 0.0

    if verbose:
        print(f"\n🔍 [BF-Search] Range: {start:.2f} → {end:.2f} | Steps: {step_num}")
        print(f"   Score range in data: [{score.min():.2f}, {score.max():.2f}]")

    for i, thr in enumerate(thresholds):
        res = calc_seq(score, label, thr, calc_latency=True)
        current_f1 = res[0]

        if current_f1 > best_metrics[0]:
            best_metrics   = tuple(res[:4])
            best_threshold = thr
            best_latency   = res[-1] if len(res) > 8 else 0.0

        if verbose and i % display_freq == 0:
            print(f"   Step {i:4d}/{step_num} | Thr={thr:10.2f} | F1={current_f1:.4f} | BestF1={best_metrics[0]:.4f}")

    if verbose:
        print("\n" + "=" * 50)
        print("🏆 BF-SEARCH FINAL RESULTS")
        print(f"   Best Threshold : {best_threshold:.4f}")
        print(f"   F1-Score       : {best_metrics[0]:.4f}")
        print(f"   Precision      : {best_metrics[1]:.4f}")
        print(f"   Recall         : {best_metrics[2]:.4f}")
        print(f"   Accuracy       : {best_metrics[3]:.4f}")
        print(f"   Avg Latency    : {best_latency:.2f} pts")
        print("=" * 50)

    return best_metrics, best_threshold


def pot_eval(init_score, score, label, q=1e-3, level=0.02, dynamic=False):
    """
    Automatic threshold via Peaks-Over-Threshold (SPOT / Extreme Value Theory).
    Works on log-prob scores: low values = anomalies, so we use min_extrema=True.
    """
    init_score = np.asarray(init_score).flatten()
    score      = np.asarray(score).flatten()
    label      = np.asarray(label).flatten()

    # ── [IMPROVEMENT] Negate scores so SPOT finds the lower-tail extremes ────
    # SPOT is designed for upper-tail (peaks). Negating log-probs turns
    # "very low log-prob = anomaly" into "very high value = anomaly", which
    # is exactly what SPOT expects.
    s = SPOT(q)
    s.fit(-init_score, -score)
    s.initialize(level=level, min_extrema=False)   # upper tail of negated
    ret = s.run(dynamic=dynamic)

    # Convert threshold back to original (negated back)
    pot_th = -np.mean(ret['thresholds'])

    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    metrics = calc_point2point(pred, label)

    print(f"\n📊 [POT] Alarms: {len(ret['alarms'])} | Threshold: {pot_th:.4f} | Latency: {p_latency:.2f}")

    return {
        'pot-f1':          metrics[0],
        'pot-precision':   metrics[1],
        'pot-recall':      metrics[2],
        'pot-accuracy':    metrics[3],
        'pot-TP':          metrics[4],
        'pot-TN':          metrics[5],
        'pot-FP':          metrics[6],
        'pot-FN':          metrics[7],
        'pot-threshold':   pot_th,
        'pot-latency':     p_latency,
        'pot-alarms':      len(ret['alarms'])
    }
