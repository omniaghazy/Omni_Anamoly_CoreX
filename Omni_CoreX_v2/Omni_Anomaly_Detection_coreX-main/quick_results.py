
import pickle
import numpy as np
import os

result_path = r'd:\Omni_Anomaly_Detection_coreX\results\RobotArm_coreX_v1\test_score.pkl'
label_path = r'd:\Omni_Anomaly_Detection_coreX\data\processed\RobotArm_test_label.pkl'

def get_results():
    if not os.path.exists(result_path):
        print(f"Result file not found: {result_path}")
        return
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return
        
    with open(result_path, 'rb') as f:
        scores = pickle.load(f)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
        
    scores = np.array(scores)
    if scores.ndim > 1:
        scores = np.sum(scores, axis=-1)
    
    labels = np.array(labels).flatten()
    
    # Align lengths
    min_len = min(len(labels), len(scores))
    labels = labels[-min_len:]
    scores = scores[-min_len:]
    
    threshold = np.mean(scores) + 3 * np.std(scores)
    predictions = (scores > threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"--- 100 Epoch Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Detected Anomalies: {int(np.sum(predictions))} / {int(np.sum(labels))} actual")

if __name__ == "__main__":
    get_results()
