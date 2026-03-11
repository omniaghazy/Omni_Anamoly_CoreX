import os
import re

target_dir = r"X:\Omnia_CoreX\Omni_CoreX_v2\Omni_Anomaly_Detection_coreX-main\omni_anomaly"

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return
        
    original = content
    # Standardize TF1 core imports
    content = re.sub(r'import tensorflow as tf', 'import tensorflow.compat.v1 as tf', content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Standardized TF imports in {filepath}")

for root, _, files in os.walk(target_dir):
    for name in files:
        if name.endswith('.py'):
            process_file(os.path.join(root, name))
