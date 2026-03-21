# import numpy as np
# import pandas as pd
# import os

# --- 1. ورشة الأعطال (The Workshop) ---

# def inject_spike(data, sensor_idx, multiplier=2.5):
#     """عطل مفاجئ في لقطة واحدة"""
#     new_data = data.copy()
#     idx = np.random.randint(0, len(data))
#     new_data[idx, sensor_idx] *= multiplier
#     label = np.zeros(len(data))
#     label[idx] = 1
#     return new_data, label

# def inject_drift(data, sensor_idx, start_idx, duration, slope=0.02):
#     """عطل تدريجي (زحف)"""
#     new_data = data.copy()
#     label = np.zeros(len(data))
#     for i in range(duration):
#         if start_idx + i < len(data):
#             new_data[start_idx + i, sensor_idx] += (i * slope)
#             label[start_idx + i] = 1
#     return new_data, label

# def inject_noise_pattern(data, sensor_idx, start_idx, duration, level=0.5):
#     """عطل في شكل الموجة (رعشة)"""
#     new_data = data.copy()
#     label = np.zeros(len(data))
#     noise = np.random.normal(0, level, duration)
#     for i in range(duration):
#         if start_idx + i < len(data):
#             new_data[start_idx + i, sensor_idx] += noise[i]
#             label[start_idx + i] = 1
#     return new_data, label

# # --- 2. المايسترو (The Master Generator) ---

# def generate_test_data(normal_csv_path, output_dir="data/RobotArm"):
#     # قراءة الداتا النورمال اللي لقيناها
#     df = pd.read_csv(normal_csv_path)
#     data = df.values.astype(np.float32)
    
#     test_data = data.copy()
#     final_labels = np.zeros(len(data))
    
#     # هنحقن مثلاً 4 أعطال في أماكن مختلفة (ديناميك)
#     num_anomalies = 4
#     for _ in range(num_anomalies):
#         # اختيارات عشوائية
#         f_type = np.random.choice(['spike', 'drift', 'noise'])
#         s_idx = np.random.randint(0, data.shape[1])
#         start = np.random.randint(50, len(data) - 200)
        
#         if f_type == 'spike':
#             test_data, l = inject_spike(test_data, s_idx)
#         elif f_type == 'drift':
#             test_data, l = inject_drift(test_data, s_idx, start, duration=150)
#         else:
#             test_data, l = inject_noise_pattern(test_data, s_idx, start, duration=150)
            
#         final_labels = np.logical_or(final_labels, l).astype(int)

#     # حفظ الملفات في الفولدر الجديد
#     os.makedirs(output_dir, exist_ok=True)
#     pd.DataFrame(test_data).to_csv(f"{output_dir}/test.csv", index=False, header=None)
#     pd.DataFrame(final_labels).to_csv(f"{output_dir}/test_label.csv", index=False, header=None)
    
#     print(f"🚀 Success! Generated test data and labels in: {output_dir}")


import numpy as np
import pandas as pd
import os

# --- 1. ورشة الأعطال (The Workshop) ---

def inject_spike(data, sensor_idx, start_idx, multiplier=None):
    """
    نسخة coreX: بتحقن عطل مفاجئ (Spike) وبترجع الداتا المحقونة والـ Label بتاعها.
    التعديل: شلنا الـ .copy() عشان التعديلات تسمع فوق بعضها في الـ Test Set.
    """
    if multiplier is None:
        # بنختار رقم عشوائي كبير عشان الموديل يلاحظه (بين 5 و 10 أضعاف)
        multiplier = np.random.uniform(5.0, 10.0) 
        
    # بنعدل في الداتا اللي داخلة مباشرة عشان الـ Loop اللي بره تحس بالتغيير
    new_data = data 
    
    # لو القيمة الأصلية صفر أو قريبة جداً من الصفر، الضرب مش هينفع، فبنضيف قيمة ثابتة
    if abs(new_data[start_idx, sensor_idx]) < 1e-6:
        new_data[start_idx, sensor_idx] = multiplier 
    else:
        new_data[start_idx, sensor_idx] *= multiplier
        
    # بنعمل مصفوفة أصفار بنفس طول الداتا ونحط 1 في مكان العطل بس
    label = np.zeros(len(data), dtype=np.int32)
    label[start_idx] = 1
    
    return new_data, label


def inject_drift(data, sensor_idx, start_idx, duration=100):
    """
    نسخة coreX: عطل الزحف التدريجي (Drift).
    بيحسب الميل بناءً على الـ Standard Deviation عشان العطل يكون واقعي.
    """
    # التعديل: بنعدل في الداتا مباشرة عشان التغيير يبقى تراكمي
    new_data = data 
    label = np.zeros(len(data), dtype=np.int32)
    
    end_idx = min(start_idx + duration, len(data))
    actual_duration = end_idx - start_idx
    
    # حساب الميل (Slope) بناءً على انحراف المعياري للسنسور (أدق)
    # بنزود 1e-6 عشان لو السنسور ثابت تماماً (Std = 0) ميعملش Error
    std_val = np.std(data[:, sensor_idx]) + 1e-6
    slope = (std_val * 3) / actual_duration # زحف بمقدار 3 انحرافات معيارية
    
    # بنعمل مصفوفة فيها قيم بتزيد تدريجياً (Linear increase)
    drift_values = np.arange(actual_duration) * slope
    
    # بنضيف الزحف ده على قيم السنسور الأصلية
    new_data[start_idx:end_idx, sensor_idx] += drift_values
    
    # بنعلم على كل النقط اللي حصل فيها الزحف إنها عطل (1)
    label[start_idx:end_idx] = 1
    
    return new_data, label


def inject_stuck_at(data, sensor_idx, start_idx, duration=100):
    """
    نسخة coreX: عطل التعليق (Stuck-at).
    السنسور بيفضل قاري نفس القيمة اللي كان قاريها لحظة بداية العطل.
    """
    # التعديل: بنعدل في الداتا مباشرة عشان التغيير يبقى تراكمي (Cumulative)
    new_data = data 
    label = np.zeros(len(data), dtype=np.int32)
    
    end_idx = min(start_idx + duration, len(data))
    
    # بناخد القيمة اللي السنسور "قفش" عندها
    stuck_value = data[start_idx, sensor_idx]
    
    # بنثبت القيمة دي طول فترة الـ duration
    new_data[start_idx:end_idx, sensor_idx] = stuck_value
    
    # بنعلم على الفترة دي كلها إنها عطل (1)
    label[start_idx:end_idx] = 1
    
    return new_data, label


def inject_noise(data, sensor_idx, start_idx, duration=100, noise_level=0.5):
    label = np.zeros(len(data), dtype=np.int32)
    end_idx = min(start_idx + duration, len(data))
    
    # بنعمل دوشة عشوائية (Gaussian Noise) ونضيفها على الداتا
    noise = np.random.normal(0, noise_level, size=(end_idx - start_idx))
    data[start_idx:end_idx, sensor_idx] += noise
    
    label[start_idx:end_idx] = 1
    return data, label


def inject_bias(data, sensor_idx, start_idx, duration=100, bias_value=2.0):
    label = np.zeros(len(data), dtype=np.int32)
    end_idx = min(start_idx + duration, len(data))
    
    # بنزود قيمة ثابتة (Offset) على كل القراءات في الفترة دي
    data[start_idx:end_idx, sensor_idx] += bias_value
    
    label[start_idx:end_idx] = 1
    return data, label


def inject_dead_sensor(data, sensor_idx, start_idx, duration=100):
    label = np.zeros(len(data), dtype=np.int32)
    end_idx = min(start_idx + duration, len(data))
    
    # السنسور بيموت ويدي صفر تماماً
    data[start_idx:end_idx, sensor_idx] = 0.0
    
    label[start_idx:end_idx] = 1
    return data, label


# --- 2. المايسترو (The Master) ---

# def create_anomaly_test_set(normal_test_data, all_columns, numeric_features, num_anomalies=10):
#     """
#     نسخة coreX النهائية: 
#     - بتفلتر أعمدة السيكل والزمن.
#     - بتوزع الـ 6 أعطال بذكاء على السنسورات المناسبة.
#     - بتضمن إن الوحايد تتسيف صح.
#     """
    
#     # 1. التجهيز
#     test_data = normal_test_data.copy().astype(np.float32)
#     final_labels = np.zeros(len(test_data), dtype=np.int32)
#     n_samples, n_features = test_data.shape
    
#     # قائمة الممنوعات (Blacklist) عشان ما نهبدش في السيكل أو التايم
#     forbidden = ['currentcycle', 'timestamp', 'num', 'stop', 'lost']
    
#     # تنقية السنسورات المسموح بالحقن فيها بس (Numeric & Not Forbidden)
#     allowed_indices = []
#     for i, col in enumerate(all_columns):
#         col_lower = col.lower()
#         if any(f in col_lower for f in forbidden):
#             continue # ابعد عن الممنوعات
#         if col in numeric_features:
#             allowed_indices.append(i)

#     if not allowed_indices:
#         print("⚠️ Warning: No valid sensors found for injection!")
#         return test_data, final_labels

#     buffer = 15 
#     segment_size = n_samples // num_anomalies
    
#     print(f"--- [coreX Info] Starting smart injection on {len(allowed_indices)} valid sensors...")

#     for i in range(num_anomalies):
#         start_zone = i * segment_size + buffer
#         end_zone = (i + 1) * segment_size - buffer
        
#         if end_zone - start_zone < 150: continue
            
#         start_idx = np.random.randint(start_zone, end_zone - 150)
        
#         # اختيار سنسور عشوائي من المسموح بيهم بس
#         s_idx = np.random.choice(allowed_indices)
#         s_name = all_columns[s_idx].lower()
        
#         # اختيار نوع العطل (الـ 6 أنواع)
#         f_type = np.random.choice(['spike', 'drift', 'stuck', 'noise', 'bias', 'dead'])
        
#         # 3. الحقن الذكي (التراكمي)
#         if f_type == 'spike':
#             test_data, l = inject_spike(test_data, s_idx, start_idx)
#         elif f_type == 'drift':
#             test_data, l = inject_drift(test_data, s_idx, start_idx, duration=150)
#         elif f_type == 'stuck':
#             test_data, l = inject_stuck_at(test_data, s_idx, start_idx, duration=100)
#         elif f_type == 'noise':
#             test_data, l = inject_noise(test_data, s_idx, start_idx, duration=120)
#         elif f_type == 'bias':
#             test_data, l = inject_bias(test_data, s_idx, start_idx, duration=100)
#         elif f_type == 'dead':
#             test_data, l = inject_dead_sensor(test_data, s_idx, start_idx, duration=100)
            
#         # دمج الـ Label الجديد مع النهائي (OR)
#         final_labels = np.logical_or(final_labels, l).astype(np.int32)
        
#     print(f"✅ Success! Injected {np.sum(final_labels == 1)} anomaly points.")
#     return test_data, final_labels

def create_anomaly_test_set(normal_test_data, all_columns, numeric_features, num_anomalies=10):
    # 1. التجهيز
    test_data = normal_test_data.copy().astype(np.float32)
    final_labels = np.zeros(len(test_data), dtype=np.int32)
    n_samples, n_features = test_data.shape
    
    # قائمة الممنوعات
    forbidden = ['currentcycle', 'timestamp', 'num', 'stop', 'lost']
    
    # تنقية السنسورات
    allowed_indices = []
    for i, col in enumerate(all_columns):
        col_lower = col.lower()
        if any(f in col_lower for f in forbidden):
            continue 
        if col in numeric_features:
            allowed_indices.append(i)

    if not allowed_indices:
        print("[WARN] Warning: No valid sensors found for injection!")
        return test_data, final_labels

    buffer = 15 
    segment_size = n_samples // num_anomalies
    
    print(f"--- [coreX Focused Mode] Injecting only OmniAnomaly-friendly types...")

    for i in range(num_anomalies):
        start_zone = i * segment_size + buffer
        end_zone = (i + 1) * segment_size - buffer
        
        if end_zone - start_zone < 150: continue
            
        start_idx = np.random.randint(start_zone, end_zone - 150)
        s_idx = np.random.choice(allowed_indices)
        
        # --- [تعديل coreX] اختيار الأعطال اللي الأومني "أستاذ" فيها بس ---
        # Drift: عشان نختبر قدرة الـ VAE على كشف الانحراف التدريجي.
        # Stuck: عشان نختبر كسر الـ Correlation بين السنسورز.
        # Spike: عشان نختبر الـ Point Anomalies والـ Reconstruction Error السريع.
        
        f_type = np.random.choice(['spike', 'drift', 'stuck'])
        
        # 3. الحقن الذكي
        if f_type == 'spike':
            test_data, l = inject_spike(test_data, s_idx, start_idx)
        elif f_type == 'drift':
            test_data, l = inject_drift(test_data, s_idx, start_idx, duration=150)
        elif f_type == 'stuck':
            test_data, l = inject_stuck_at(test_data, s_idx, start_idx, duration=100)
            
        # ملاحظة: تم استبعاد (noise, bias, dead) لأنها قد لا تظهر بوضوح في الـ Reconstruction Probability
        # أو قد تتداخل مع الـ Normal Noise بتاع السنسورات.

        # دمج الـ Label
        final_labels = np.logical_or(final_labels, l).astype(np.int32)
        
    print(f"[OK] Success! Injected {np.sum(final_labels == 1)} points (Spikes/Drifts/Stucks).")
    return test_data, final_labels

    
# --- 3. نقطة التشغيل (coreX Debugger) ---
if __name__ == "__main__":
    your_file = r"data/RobotArm/all_data.csv" # يفضل تستخدمي r قبل المسار عشان الـ backslashes
    
    if os.path.exists(your_file):
        df = pd.read_csv(your_file)
        
        # بنفلتر الأعمدة اللي "ينفع" تكون رقمية بس عشان التجربة
        # بنختار كل الأعمدة ما عدا الـ Timestamp والـ ID
        numeric_df = df.select_dtypes(include=[np.number])
        all_cols = numeric_df.columns
        
        print(f"[OK] Testing: Data Loaded. Shape: {numeric_df.shape}")

        # تجربة سريعة بـ 10 أعطال
        test_data, labels = create_anomaly_test_set(
            numeric_df.values, 
            all_cols, 
            all_cols, # بنجرب نحقن في كل الأعمدة الرقمية المتاحة
            num_anomalies=10
        )
        
        # أهم سطرين في الكون دلوقتي:
        total_anomalies = np.sum(labels)
        print(f"[OK] Total anomaly points (1s) found: {total_anomalies}")
        
        if total_anomalies > 0:
            print("[OK] The injection logic is working and saving labels.")
        else:
            print("[WARN] Still Zero? Check if start_idx and buffer logic are skipping segments.")
            
    else:
        print(f"[ERR] Error: File '{your_file}' not found. Check your folders!")