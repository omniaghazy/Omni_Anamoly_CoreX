import pickle
import numpy as np
import pandas as pd
import os

# المسارات بتاعة ملفات الـ processed اللي الموديل بيقرأ منها
test_data_path = r'data\processed\RobotArm_test.pkl'
label_data_path = r'data\processed\RobotArm_test_label.pkl'

print("🕵️--- [coreX Inspector] Checking your PKL files ---")

if os.path.exists(test_data_path) and os.path.exists(label_data_path):
    # 1. فتح ملفات الـ Binary (PKL)
    with open(test_data_path, 'rb') as f:
        x_test = pickle.load(f)
    with open(label_data_path, 'rb') as f:
        y_test = pickle.load(f)

    print(f"✅ Data Loaded! \n   - Features (X) shape: {x_test.shape} \n   - Labels (Y) shape: {y_test.shape}")

    # 2. تحويل أول 10 سنسورات لـ DataFrame عشان نعرف نعرضهم
    num_sensors_to_show = 10
    columns = [f'Sensor_{i}' for i in range(num_sensors_to_show)]
    df_check = pd.DataFrame(x_test[:, :num_sensors_to_show], columns=columns)
    
    # 3. إضافة عمود الـ Label (اللي إنتي عايزة تطمني عليه)
    df_check['TARGET_LABEL'] = y_test

    # 4. البحث عن الوحايد (Anomalies)
    anomalies_only = df_check[df_check['TARGET_LABEL'] == 1]

    if len(anomalies_only) > 0:
        print(f"\n🔥 Found {len(anomalies_only)} anomaly points in the PKL!")
        print("-" * 50)
        print("Showing a sample of rows where Label is 1:")
        # هنعرض أول 15 سطر فيهم أعطال
        print(anomalies_only.head(15).to_string()) 
        print("-" * 50)
    else:
        print("\n❌ Sad News: All Labels in this PKL are still ZERO (0).")
        print("This means the injection didn't save correctly or was skipped.")

else:
    print(f"❌ Error: Could not find files at: \n {test_data_path} \n {label_data_path}")