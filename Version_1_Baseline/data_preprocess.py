import ast
import csv
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pickle import dump
from anomaly_factory import create_anomaly_test_set
import shutil
from omni_anomaly.utils import initial_data_inventory, preprocess
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)





# def auto_organize_input_data(specific_file=None):          #omnia 4/3
#     base_data_path = os.path.join('data', 'RobotArm')
#     processed_path = os.path.join('data', 'processed')
#     os.makedirs(base_data_path, exist_ok=True)
#     os.makedirs(processed_path, exist_ok=True)

#     if specific_file and os.path.exists(specific_file):
#         source_file = specific_file
#     else:
#         data_extensions = ('.csv', '.xlsx', '.xls')
#         files_nearby = [f for f in os.listdir('.') if f.endswith(data_extensions)]
#         if not files_nearby:
#             if os.path.exists(os.path.join(base_data_path, 'all_data.csv')):
#                 return True
#             return False
#         source_file = files_nearby[0] 

#     target_file = os.path.join(base_data_path, 'all_data.csv')

#     try:
#         if source_file.endswith(('.xlsx', '.xls')):
#             df = pd.read_excel(source_file) 
#             df.to_csv(target_file, index=False, encoding='utf-8')
#             print("--- [Success] Excel Converted to CSV!")
#         else:
#             shutil.copy(source_file, target_file)
#             print("--- [Success] CSV Moved and Renamed!")
            
#         return True
#     except Exception as e:
#         print("--- [Error] " + str(e))
#         return False


def save_processed_data(data, category, dataset):
    """
    الفانكشن دي مهمتها الحفظ (Saving) 
    بتاخد الـ Array الجاهز وتحفظه بصيغة pkl في فولدر data/processed
    """
    try:
        # 1. تحديد الفولدر بشكل صريح
        folder = os.path.join('data', 'processed')
        
        # 2. التأكد إن الفولدر موجود (لو مش موجود اعمله)
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # 3. تحديد المسار النهائي للملف
        output_path = os.path.join(folder, f"{dataset}_{category}.pkl")
        
        # 4. حفظ الداتا (الـ Dump)
        with open(output_path, "wb") as file:
            pickle.dump(data, file, protocol=4)
            print(f"📍 Absolute Path: {os.path.abspath(output_path)}")
            
        print(f"✅ [File Saved] {dataset}_{category}.pkl is ready in {folder}")
        
    except Exception as e:
        print(f"❌ Error saving {category}: {str(e)}")


import matplotlib.pyplot as plt
import numpy as np

def visualize_sensor_data(data_array, column_names):
    """
    الدالة دي بتاخد المصفوفة وأسماء العواميد وترسمهم صح
    """
    # هنفترض إننا عايزين نرسم أول 3 سنسورز بس عشان الزحمة
    num_sensors_to_plot = 3 
    
    plt.figure(figsize=(12, 8))
    
    for i in range(num_sensors_to_plot):
        plt.subplot(num_sensors_to_plot, 1, i+1)
        
        # هنا بقى السحر: بنستخدم column_names عشان نكتب اسم السنسور الحقيقي
        sensor_name = column_names[i]
        
        plt.plot(data_array[:, i], label=f'Sensor: {sensor_name}')
        plt.title(f'Reading for {sensor_name}')
        plt.ylabel('Normalized Value')
        plt.legend()

    plt.tight_layout()
    plt.show()

# --- طريقة الاستخدام جوه مشروعك ---
# بعد ما تنادي الـ get_data أو الـ load_data:
# train_final, test_final, test_labels, all_processed_cols = load_data()

# جربي تنادي الفانكشن دي:
# visualize_sensor_data(train_final, all_processed_cols)
def load_data(dataset='RobotArm'):

    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    dataset_folder = os.path.join('data', 'RobotArm')
    from omni_anomaly.utils import unpack_robot_data, initial_data_inventory, advanced_data_visualizer

    # 1. حددي مكان الملفات
    file_path = os.path.join('data', 'RobotArm', 'all_data.csv')
    output_file = os.path.join('data', 'RobotArm', 'ready_for_ai.csv')

    print("--- [Step 1] Loading Raw Data...")

    if os.path.exists(file_path):
        # قراءة الداتا المكلكعة
        df_raw = pd.read_csv(file_path)
        
        # --- تشغيل الدالة الأولى: Unpacking ---
        # دي اللي بتفك الشنط وتحول الـ 14 عمود لـ 68
        print("\n--- [Step 2] Unpacking Vectors (Making it AI-Ready)...")
        df_unpacked = unpack_robot_data(df_raw)
        
        # حفظ الملف عشان نشوفه بعنينا
        df_unpacked.to_csv(output_file, index=False)
        print(f"✅ Unpacked file saved at: {output_file}")

        # --- تشغيل الدالة الثانية: Inventory (الجرد) ---
        # بنعمل جرد للداتا "المفكوكه" عشان نشوف الـ 68 عمود
        print("\n--- [Step 3] Running Data Inventory Audit...")
        df_clean, sensors = initial_data_inventory(df_unpacked)

        # --- تشغيل الدالة الثالثة: Visualizer (الرسومات) ---
        print("\n--- [Step 4] Launching Visualizations...")
        # ملحوظة: لو مش عايزة الرسومات توقف الكود، بنستخدم حيلة plt.show(block=False) 
        # بس الأفضل نشغلها عادي ونقفلها عشان Matplotlib في بايثون 3.6 ساعات بيهنج
        # advanced_data_visualizer(df_clean)
        
        print("\n--- [Success] Everything finished! ---")

    else:
        print(f"❌ Error: Could not find {file_path}")
        return None, None, None, None




    all_cleaned_cols = df_clean.columns
    print(f"✅ Step 2: Inventory done. Found {len(sensors)} numeric sensors.")

    from feature_engineering import remove_constant_features, remove_correlated_features
    print("\n--- [Step 3.5] Feature Selection & Engineering...")
    df_clean = remove_constant_features(df_clean, sensors)
    df_clean, sensors = remove_correlated_features(df_clean, threshold=0.95)
    
    # Keep only numeric sensors for the data array to prevent non-numeric casting errors
    df_clean_numeric = df_clean[sensors]
    all_cleaned_cols = df_clean_numeric.columns
    print(f"✅ Step 3.5: Feature selection done. Found {len(sensors)} numeric sensors after selection.")

    # 3. تحويل الداتا لـ Array
    data_array = df_clean_numeric.values.astype(np.float32)
    print(f"✅ Step 3: Data converted to float32 array. Shape: {data_array.shape}")

    # 4. التقسيم الزمني (TimeSeriesSplit)
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(data_array):
        raw_train = data_array[train_index]
        raw_test = data_array[test_index] # ده الاسم الصح اللي هنستخدمه تحت
    
    print(f"✅ Step 4: TimeSeriesSplit successful.")

    # 5. حقن الأعطال (Injection)
    # استعملنا raw_test بدل الاسم القديم اللي كان عامل Error


    test_faulty, test_labels = create_anomaly_test_set(
        raw_test, 
        all_cleaned_cols, 
        sensors, 
        num_anomalies=10
    )
    print(f"✅ Step 5: Anomalies injected successfully.")

    # 6. الـ Scaling (المكواة)
    # بنعمل سكيل للـ Train وبنستخدم نفس السكيلر للـ Test
    my_scaler = MinMaxScaler()
    train_final = my_scaler.fit_transform(raw_train)

    test_final = my_scaler.transform(test_faulty)
    
    print("visualize_sensor_data(train_final, all_cleaned_cols)")
    # visualize_sensor_data(train_final, all_cleaned_cols)
    print("✅ Step 6: Selective Scaling done.")

    # 7. حفظ البيانات (عشان OmniAnomaly يشوفها)
    save_processed_data(train_final, "train", dataset)
    save_processed_data(test_final, "test", dataset)
    save_processed_data(test_labels, "test_label", dataset)

# ... (آخر سطرين في الفانكشن) ...
    print(f"🚀 [coreX] Data ready! Anomalies injected: {np.sum(test_labels == 1)}")
    
    # هنرجع الـ 4 حاجات اللي الـ Script مستنيهم بره بالظبط
    return train_final, test_final, test_labels, all_cleaned_cols






if __name__ == "__main__":
    target = "RobotArm"
    print(f"CoreX Preprocessing System Starting for: {target}")
    
    # 1. تشغيل البروسيس
    # زودي متغير رابع (مثلاً اسمه cols) عشان بايثون ميزعلش

    train_data, test_data, test_labels, cols = load_data(target)
    
    if train_data is not None:
        # 2. الحفظ الإجباري
        print("--- [Action] Starting manual save...")
        save_processed_data(train_data, 'train', target)
        save_processed_data(test_data, 'test', target)
        save_processed_data(test_labels, 'test_label', target)
        
        print("\n🚀 ALL OPERATIONS COMPLETE. CHECK 'data/processed' NOW!")
    else:
        print("❌ Pipeline failed. Check errors above.")





