import os
import pandas as pd
import numpy as np
import ast
# Import functions from your utils file
# تأكدي إن الدوال دي موجودة في ملف utils.py في نفس الفولدر أو في omni_anomaly/utils.py
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