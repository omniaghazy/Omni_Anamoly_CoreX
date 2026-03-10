# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np



prefix = "processed"

def save_z(z, full_path):
    """
    coreX Version: بتسيف الـ z في المسار اللي بنحدده بالظبط
    """
    # 1. بنفصل اسم الملف عن الفولدر عشان نأمن نفسنا
    folder = os.path.dirname(full_path)
    
    # 2. لو الفولدر مش موجود.. ابنيه فوراً
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        print(f"---> [coreX] Created folder: {folder}")
    
    # 3. بنأكد إن الامتداد .npy موجود
    if not full_path.endswith('.npy'):
        full_path += '.npy'
    
    # 4. الحفظ
    np.save(full_path, z)
    print(f"---> [Success] Latent space saved to: {full_path}")






def get_data_dim(dataset):
    """
    Dynamically detects the number of sensors from CSV or PKL files.
    Updated to support the directory structure: data/processed/
    """
    import pandas as pd
    import pickle
    import os

    # 1. Define Paths based on your actual folder structure
    csv_path = os.path.join('data', dataset, 'train.csv')
    # According to your screenshot, 'processed' is inside 'data'
    pkl_path = os.path.join('data', 'processed', f'{dataset}_train.pkl')

    # 2. Check for data source
    # coreX Fix: We prioritize PKL because it matches the exact dimension after feature selection
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                actual_dim = data.shape[1]
                print(f"--- [coreX Info]: Loading dimension from PKL: {pkl_path} ---")
                print(f"--- [coreX Info]: Detected {actual_dim} sensors from PKL after feature selection ---")
                return actual_dim
        except Exception as pkl_e:
            raise Exception(f"Error reading PKL file: {pkl_e}")
            
    if not os.path.exists(csv_path):
        # If both are missing
        raise FileNotFoundError(f"Data not found! Checked: {csv_path} and {pkl_path}")

    # 3. Original CSV Logic (if CSV exists)
    try:
        # Read only headers
        df = pd.read_csv(csv_path, nrows=0)
        
        # Exclude non-feature columns
        non_feature_cols = ['Timestamp', 'time', 'Time', 'date', 'Unnamed: 0', 'Num']
        actual_features = [c for c in df.columns if c not in non_feature_cols]
        
        actual_dim = len(actual_features)
        print(f"--- [coreX Info]: Detected {actual_dim} sensors from CSV: {csv_path} ---")
        return actual_dim
        
    except Exception as e:
        raise Exception(f"Error reading CSV dimension: {e}")
        





def get_data(dataset, max_train_size=None, do_preprocess=False):        #omnia 4/3
    pkl_folder = os.path.join('data', 'processed')
    
    print(f"\n" + "="*50)
    print(f"--- [coreX Loader] Targeting: {dataset} ---")
    print(f"--- [coreX Loader] Path: {pkl_folder} ---")
    
    try:
        train_path = os.path.join(pkl_folder, f'{dataset}_train.pkl')
        test_path = os.path.join(pkl_folder, f'{dataset}_test.pkl')
        label_path = os.path.join(pkl_folder, f'{dataset}_test_label.pkl')

        with open(train_path, "rb") as f:
            x_train = pickle.load(f)
        with open(test_path, "rb") as f:
            x_test = pickle.load(f)
        with open(label_path, "rb") as f:
            y_test = pickle.load(f)

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        n_anomalies = int(np.sum(y_test == 1))
        unique_labels = np.unique(y_test)
        
        print(f"[OK] [coreX Audit] Data Loaded Successfully!")
        print(f"--- Train Shape: {x_train.shape}")
        print(f"--- Test Shape: {x_test.shape}")
        print(f"!!! Anomalies Found (1s): {n_anomalies}")
        print(f"--- Unique Labels: {unique_labels}")
        print("="*50 + "\n")

        if max_train_size:
            x_train = x_train[:max_train_size]

        return (x_train, None), (x_test, y_test)

    except FileNotFoundError as e:
        print(f"[ERR] [coreX Error] Missing files in {pkl_folder}!")
        print(f"Missing File: {e.filename}")
        print("[TIP] Action: Run 'python data_preprocess.py' again to generate these files.")
        raise
    except Exception as e:
        print(f"[ERR] [coreX Unexpected Error]: {e}")
        raise











import pandas as pd
import numpy as np
import pandas as pd
import ast

def unpack_robot_data(df):
    vector_cols = [
        'target_q', 'target_qd', 'target_qdd', 'target_current', 
        'target_moment', 'actual_q', 'actual_qd', 'actual_current', 
        'joint_temperatures', 'target_TCP_pose', 'actual_TCP_pose'
    ]
    
    new_df = pd.DataFrame()
    
    if 'timestamp' in df.columns:
        new_df['timestamp'] = df['timestamp']
    
    for col in vector_cols:
        if col in df.columns:
            try:
                # تحويل النص لـ List أرقام
                expanded = df[col].apply(lambda x: ast.literal_eval(x.strip()) if isinstance(x, str) else x).tolist()
                
                # بنعرف عدد المفاصل أو العناصر كام (ديناميكي)
                num_elements = len(expanded[0])
                col_names = [col + "_" + str(i) for i in range(num_elements)]
                
                temp_df = pd.DataFrame(expanded, columns=col_names, index=df.index)
                new_df = pd.concat([new_df, temp_df], axis=1)
                print("[OK] Unpacked: " + col + " (" + str(num_elements) + " elements)")
            except Exception as e:
                print("[WARN] Skipping " + col + " because: " + str(e))
                
    if 'robot_mode' in df.columns:
        new_df['robot_mode'] = df['robot_mode']
        
    return new_df

# --- طريقة الاستخدام ---
# لو عندك ملف اسمه data.csv
# raw_data = pd.read_csv('data.csv')
# clean_data = unpack_robot_data(raw_data)
# clean_data.to_csv('ready_for_ai.csv', index=False)

def initial_data_inventory(df):
    """
    دالة الجرد الشاملة لـ coreX (النسخة الكاملة):
    تجمع بين الكود الأصلي + رؤية البيانات + أنواع البيانات لجميع العواميد + الإحصائيات.
    """
    print("\n" + "="*60)
    print("[coreX Data Audit] Full Detailed Inventory Starting...")
    print("="*60)

    # 1. الأبعاد الأساسية (من كودك)
    print(f"--- Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")

    # 2. [إضافة] أنواع البيانات لكل عمود (Data Types)
    # دي عشان تشوفي بايثون فاهم كل عمود إيه بالظبط
    print("\n--- [Full Column Data Types]:")
    print(df.dtypes)

    # 3. [إضافة] رؤية جزء من الداتا (Snapshot)
    print("\n--- [Data Snapshot - First 5 Rows]:")
    print(df.head(5))

    # 4. جرد أنواع البيانات (من كودك)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\n--- Numerical Sensors: {len(numeric_cols)}")
    print(f"--- Categorical Columns: {len(categorical_cols)}")
    if categorical_cols:
        print(f"--- List of Categorical: {categorical_cols}")

    # 5. جرد القيم المفقودة (من كودك)
    missing_data = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_report = pd.concat([missing_data, missing_percent], axis=1, keys=['Total NaN', 'Percent %'])
    
    print("[WARN] [Missing Values Report]:")
    actual_missing = missing_report[missing_report['Total NaN'] > 0]
    if not actual_missing.empty:
        print(actual_missing)
    else:
        print("[OK] No missing values found! Clean sheet.")

    # 6. الإحصائيات الوصفية (من كودك + تطوير)
    print("\n--- [Statistical Summary - Numerical]:")
    if numeric_cols:
        description = df[numeric_cols].describe().T
        # عرض الإحصائيات الأساسية + الـ Median (50%) عشان المقارنة
        print(description[['mean', 'std', 'min', '50%', 'max']])

    # 7. [إضافة] إحصائيات الـ Categorical (عشان محرمكيش من معلومة)
    if categorical_cols:
        print("\n--- [Statistical Summary - Categorical]:")
        # بيوريكي الـ Unique values والـ Top (أكتر حاجة متكررة)
        print(df[categorical_cols].describe().T)

    # 8. كشف العواميد الثابتة (من كودك)
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    if constant_cols:
        print(f"\n[WARN] [Warning] Constant Columns detected (No info): {constant_cols}")

    # 9. [إضافة من تفكيري] كشف العواميد اللي فيها أصفار كتير (Zeros Detection)
    # ده مهم جداً في الروبوت عشان نعرف السنسورز اللي مش بتتحرك
    zeros_report = (df[numeric_cols] == 0).sum()
    significant_zeros = zeros_report[zeros_report > (0.1 * len(df))] # لو أكتر من 10% أصفار
    if not significant_zeros.empty:
        print("\n[NOTE] [Note] Columns with high zero-counts (>10%):")
        print(significant_zeros)

    print("\n" + "="*60)
    
    # رجوع الداتا المنظفة (مبدئياً) وأسامي السنسورز الرقمية
    return df, numeric_cols


# --- طريقة الاستخدام ---
# df = pd.read_csv('data/RobotArm/all_data.csv')
# df_clean, sensors = initial_data_inventory(df)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def advanced_data_visualizer(df):
    """
    دالة التحليل البصري الشامل لـ coreX:
    - Distribution
    - Boxplots
    - Correlation Heatmap
    """
    # هنركز بس على السنسورز اللي أرقام (Numeric)
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("[ERR] No numeric data to plot!")
        return

    # 1. الـ Correlation Heatmap (خريطة العلاقات)
    plt.figure(figsize=(12, 10))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("--- Sensors Correlation Heatmap")
    plt.show()

    # 2. الـ Distribution & Boxplot (لأول 6 سنسورز مثلاً عشان الزحمة)
    cols_to_show = numeric_df.columns[:6] 
    
    for col in cols_to_show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # الرسمة الأولى: Distribution
        sns.histplot(numeric_df[col], kde=True, ax=ax1, color='skyblue')
        ax1.set_title(f'Distribution of {col}')
        
        # الرسمة الثانية: Boxplot
        sns.boxplot(x=numeric_df[col], ax=ax2, color='salmon')
        ax2.set_title(f'Boxplot (Outliers) of {col}')
        
        plt.tight_layout()
        plt.show()

    # 3. نصيحة للـ Feature Engineering بناءً على الـ Correlation
    high_corr = np.where(np.abs(corr_matrix) > 0.95)
    high_corr_list = [(corr_matrix.index[x], corr_matrix.columns[y]) 
                      for x, y in zip(*high_corr) if x != y and x < y]
    
    if high_corr_list:
        print("\n[TIP] [Feature Engineering Tip]:")        
        print("The following sensors are highly correlated (>95%). Consider dropping one of them:")
        for pair in high_corr_list:
            print(f" - {pair[0]} & {pair[1]}")

# --- طريقة الاستخدام ---
# advanced_data_visualizer(df)






def preprocess(data_array, all_columns, numeric_cols, scaler=None):


    # 1. التأكد من النوع والمصفوفة
    data_array = np.asarray(data_array, dtype=np.float32)
    
    # 2. تحويل لـ DataFrame عشان نعرف نتحكم بالأعمدة بأساميها
    # التأكد إن عدد الأسامي قد عدد العواميد فعلاً
    if len(all_columns) != data_array.shape[1]:
        # لو فيه اختلاف، بنخترع أسامي عشان الكود ميفصلش
        all_columns = [f'feat_{i}' for i in range(data_array.shape[1])]
        # وبنعتبرهم كلهم numeric احتياطي
        numeric_cols = all_columns

    df_temp = pd.DataFrame(data_array, columns=all_columns)

    # 3. علاج الـ Nulls (احتياطي زيادة)
    if df_temp.isnull().values.any():
        df_temp = df_temp.ffill().bfill().fillna(0)

    # 4. الـ Scaling
    # لازم نتأكد إن الـ numeric_cols اللي جاية موجودة فعلاً في الداتا دي
    cols_to_scale = [c for c in numeric_cols if c in df_temp.columns]

    if scaler is None:
        # حالة الـ Train
        scaler = MinMaxScaler(feature_range=(0, 1))
        if cols_to_scale:
            df_temp[cols_to_scale] = scaler.fit_transform(df_temp[cols_to_scale])
        print(f'[OK] [Success] Train Preprocessing: {len(cols_to_scale)} sensors scaled.')
    else:
        # حالة الـ Test
        if cols_to_scale:
            df_temp[cols_to_scale] = scaler.transform(df_temp[cols_to_scale])
        print(f'[OK] [Success] Test Preprocessing: Scaled using Train parameters.')

    return df_temp.values.astype(np.float32), scaler


    


def minibatch_slices_iterator(length, batch_size, ignore_incomplete_batch=False):
    """
    بتقسم الداتا لقطع صغيرة (Slices) عشان الموديل يتدرب عليها بالدور.
    """
    # بنمشي من 0 لحد الطول الكلي، وبننط كل مرة بمقدار الـ batch_size
    for start in range(0, length, batch_size):
        end = start + batch_size
        
        # لو وصلنا للآخر والحتة اللي فاضلة أصغر من الـ batch_size
        if end > length:
            if ignore_incomplete_batch:
                break # ارمي الحتة الأخيرة دي مش عايزينها
            end = length # خد اللي فاضل لحد آخر نقطة في الداتا
            
        yield slice(start, end)





class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.

    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.

    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, batch_size, excludes=None,
                    shuffle=False, ignore_incomplete_batch=False):
            # 1. التأكد إن البرامترات منطقية
            if window_size < 1:
                raise ValueError('`window_size` must be at least 1')
            if array_size < window_size:
                raise ValueError('`array_size` must be at least as large as `window_size`')

            # 2. تجهيز الـ Mask (المصفاة)
            if excludes is not None:
                # استخدمنا bool العادية بدل np.bool عشان النسخ الجديدة
                excludes = np.asarray(excludes, dtype=bool)
                expected_shape = (array_size,)
                if excludes.shape != expected_shape:
                    raise ValueError(f'The shape of `excludes` is expected to be {expected_shape}, but got {excludes.shape}')
                mask = np.logical_not(excludes)
            else:
                mask = np.ones([array_size], dtype=bool)

            # 3. استبعاد أول كذا نقطة (لأنهم ميعملوش شباك كامل)
            mask[: window_size - 1] = False

            # 4. استبعاد أي شباك بيلمس نقطة "بايظة" (Exclude logic)
            if excludes is not None:
                where_excludes = np.where(excludes)[0]
                for k in range(1, window_size):
                    also_excludes = where_excludes + k
                    # بنضمن إننا مش بنخرج بره حدود المصفوفة
                    also_excludes = also_excludes[also_excludes < array_size]
                    mask[also_excludes] = False

            # 5. توليد الـ Indices النهائية (نهايات الشبابيك)
            indices = np.arange(array_size)[mask]
            self._indices = indices.reshape([-1, 1])

            # 6. الـ Offsets (دي المسطرة اللي بتحدد بداية ونهاية كل شباك)
            self._offsets = np.arange(-window_size + 1, 1)

            # 7. تخزين المتغيرات في الكلاس
            self._array_size = array_size
            self._window_size = window_size
            self._batch_size = batch_size
            self._shuffle = shuffle
            self._ignore_incomplete_batch = ignore_incomplete_batch
            
            print(f"--- [BatchSlidingWindow] Done! Windows found: {len(self._indices)} ---")

        

    def get_iterator(self, arrays):
            """
            الماكينة اللي بتقطع الداتا لشبابيك (Windows) وبتطلع Batches جاهزة للموديل.
            """
            # 1. التأكد إن الداتا جاهزة وتحويلها لـ Tuple
            arrays = tuple(np.asarray(a) for a in arrays)
            if not arrays:
                raise ValueError('`arrays` must not be empty')

            # 2. إنشاء خريطة ترتيب (Order Map) عشان الـ Shuffle ميبوظش الداتا الأصلية
            num_samples = len(self._indices)
            order = np.arange(num_samples)
            if self._shuffle:
                np.random.shuffle(order)

            # 3. الـ Loop الذكي باستخدام الموزع بتاعنا (Minibatch Slices)
            for s in minibatch_slices_iterator(
                    length=num_samples,
                    batch_size=self._batch_size,
                    ignore_incomplete_batch=self._ignore_incomplete_batch):
                
                # بنجيب العناوين للـ Batch ده بالترتيب (سواء متلخبط أو بالدور)
                current_order = order[s]
                
                # حساب مصفوفة العناوين لكل نقطة في كل شباك جوه الـ Batch ده
                # النتيجة هنا بتكون مصفوفة 2D فيها كل العناوين اللي هنحتاجها
                window_indices = self._indices[current_order] + self._offsets
                
                # >>> اللمسة السحرية: سحب الداتا كلها مرة واحدة (Vectorized Indexing)
                # ده بيخلي الداتا تطلع 3D للموديل: (Batch, Window, Sensors)
                yield tuple(a[window_indices] for a in arrays)
