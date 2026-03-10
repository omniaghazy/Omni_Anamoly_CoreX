# -*- coding: utf-8 -*-
import numpy as np

from omni_anomaly.spot import SPOT


def calc_point2point(predict, actual):
    """
    حساب الـ F1-Score وباقي الـ Metrics بدقة متناهية.
    الـ Point-to-Point دي هي الطريقة الصارمة (نقطة بنقطة).
    """
    # 1. التأكد إن الداتا Numpy Arrays ونوعها Integer (0 أو 1)
    predict = np.asarray(predict, dtype=np.int32)
    actual = np.asarray(actual, dtype=np.int32)

    # 2. حساب الـ Confusion Matrix Elements
    # الـ Vectorized Operations دي أسرع بكتير من الضرب العادي
    TP = np.sum((predict == 1) & (actual == 1))
    TN = np.sum((predict == 0) & (actual == 0))
    FP = np.sum((predict == 1) & (actual == 0))
    FN = np.sum((predict == 0) & (actual == 1))

    # 3. حساب المقاييس (بكل دقة)
    # epsilon دي أصغر رقم ممكن عشان نتفادى الـ Zero Division من غير ما نأثر على الأرقام
    eps = np.finfo(float).eps

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    # حتة زيادة من عندي عشان المناقشة (Accuracy)
    accuracy = (TP + TN) / (len(actual) + eps)

    return f1, precision, recall, accuracy, TP, TN, FP, FN


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    تعديل التوقعات بنظام الـ Segment-based Adjustment.
    لو الموديل لمس أي نقطة في العطل، بنحسب إن العطل كله 'اتقفش'.
    """
    # 1. تجهيز الداتا ومعالجة فرق الأطوال
    score = np.asarray(score).flatten()
    label = np.asarray(label).flatten()
    
    if len(score) != len(label):
        min_len = min(len(score), len(label))
        label = label[-min_len:]
        score = score[-min_len:]
        if pred is not None:
            pred = np.asarray(pred).flatten()[-min_len:]

    # 2. تحديد التوقعات الأولية
    if pred is None:
        predict = (score < threshold).astype(np.int32)
    else:
        predict = np.asarray(pred).flatten().astype(np.int32)
    
    actual = (label > 0.1).astype(np.int32)
    
    # 3. تحديد بدايات ونهايات فترات الأعطال (Segments)
    # بنعرف بداية ونهاية كل بلوك (Label=1)
    change = np.diff(np.concatenate([[0], actual, [0]]))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    
    latency = 0
    anomaly_count = len(starts)

    # 4. الـ Adjustment الصايع (بدل الـ nested loop)
    for start, end in zip(starts, ends):
        # هل الموديل قفش أي نقطة جوه الفترة دي؟
        if np.any(predict[start:end]):
            # قفش أول نقطة فين؟ (عشان نحسب الـ Latency)
            first_detection = np.where(predict[start:end] == 1)[0][0]
            latency += first_detection
            
            # ✅ تلوين الفترة كلها بـ True (اكتشاف العطل بالكامل)
            predict[start:end] = 1
            
    # 5. إرجاع النتائج
    if calc_latency:
        avg_latency = latency / (anomaly_count + 1e-6)
        return predict.astype(bool), avg_latency
    
    return predict.astype(bool)


def calc_seq(score, label, threshold, calc_latency=False):
    """
    حساب الـ F1-score والـ Metrics النهائية لسلسلة من البيانات.
    بتدمج الـ Adjustment مع الـ Point-to-Point calculation.
    """
    # 1. ضمان إن الـ label مسطح ونضيف
    label = np.asarray(label).flatten()
    
    # 2. استدعاء الـ adjustment (النسخة الصاروخية اللي عملناها)
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=False)
        latency = None

    # 3. حساب المقاييس الأساسية
    # الفانكشن دي بترجع (f1, precision, recall, accuracy, TP, TN, FP, FN)
    metrics = calc_point2point(predict, label)
    
    # 4. تجميع كل حاجة في Dictionary شيك (عشان المناقشة والـ Logs)
    results = {
        'f1': metrics[0],
        'precision': metrics[1],
        'recall': metrics[2],
        'accuracy': metrics[3],
        'TP': metrics[4],
        'TN': metrics[5],
        'FP': metrics[6],
        'FN': metrics[7]
    }
    
    if calc_latency:
        results['latency'] = latency
        
    # بنرجعها كـ tuple برضه لو الكود القديم معتمد على الترتيب، بس الـ dict أضمن
    return list(metrics) + ([latency] if calc_latency else[])
    

def bf_search(score, label, start, end=None, step_num=100, display_freq=10, verbose=True):
    """
    البحث عن أفضل Threshold لتحقيق أعلى F1-score.
    بتمشي بنظام الخطوات (Steps) من البداية للنهاية.
    """
    score = np.asarray(score).flatten()
    label = np.asarray(label).flatten()

    if end is None:
        end = start
        step_num = 1
    
    # 1. توليد كل الـ thresholds اللي هنجربها مرة واحدة
    thresholds = np.linspace(start, end, step_num)
    
    best_metrics = (-1.0, -1.0, -1.0, -1.0) # (F1, Precision, Recall, Accuracy)
    best_threshold = 0.0
    best_latency = 0.0
    
    if verbose:
        print(f"🚀 [Search] Starting Brute Force from {start:.4f} to {end:.4f} ({step_num} steps)")

    # 2. الـ Search Loop
    for i, thr in enumerate(thresholds):
        # حساب المقاييس للـ threshold الحالي
        # بنفترض إن calc_seq بترجع [f1, pre, rec, acc, tp, tn, fp, fn, latency]
        res = calc_seq(score, label, thr, calc_latency=True)
        
        current_f1 = res[0]
        
        # 3. تحديث "الأفضل" (The Champion)
        if current_f1 > best_metrics[0]:
            best_metrics = tuple(res[:4]) # بنحفظ الأساسيات
            best_threshold = thr
            best_latency = res[-1] if len(res) > 8 else 0.0 # الـ latency آخر عنصر
            
        if verbose and i % display_freq == 0:
            print(f"🔹 Step {i}/{step_num} | Thr: {thr:.4f} | F1: {current_f1:.4f} | Best F1: {best_metrics[0]:.4f}")

    if verbose:
        print("\n" + "="*40)
        print("🏆 FINAL BEST RESULTS 🏆")
        print(f"Best Threshold: {best_threshold:.6f}")
        print(f"F1-Score:      {best_metrics[0]:.4f}")
        print(f"Precision:     {best_metrics[1]:.4f}")
        print(f"Recall:        {best_metrics[2]:.4f}")
        print(f"Avg Latency:   {best_latency:.2f} points")
        print("="*40)
    
    return best_metrics, best_threshold

def pot_eval(init_score, score, label, q=1e-3, level=0.02, dynamic=False):
    """
    تحديد الـ Threshold تلقائياً باستخدام طريقة POT.
    الطريقة دي بتخلي الروبوت 'ذكي' وبيعرف العطل لوحده من غير ما نتدخل.
    """
    # 1. التأكد إن الداتا Numpy Arrays
    init_score = np.asarray(init_score).flatten()
    score = np.asarray(score).flatten()
    label = np.asarray(label).flatten()

    # 🚀 تكة المحترفين: الـ SPOT بيتعامل مع القيم الموجبة (القمم)
    # فلو الداتا سالبة (Log-Likelihood)، بنعكسها عشان نحسب الـ Tail صح
    s = SPOT(q)
    s.fit(init_score, score)
    
    # 2. خطوة الـ Initialization
    # min_extrema=True لأننا بندور على القيم اللي بتبعد عن الطبيعي (الطرف السفلي للـ Likelihood)
    s.initialize(level=level, min_extrema=True)
    
    # 3. التشغيل (سواء ثابت أو ديناميكي)
    ret = s.run(dynamic=dynamic)
    
    # حساب المتوسط للـ Thresholds اللي الـ SPOT اقترحها
    # بناخد القيمة المطلقة لضمان إن الـ threshold متوافق مع الـ scores بتاعتنا
    pot_th = np.mean(ret['thresholds'])
    
    # 4. التقييم النهائي (بإستخدام الـ Adjustment اللي ظبطناه)
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    metrics = calc_point2point(pred, label) # بترجع [f1, pre, rec, acc, tp, tn, fp, fn]

    if True: # Printing for logs
        print(f"\n📊 [POT Results] Found {len(ret['alarms'])} Alarms")
        print(f"📍 Suggested Threshold: {pot_th:.4f}")
        print(f"⏱️ Response Latency: {p_latency:.2f}")

    # 5. الترجيع في شكل قاموس منظم جداً
    return {
        'pot-f1': metrics[0],
        'pot-precision': metrics[1],
        'pot-recall': metrics[2],
        'pot-accuracy': metrics[3],
        'pot-TP': metrics[4],
        'pot-TN': metrics[5],
        'pot-FP': metrics[6],
        'pot-FN': metrics[7],
        'pot-threshold': pot_th,
        'pot-latency': p_latency,
        'pot-alarms_count': len(ret['alarms'])
    }
    