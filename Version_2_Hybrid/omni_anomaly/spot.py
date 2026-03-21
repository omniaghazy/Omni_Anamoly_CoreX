#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:08:16 2016

@author: Alban Siffer 
@company: Amossys
@license: GNU GPLv3
"""

from math import log, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.optimize import minimize

# colors for plot
deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'

"""
================================= MAIN CLASS ==================================
"""


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
            """
            Constructor المعدل ليكون أكثر مرونة
            """
            # 1. التأكد من منطقية مستوى المخاطرة (Risk Level)
            if not (0 < q < 1):
                raise ValueError("يا هندسة الـ q لازم تكون بين 0 و 1 (مثلاً 1e-4)")
                
            self.proba = q
            
            # 2. المتغيرات الأساسية (زي ما هي)
            self.extreme_quantile = None
            self.data = None
            self.init_data = None
            self.init_threshold = None
            self.peaks = None
            self.n = 0
            self.Nt = 0
            
            # 3. إضافات "الـ CoreX" للتحليل العميق
            # هنخزن التاريخ بتاع القيم عشان لو حبيتي ترسميهم بعدين
            self.history = {
                'gamma': [],
                'sigma': [],
                'thresholds': []
            }



    def __str__(self):
        """
        بيطلع ملخص محترم للحالة الحالية للـ Object
        """
        if self.data is None:
            return "❌ SPOT Object: No data imported yet."

        # بنستخدم f-strings عشان الكود يبقى "نضيف" وقابل للقراءة
        s =  "========================================\n"
        s += "   Streaming SPOT Algorithm Summary     \n"
        s += "========================================\n"
        s += f"🔹 Risk Level (q)      : {self.proba}\n"
        s += f"🔹 Init Data Size      : {self.init_data.size}\n"
        s += f"🔹 Stream Data Size    : {self.data.size}\n"
        
        if self.n == 0:
            s += "🔸 Status              : ⚠️ Not Initialized\n"
        else:
            s += "🔸 Status              : ✅ Initialized\n"
            s += f"🔹 Initial Threshold   : {self.init_threshold:.4f}\n"
            s += f"🔹 Extreme Quantile    : {self.extreme_quantile if self.extreme_quantile else 'N/A'}\n"
            s += f"🔹 Number of Peaks (Nt): {self.Nt}\n"
            
            # حساب التقدم
            run_count = self.n - self.init_data.size
            if run_count > 0:
                percent = (run_count / self.n) * 100
                s += f"🔹 Algorithm Run       : Yes ({run_count} points, {percent:.1f}%)\n"
            else:
                s += "🔹 Algorithm Run       : No (Still in calibration)\n"
        
        s += "========================================\n"
        return s
        
        
    def fit(self, init_data, data):
        """
        Import data to SPOT object
        
        Parameters
	    ----------
	    init_data : list, numpy.array or pandas.Series
		    initial batch to calibrate the algorithm
            
        data : numpy.array
		    data for the run (list, np.array or pd.series)
	
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
            """
            دالة إضافة الداتا مع التأكد من سلامتها وسرعة الأداء
            """
            # 1. تحويل الداتا لنوع NumPy بذكاء
            if isinstance(data, (list, np.ndarray, pd.Series)):
                new_data = np.array(data).flatten() # نضمن إنها صف واحد
            else:
                print(f"❌ Type {type(data)} not supported. Use list, numpy array or pandas series.")
                return

            # 2. فحص القيم الفاضية (Data Cleaning)
            if np.any(np.isnan(new_data)):
                print("⚠️ Warning: Data contains NaNs. Dropping them to save the algorithm.")
                new_data = new_data[~np.isnan(new_data)]

            # 3. الأداء (Performance Trick)
            # لو الـ self.data لسه بـ None بنحط الداتا الجديدة فوراً
            if self.data is None:
                self.data = new_data
            else:
                # في الـ Streaming الكبير، الأفضل نستخدم list مؤقتاً لو بنضيف نقطة بنقطة
                # بس هنا هنلتزم بالـ numpy مع تحسين بسيط
                self.data = np.concatenate([self.data, new_data])
                
            return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
            """
            دالة المعايرة (Calibration) - تحديد الثريشولد المبدئي وحساب معاملات التوزيع
            """
            # 1. التعامل مع القيم الصغرى (لو بنراقب هبوط مفاجئ في داتا الروبوت)
            if min_extrema:
                self.init_data = -self.init_data
                self.data = -self.data
                level = 1 - level

            # 2. تأمين الـ level (بلاش floor اللي مش متعرفة، نستخدم int مباشرة)
            # نضمن إن الـ level دايمًا كسر بين 0 و 1
            level = level - np.floor(level) if level >= 1 else level
            
            n_init = self.init_data.size
            S = np.sort(self.init_data)  # ترتيب الداتا عشان نجيب الـ Quantile

            # 3. تحديد الـ Initial Threshold (تجنب الـ Index Error)
            idx = int(level * n_init)
            if idx >= n_init: idx = n_init - 1
            self.init_threshold = S[idx]

            # 4. استخراج الـ Peaks (الزيادات فوق الثريشولد المبدئي)
            self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
            self.Nt = self.peaks.size
            self.n = n_init

            # 5. حماية الكود: لو الـ Peaks قليلة جداً الـ Grimshaw هيفشل
            if self.Nt < 2:
                if verbose: print("⚠️ Warning: Too few peaks for Grimshaw. Check your 'level' or data.")
                # هنحط قيم افتراضية بدل ما الكود يـ Crash
                self.extreme_quantile = self.init_threshold 
                return

            if verbose:
                print(f'Initial threshold : {self.init_threshold}')
                print(f'Number of peaks : {self.Nt}')
                print('Grimshaw maximum log-likelihood estimation ... ', end='')

            # 6. المطبخ الرياضي (حساب Gamma و Sigma)
            try:
                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)
                
                # تخزين في الـ History (للتحليل في مشروع التخرج)
                if hasattr(self, 'history'):
                    self.history['gamma'].append(g)
                    self.history['sigma'].append(s)
            except:
                if verbose: print('[Failed]')
                self.extreme_quantile = self.init_threshold
                return

            if verbose:
                print('[done]')
                print(f'\tγ (gamma) = {g:.6f}')
                print(f'\tσ (sigma) = {s:.6f}')
                print(f'\tL (Likelihood) = {l:.6f}')
                print(f'Extreme quantile (prob = {self.proba}): {self.extreme_quantile}')

            return


    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        دالة البحث عن الجذور - معدلة لتكون أكثر دقة وسرعة
        """
        # 1. توليد نقاط البداية (Initial guesses)
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)
        else:
            X0 = np.array([(bounds[0] + bounds[1]) / 2]) # Default middle point

        # 2. تعريف دالة الهدف (Objective Function) 
        # الهدف إننا نخلي f(x)^2 أقل ما يمكن (يعني يقرب من الصفر)
        def objFun(X, f, j_func):
            # بنحسب قيمة الدالة والمشتقة لكل نقطة
            f_vals = np.array([f(x) for x in X])
            j_vals = np.array([j_func(x) for x in X])
            
            g = np.sum(f_vals**2) # مجموع المربعات
            gradient = 2 * f_vals * j_vals # المشتقة بتاع مجموع المربعات
            return g, gradient

        # 3. عملية الـ Optimization باستخدام L-BFGS-B
        from scipy.optimize import minimize
        
        opt = minimize(lambda X: objFun(X, fun, jac), 
                       X0,
                       method='L-BFGS-B',
                       jac=True, 
                       bounds=[bounds] * len(X0),
                       tol=1e-8) # ضفنا Tolerance عشان الدقة

        # 4. تصفية النتائج (Filtering)
        # لازم نسيف التقريب في المتغير X
        X = np.round(opt.x, decimals=5)
        
        # نرجع القيم الفريدة (Unique) اللي فعلاً بتحقق إن الدالة قريبة من الصفر
        valid_roots = [x for x in np.unique(X) if abs(fun(x)) < 1e-4]
        
        return np.array(valid_roots)



    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
        """
        حساب الـ Log-Likelihood لتوزيع GPD
        المعادلة بتختلف حسب قيمة Gamma (الـ Shape parameter)
        """
        n = Y.size
        
        # التأكد إن sigma دايمًا موجبة عشان الـ log ميزعلش
        if sigma <= 0:
            return -np.inf

        if abs(gamma) > 1e-8:
            # الحالة العامة لما gamma مش بصفر
            tau = gamma / sigma
            # لازم نضمن إن (1 + tau * Y) دايمًا أكبر من الصفر
            # لو فيه قيمة خلت القوس ده سالب، الاحتمالية بتبقى صفر (يعني -inf في اللوج)
            if np.any(1 + tau * Y <= 0):
                return -np.inf
                
            L = -n * np.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            # حالة خاصة لما gamma تقرب من الصفر (التوزيع بيتحول لـ Exponential)
            # التصحيح الرياضي هنا: L = -n*log(sigma) - (1/sigma)*sum(Y)
            L = -n * np.log(sigma) - (Y.sum() / sigma)
            
        return L


    def _grimshaw(self, epsilon=1e-8, n_points=10):
            """
            حساب معاملات GPD باستخدام خدعة Grimshaw
            الهدف: إيجاد أفضل قيمة لـ gamma و sigma بناءً على الـ peaks
            """
            # تعريف الدوال المساعدة (Internal Helper Functions)
            def u(s):
                return 1 + np.log(s).mean()

            def v(s):
                return np.mean(1 / s)

            def w(Y, t):
                s = 1 + t * Y
                return u(s) * v(s) - 1

            def jac_w(Y, t):
                # حماية من القسمة على صفر لو t صغيرة جداً
                if abs(t) < 1e-12: t = 1e-12
                
                s = 1 + t * Y
                us = u(s)
                vs = v(s)
                jac_us = (1 / t) * (1 - vs)
                jac_vs = (1 / t) * (-vs + np.mean(1 / s**2))
                return us * jac_vs + vs * jac_us

            # حساب الخصائص الأساسية للـ Peaks
            Ym = self.peaks.min()
            YM = self.peaks.max()
            Ymean = self.peaks.mean()

            # تحديد فترات البحث عن الجذور (Bounds)
            a = -1 / YM
            if abs(a) < 2 * epsilon:
                epsilon = abs(a) / n_points

            a = a + epsilon
            b = 2 * (Ymean - Ym) / (Ymean * Ym)
            c = 2 * (Ymean - Ym) / (Ym**2)

            # 1. البحث عن الجذور في الناحية السالبة والموجبة
            # بنستخدم self._rootsFinder عشان نضمن استدعاء الدالة من الـ object الحالي
            left_zeros = self._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (a + epsilon, -epsilon),
                                        n_points, 'regular')

            right_zeros = self._rootsFinder(lambda t: w(self.peaks, t),
                                            lambda t: jac_w(self.peaks, t),
                                            (b, c),
                                            n_points, 'regular')

            # دمج كل الجذور المحتملة
            zeros = np.concatenate((left_zeros, right_zeros))

            # 2. تعيين الحالة الافتراضية (Initial best guess)
            # لو ملقيناش جذور، بنفترض إن التوزيع Exponential (gamma=0)
            gamma_best = 0.0
            sigma_best = Ymean
            ll_best = self._log_likelihood(self.peaks, gamma_best, sigma_best)

            # 3. المقارنة بين كل الجذور لاختيار اللي بيدي أعلى Likelihood
            for z in zeros:
                s_val = 1 + z * self.peaks
                # التأكد إن القيم موجبة عشان الـ log
                if np.any(s_val <= 0):
                    continue
                    
                gamma = u(s_val) - 1
                sigma = gamma / z
                
                # حساب الـ Log-Likelihood للمقارنة
                ll = self._log_likelihood(self.peaks, gamma, sigma)
                
                if ll > ll_best:
                    gamma_best = gamma
                    sigma_best = sigma
                    ll_best = ll

            return gamma_best, sigma_best, ll_best


    def _quantile(self, gamma, sigma):
            """
            حساب قيمة العتبة القصوى (Extreme Quantile) 
            بناءً على معاملات توزيع GPD ومستوى المخاطرة q
            """
            # r هو النسبة بين عدد النقاط الكلي ومستوى المخاطرة بالنسبة لعدد الـ Peaks
            # r = (n * q) / Nt
            r = (self.n * self.proba) / self.Nt
            
            # التأكد إن r قيمتها منطقية عشان الـ log ميزعلش
            if r <= 0:
                return self.init_threshold

            if abs(gamma) > 1e-8:
                # المعادلة العامة في حالة وجود انحراف (gamma != 0)
                # z_q = t + (sigma/gamma) * ( (n*q/Nt)^-gamma - 1 )
                return self.init_threshold + (sigma / gamma) * (np.power(r, -gamma) - 1)
            else:
                # حالة خاصة لما التوزيع يكون Exponential (gamma يقترب من الصفر)
                # z_q = t - sigma * log(n*q/Nt)
                return self.init_threshold - sigma * np.log(r)


    def run(self, with_alarm=True, dynamic=True):
            """
            تشغيل خوارزمية SPOT على سيل البيانات (Stream)
            """
            import tqdm # نضمن إن مكتبة tqdm موجودة للـ progress bar
            
            # 1. التأكد إننا عملنا initialize الأول
            if self.n > self.init_data.size:
                print('⚠️ Warning: Algorithm already run. Please initialize again if needed.')
                return {}

            thresholds_history = []
            alarms_indexes = []
            
            # 2. اللوب الرئيسية على الداتا
            # استخدام tqdm بيعرفك فاضل وقت قد إيه (مهم جداً في مشروع التخرج)
            for i in tqdm.tqdm(range(self.data.size), desc="Processing Stream"):
                current_val = self.data[i]
                
                # --- الحالة الأولى: لو مش عايزين "ديناميكية" (Static Threshold) ---
                if not dynamic:
                    if current_val > self.init_threshold and with_alarm:
                        alarms_indexes.append(i)
                    # في الحالة الـ Static، الـ extreme_quantile بيفضل ثابت
                
                # --- الحالة الثانية: الـ SPOT الحقيقي (Dynamic Threshold) ---
                else:
                    # أ: هل القيمة تعدت العتبة القصوى؟ (حالة Alarm)
                    if current_val > self.extreme_quantile:
                        if with_alarm:
                            alarms_indexes.append(i)
                        else:
                            # لو مش عايزين Alarm، بنعتبرها Peak جديدة ونحدث الموديل
                            self._update_model(current_val)
                    
                    # ب: هل القيمة بين الـ Initial Threshold والـ Extreme Quantile؟
                    elif current_val > self.init_threshold:
                        # بنعتبرها Peak طبيعية بنعلم بيها الموديل (Learning)
                        self._update_model(current_val)
                    
                    # ج: القيمة تحت الثريشولد (طبيعية جداً)
                    else:
                        self.n += 1

                # تسجيل قيمة الثريشولد في اللحظة دي
                thresholds_history.append(self.extreme_quantile)

            return {'thresholds': thresholds_history, 'alarms': alarms_indexes}

    def _update_model(self, val):
            """
            دالة مساعدة (Helper) لتحديث المعاملات بدل التكرار في الكود
            """
            new_peak = val - self.init_threshold
            # أسرع من np.append في الـ Loops
            self.peaks = np.concatenate([self.peaks, [new_peak]])
            self.Nt += 1
            self.n += 1
            
            # تحديث الحسابات الإحصائية
            g, s, l = self._grimshaw()
            self.extreme_quantile = self._quantile(g, s)



    def plot(self, run_results, with_alarm=True):
            """
            رسم نتائج الـ SPOT: البيانات، الثريشولد، والإنذارات
            """
            import matplotlib.pyplot as plt

            # 1. تجهيز البيانات
            x = range(self.data.size)
            K = run_results.keys()
            
            # إنشاء الشكل (Figure) بمقاس مريح للعين
            plt.figure(figsize=(12, 6))
            
            # 2. رسم البيانات الأصلية (السنسور بتاع الروبوت)
            ts_fig, = plt.plot(x, self.data, color='#5D8AA8', label='Stream Data', alpha=0.8)
            fig_elements = [ts_fig]

            # 3. رسم الثريشولد (العتبة المتغيرة)
            if 'thresholds' in K:
                th = run_results['thresholds']
                # لو طول الثريشولد أقل من الداتا (بسبب الـ initialization) هنظبط الـ x
                x_th = range(self.data.size - len(th), self.data.size)
                th_fig, = plt.plot(x, th, color='#FF9933', lw=2, ls='--', label='Dynamic Threshold')
                fig_elements.append(th_fig)

            # 4. رسم الإنذارات (القط المهرش في الداتا)
            if with_alarm and ('alarms' in K):
                alarm_idx = run_results['alarms']
                if len(alarm_idx) > 0:
                    al_fig = plt.scatter(alarm_idx, self.data[alarm_idx], 
                                        color='red', marker='x', s=50, label='Alarms')
                    fig_elements.append(al_fig)

            # 5. تظبيط شكل الرسمة (The Finishing Touches)
            plt.title(f'SPOT Anomaly Detection (q={self.proba})', fontsize=14)
            plt.xlabel('Time Step / Observations', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(loc='upper left')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.xlim((0, self.data.size))
            
            plt.show() # عشان يعرض الرسمة فوراً
            
            return fig_elements


"""
============================ UPPER & LOWER BOUNDS =============================
"""



class biSPOT:
    """
    This class allows to run biSPOT algorithm on univariate dataset (upper and lower bounds)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
            """
            الـ Constructor بتاع الـ biSPOT
            هنا بنجهز المخازن (Dictionaries) اللي هتشيل القيم للناحيتين:
            'up' (للمراقبة من فوق) و 'down' (للمراقبة من تحت)
            """
            # مستوى المخاطرة (Risk Level)
            self.proba = q
            
            # بنجهز مكان للداتا (هتتملي لما ننادي على fit)
            self.data = None
            self.init_data = None
            
            # عداد النقاط اللي الموديل شافها
            self.n = 0
            
            # قائمة الإنذارات (مهمة عشان الـ __str__ والـ plot)
            self.alarm = []

            # تعريف القواميس بشكل مباشر (أسرع وأضمن من dict.copy)
            # بنحط None كقيمة مبدئية لحد ما نعمل initialize
            self.extreme_quantile = {'up': None, 'down': None}
            self.init_threshold = {'up': None, 'down': None}
            self.peaks = {'up': None, 'down': None}
            self.gamma = {'up': None, 'down': None}
            self.sigma = {'up': None, 'down': None}
            
            # عدد الـ Peaks اللي لقطناها في كل ناحية
            self.Nt = {'up': 0, 'down': 0}
            
            # مخزن لتاريخ الـ Parameters (مفيد جداً لو حبيتي ترسمي التغير في Gamma أو Sigma)
            self.history = {'up': {'gamma': [], 'sigma': []}, 
                            'down': {'gamma': [], 'sigma': []}}



    def __str__(self):
            """
            عرض حالة الـ Object بشكل منظم واحترافي
            """
            s = '--- biSPOT (Bidirectional Streaming Peaks-Over-Threshold) ---\n'
            s += f'Detection level q = {self.proba}\n'
            
            # 1. حالة البيانات
            if self.data is not None:
                s += 'Data imported : Yes\n'
                s += f'\t - Initialization batch : {self.init_data.size} values\n'
                s += f'\t - Stream batch         : {self.data.size} values\n'
            else:
                s += 'Data imported : No\n'
                return s # هنا بنوقف لو مفيش داتا أصلاً

            # 2. حالة الـ Initialization
            if self.n == 0:
                s += 'Algorithm initialized : No\n'
            else:
                s += 'Algorithm initialized : Yes\n'
                s += f'\t - Upper initial threshold : {self.init_threshold["up"]}\n'
                s += f'\t - Lower initial threshold : {self.init_threshold["down"]}\n'

                # 3. حالة التشغيل (Run status)
                r = self.n - self.init_data.size
                if r > 0:
                    s += 'Algorithm run : Yes\n'
                    s += f'\t - Observations processed : {r} ({100 * r / self.n:.2f} %)\n'
                    # تأمين الـ alarm لو مش موجودة كـ Attribute
                    num_alarms = len(self.alarm) if hasattr(self, 'alarm') else 0
                    s += f'\t - Triggered alarms       : {num_alarms} ({100 * num_alarms / self.n:.2f} %)\n'
                else:
                    s += 'Algorithm run : No\n'
                    s += f'\t - Number of peaks (Up)   : {self.Nt["up"]}\n'
                    s += f'\t - Number of peaks (Down) : {self.Nt["down"]}\n'
                    s += f'\t - Upper Extreme Quantile : {self.extreme_quantile["up"]}\n'
                    s += f'\t - Lower Extreme Quantile : {self.extreme_quantile["down"]}\n'
            
            s += '-------------------------------------------------------'
            return s


            
    def fit(self, init_data, data):
            """
            إدخال البيانات وتجهيزها للـ biSPOT
            بنحول أي نوع (List, Series, Array) لـ NumPy Array عشان الحسابات
            """
            
            # دالة داخلية صغيرة عشان منكررش الكود (Helper)
            def convert_to_numpy(d):
                if isinstance(d, list):
                    return np.array(d)
                elif isinstance(d, np.ndarray):
                    return d
                elif isinstance(d, pd.Series):
                    return d.values
                return None

            # 1. تحويل الداتا الأساسية (Stream Data)
            self.data = convert_to_numpy(data)
            if self.data is None:
                print(f'❌ Error: Data format ({type(data)}) is not supported')
                return

            # 2. تحويل بيانات الـ Initialization (الـ Calibration batch)
            # حالة إن init_data رقم صحيح (عدد نقاط معين)
            if isinstance(init_data, int):
                self.init_data = self.data[:init_data]
                self.data = self.data[init_data:]
                
            # حالة إن init_data نسبة مئوية (مثلاً 0.15 يعني خد أول 15% من الداتا)
            elif isinstance(init_data, float) and (0 < init_data < 1):
                # بنستخدم self.data.size عشان نضمن إننا بنتعامل مع الـ array المحول
                r = int(init_data * self.data.size)
                self.init_data = self.data[:r]
                self.data = self.data[r:]
                
            # حالة إن init_data مصفوفة جاهزة
            else:
                self.init_data = convert_to_numpy(init_data)
                if self.init_data is None:
                    print('❌ Error: The initial data cannot be set')
                    return

            # للتأكيد: بنخلي الداتا Flatten (بُعد واحد) عشان حسابات الـ GPD
            self.data = self.data.flatten()
            self.init_data = self.init_data.flatten()



    def add(self, data):
            """
            إضافة بيانات جديدة للـ stream اللي موجود فعلاً
            مفيدة جداً لو الداتا بتيجي على دفعات (Batches)
            """
            # تحويل الداتا الجديدة لـ NumPy Array
            if isinstance(data, list):
                new_data = np.array(data)
            elif isinstance(data, np.ndarray):
                new_data = data
            elif isinstance(data, pd.Series):
                new_data = data.values
            else:
                print(f'❌ Error: This data format ({type(data)}) is not supported')
                return

            # التأكد إن الداتا الجديدة والقديمة "مفرودين" (Flattened) قبل الدمج
            # np.concatenate أسرع في التعامل مع الذاكرة من np.append
            if self.data is None:
                self.data = new_data.flatten()
            else:
                self.data = np.concatenate([self.data, new_data.flatten()])
                
            # مفيش داعي لـ return هنا لأننا بنعدل في الـ Object نفسه


    def initialize(self, verbose=True):
            """
            مرحلة المعايرة (Calibration):
            1. تحديد الـ Initial Thresholds (فوق وتحت).
            2. استخراج الـ Initial Peaks.
            3. حساب معاملات GPD باستخدام Grimshaw.
            """
            n_init = self.init_data.size

            # 1. ترتيب البيانات لحساب الـ Quantiles التجريبية
            S = np.sort(self.init_data)
            
            # تحديد العتبة العلوية (أعلى 2%) والسفلية (أقل 2%)
            # ملحوظة: الـ 0.98 و 0.02 دي قيم متعارف عليها في الـ SPOT
            self.init_threshold['up'] = S[int(0.98 * n_init)]
            self.init_threshold['down'] = S[int(0.02 * n_init)]

            # 2. استخراج الـ initial peaks (القيم المتطرفة عن العتبة)
            # الناحية اللي فوق: القيمة - العتبة
            self.peaks['up'] = self.init_data[self.init_data > self.init_threshold['up']] - self.init_threshold['up']
            
            # الناحية اللي تحت: العتبة - القيمة (عشان تطلع موجبة للـ GPD)
            self.peaks['down'] = self.init_threshold['down'] - self.init_data[self.init_data < self.init_threshold['down']]
            
            self.Nt['up'] = self.peaks['up'].size
            self.Nt['down'] = self.peaks['down'].size
            self.n = n_init

            if verbose:
                print(f"Initial threshold : {self.init_threshold}")
                print(f"Number of peaks : {self.Nt}")
                print('Grimshaw maximum log-likelihood estimation ... ', end='')

            # 3. تشغيل Grimshaw لكل ناحية بشكل منفصل
            l = {'up': None, 'down': None}
            for side in ['up', 'down']:
                # تأكدي إن الـ peaks مش فاضية قبل ما تشغلي Grimshaw
                if self.Nt[side] > 1:
                    g, s, l[side] = self._grimshaw(side)
                    self.extreme_quantile[side] = self._quantile(side, g, s)
                    self.gamma[side] = g
                    self.sigma[side] = s
                else:
                    # لو مفيش peaks كفاية، بنحط قيم افتراضية
                    self.gamma[side] = 0
                    self.sigma[side] = self.init_data.std() # تقريب أولي
                    self.extreme_quantile[side] = self.init_threshold[side]

            # 4. طباعة الجدول النهائي (التقرير)
            if verbose:
                print('[done]')
                ltab = 20
                form = '\t' + '%20s' + '%20.4f' + '%20.4f'
                print('\t' + 'Parameters'.rjust(ltab) + 'Upper'.rjust(ltab) + 'Lower'.rjust(ltab))
                print('\t' + '-' * ltab * 3)
                # استخدام اليونيكود لرسم الرموز الرياضية (Gamma و Sigma)
                print(form % (chr(0x03B3), self.gamma['up'], self.gamma['down']))
                print(form % (chr(0x03C3), self.sigma['up'], self.sigma['down']))
                print(form % ('Likelihood', l['up'] if l['up'] else 0, l['down'] if l['down'] else 0))
                print(form % ('Extreme Quantile', self.extreme_quantile['up'], self.extreme_quantile['down']))
                print('\t' + '-' * ltab * 3)



    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
            """
            البحث عن الجذور المحتملة للدالة (حيث f(x) = 0)
            بتحول المشكلة لـ Minimization Problem لـ f(x)^2
            """
            # 1. تحديد نقاط البداية (Starting Points) للبحث
            if method == 'regular':
                step = (bounds[1] - bounds[0]) / (npoints + 1)
                X0 = np.arange(bounds[0] + step, bounds[1], step)
            elif method == 'random':
                X0 = np.random.uniform(bounds[0], bounds[1], npoints)
            else:
                X0 = np.array([(bounds[0] + bounds[1]) / 2]) # Default middle point

            # 2. تعريف الدالة الهدف (Objective Function)
            # إحنا بنحاول نصفر f(x)، فبنعمل minimize لـ f(x)^2
            def objFun(X, f, j_func):
                # X هنا عبارة عن Vector فيه كل النقاط اللي بنجربها
                f_vals = np.array([f(x) for x in X])
                j_vals = np.array([j_func(x) for x in X])
                
                # g = sum of squares
                g = np.sum(f_vals**2)
                # الـ Gradient بتاع f(x)^2 هو 2 * f(x) * f'(x)
                jac_vec = 2 * f_vals * j_vals
                return g, jac_vec

            # 3. عملية الـ Optimization باستخدام L-BFGS-B
            # دي خوارزمية شاطرة جداً في البحث داخل حدود (Bounds)
            opt = minimize(lambda X: objFun(X, fun, jac), 
                        X0,
                        method='L-BFGS-B',
                        jac=True, 
                        bounds=[bounds] * len(X0))

            # 4. تنظيف النتائج
            X = opt.x
            # لازم تعملي Assignment للتقريب، np.round مش بتغير في مكانها
            X = np.round(X, decimals=5)
            
            # بنرجع القيم الفريدة (عشان لو كذا نقطة وصلت لنفس الجذر)
            return np.unique(X)

    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
            """
            حساب الـ Log-Likelihood لتوزيع GPD
            الهدف: قياس مدى جودة المعاملات (gamma, sigma) في تمثيل الـ Peaks
            """
            n = Y.size
            if n == 0:
                return -np.inf # لو مفيش داتا، الاحتمالية بصفر (سالب مالانهاية في الـ log)

            # بنستخدم np.abs عشان نضمن إننا بنتعامل مع قيم قريبة جداً من الصفر صح
            if abs(gamma) > 1e-8:
                tau = gamma / sigma
                # لازم نتأكد إن (1 + tau * Y) دايماً موجبة عشان الـ log ميزعلش
                arg = 1 + tau * Y
                if np.any(arg <= 0):
                    return -np.inf
                    
                # المعادلة الرسمية للـ GPD Log-Likelihood
                # L = -n*ln(sigma) - (1 + 1/gamma) * sum(ln(1 + gamma*Y/sigma))
                L = -n * np.log(sigma) - (1 + (1 / gamma)) * (np.log(arg)).sum()
            else:
                # حالة خاصة لما gamma تؤول للصفر (التوزيع بيتحول لـ Exponential)
                # L = -n*ln(mean(Y)) - n
                # ملحوظة: الكود القديم كان فيه غلطة في معادلة الـ else، دي النسخة الصح:
                L = -n * np.log(Y.mean()) - n
                
            return L



    def _grimshaw(self, side, epsilon=1e-8, n_points=10):
            """
            حساب معاملات GPD (gamma & sigma) باستخدام خدعة Grimshaw
            الهدف: إيجاد أحسن معاملات توصف الـ Peaks المتطرفة.
            """
            
            # 1. تعريف الدوال المساعدة (Grimshaw Helper Functions)
            def u(s):
                return 1 + np.log(s).mean()

            def v(s):
                return np.mean(1 / s)

            def w(Y, t):
                s = 1 + t * Y
                return u(s) * v(s) - 1

            def jac_w(Y, t):
                s = 1 + t * Y
                us, vs = u(s), v(s)
                jac_us = (1 / t) * (1 - vs)
                jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
                return us * jac_vs + vs * jac_us

            # 2. تجهيز البيانات للناحية المطلوبة (Side: up or down)
            Y = self.peaks[side]
            if Y.size == 0:
                return 0, self.init_data.std(), -np.inf

            Ym = Y.min()
            YM = Y.max()
            Ymean = Y.mean()

            # 3. تحديد نطاق البحث عن الجذور (Bounds)
            a = -1 / YM
            if abs(a) < 2 * epsilon:
                epsilon = abs(a) / n_points

            a = a + epsilon
            # b و c هما حدود المنطقة التانية للبحث
            b = 2 * (Ymean - Ym) / (Ymean * Ym)
            c = 2 * (Ymean - Ym) / (Ym ** 2)

            # 4. البحث عن الجذور (Roots) باستخدام الـ staticmethod اللي صلحناه
            # بننادي عليه بـ self._rootsFinder أو biSPOT._rootsFinder
            left_zeros = self._rootsFinder(lambda t: w(Y, t),
                                        lambda t: jac_w(Y, t),
                                        (a + epsilon, -epsilon),
                                        n_points, 'regular')

            right_zeros = self._rootsFinder(lambda t: w(Y, t),
                                            lambda t: jac_w(Y, t),
                                            (b, c),
                                            n_points, 'regular')

            # دمج كل الجذور المحتملة
            zeros = np.concatenate((left_zeros, right_zeros))

            # 5. اختيار أفضل جذر (اللي بيدي أعلى Likelihood)
            # بنبدأ بالحالة الافتراضية (gamma = 0)
            gamma_best = 0
            sigma_best = Ymean
            ll_best = self._log_likelihood(Y, gamma_best, sigma_best)

            for z in zeros:
                # التحقق من أن القيمة ليست صفرية تماماً لتجنب ZeroDivision
                if abs(z) < 1e-12: continue
                
                gamma = u(1 + z * Y) - 1
                sigma = gamma / z
                ll = self._log_likelihood(Y, gamma, sigma)
                
                if ll > ll_best:
                    gamma_best = gamma
                    sigma_best = sigma
                    ll_best = ll

            return gamma_best, sigma_best, ll_best


    def _quantile(self, side, gamma, sigma):
            """
            حساب الـ Extreme Quantile (العتبة النهائية)
            دي المعادلة اللي بدمج الـ Initial Threshold مع توزيع GPD
            """
            # r هو النسبة بين احتمال المخاطرة (q) ونسبة الـ Peaks اللي لقطناها
            # r = (n * q) / Nt
            if self.Nt[side] == 0:
                return self.init_threshold[side]
                
            r = (self.n * self.proba) / self.Nt[side]

            # الناحية اللي فوق (Upper Bound)
            if side == 'up':
                if abs(gamma) > 1e-8:
                    # المعادلة: t + (sigma/gamma) * ( (r^-gamma) - 1 )
                    return self.init_threshold['up'] + (sigma / gamma) * (np.power(r, -gamma) - 1)
                else:
                    # لو gamma بصفر بنستخدم الـ Exponential limit
                    return self.init_threshold['up'] - sigma * np.log(r)
            
            # الناحية اللي تحت (Lower Bound)
            elif side == 'down':
                if abs(gamma) > 1e-8:
                    # بنطرح هنا لأننا بننزل تحت الـ threshold السفلي
                    return self.init_threshold['down'] - (sigma / gamma) * (np.power(r, -gamma) - 1)
                else:
                    return self.init_threshold['down'] + sigma * np.log(r)
            
            else:
                print('❌ Error: The side must be either "up" or "down"')
                return None


    def run(self, with_alarm=True):
            """
            تشغيل الـ biSPOT على تدفق البيانات (Stream)
            """
            # التأكد إننا عملنا initialize الأول
            if self.n > self.init_data.size:
                print('⚠️ Warning: Algorithm already run. Please re-initialize if needed.')
                # بنكمل عادي بس بندي تنبيه

            # مخازن للنتائج
            thup = []
            thdown = []
            alarm = []
            
            # بنسجل الـ alarm في الـ object نفسه برضه عشان الـ __str__
            self.alarm = alarm 

            # اللوب بتمشي على كل نقطة في الـ stream
            # tqdm بتعمل شريط تقدم (Progress Bar) شكلة شيك جداً في الـ console
            for i in tqdm.tqdm(range(self.data.size)):
                val = self.data[i]
                
                # --- الجزء الأول: مراقبة الناحية العلوية (UP) ---
                if val > self.extreme_quantile['up']:
                    if with_alarm:
                        alarm.append(i)
                    else:
                        # لو مش عايزين إنذار، بنعتبرها نقطة طبيعية ونحدث العتبة
                        self.peaks['up'] = np.append(self.peaks['up'], val - self.init_threshold['up'])
                        self.Nt['up'] += 1
                        g, s, l = self._grimshaw('up')
                        self.extreme_quantile['up'] = self._quantile('up', g, s)
                
                elif val > self.init_threshold['up']:
                    # نقطة بين العتبة العادية والعتبة القصوى (Peak طبيعي)
                    self.peaks['up'] = np.append(self.peaks['up'], val - self.init_threshold['up'])
                    self.Nt['up'] += 1
                    g, s, l = self._grimshaw('up')
                    self.extreme_quantile['up'] = self._quantile('up', g, s)

                # --- الجزء الثاني: مراقبة الناحية السفلية (DOWN) ---
                elif val < self.extreme_quantile['down']:
                    if with_alarm:
                        alarm.append(i)
                    else:
                        self.peaks['down'] = np.append(self.peaks['down'], self.init_threshold['down'] - val)
                        self.Nt['down'] += 1
                        g, s, l = self._grimshaw('down')
                        self.extreme_quantile['down'] = self._quantile('down', g, s)
                
                elif val < self.init_threshold['down']:
                    self.peaks['down'] = np.append(self.peaks['down'], self.init_threshold['down'] - val)
                    self.Nt['down'] += 1
                    g, s, l = self._grimshaw('down')
                    self.extreme_quantile['down'] = self._quantile('down', g, s)

                # في كل الحالات بنزود عداد المشاهدات
                self.n += 1
                
                # تسجيل العتبات الحالية للرسم البياني لاحقاً
                thup.append(self.extreme_quantile['up'])
                thdown.append(self.extreme_quantile['down'])

            return {'upper_thresholds': thup, 'lower_thresholds': thdown, 'alarms': alarm}




    def plot(self, run_results, with_alarm=True):
            """
            رسم نتائج الـ biSPOT (البيانات، العتبات، والإنذارات)
            """
            import matplotlib.pyplot as plt # التأكد من استدعاء المكتبة
            
            # تعريف الألوان لو مش متعرفة بره الكلاس
            air_force_blue = '#5D8AA8'
            deep_saffron = '#FF9933'
            
            x = range(self.data.size)
            K = run_results.keys()
            fig_list = []

            # 1. رسم البيانات الأساسية (The Stream)
            ts_fig, = plt.plot(x, self.data, color=air_force_blue, label='Robot Data', alpha=0.8)
            fig_list.append(ts_fig)

            # 2. رسم العتبة العلوية (Upper Threshold)
            if 'upper_thresholds' in K:
                thup = run_results['upper_thresholds']
                uth_fig, = plt.plot(x, thup, color=deep_saffron, lw=2, ls='--', label='Upper Threshold')
                fig_list.append(uth_fig)

            # 3. رسم العتبة السفلية (Lower Threshold)
            if 'lower_thresholds' in K:
                thdown = run_results['lower_thresholds']
                lth_fig, = plt.plot(x, thdown, color=deep_saffron, lw=2, ls='--', label='Lower Threshold')
                fig_list.append(lth_fig)

            # 4. رسم الإنذارات (Alarms) باللون الأحمر
            if with_alarm and ('alarms' in K):
                alarm_indices = run_results['alarms']
                if len(alarm_indices) > 0:
                    al_fig = plt.scatter(alarm_indices, self.data[alarm_indices], 
                                        color='red', marker='x', s=50, label='Alarms', zorder=5)
                    fig_list.append(al_fig)

            # تنسيق الرسمة
            plt.xlim((0, self.data.size))
            plt.title('biSPOT Anomaly Detection for Robot Arm')
            plt.xlabel('Time Steps')
            plt.ylabel('Sensor Value')
            plt.legend(loc='best') # إضافة دليل الرسمة
            plt.grid(True, linestyle=':', alpha=0.6) # إضافة شبكة خفيفة
            
            return fig_list


"""
================================= WITH DRIFT ==================================
"""



def backMean(X, d):
    """
    حساب المتوسط المتحرك (Moving Average)
    
    Parameters
    ----------
    X : numpy.array
        الداتا الخام اللي جاية من الحساسات
    d : int
        حجم النافذة (Window Size) - كل ما يكبر، الـ Smoothing يزيد
    """
    # تحويل الداتا لـ NumPy Array لو مش كدة
    X = np.asarray(X)
    M = []
    
    # 1. حساب أول نافذة (Initial Window)
    w = X[:d].sum()
    M.append(w / d)
    
    # 2. التحرك بالنافذة (Sliding)
    # بنشيل أول عنصر دخل ونزود العنصر الجديد اللي عليه الدور
    for i in range(d, len(X)):
        w = w - X[i - d] + X[i]
        M.append(w / d)
    
    return np.array(M)


class dSPOT:
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper-bound)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    depth : int
        Number of observations to compute the moving average
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q, depth):
            """
            التهيئة (Initialization) لكلاس dSPOT
            q: احتمال المخاطرة (Risk level)، مثلاً 10^-4
            depth: حجم الـ Moving Window اللي بنحسب فيه المتوسط
            """
            self.proba = q
            self.depth = depth
            
            # --- متغيرات الداتا ---
            self.data = None        # الـ Stream اللي بنراقبه
            self.init_data = None   # داتا المعايرة (Training Batch)
            self.W = None           # الـ Window الحالية اللي بنحسب منها الـ Mean (لمستنا هنا)

            # --- متغيرات العتبات والنتائج ---
            self.extreme_quantile = None   # العتبة النهائية (Zq)
            self.init_threshold = None      # العتبة الأولية الثابتة للفروقات (t)
            self.extreme_quantiles = []    # مخزن لكل العتبات اللي اتحسبت (عشان الـ Plot)
            
            # --- متغيرات الـ Peaks (الفروقات عن المتوسط) ---
            self.peaks = None             # الفروقات اللي عدت العتبة (Excesses)
            self.n = 0                    # إجمالي عدد النقاط
            self.Nt = 0                   # عدد الـ Peaks
            
            # --- مخزن الإنذارات ---
            self.alarm = [] 
            
            # --- معاملات GPD ---
            self.gamma = None
            self.sigma = None

            # لمسة إضافية: تحديد إننا شغالين Upper-bound
            self.side = 'up'



    def __str__(self):
            """
            عرض حالة الموديل بشكل منظم (التقرير الملخص)
            """
            s = '\n' + '='*40 + '\n'
            s += '⭐ dSPOT: Streaming Anomaly Detection ⭐\n'
            s += '='*40 + '\n'
            s += f'🔹 Detection level (q) : {self.proba}\n'
            s += f'🔹 Window depth (d)    : {self.depth}\n'
            s += '-'*40 + '\n'

            # 1. حالة البيانات (Data Status)
            if self.data is not None:
                s += '✅ Data Status: Imported\n'
                s += f'\t- Initialization Batch : {self.init_data.size} points\n'
                s += f'\t- Streaming Data       : {self.data.size} points\n'
            else:
                s += '❌ Data Status: No data imported yet.\n'
                s += '='*40 + '\n'
                return s

            # 2. حالة المعايرة (Initialization Status)
            if self.n == 0:
                s += '❌ Algorithm Status: Not initialized.\n'
            else:
                s += '✅ Algorithm Status: Initialized\n'
                s += f'\t- Initial Threshold (t) : {self.init_threshold:.4f}\n'
                s += f'\t- Number of Peaks (Nt)  : {self.Nt}\n'

                # 3. حالة التشغيل (Run Status)
                # r هو عدد النقاط اللي الموديل عالجها فعلياً في الـ Stream
                # n هو إجمالي النقاط (init + stream processed)
                r = self.n - (self.init_data.size - self.depth)
                
                if r > 0:
                    s += '🚀 Run Status: Completed/Running\n'
                    s += f'\t- Processed Points : {r} points\n'
                    s += f'\t- Triggered Alarms : {len(self.alarm)} ({ (len(self.alarm)/r)*100 :.2f}%)\n'
                    s += f'\t- Current Quantile : {self.extreme_quantile:.4f}\n'
                else:
                    s += '⏳ Run Status: Ready to run (Waiting for stream)\n'
            
            s += '='*40 + '\n'
            return s


    def fit(self, init_data, data):
            """
            استيراد وتجهيز البيانات لكائن dSPOT
            init_data: داتا المعايرة (ممكن تكون عدد، نسبة، أو array)
            data: داتا الـ stream الأساسية
            """
            import pandas as pd
            import numpy as np

            # --- لمسة 1: دالة التحويل الموحدة ---
            def to_ndarray(obj):
                if isinstance(obj, list):
                    return np.array(obj)
                if isinstance(obj, np.ndarray):
                    return obj
                if isinstance(obj, pd.Series):
                    return obj.values
                return None

            # تحويل الداتا الأساسية
            tmp_data = to_ndarray(data)
            if tmp_data is None:
                print(f'❌ Error: Data format ({type(data)}) is not supported')
                return
            
            # --- لمسة 2: تنظيف البيانات (لمسة CoreX) ---
            # بنشيل أي NaN أو أرقام غير منطقية عشان الـ Optimizer ميفصلش
            self.data = tmp_data[np.isfinite(tmp_data)]

            # --- لمسة 3: تقسيم بيانات المعايرة بذكاء ---
            # الحالة أ: لو باعتة عدد نقاط معين (int)
            if isinstance(init_data, int):
                if init_data < self.depth:
                    print(f'⚠️ Warning: init_data size ({init_data}) < depth ({self.depth})!')
                self.init_data = self.data[:init_data]
                self.data = self.data[init_data:]
                
            # الحالة ب: لو باعتة نسبة مئوية (float)
            elif isinstance(init_data, float) and (0 < init_data < 1):
                r = int(init_data * self.data.size)
                self.init_data = self.data[:r]
                self.data = self.data[r:]
                
            # الحالة ج: لو باعتة array جاهزة
            else:
                self.init_data = to_ndarray(init_data)
                if self.init_data is None:
                    print('❌ Error: The initial data cannot be set')
                    return

            print(f"✅ Data Fitted: {self.init_data.size} points for init, {self.data.size} for stream.")

    def add(self, data):
            """
            إضافة بيانات جديدة للـ Stream الحالي
            """
            import pandas as pd
            import numpy as np

            # --- لمسة 1: توحيد النوع (Re-using logic) ---
            if isinstance(data, list):
                new_data = np.array(data)
            elif isinstance(data, np.ndarray):
                new_data = data
            elif isinstance(data, pd.Series):
                new_data = data.values
            else:
                print(f'❌ Error: Cannot add data of type {type(data)}')
                return

            # --- لمسة 2: الفلترة (لمسة الأمان) ---
            # بنضمن إننا مش بنزود NaN بوظ الحسابات اللي جاية
            new_data = new_data[np.isfinite(new_data)]

            # --- لمسة 3: التحديث الذكي ---
            if self.data is None:
                self.data = new_data
            else:
                self.data = np.concatenate([self.data, new_data])
                
            print(f"➕ Added {new_data.size} new points. Total stream: {self.data.size}")


    def initialize(self, verbose=True):
            """
            مرحلة المعايرة: تحويل البيانات لـ 'فروقات' وحساب أول عتبة قصوى
            """
            # 1. حساب عدد النقاط الفعلية بعد استقطاع الـ depth
            n_init = self.init_data.size - self.depth

            # 2. لمسة الـ dSPOT: التعامل مع الفروقات (Local Fluctuations)
            # بنجيب المتوسط المتحرك ونطرحه من الداتا الأصلية
            M = backMean(self.init_data, self.depth)
            
            # T هي الـ 'Residuals' أو الفروقات.. دي اللي بنراقبها فعلياً
            # M[:-1] عشان ناخد المتوسطات اللي قبل النقطة الحالية
            T = self.init_data[self.depth:] - M[:-1] 

            # 3. حساب العتبة الأولية (Initial Threshold)
            # بنرتب الفروقات وناخد الـ Quantile رقم 98% (ده المتعارف عليه في الأبحاث)
            S = np.sort(T)
            self.init_threshold = S[int(0.98 * n_init)]

            # 4. تحديد الـ Peaks (القيم اللي عدت العتبة)
            # بنخزن بس "مقدار الزيادة" عن العتبة (Excesses)
            self.peaks = T[T > self.init_threshold] - self.init_threshold
            self.Nt = self.peaks.size
            self.n = n_init

            if verbose:
                print(f'✅ Initialization Started...')
                print(f'   - Initial threshold (t) : {self.init_threshold:.4f}')
                print(f'   - Number of peaks (Nt)  : {self.Nt}')
                print('   - Running Grimshaw MLE ... ', end='')

            # 5. حساب معاملات GPD (gamma و sigma)
            # تريكة: لازم نتأكد إن الدوال دي موجودة جوه الكلاس
            g, s, l = self._grimshaw()
            self.gamma, self.sigma = g, s # بنخزنهم في الـ object
            
            # 6. حساب الـ Extreme Quantile (العتبة النهائية اللي بتطلع Alarm)
            self.extreme_quantile = self._quantile(g, s)

            if verbose:
                print('[Done]')
                print(f'\tγ (Shape) = {g:.4f}')
                print(f'\tσ (Scale) = {s:.4f}')
                print(f'\tL (Log-Likelihood) = {l:.4f}')
                print(f'🎯 Extreme quantile (z_q) : {self.extreme_quantile:.4f}')

            return


    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
            """
            البحث عن الجذور المحتملة للدالة (Finding roots)
            بلمستنا: خليناها Static عشان تقدر تشتغل كدالة مساعدة من غير زحمة الـ self
            """
            from scipy.optimize import minimize # لازم نستدعيها هنا أو في أول الملف
            import numpy as np

            # 1. تجهيز نقاط البداية (Starting points)
            if method == 'regular':
                step = (bounds[1] - bounds[0]) / (npoints + 1)
                X0 = np.arange(bounds[0] + step, bounds[1], step)
            elif method == 'random':
                X0 = np.random.uniform(bounds[0], bounds[1], npoints)
            else:
                X0 = np.array([(bounds[0] + bounds[1]) / 2]) # حالة احتياطية

            # 2. تعريف دالة الهدف (Objective Function)
            # إحنا بنحول مشكلة البحث عن جذر لمشكلة Optimization (تقليل المربعات)
            def objFun(X, f, j_func):
                fx = np.array([f(x) for x in X])
                g = np.sum(fx ** 2) # مجموع المربعات (عايزين نوصل لصفر)
                
                # حساب الـ Gradient (الاشتقاق) عشان نسرع الحل
                j = 2 * fx * np.array([j_func(x) for x in X])
                return g, j

            # 3. عملية البحث باستخدام L-BFGS-B (دي أسرع وأدق حاجة للـ Bounds)
            opt = minimize(lambda X: objFun(X, fun, jac), X0,
                        method='L-BFGS-B',
                        jac=True, 
                        bounds=[bounds] * len(X0))

            # 4. تنقية النتائج
            X = np.round(opt.x, decimals=5) # تقريب عشان نشيل الفروقات البسيطة
            return np.unique(X) # بنرجع القيم الفريدة بس (بدون تكرار)

    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
            """
            حساب الـ Log-Likelihood لتوزيع GPD
            بلمستنا: خليناها Static و Robust ضد القيم الصفرية
            """
            import numpy as np
            
            n = Y.size
            # تأكيد إن الـ sigma دايماً موجبة عشان الـ log ميزعلش
            if sigma <= 0:
                return -np.inf 

            if abs(gamma) > 1e-8: # لو الجاما مش بصفر
                tau = gamma / sigma
                # تريكة: لازم نضمن إن (1 + tau * Y) دايماً موجبة
                arg = 1 + tau * Y
                if np.any(arg <= 0):
                    return -np.inf
                    
                L = -n * np.log(sigma) - (1 + (1 / gamma)) * (np.log(arg)).sum()
            else:
                # حالة خاصة: لما gamma تقترب من الصفر (توزيع إكسبوننشال)
                # الـ Likelihood بتكون: n*log(1/sigma) - (1/sigma)*sum(Y)
                L = -n * np.log(sigma) - (1 / sigma) * Y.sum()
                
            return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
            """
            حساب معاملات GPD باستخدام خدعة Grimshaw
            """
            # الدوال المساعدة للحسابات الرياضية
            def u(s):
                # بنستخدم np.clip عشان نضمن إننا مش بنحسب log لصفر
                return 1 + np.log(np.maximum(s, epsilon)).mean()

            def v(s):
                return np.mean(1 / s)

            def w(Y, t):
                s = 1 + t * Y
                return u(s) * v(s) - 1

            def jac_w(Y, t):
                s = 1 + t * Y
                us = u(s)
                vs = v(s)
                # مشتقات الدالة عشان نسرع الـ Root Finder
                jac_us = (1 / t) * (1 - vs)
                jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
                return us * jac_vs + vs * jac_us

            # 1. تحديد حدود البحث (The Bounds)
            Ym = self.peaks.min()
            YM = self.peaks.max()
            Ymean = self.peaks.mean()

            a = -1 / YM
            if abs(a) < 2 * epsilon:
                epsilon = abs(a) / n_points

            a = a + epsilon
            b = 2 * (Ymean - Ym) / (Ymean * Ym)
            c = 2 * (Ymean - Ym) / (Ym ** 2)

            # 2. البحث عن الجذور (Zeros)
            # تريكة CoreX: بننادي الـ static methods من الكلاس الحالي (self)
            left_zeros = self._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (a + epsilon, -epsilon),
                                        n_points, 'regular')

            right_zeros = self._rootsFinder(lambda t: w(self.peaks, t),
                                            lambda t: jac_w(self.peaks, t),
                                            (b, c),
                                            n_points, 'regular')

            # دمج كل الحلول الممكنة
            zeros = np.concatenate((left_zeros, right_zeros))

            # 3. اختيار أفضل "جذر" بيحقق أعلى Log-Likelihood
            # البداية دايماً بنفترض إن التوزيع Exponential (gamma = 0)
            gamma_best = 0
            sigma_best = Ymean
            ll_best = self._log_likelihood(self.peaks, gamma_best, sigma_best)

            for z in zeros:
                # تحويل الجذر (z) لمعاملات التوزيع
                gamma = u(1 + z * self.peaks) - 1
                sigma = gamma / z
                
                # تقييم الحل
                ll = self._log_likelihood(self.peaks, gamma, sigma)
                if ll > ll_best:
                    gamma_best = gamma
                    sigma_best = sigma
                    ll_best = ll

            return gamma_best, sigma_best, ll_best



    def _quantile(self, gamma, sigma):
            """
            حساب الـ Quantile النهائي (العتبة القصوى)
            gamma, sigma: معاملات توزيع GPD اللي لسه حاسبينهم في Grimshaw
            """
            import numpy as np

            # 1. حساب النسبة (r)
            # دي بتمثل احتمال ظهور Peak جديد بناءً على اللي شفناه قبل كدة
            # Nt: عدد الـ Peaks القديمة، n: إجمالي النقط، proba: مستوى المخاطرة اللي حددتيه
            r = (self.n * self.proba) / self.Nt
            
            # 2. حساب العتبة بناءً على قيمة Gamma
            # لو الجاما مش بصفر (توزيع Pareto العام)
            if abs(gamma) > 1e-8:
                # بنستخدم np.power عشان السرعة والدقة مع الـ Numpy arrays
                # المعادلة دي بتحسب الـ 'Tail' بتاع التوزيع
                zq = self.init_threshold + (sigma / gamma) * (np.power(r, -gamma) - 1)
            else:
                # لو الجاما بصفر (توزيع Exponential)
                # بنستخدم np.log عشان نضمن إن الكود شغال صح
                zq = self.init_threshold - sigma * np.log(r)
                
            return zq


    def run(self, with_alarm=True):
            """
            تشغيل الـ dSPOT على بيانات الـ Stream
            """
            import tqdm
            import numpy as np

            # تأكيد إننا مكررناش الـ run من غير ما نصفر الـ n
            if self.n > self.init_data.size:
                print('⚠️ Warning: الموديل اشتغل قبل كدة، يفضل تعملي initialize تاني.')
                #return {} # ممكن نشيل دي لو عايزين نكمل عادي

            # تجهيز الـ Moving Window (W)
            W = self.init_data[-self.depth:].tolist() # بنحولها لـ list عشان الـ pop والـ append أسرع

            thresholds = [] # مخزن الـ Zq + Mi
            self.alarm = [] # تصفير الإنذارات قبل البدء

            print(f"🚀 Streaming started on {self.data.size} points...")
            
            for i in tqdm.tqdm(range(self.data.size)):
                # 1. حساب المتوسط المحلي الحالي (Local Mean)
                Mi = np.mean(W)
                current_value = self.data[i]
                diff = current_value - Mi # الفرق اللي بنراقبه

                # 2. فحص حالة الإنذار (Anomaly)
                if diff > self.extreme_quantile:
                    if with_alarm:
                        self.alarm.append(i)
                    else:
                        # لو مش عايزين إنذار، بنعتبرها Peak طبيعي ونحدث الموديل (تريكة الـ Assumption)
                        self.peaks = np.append(self.peaks, diff - self.init_threshold)
                        self.Nt += 1
                        # تحديث المعاملات والعتبة
                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)

                # 3. فحص حالة الـ Peak (قيمة عالية بس مش شاذة)
                elif diff > self.init_threshold:
                    self.peaks = np.append(self.peaks, diff - self.init_threshold)
                    self.Nt += 1
                    # تحديث المعاملات والعتبة (Adaptation)
                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)

                # 4. تحديث العداد والنافذة (Sliding Window)
                self.n += 1
                W.pop(0)         # شيل أقدم نقطة
                W.append(current_value) # ضيف أحدث نقطة

                # تسجيل العتبة النهائية (Zq + Mi) عشان الرسم
                thresholds.append(self.extreme_quantile + Mi)

            return {'thresholds': thresholds, 'alarms': self.alarm}



    def plot(self, run_results, with_alarm=True):
            """
            رسم نتائج الـ dSPOT بشكل احترافي
            """
            import matplotlib.pyplot as plt
            
            # تعريف الألوان لو مش موجودة (لمسة جمالية)
            colors = {
                'data': '#5D8AA8',      # Air Force Blue
                'threshold': '#FF9933', # Deep Saffron
                'alarm': '#FF0000'      # Red
            }

            plt.figure(figsize=(12, 6))
            x = range(self.data.size)
            K = run_results.keys()

            # 1. رسم الداتا الأساسية (Sensor Stream)
            ts_fig, = plt.plot(x, self.data, color=colors['data'], label='Sensor Data', alpha=0.8)
            fig = [ts_fig]

            # 2. رسم العتبة الديناميكية (Dynamic Threshold)
            if 'thresholds' in K:
                th = run_results['thresholds']
                th_fig, = plt.plot(x, th, color=colors['threshold'], lw=2, ls='--', label='Dynamic Threshold ($Z_q$)')
                fig.append(th_fig)

            # 3. رسم الإنذارات (Alarms) كـ Scatter points
            if with_alarm and ('alarms' in K):
                alarm_idx = run_results['alarms']
                if len(alarm_idx) > 0:
                    alarm_fig = plt.scatter(alarm_idx, self.data[alarm_idx], 
                                            color=colors['alarm'], label='Anomaly/Alarm', 
                                            zorder=5, s=40, edgecolors='black')
                    # fig.append(alarm_fig) # الـ scatter بيرجع PathCollection مش Line2D

            # --- لمسات التنسيق النهائية ---
            plt.title(f'dSPOT Anomaly Detection (q={self.proba})', fontsize=14)
            plt.xlabel('Time Steps', fontsize=12)
            plt.ylabel('Sensor Value / Residuals', fontsize=12)
            plt.xlim((0, self.data.size))
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='upper right')
            
            # تريكة: تحسين شكل الرسمة عشان متبقاش مضغوطة
            plt.tight_layout()
            plt.show()

            return fig
            


"""
=========================== DRIFT & DOUBLE BOUNDS =============================
"""


class bidSPOT:
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper and lower bounds)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    depth : int
        Number of observations to compute the moving average
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4, depth=10):
            """
            Initialization لـ bidSPOT (مراقبة ثنائية الاتجاه)
            """
            self.proba = q
            self.depth = depth
            self.data = None
            self.init_data = None
            self.n = 0
            self.alarm = [] # مهم جداً عشان نسجل الـ indexes اللي حصل فيها مشاكل

            # بدل ما نكرر dict.copy كل شوية، هنعملها بشكل Pythonic أنضف
            # 'up' للزيادات المفاجئة و 'down' للانخفاضات المفاجئة
            sides = ['up', 'down']
            
            self.extreme_quantile = {side: None for side in sides}
            self.init_threshold   = {side: None for side in sides}
            self.peaks            = {side: None for side in sides}
            self.gamma            = {side: None for side in sides}
            self.sigma            = {side: None for side in sides}
            self.Nt               = {side: 0 for side in sides}
            
            # ملحوظة: الـ Nt هنا هي عدد المرات اللي الداتا عدت فيها الـ Threshold الابتدائي



    def __str__(self):
            """
            عرض حالة الموديل بشكل منظم (Dashboard view)
            """
            s = '--- bidSPOT (Bi-directional DSPOT) Status ---\n'
            s += f'Detection level (q) : {self.proba}\n'
            
            # 1. فحص استيراد البيانات
            if self.data is not None:
                s += 'Data imported       : Yes\n'
                s += f'\t- Initial batch : {self.init_data.size} values\n'
                s += f'\t- Stream size   : {self.data.size} values\n'
            else:
                s += 'Data imported       : No\n'
                return s # بنوقف هنا لو مفيش داتا أصلاً

            # 2. فحص المعايرة (Initialization)
            if self.n == 0:
                s += 'Algorithm status    : Not Initialized\n'
            else:
                s += 'Algorithm status    : Initialized ✅\n'
                s += f'\t- Initial thresholds: Up={self.init_threshold["up"]:.4f}, Down={self.init_threshold["down"]:.4f}\n'
                s += f'\t- Number of peaks   : Up={self.Nt["up"]}, Down={self.Nt["down"]}\n'

                # 3. فحص التشغيل (Run phase)
                # r هو عدد النقط اللي الموديل شافها في الـ Stream فعلياً
                r = self.n - (self.init_data.size - self.depth)
                if r > 0:
                    s += 'Execution status    : Run Completed/Active\n'
                    s += f'\t- Observations processed : {r} ({100 * r / self.n:.2f} %)\n'
                    # بنعرض الـ Alarms لو القائمة موجودة
                    alarm_count = len(self.alarm) if hasattr(self, 'alarm') else 0
                    s += f'\t- Triggered alarms       : {alarm_count} ({100 * alarm_count / r:.2f} %)\n'
                else:
                    s += 'Execution status    : Ready to Run (Waiting for stream)\n'
                    s += f'\t- Upper Extreme Quantile : {self.extreme_quantile["up"]:.4f}\n'
                    s += f'\t- Lower Extreme Quantile : {self.extreme_quantile["down"]:.4f}\n'
            
            s += '-------------------------------------------'
            return s


    def fit(self, init_data, data):
            """
            استيراد البيانات وتجهيزها للمعايرة والتشغيل
            """
            import numpy as np
            import pandas as pd

            # 1. تجهيز الـ Data الأساسية (الـ Stream اللي هنراقبه)
            if isinstance(data, list):
                self.data = np.array(data)
            elif isinstance(data, np.ndarray):
                self.data = data
            elif isinstance(data, pd.Series):
                self.data = data.values
            else:
                print(f'❌ النوع ده ({type(data)}) مش مدعوم يا هندسة!')
                return

            # 2. تجهيز الـ init_data (بيانات المعايرة)
            if isinstance(init_data, list):
                self.init_data = np.array(init_data)
            elif isinstance(init_data, np.ndarray):
                self.init_data = init_data
            elif isinstance(init_data, pd.Series):
                self.init_data = init_data.values
            
            # لو المستخدم باعت رقم صحيح (معناه خد أول N نقطة للمعايرة)
            elif isinstance(init_data, int):
                self.init_data = self.data[:init_data]
                self.data = self.data[init_data:]
                
            # لو المستخدم باعت نسبة مئوية (مثلاً 0.2 يعني خد أول 20%)
            # صلحنا الـ & وخليناها and
            elif isinstance(init_data, float) and (0 < init_data < 1):
                r = int(init_data * self.data.size)
                self.init_data = self.data[:r]
                self.data = self.data[r:]
            else:
                print('❌ مش قادر أحدد بيانات المعايرة (init_data) صح.')
                return
            
            # لمسة CoreX: تنظيف سريع من الـ NaNs عشان الـ Grimshaw ميهنجش
            self.data = self.data[np.isfinite(self.data)]
            self.init_data = self.init_data[np.isfinite(self.init_data)]



    def add(self, data):
            """
            إضافة بيانات جديدة للـ Stream اللي الموديل شغال عليه
            """
            import numpy as np
            import pandas as pd

            # 1. التأكد من نوع البيانات وتحويلها لـ Numpy
            if isinstance(data, list):
                new_data = np.array(data)
            elif isinstance(data, np.ndarray):
                new_data = data
            elif isinstance(data, pd.Series):
                new_data = data.values
            else:
                print(f'⚠️ النوع ده ({type(data)}) مش مدعوم في الإضافة يا هندسة!')
                return

            # 2. التأمين: لو نادى add قبل fit (يعني self.data لسه None)
            if self.data is None:
                self.data = new_data
            else:
                # دمج البيانات القديمة مع الجديدة
                self.data = np.append(self.data, new_data)
            
            # تنظيف سريع برضه عشان نضمن إن مفيش NaN دخلت في النص
            self.data = self.data[np.isfinite(self.data)]



    def initialize(self, verbose=True):
            """
            مرحلة المعايرة (Calibration) للاتجاهين
            """
            import numpy as np

            # 1. حساب الـ Residuals (T)
            # n_init هو عدد النقط المتاحة بعد خصم الـ depth بتاع المتوسط المتحرك
            n_init = self.init_data.size - self.depth

            # بنستخدم الدالة المساعدة backMean اللي حسبناها قبل كدة
            M = backMean(self.init_data, self.depth)
            
            # T هي الفرق بين كل نقطة والمتوسط اللي قبلها (Detrending)
            T = self.init_data[self.depth:] - M[:-1]

            # 2. تحديد العتبات الابتدائية (Empirical Quantiles)
            S = np.sort(T)
            # الـ Upper عند 98% والـ Lower عند 2%
            self.init_threshold['up'] = S[int(0.98 * n_init)]
            self.init_threshold['down'] = S[int(0.02 * n_init)]

            # 3. استخراج الـ Peaks الابتدائية
            # للاتجاه اللي فوق: النقط اللي عدت العتبة العليا
            self.peaks['up'] = T[T > self.init_threshold['up']] - self.init_threshold['up']
            
            # للاتجاه اللي تحت: النقط اللي نزلت عن العتبة السفلى
            # تريكة CoreX: بنضرب في سالب عشان نحولها لـ "Peaks" موجبة فيدخلوا الـ GPD صح
            self.peaks['down'] = -(T[T < self.init_threshold['down']] - self.init_threshold['down'])

            self.Nt['up'] = self.peaks['up'].size
            self.Nt['down'] = self.peaks['down'].size
            self.n = n_init

            if verbose:
                print(f'✅ Initialization done. n={n_init}')
                print(f'   Upper Threshold: {self.init_threshold["up"]:.4f} ({self.Nt["up"]} peaks)')
                print(f'   Lower Threshold: {self.init_threshold["down"]:.4f} ({self.Nt["down"]} peaks)')
                print('📊 Estimating GPD parameters (Grimshaw)... ', end='')

            # 4. تشغيل الـ Grimshaw لكل اتجاه
            l = {'up': None, 'down': None}
            for side in ['up', 'down']:
                g, s, log_lik = self._grimshaw(side)
                self.extreme_quantile[side] = self._quantile(side, g, s)
                self.gamma[side] = g
                self.sigma[side] = s
                l[side] = log_lik

            # 5. عرض النتائج بشكل شيك للمناقشة
            if verbose:
                print('[Done]')
                ltab = 20
                header = f"{'Parameters':>{ltab}}{'Upper':>{ltab}}{'Lower':>{ltab}}"
                line = '-' * (ltab * 3)
                print(f'\t{header}\n\t{line}')
                print(f"\t{'Gamma (γ)':>{ltab}}{self.gamma['up']:>{ltab}.4f}{self.gamma['down']:>{ltab}.4f}")
                print(f"\t{'Sigma (σ)':>{ltab}}{self.sigma['up']:>{ltab}.4f}{self.sigma['down']:>{ltab}.4f}")
                print(f"\t{'Ext. Quantile':>{ltab}}{self.extreme_quantile['up']:>{ltab}.4f}{self.extreme_quantile['down']:>{ltab}.4f}")
                print(f'\t{line}')


    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
            """
            البحث عن الجذور المحتملة لدالة Grimshaw
            fun: الدالة (w), jac: المشتقة (jac_w)
            """
            import numpy as np
            from scipy.optimize import minimize

            # 1. تحديد نقط البداية (Starting points)
            if method == 'regular':
                step = (bounds[1] - bounds[0]) / (npoints + 1)
                X0 = np.arange(bounds[0] + step, bounds[1], step)
            elif method == 'random':
                X0 = np.random.uniform(bounds[0], bounds[1], npoints)
            else:
                X0 = np.array([(bounds[0] + bounds[1]) / 2])

            # 2. تعريف دالة الهدف (Objective Function)
            # إحنا بنحاول نخلي f(x) تقرب من الصفر، فبنصغر f(x)^2
            def objFun(X, f, j_func):
                fx = np.array([f(x) for x in X])
                g = np.sum(fx ** 2) # مجموع المربعات
                
                # الـ Gradient (المشتقة)
                jx = 2 * fx * np.array([j_func(x) for x in X])
                return g, jx

            # 3. عملية الـ Optimization (L-BFGS-B)
            # دي خوارزمية سريعة جداً وبتحترم الحدود (bounds)
            res = minimize(lambda X: objFun(X, fun, jac), 
                        X0,
                        method='L-BFGS-B',
                        jac=True, 
                        bounds=[bounds] * len(X0))

            # 4. تنظيف النتايج
            # لازم نخزن الـ round عشان np.unique تشتغل صح وتلم الجذور القريبة من بعض
            X = np.round(res.x, decimals=5)
            return np.unique(X)


    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
            """
            حساب الـ Log-Likelihood لتوزيع GPD
            دي اللي بتقولنا الموديل "واثق" قد إيه في حساباته
            """
            import numpy as np
            
            n = Y.size
            # حماية من القسمة على صفر أو لو sigma مش منطقية
            if sigma <= 0:
                return -np.inf

            if abs(gamma) > 1e-9: # حالة إن جاما مش بصفر
                tau = gamma / sigma
                # لازم نضمن إن (1 + tau * Y) دايماً موجب عشان الـ log
                if np.any(1 + tau * Y <= 0):
                    return -np.inf
                
                L = -n * np.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
            else:
                # حالة جاما = 0 (توزيع Exponential)
                # المعادلة هنا بتتبسط لـ n * (1 + log(mean)) بالسالب أو حسب الـ derivation
                L = -n * np.log(Y.mean()) - n
                
            return L

    def _grimshaw(self, side, epsilon=1e-8, n_points=10):
            """
            حساب معاملات GPD باستخدام خدعة Grimshaw
            side: 'up' (للمشاكل فوق) أو 'down' (للمشاكل تحت)
            """
            import numpy as np

            # 1. تعريف الدوال المساعدة (Internal Functions)
            def u(s):
                return 1 + np.log(s).mean()

            def v(s):
                return np.mean(1 / s)

            def w(Y, t):
                s = 1 + t * Y
                return u(s) * v(s) - 1

            def jac_w(Y, t):
                s = 1 + t * Y
                us, vs = u(s), v(s)
                # المشتقات عشان الـ Roots Finder يكون سريع
                jac_us = (1 / t) * (1 - vs)
                jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
                return us * jac_vs + vs * jac_us

            # 2. تجهيز البيانات
            Y = self.peaks[side]
            if Y.size == 0:
                # لو مفيش Peaks، بنفترض معاملات افتراضية (Normal state)
                return 0, 1e-4, -np.inf

            Ym = Y.min()
            YM = Y.max()
            Ymean = Y.mean()

            # 3. تحديد فترات البحث عن الجذور (The Bounds)
            # دي فترات رياضية ثابتة في بحث Grimshaw
            a = -1 / YM
            if abs(a) < 2 * epsilon:
                epsilon = abs(a) / n_points

            a = a + epsilon
            # الفترات اللي بندور فيها على الحلول الممكنة لـ t
            b = 2 * (Ymean - Ym) / (Ymean * Ym) if Ymean * Ym != 0 else epsilon
            c = 2 * (Ymean - Ym) / (Ym ** 2) if Ym != 0 else epsilon

            # 4. البحث عن الجذور في الناحيتين (Left & Right)
            # بننادي الـ _rootsFinder اللي ظبطناها سوا
            left_zeros = self._rootsFinder(lambda t: w(Y, t),
                                        lambda t: jac_w(Y, t),
                                        (a + epsilon, -epsilon),
                                        n_points, 'regular')

            right_zeros = self._rootsFinder(lambda t: w(Y, t),
                                            lambda t: jac_w(Y, t),
                                            (b, c),
                                            n_points, 'regular')

            # دمج كل الحلول الممكنة
            zeros = np.concatenate((left_zeros, right_zeros))

            # 5. اختيار الحل الأفضل (The Champion)
            # بنبدأ بـ gamma=0 كحل افتراضي
            gamma_best = 0
            sigma_best = Ymean
            ll_best = self._log_likelihood(Y, gamma_best, sigma_best)

            # بنجرب كل "الجذور" اللي لقيناها ونشوف مين بيدي Log-Likelihood أعلى
            for z in zeros:
                if abs(z) < epsilon: continue # بنطنش الصفر لأنه متجرب
                
                s_val = 1 + z * Y
                if np.any(s_val <= 0): continue # حماية من الـ log(-x)
                    
                gamma = u(s_val) - 1
                sigma = gamma / z
                
                if sigma <= 0: continue # الـ sigma لازم تكون موجبة
                    
                ll = self._log_likelihood(Y, gamma, sigma)
                if ll > ll_best:
                    gamma_best = gamma
                    sigma_best = sigma
                    ll_best = ll

            return gamma_best, sigma_best, ll_best


    def _quantile(self, side, gamma, sigma):
            """
            حساب الـ Extreme Quantile (عتبة الإنذار النهائية)
            side: 'up' (للانحرافات العلوية) أو 'down' (للانحرافات السفلية)
            """
            import numpy as np

            # 1. حساب النسبة r (تكرار الـ Peaks بالنسبة للمخاطرة المطلوبة q)
            # لو مفيش Peaks خالص، بنحط رقم صغير جداً عشان المعادلة متبوظش
            nt_val = max(self.Nt[side], 1)
            r = (self.n * self.proba) / nt_val

            # 2. الحساب بناءً على الاتجاه (Side) وقيمة Gamma
            if side == 'up':
                # الاتجاه العلوي: العتبة = العتبة الابتدائية + الزيادة المتوقعة من GPD
                if abs(gamma) > 1e-9:
                    return self.init_threshold['up'] + (sigma / gamma) * (pow(r, -gamma) - 1)
                else:
                    # حالة Gamma = 0 (توزيع Exponential)
                    return self.init_threshold['up'] - sigma * np.log(r)

            elif side == 'down':
                # الاتجاه السفلي: العتبة = العتبة الابتدائية - النقص المتوقع
                # تذكري: إحنا في الـ initialize ضربنا الـ Peaks في سالب، فدلوقتي بنطرحها
                if abs(gamma) > 1e-9:
                    return self.init_threshold['down'] - (sigma / gamma) * (pow(r, -gamma) - 1)
                else:
                    return self.init_threshold['down'] + sigma * np.log(r)
            else:
                print('❌ خطأ: الاتجاه (side) لازم يكون up أو down')
                return None


    def run(self, with_alarm=True):
            """
            تشغيل الموديل على الـ Stream ومعالجة البيانات نقطة بنقطة
            """
            import tqdm
            import numpy as np

            # 1. حماية: نضمن إن الموديل حصل له Initialization
            if self.n == 0:
                print('⚠️ لازم تعمل initialize الأول يا هندسة قبل ما تشغل الـ run!')
                return {}

            # actual normal window: النافذة اللي بنحسب منها المتوسط المتحرك الحالي
            W = self.init_data[-self.depth:].copy()

            # قوايم لتخزين النتائج
            thup = []
            thdown = []
            self.alarm = [] # بنخزن الـ alarms في الـ object نفسه عشان الـ __str__
            
            # 2. الـ Loop الرئيسية على كل نقطة في الـ Stream
            for i in tqdm.tqdm(range(self.data.size), desc="Streaming Detection"):
                Mi = W.mean()
                Ni = self.data[i] - Mi # الـ Residual الحالي
                
                triggered = False # Flag عشان نعرف النقطة دي عملت Alarm ولا لأ

                # --- حالة الاتجاه العلوي (Up Side) ---
                if Ni > self.extreme_quantile['up']:
                    if with_alarm:
                        self.alarm.append(i)
                        triggered = True
                    else:
                        self._update_threshold(Ni, 'up')
                
                elif Ni > self.init_threshold['up']:
                    self._update_threshold(Ni, 'up')

                # --- حالة الاتجاه السفلي (Down Side) ---
                elif Ni < self.extreme_quantile['down']:
                    if with_alarm:
                        self.alarm.append(i)
                        triggered = True
                    else:
                        self._update_threshold(Ni, 'down')
                
                elif Ni < self.init_threshold['down']:
                    self._update_threshold(Ni, 'down')

                # 3. تحديث الحالة العامة
                if not triggered:
                    # لو مفيش Alarm، بنحدث الـ n وبندخل النقطة في الـ Window
                    self.n += 1
                    W = np.append(W[1:], self.data[i])
                else:
                    # في حالة الـ Alarm، غالباً مش بنحدث الـ Window عشان "منلوثش" المتوسط بقيمة خربانة
                    self.n += 1

                # تسجيل العتبات الحالية (بالإضافة للمتوسط عشان تترسم صح فوق الداتا)
                thup.append(self.extreme_quantile['up'] + Mi)
                thdown.append(self.extreme_quantile['down'] + Mi)

            return {'upper_thresholds': thup, 'lower_thresholds': thdown, 'alarms': self.alarm}

    def _update_threshold(self, Ni, side):
            """ دالة مساعدة لتحديث العتبات (Refactoring للكود المتكرر) """
            import numpy as np
            # إضافة الـ Peak الجديد
            peak_val = Ni - self.init_threshold[side] if side == 'up' else -(Ni - self.init_threshold[side])
            self.peaks[side] = np.append(self.peaks[side], peak_val)
            self.Nt[side] += 1
            
            # إعادة حساب المعايرة (Re-calibration)
            g, s, l = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)


    def plot(self, run_results, with_alarm=True):
            """
            رسم نتايج الـ biDSPOT وتوضيح الـ Alarms
            """
            import matplotlib.pyplot as plt
            import numpy as np

            # تعريف الألوان (CoreX Style)
            data_color = '#003049'     # أزرق غامق للداتا
            threshold_color = '#f77f00' # برتقالي للعتبات (Thresholds)
            alarm_color = '#d62828'     # أحمر فاقع للـ Alarms

            plt.figure(figsize=(12, 6))
            x = np.arange(self.data.size)
            K = run_results.keys()
            plots_list = []

            # 1. رسم بيانات الحساس الأساسية
            ts_fig, = plt.plot(x, self.data, color=data_color, label='Sensor Data', alpha=0.8)
            plots_list.append(ts_fig)

            # 2. رسم الـ Upper Threshold
            if 'upper_thresholds' in K:
                thup = run_results['upper_thresholds']
                uth_fig, = plt.plot(x, thup, color=threshold_color, lw=1.5, ls='--', label='Upper Bound')
                plots_list.append(uth_fig)

            # 3. رسم الـ Lower Threshold
            if 'lower_thresholds' in K:
                thdown = run_results['lower_thresholds']
                lth_fig, = plt.plot(x, thdown, color=threshold_color, lw=1.5, ls='--', label='Lower Bound')
                plots_list.append(lth_fig)

            # 4. رسم الـ Alarms (نقط حمراء مكان العطل)
            if with_alarm and ('alarms' in K):
                alarm_idx = run_results['alarms']
                if len(alarm_idx) > 0:
                    al_fig = plt.scatter(alarm_idx, self.data[alarm_idx], 
                                        color=alarm_color, label='Anomalies', s=40, zorder=5)
                    plots_list.append(al_fig)

            # تنسيق الرسمة عشان تكون جاهزة لمشروع التخرج
            plt.title(f'Robot Arm Predictive Maintenance (q={self.proba})', fontsize=14)
            plt.xlabel('Time Steps / Observations', fontsize=12)
            plt.ylabel('Sensor Value (Residual-based)', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.xlim(0, self.data.size)
            
            plt.tight_layout()
            plt.show()

            return plots_list
