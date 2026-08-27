import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import warnings

# غیرفعال کردن هشدارهای غیرمهم برای خروجی تمیز
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
#  بخش 1: کلاس حل‌گر هوشمند (The AI Engine)
# ==============================================================================
class FannoSolverAI:
    def __init__(self):
        print("Initializing AI Solver...")
        self.models = {}
        self.scalers = {}
        self._build_and_train_networks()
        print("AI Solver Ready! 🚀")

    def _fanno_physics(self, M, gamma=1.4):
        """روابط تحلیلی دقیق برای تولید داده آموزش و اعتبارسنجی"""
        M = np.array(M, dtype=float)
        M = np.clip(M, 1e-5, 10.0)
        g = gamma
        
        # محاسبات نسبت‌ها
        T_Tstar = (g + 1.0) / (2.0 + (g - 1.0) * M**2)
        P_Pstar = (1.0 / M) * np.sqrt((g + 1.0) / (2.0 + (g - 1.0) * M**2))
        P0_P0star = (1.0 / M) * (((2.0 + (g - 1.0) * M**2) / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0))))
        
        # پارامتر اصطکاک
        term1 = (1.0 - M**2) / (g * M**2)
        term2 = (g + 1.0) / (2.0 * g)
        term3 = np.log(((g + 1.0) * M**2) / (2.0 + (g - 1.0) * M**2))
        fL_D = np.abs(term1 + term2 * term3)
        
        return np.vstack([T_Tstar, P_Pstar, P0_P0star, fL_D]).T

    def _build_model(self, input_dim, output_dim):
        """ساخت شبکه عصبی عمیق"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='swish'),
            layers.Dense(128, activation='swish'),
            layers.Dense(64, activation='swish'),
            layers.Dense(output_dim, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='logcosh')
        return model

    def _build_and_train_networks(self):
        """آموزش خودکار شبکه‌ها در هنگام شروع برنامه"""
        print("   > Training Forward Model (Mach -> Properties)...")
        # داده‌سازی
        M_train = np.concatenate([np.linspace(0.1, 0.95, 3000), np.linspace(1.05, 3.5, 3000)])
        y_train = self._fanno_physics(M_train) # [T, P, P0, fLD]
        
        # مهندسی ویژگی
        X_train = np.hstack([M_train.reshape(-1,1), 1/M_train.reshape(-1,1), np.log(M_train.reshape(-1,1))])
        
        self.scalers['fwd_X'] = StandardScaler().fit(X_train)
        self.scalers['fwd_y'] = StandardScaler().fit(y_train)
        
        self.models['fwd'] = self._build_model(3, 4)
        self.models['fwd'].fit(self.scalers['fwd_X'].transform(X_train), 
                               self.scalers['fwd_y'].transform(y_train), 
                               epochs=100, batch_size=64, verbose=0)

        print("   > Training Inverse Models (Friction -> Mach)...")
        # مدل معکوس (زیرصوتی و فراصوتی جدا)
        for regime, M_range in [('sub', np.linspace(0.05, 0.99, 4000)), ('sup', np.linspace(1.01, 3.5, 4000))]:
            fLD = self._fanno_physics(M_range)[:, 3].reshape(-1, 1)
            X_inv = np.log10(fLD) # ویژگی لگاریتمی
            
            self.scalers[f'{regime}_X'] = StandardScaler().fit(X_inv)
            self.scalers[f'{regime}_y'] = MinMaxScaler().fit(M_range.reshape(-1, 1))
            
            self.models[f'{regime}'] = self._build_model(1, 1)
            self.models[f'{regime}'].fit(self.scalers[f'{regime}_X'].transform(X_inv), 
                                         self.scalers[f'{regime}_y'].transform(M_range.reshape(-1, 1)), 
                                         epochs=100, batch_size=32, verbose=0)

    def get_properties(self, M):
        """دریافت خواص از عدد ماخ"""
        M = np.array([M])
        X = np.hstack([M.reshape(-1,1), 1/M.reshape(-1,1), np.log(M.reshape(-1,1))])
        X_s = self.scalers['fwd_X'].transform(X)
        y_s = self.models['fwd'].predict(X_s, verbose=0)
        props = self.scalers['fwd_y'].inverse_transform(y_s)[0]
        return {'T/T*': props[0], 'P/P*': props[1], 'P0/P0*': props[2], '4fL*/D': props[3]}

    def get_mach_from_friction(self, fLD, regime='sup'):
        """یافتن ماخ از پارامتر اصطکاک"""
        X = np.log10(np.array([[fLD]]))
        X_s = self.scalers[f'{regime}_X'].transform(X)
        y_s = self.models[f'{regime}'].predict(X_s, verbose=0)
        return self.scalers[f'{regime}_y'].inverse_transform(y_s)[0][0]

# ==============================================================================
#  بخش 2: حل مسئله مهندسی نمونه (Sample Problem)
# ==============================================================================
def solve_engineering_problem():
    solver = FannoSolverAI()
    
    print("\n" + "="*60)
    print(" SAMPLE PROBLEM: Supersonic Flow in a Duct with Friction")
    print("="*60)
    
    # --- صورت مسئله ---
    # ورودی‌ها
    M1 = 2.5          # عدد ماخ ورودی (فراصوتی)
    f = 0.005         # ضریب اصطکاک
    L = 1.0           # طول لوله (متر)
    D = 0.1           # قطر لوله (متر)
    
    param_4fL_D_actual = (4 * f * L) / D
    
    print(f"Given Inputs:")
    print(f"  Inlet Mach (M1) = {M1}")
    print(f"  Duct Parameter (4fL/D) = {param_4fL_D_actual}")
    print("-" * 60)

    # --- گام 1: محاسبه خواص در ورودی (M1) ---
    props_1 = solver.get_properties(M1)
    fLD_1_star = props_1['4fL*/D']
    
    print(f"Step 1: Inlet Properties (via AI Forward Model)")
    print(f"  AI Prediction -> 4fL*/D|_1 = {fLD_1_star:.5f}")
    
    # --- گام 2: محاسبه طول تا خفگی در خروجی ---
    # فرمول: L*_2 = L*_1 - L_actual
    # بنابراین: 4fL*/D|_2 = 4fL*/D|_1 - 4fL/D_actual
    fLD_2_star = fLD_1_star - param_4fL_D_actual
    
    print(f"\nStep 2: Calculate Exit Friction Parameter")
    print(f"  4fL*/D|_2 = {fLD_1_star:.5f} - {param_4fL_D_actual:.5f} = {fLD_2_star:.5f}")
    
    if fLD_2_star < 0:
        print("  WARNING: Flow is choked! Length exceeds maximum possible length.")
        return

    # --- گام 3: یافتن ماخ خروجی (M2) ---
    # چون ورودی فراصوتی (Supersonic) است و خفگی رخ نداده، خروجی هم در شاخه فراصوتی می‌ماند
    M2_ai = solver.get_mach_from_friction(fLD_2_star, regime='sup')
    
    print(f"\nStep 3: Find Exit Mach (via AI Inverse Model)")
    print(f"  AI Prediction -> Exit Mach (M2) = {M2_ai:.5f}")

    # --- گام 4: محاسبه خواص خروجی (M2) ---
    props_2 = solver.get_properties(M2_ai)
    
    # محاسبه افت فشار رکود
    P0_ratio_ai = props_2['P0/P0*'] / props_1['P0/P0*']
    
    print(f"\nStep 4: Calculate Stagnation Pressure Ratio (P02/P01)")
    print(f"  P02/P01 = (P02/P0*) / (P01/P0*) = {props_2['P0/P0*']:.4f} / {props_1['P0/P0*']:.4f}")
    print(f"  AI Result -> P02/P01 = {P0_ratio_ai:.4f}")

    # ==========================================================================
    #  بخش 3: مقایسه با حل تحلیلی (Validation)
    # ==========================================================================
    print("\n" + "="*60)
    print(" VERIFICATION: AI vs Analytical Solution")
    print("="*60)
    
    # حل دقیق تحلیلی (بدون AI)
    # 1. محاسبه fLD1 دقیق
    g = 1.4
    term1 = (1-M1**2)/(g*M1**2) + (g+1)/(2*g)*np.log(((g+1)*M1**2)/(2+(g-1)*M1**2))
    fLD_1_exact = abs(term1)
    
    # 2. fLD2 دقیق
    fLD_2_exact = fLD_1_exact - param_4fL_D_actual
    
    # 3. حل عددی برای پیدا کردن M2 (چون فرمول بسته برای M(fLD) وجود ندارد)
    from scipy.optimize import fsolve
    def resid(m):
        val = (1-m**2)/(g*m**2) + (g+1)/(2*g)*np.log(((g+1)*m**2)/(2+(g-1)*m**2))
        return abs(val) - fLD_2_exact
    
    M2_exact = fsolve(resid, 1.5)[0] # حدس اولیه 1.5 (فراصوتی)
    
    # 4. محاسبه P0
    def get_p0(m): return (1/m)*((2+(g-1)*m**2)/(g+1))**((g+1)/(2*(g-1)))
    P0_ratio_exact = get_p0(M2_exact) / get_p0(M1)

    # جدول مقایسه
    df = pd.DataFrame({
        'Parameter': ['Exit Mach (M2)', 'Stagnation Press Ratio (P02/P01)', 'Friction Param State 1'],
        'AI Prediction': [M2_ai, P0_ratio_ai, fLD_1_star],
        'Analytical Exact': [M2_exact, P0_ratio_exact, fLD_1_exact],
    })
    df['Error (%)'] = 100 * abs(df['AI Prediction'] - df['Analytical Exact']) / df['Analytical Exact']
    
    print(df.to_string(index=False, float_format="%.5f"))
    
    # --- رسم نمودار مسیر فرآیند ---
    plot_process_on_fanno_line(solver, M1, M2_ai)

def plot_process_on_fanno_line(solver, M1, M2):
    """رسم مسیر جریان روی نمودار T-s"""
    M_range = np.linspace(1.01, 3.0, 200) # رسم شاخه فراصوتی
    T_vals = []
    S_vals = [] # تغییر آنتروپی
    
    # تولید خط فانو با AI
    for m in M_range:
        p = solver.get_properties(m)
        T_vals.append(p['T/T*'])
        # ds/cp = ln(T/T*) - (g-1)/g ln(P/P*)
        s = np.log(p['T/T*']) - (0.4/1.4)*np.log(p['P/P*'])
        S_vals.append(s)
        
    # نقاط ورودی و خروجی
    p1 = solver.get_properties(M1)
    s1 = np.log(p1['T/T*']) - (0.4/1.4)*np.log(p1['P/P*'])
    
    p2 = solver.get_properties(M2)
    s2 = np.log(p2['T/T*']) - (0.4/1.4)*np.log(p2['P/P*'])

    plt.figure(figsize=(10, 8))
    plt.plot(S_vals, T_vals, 'k-', linewidth=2, label='Fanno Line (AI Generated)')
    plt.plot([s1, s2], [p1['T/T*'], p2['T/T*']], 'r-o', linewidth=3, markersize=10, label='Process Path')
    
    plt.annotate('Inlet (M1=2.5)', xy=(s1, p1['T/T*']), xytext=(s1-0.1, p1['T/T*']-0.1), arrowprops=dict(facecolor='blue'))
    plt.annotate('Exit (M2)', xy=(s2, p2['T/T*']), xytext=(s2+0.05, p2['T/T*']+0.1), arrowprops=dict(facecolor='blue'))
    
    plt.xlabel(r"Entropy Change $(s-s^*)/c_p$")
    plt.ylabel(r"Temperature Ratio $T/T^*$")
    plt.title("Solution Path on Fanno Line (Supersonic Branch)")
    plt.grid(True)
    plt.legend()
    plt.savefig("Fanno_Problem_Solution.png")
    print("\nProcess path plotted and saved as 'Fanno_Problem_Solution.png'")
    plt.show()

# اجرا
if __name__ == "__main__":
    solve_engineering_problem()