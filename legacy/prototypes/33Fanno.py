import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import warnings
import os

# تنظیمات محیطی
warnings.filterwarnings('ignore')
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
                               epochs=150, batch_size=64, verbose=0) # Epochs increased slightly for better accuracy

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
                                         epochs=150, batch_size=32, verbose=0)

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
#  بخش 2: توابع کمکی رسم نمودار (Plotting Helpers)
# ==============================================================================
def plot_process_on_fanno_line(solver, M1, M2):
    """
    رسم مسیر جریان روی نمودار T-s (خط فانو)
    """
    print("   > Plotting Process Path on T-s Diagram...")
    
    # 1. تولید نقاط برای رسم منحنی کامل خط فانو
    M_range = np.concatenate([np.linspace(0.1, 0.99, 100), np.linspace(1.01, 3.5, 100)])
    
    T_vals = []
    S_vals = [] # تغییر آنتروپی بی بعد
    
    for m in M_range:
        p = solver.get_properties(m)
        t_ratio = p['T/T*']
        p_ratio = p['P/P*']
        # فرمول تغییر آنتروپی
        s = np.log(t_ratio) - (0.2857) * np.log(p_ratio)
        T_vals.append(t_ratio)
        S_vals.append(s)
        
    # 2. محاسبه نقاط دقیق ورودی و خروجی
    p1 = solver.get_properties(M1)
    s1 = np.log(p1['T/T*']) - (0.2857)*np.log(p1['P/P*'])
    
    p2 = solver.get_properties(M2)
    s2 = np.log(p2['T/T*']) - (0.2857)*np.log(p2['P/P*'])

    # 3. رسم نمودار
    plt.figure(figsize=(9, 7))
    plt.plot(S_vals, T_vals, 'k-', alpha=0.6, linewidth=1.5, label='Fanno Line')
    plt.plot([s1, s2], [p1['T/T*'], p2['T/T*']], 'r-o', linewidth=3, markersize=8, label='Process Path')
    
    plt.annotate(f'Inlet M1={M1}', xy=(s1, p1['T/T*']), xytext=(s1-0.5, p1['T/T*']+0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.annotate(f'Exit M2={M2:.2f}', xy=(s2, p2['T/T*']), xytext=(s2+0.2, p2['T/T*']),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    plt.xlabel(r"Entropy Change $(s-s^*)/c_p$")
    plt.ylabel(r"Temperature Ratio $T/T^*$")
    plt.title("Process Path on Fanno Line")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("Fanno_Process_Path.png", dpi=150)
    print("   > Plot saved as 'Fanno_Process_Path.png'")
    plt.show()

def plot_corrected_inverse_validation(solver):
    """
    رسم نمودار صحت سنجی مدل معکوس (اصلاح شده برای شاخه فراصوتی)
    """
    print("\nGeneratig Corrected Validation Plot...")
    
    # محاسبه حد فیزیکی برای جریان فراصوتی
    gamma = 1.4
    max_fLD_supersonic = (1/gamma) * ( ((gamma+1)/(2)) * np.log((gamma+1)/(gamma-1)) - 1 ) # ~0.8215
    
    plt.figure(figsize=(10, 7))
    
    # --- 1. زیرصوتی (Subsonic) ---
    fLD_sub = np.logspace(-4, 1.5, 200) 
    M_pred_sub = [solver.get_mach_from_friction(f, regime='sub') for f in fLD_sub]
    plt.plot(fLD_sub, M_pred_sub, 'r--', linewidth=2.5, label='AI Prediction (Subsonic)')

    # حل دقیق زیرصوتی
    M_exact_range_sub = np.linspace(0.02, 0.99, 100)
    fLD_exact_sub = solver._fanno_physics(M_exact_range_sub)[:, 3]
    plt.plot(fLD_exact_sub, M_exact_range_sub, 'k-', alpha=0.3, linewidth=5, label='Analytical (Subsonic)')

    # --- 2. فراصوتی (Supersonic) - اصلاح شده ---
    # محدود کردن دامنه به قبل از حد فیزیکی (0.99 از حد ماکزیمم)
    fLD_sup = np.logspace(-4, np.log10(max_fLD_supersonic * 0.99), 200) 
    M_pred_sup = [solver.get_mach_from_friction(f, regime='sup') for f in fLD_sup]
    plt.plot(fLD_sup, M_pred_sup, 'b--', linewidth=2.5, label='AI Prediction (Supersonic)')

    # حل دقیق فراصوتی
    M_exact_range_sup = np.linspace(1.01, 4.0, 100)
    fLD_exact_sup = solver._fanno_physics(M_exact_range_sup)[:, 3]
    plt.plot(fLD_exact_sup, M_exact_range_sup, 'k-', alpha=0.3, linewidth=5, label='Analytical (Supersonic)')

    # --- تنظیمات ---
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlabel(r"Friction Parameter $4fL^*/D$ (Log Scale)", fontsize=12)
    plt.ylabel("Mach Number", fontsize=12)
    plt.title("Inverse Model Validation: Friction -> Mach", fontsize=14)
    plt.ylim(0, 4.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("Corrected_Fanno_Plot.png", dpi=300)
    print("   > Plot saved as 'Corrected_Fanno_Plot.png'")
    plt.show()

# ==============================================================================
#  بخش 3: حل مسئله مهندسی (Problem Solver)
# ==============================================================================
def solve_engineering_problem(solver):
    print("\n" + "="*60)
    print(" SAMPLE PROBLEM: Supersonic Flow in a Duct with Friction")
    print("="*60)
    
    # --- صورت مسئله ---
    M1 = 2.5          # ورودی
    f = 0.005         
    L = 1.0           
    D = 0.1           
    
    param_4fL_D_actual = (4 * f * L) / D
    
    print(f"Given Inputs: M1={M1}, 4fL/D={param_4fL_D_actual}")

    # --- حل با AI ---
    # گام 1: خواص ورودی
    props_1 = solver.get_properties(M1)
    fLD_1_star = props_1['4fL*/D']
    print(f"Step 1: Inlet 4fL*/D = {fLD_1_star:.5f}")
    
    # گام 2: خواص خروجی
    fLD_2_star = fLD_1_star - param_4fL_D_actual
    print(f"Step 2: Exit 4fL*/D = {fLD_2_star:.5f}")
    
    if fLD_2_star < 0:
        print("WARNING: Choked Flow!")
        return

    # گام 3: یافتن ماخ خروجی
    M2_ai = solver.get_mach_from_friction(fLD_2_star, regime='sup')
    print(f"Step 3: AI Predicted Exit Mach (M2) = {M2_ai:.5f}")

    # گام 4: افت فشار
    props_2 = solver.get_properties(M2_ai)
    P0_ratio_ai = props_2['P0/P0*'] / props_1['P0/P0*']
    print(f"Step 4: P02/P01 = {P0_ratio_ai:.4f}")

    # --- اعتبارسنجی (Validation) ---
    print("\n" + "="*40)
    print(" VERIFICATION")
    print("="*40)
    
    # حل دقیق تحلیلی
    g = 1.4
    fLD_1_exact = abs((1-M1**2)/(g*M1**2) + (g+1)/(2*g)*np.log(((g+1)*M1**2)/(2+(g-1)*M1**2)))
    fLD_2_exact = fLD_1_exact - param_4fL_D_actual
    
    from scipy.optimize import fsolve
    def resid(m):
        val = (1-m**2)/(g*m**2) + (g+1)/(2*g)*np.log(((g+1)*m**2)/(2+(g-1)*m**2))
        return abs(val) - fLD_2_exact
    
    M2_exact = fsolve(resid, 1.5)[0]
    
    def get_p0(m): return (1/m)*((2+(g-1)*m**2)/(g+1))**((g+1)/(2*(g-1)))
    P0_ratio_exact = get_p0(M2_exact) / get_p0(M1)

    # جدول مقایسه
    df = pd.DataFrame({
        'Parameter': ['Exit Mach', 'P0 Ratio', 'Inlet fLD'],
        'AI Prediction': [M2_ai, P0_ratio_ai, fLD_1_star],
        'Analytical': [M2_exact, P0_ratio_exact, fLD_1_exact],
    })
    df['Error (%)'] = 100 * abs(df['AI Prediction'] - df['Analytical']) / df['Analytical']
    print(df.to_string(index=False, float_format="%.5f"))
    
    # رسم نمودار مسیر
    plot_process_on_fanno_line(solver, M1, M2_ai)

# ==============================================================================
#  اجرای اصلی برنامه (Main Execution)
# ==============================================================================
if __name__ == "__main__":
    # 1. ساخت و آموزش مدل (فقط یک بار)
    my_solver = FannoSolverAI()
    
    # 2. حل مسئله نمونه
    solve_engineering_problem(my_solver)
    
    # 3. رسم نمودار اصلاح شده (درخواست شما برای خط آبی)
    plot_corrected_inverse_validation(my_solver)