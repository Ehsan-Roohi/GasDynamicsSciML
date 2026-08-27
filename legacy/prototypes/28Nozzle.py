import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import fsolve, brentq
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. تنظیمات گرافیکی
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'figure.figsize': (24, 7) # شکل کشیده برای سه نمودار کنار هم
})

# ==========================================
# 2. توابع فیزیک
# ==========================================
GAMMA = 1.4

def get_mach_from_area_ratio(area_ratio, gamma=1.4, supersonic=True):
    def equation(M):
        if M <= 0: return 1e5
        part1 = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
        part1 = part1 ** ((gamma + 1) / (gamma - 1))
        return (1 / M**2) * part1 - area_ratio**2
    
    if area_ratio == 1.0: return 1.0
    initial_guess = 3.5 if supersonic else 0.1
    try:
        return fsolve(equation, initial_guess)[0]
    except:
        return 1.0

def normal_shock_relations(M1, gamma=1.4):
    term1 = 1 + 0.5 * (gamma - 1) * M1**2
    term2 = gamma * M1**2 - 0.5 * (gamma - 1)
    M2 = np.sqrt(term1 / term2)
    
    term_a = ((gamma + 1) / 2 * M1**2) / (1 + (gamma - 1) / 2 * M1**2)
    term_a = term_a ** (gamma / (gamma - 1))
    term_b = (2 * gamma / (gamma + 1) * M1**2 - (gamma - 1) / (gamma + 1))
    term_b = term_b ** (1 / (1 - gamma))
    p0_ratio = term_a * term_b
    return M2, p0_ratio

def calculate_pb_from_shock_location(Ae_At, As_At, gamma=1.4):
    M1 = get_mach_from_area_ratio(As_At, gamma, supersonic=True)
    M2, p02_p01 = normal_shock_relations(M1, gamma)
    At2_At1 = 1 / p02_p01
    Ae_At2 = Ae_At / At2_At1
    Me = get_mach_from_area_ratio(Ae_At2, gamma, supersonic=False)
    p_ratio_isentropic = (1 + 0.5 * (gamma - 1) * Me**2) ** (-gamma / (gamma - 1))
    Pe_P0 = p_ratio_isentropic * p02_p01
    return Pe_P0

def find_exact_shock_location_for_pb(Ae_At, target_Pb_P0):
    # حل معکوس دقیق برای یافتن محل شوک از روی فشار
    def error_func(As_candidate):
        try:
            pb_calc = calculate_pb_from_shock_location(Ae_At, As_candidate, GAMMA)
            return pb_calc - target_Pb_P0
        except:
            return 1.0
    try:
        # جستجو بین گلوگاه (1.0) و خروجی (Ae_At)
        return brentq(error_func, 1.001, Ae_At - 0.001)
    except:
        return np.nan # اگر جوابی پیدا نشد (مثلا فشار نامعتبر است)

# ==========================================
# 3. آموزش مدل (سریع)
# ==========================================
print("Generating Data & Training AI...")
X_data = []
y_data = []
np.random.seed(42)

for _ in range(12000): 
    Ae_At = np.random.uniform(1.2, 6.0) 
    As_At = np.random.uniform(1.05, Ae_At * 0.95)
    try:
        Pb_P0 = calculate_pb_from_shock_location(Ae_At, As_At, GAMMA)
        X_data.append([Ae_At, Pb_P0])
        y_data.append(As_At)
    except:
        continue

X = np.array(X_data)
y = np.array(y_data)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

model = Sequential([
    Input(shape=(2,)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=100, batch_size=64, verbose=0)
print("Training Complete.")

# ==========================================
# 4. رسم هیت‌مپ مقایسه‌ای (Analytical vs AI)
# ==========================================
def plot_comparative_heatmaps(model, scaler):
    print("Calculating Heatmaps (This takes a few seconds)...")
    
    # رزولوشن گرید (تعداد پیکسل‌ها)
    res = 100 
    ae_vals = np.linspace(1.5, 6.0, res)
    pb_vals = np.linspace(0.2, 0.9, res)
    
    AE, PB = np.meshgrid(ae_vals, pb_vals)
    
    # آرایه‌ها برای ذخیره نتایج
    Shock_AI = np.zeros_like(AE)
    Shock_Exact = np.zeros_like(AE)
    
    # حلقه روی تمام پیکسل‌ها برای محاسبه مقادیر
    # نکته: برای سرعت بیشتر، AI را برداری حساب می‌کنیم اما Exact را لوپ می‌زنیم
    
    # 1. محاسبه AI (سریع)
    inputs = np.column_stack((AE.ravel(), PB.ravel()))
    inputs_scaled = scaler.transform(inputs)
    Shock_AI_flat = model.predict(inputs_scaled, verbose=0).flatten()
    Shock_AI = Shock_AI_flat.reshape(AE.shape)
    
    # 2. محاسبه Exact (کندتر - حل عددی برای هر نقطه)
    for i in range(res):
        for j in range(res):
            ae = AE[i, j]
            pb = PB[i, j]
            # حل دقیق
            val_exact = find_exact_shock_location_for_pb(ae, pb)
            Shock_Exact[i, j] = val_exact

    # فیلتر کردن نقاط غیر فیزیکی (ماسک کردن)
    # اگر شوک محاسبه شده خارج از نازل باشد، آن را نمایش نمی‌دهیم
    # شرط: 1.0 < As < Ae
    
    # ماسک برای AI
    mask_ai = (Shock_AI < 1.0) | (Shock_AI > AE)
    Shock_AI_masked = np.ma.masked_where(mask_ai, Shock_AI)
    
    # ماسک برای Exact (خودش NaN برمی‌گرداند اگر جواب نباشد)
    Shock_Exact_masked = np.ma.masked_invalid(Shock_Exact)
    
    # محاسبه خطا (فقط در نقاطی که هر دو جواب دارند)
    Error_Map = np.abs(Shock_Exact_masked - Shock_AI_masked)

    # --- رسم نمودارها ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 7))
    
    # تنظیمات مشترک کانتورها
    levels = np.linspace(1.0, 6.0, 21)
    cmap = 'viridis' # طیف رنگی اصلی
    
    # 1. نمودار دقیق (Ground Truth)
    c1 = ax1.contourf(PB, AE, Shock_Exact_masked, levels=levels, cmap=cmap)
    plt.colorbar(c1, ax=ax1, label=r'Shock Location ($A_s/A_t$)')
    ax1.set_title('Analytical Solution (Theory)', fontweight='bold')
    
    # 2. نمودار هوش مصنوعی (AI)
    c2 = ax2.contourf(PB, AE, Shock_AI_masked, levels=levels, cmap=cmap)
    plt.colorbar(c2, ax=ax2, label=r'Shock Location ($A_s/A_t$)')
    ax2.set_title('AI Prediction (Neural Network)', fontweight='bold')
    
    # 3. نمودار خطا (Error)
    # برای خطا از رنگ‌های متفاوت (مثل inferno) استفاده می‌کنیم تا بهتر دیده شود
    c3 = ax3.contourf(PB, AE, Error_Map, levels=20, cmap='inferno')
    cbar3 = plt.colorbar(c3, ax=ax3)
    cbar3.set_label('Absolute Error')
    ax3.set_title('Difference (|Theory - AI|)', fontweight='bold')

    # برچسب‌گذاری محورها
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(r'Back Pressure Ratio ($P_b/P_0$)', fontsize=16)
        ax.set_ylabel(r'Nozzle Geometry ($A_e/A_t$)', fontsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('comparative_heatmap.png', dpi=150)
    print("Plot saved as 'comparative_heatmap.png'")
    plt.show()

# ==========================================
# 5. اجرا
# ==========================================
if __name__ == "__main__":
    plot_comparative_heatmaps(model, scaler_X)