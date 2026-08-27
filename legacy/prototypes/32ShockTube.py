import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. تنظیمات گرافیکی
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.figsize': (22, 18),
    'axes.grid': False # برای کانتور معمولا گرید نمی‌گذارند
})

# ==========================================
# 2. فیزیک و مدل (همان توابع قبلی)
# ==========================================
GAMMA = 1.4
R_GAS = 287.0

def solve_shock_strength_exact(P4_P1, gamma=1.4):
    a1_a4 = 1.0
    def equation(P21):
        if P21 <= 1: return 1e6
        term1 = (gamma - 1) * (a1_a4) * (P21 - 1)
        term2 = np.sqrt(2 * gamma * (2 * gamma + (gamma + 1) * (P21 - 1)))
        bracket = 1 - (term1 / term2)
        if bracket <= 0: return 1e6
        rhs = P21 * (bracket ** (-2 * gamma / (gamma - 1)))
        return rhs - P4_P1
    try:
        return fsolve(equation, x0=3.0)[0]
    except:
        return 1.0

# آموزش سریع مدل (برای اینکه کد مستقل کار کند)
print("Training AI Model...")
P4_P1_data = np.logspace(0.1, 2.5, 4000)
P2_P1_data = np.array([solve_shock_strength_exact(p) for p in P4_P1_data])
X = np.log10(P4_P1_data).reshape(-1, 1)
y = P2_P1_data.reshape(-1, 1)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
model = Sequential([Dense(64, activation='relu', input_shape=(1,)), Dense(64, activation='relu'), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=0)
print("Training Done.")

# ==========================================
# 3. تولید داده‌های ماتریسی برای کانتور x-t
# ==========================================
def generate_xt_data(P4_P1_input, model, scaler, t_max=0.01, L=10.0, nx=500, nt=500):
    # پیش‌بینی قدرت شوک با AI
    x_in = scaler.transform(np.array([[np.log10(P4_P1_input)]]))
    P2_P1 = model.predict(x_in, verbose=0)[0][0]
    
    # محاسبه پارامترهای ثابت جریان
    P1, T1 = 100000.0, 300.0
    a1 = np.sqrt(GAMMA * R_GAS * T1)
    
    P2 = P2_P1 * P1
    T2 = T1 * P2_P1 * (( (GAMMA+1)/(GAMMA-1) + P2_P1 ) / (1 + (GAMMA+1)/(GAMMA-1)*P2_P1 ))
    a2 = np.sqrt(GAMMA * R_GAS * T2)
    Ms = np.sqrt( (GAMMA+1)/(2*GAMMA)*(P2_P1 - 1) + 1 )
    Cs = Ms * a1
    up = Cs * (1 - 1/( (GAMMA+1)*Ms**2 / ((GAMMA-1)*Ms**2 + 2) )) 
    
    P3, u3 = P2, up
    P4 = P4_P1_input * P1
    T4, T1 = 300.0, 300.0
    a4 = np.sqrt(GAMMA * R_GAS * T4)
    T3 = T4 * (P3/P4)**((GAMMA-1)/GAMMA)
    a3 = np.sqrt(GAMMA * R_GAS * T3)
    
    # شبکه‌بندی زمان و مکان
    t_vals = np.linspace(0.0001, t_max, nt) # زمان از نزدیک صفر
    x_vals = np.linspace(-L/2, L/2, nx)
    
    # ماتریس‌های دوبعدی برای ذخیره فشار و دما
    Pressure_Matrix = np.zeros((nt, nx))
    Temperature_Matrix = np.zeros((nt, nx))
    
    for i, t in enumerate(t_vals):
        # موقعیت موج‌ها در زمان t
        x_shock = Cs * t
        x_contact = up * t
        x_tail = (u3 - a3) * t
        x_head = -a4 * t
        
        # پر کردن سطر i ام ماتریس (مکان‌های مختلف در زمان ثابت)
        for j, x in enumerate(x_vals):
            if x > x_shock: # ناحیه 1
                P, T = P1, T1
            elif x > x_contact: # ناحیه 2
                P, T = P2, T2
            elif x > x_tail: # ناحیه 3
                P, T = P3, T3
            elif x > x_head: # ناحیه فن انبساطی
                u_local = 2/(GAMMA+1) * (a4 + x/t)
                a_local = a4 - (GAMMA-1)/2 * u_local
                T_local = T4 * (a_local/a4)**2
                P_local = P4 * (T_local/T4)**(GAMMA/(GAMMA-1))
                P, T = P_local, T_local
            else: # ناحیه 4
                P, T = P4, T4
            
            Pressure_Matrix[i, j] = P / 1000.0 # kPa
            Temperature_Matrix[i, j] = T
            
    return x_vals, t_vals * 1000, Pressure_Matrix, Temperature_Matrix # زمان به میلی‌ثانیه

# ==========================================
# 4. رسم نمودار کانتور x-t
# ==========================================
target_ratio = 40.0
x, t, P_mat, T_mat = generate_xt_data(target_ratio, model, scaler_X, t_max=0.008)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# کانتور فشار
cp1 = ax1.contourf(x, t, P_mat, levels=50, cmap='jet')
cbar1 = fig.colorbar(cp1, ax=ax1)
cbar1.set_label('Pressure (kPa)', fontweight='bold')
ax1.set_title(f'Pressure x-t Diagram ($P_4/P_1={target_ratio}$)', fontweight='bold', pad=15)
ax1.set_xlabel('Tube Position (m)', fontweight='bold')
ax1.set_ylabel('Time (ms)', fontweight='bold')

# کانتور دما
cp2 = ax2.contourf(x, t, T_mat, levels=50, cmap='inferno')
cbar2 = fig.colorbar(cp2, ax=ax2)
cbar2.set_label('Temperature (K)', fontweight='bold')
ax2.set_title(f'Temperature x-t Diagram ($P_4/P_1={target_ratio}$)', fontweight='bold', pad=15)
ax2.set_xlabel('Tube Position (m)', fontweight='bold')
ax2.set_ylabel('Time (ms)', fontweight='bold')

# اضافه کردن خطوط راهنما (برای شبیه‌سازی دقیق شکل کتاب)
# محاسبه شیب خطوط برای رسم روی کانتور
# برای سادگی، خط شوک و سطح تماس را روی نمودار دما می‌کشیم
# سرعت شوک و ... را دوباره حساب می‌کنیم (صرفا برای رسم خط)
# (در عمل این‌ها در تابع داخلی حساب شده‌اند، اینجا تقریبی رسم می‌کنیم یا باید از تابع خروجی بگیریم)
# اما کانتورها به وضوح مرزها را نشان می‌دهند پس خط اضافه لازم نیست.

plt.tight_layout()
plt.savefig('shock_tube_xt_contour.png', dpi=300)
print("Plot saved as 'shock_tube_xt_contour.png'")
plt.show()