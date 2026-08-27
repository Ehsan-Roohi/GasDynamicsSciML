import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
#  بخش اول: تعریف توابع فیزیکی (Rayleigh Flow Physics)
# ==============================================================================
def rayleigh_ratios(M, gamma=1.4):
    """ محاسبه تمام نسبت‌های جریان ریلی برای مسئله مستقیم """
    M = np.asarray(M, dtype=float)
    M = np.maximum(M, 1e-6) # جلوگیری از تقسیم بر صفر
    g = gamma
    
    P_Pstar = (g + 1.0) / (1.0 + g * M**2)
    T_Tstar = ((g + 1.0)**2 * M**2) / (1.0 + g * M**2)**2
    rho_rhostar = (1.0 + g * M**2) / ((g + 1.0) * M**2)
    u_ustar = (g + 1.0) * M**2 / (1.0 + g * M**2)
    
    term1 = (g + 1.0) / (1.0 + g * M**2)
    term2 = ((2.0 / (g + 1.0)) * (1.0 + (g - 1.0) * M**2 / 2.0)) ** (g / (g - 1.0))
    P0_P0star = term1 * term2

    return np.vstack([T_Tstar, P_Pstar, rho_rhostar, u_ustar, P0_P0star]).T

def get_T_for_inverse(M, gamma=1.4):
    """ محاسبه فقط نسبت دما برای مسئله معکوس """
    M = np.array(M)
    return ((gamma + 1)**2 * M**2) / (1 + gamma * M**2)**2

# ==============================================================================
#  بخش دوم: حل مسئله مستقیم (Forward Problem: Mach -> Properties)
# ==============================================================================
print("--- 1. Solving Forward Problem (Mach -> Properties) ---")

# 1. تولید داده
M_min, M_max = 0.2, 3.0
n_samples = 2000
M_vals = np.linspace(M_min, M_max, n_samples)
y_vals = rayleigh_ratios(M_vals)

# 2. مهندسی ویژگی (Feature Engineering) برای دقت بالا
M_col = M_vals.reshape(-1, 1)
X_enhanced = np.hstack([
    M_col, 
    M_col**2, 
    1.0 / (M_col + 1e-6), 
    1.0 / (M_col**2 + 1e-6)
])

# 3. تقسیم و اسکیل کردن
X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y_vals, test_size=0.2, random_state=42)

scaler_X_fwd = StandardScaler()
scaler_y_fwd = StandardScaler()

X_train_s = scaler_X_fwd.fit_transform(X_train)
y_train_s = scaler_y_fwd.fit_transform(y_train)
X_test_s = scaler_X_fwd.transform(X_test)

# 4. آموزش مدل
model_fwd = MLPRegressor(hidden_layer_sizes=(128, 128, 64), activation='tanh', 
                         solver='adam', alpha=1e-6, max_iter=5000, tol=1e-8, random_state=42)
model_fwd.fit(X_train_s, y_train_s)

# 5. ارزیابی
r2_fwd = model_fwd.score(X_test_s, scaler_y_fwd.transform(y_test))
print(f"Forward Model R2 Score: {r2_fwd:.6f}")

# ==============================================================================
#  بخش سوم: حل مسئله معکوس (Inverse Problem: T/T* -> Mach)
# ==============================================================================
print("--- 2. Solving Inverse Problem (T/T* -> Mach) ---")

# تولید داده‌های جداگانه برای زیرصوتی و فراصوتی
# کمی فاصله از 1.0 برای جلوگیری از تکینگی در آموزش
M_sub = np.linspace(0.05, 0.99, 1000).reshape(-1, 1)
M_sup = np.linspace(1.01, 3.00, 1000).reshape(-1, 1)

T_sub = get_T_for_inverse(M_sub)
T_sup = get_T_for_inverse(M_sup)

# --- مدل زیرصوتی (Subsonic) ---
scaler_T_sub = StandardScaler()
scaler_M_sub = StandardScaler()
X_sub_s = scaler_T_sub.fit_transform(T_sub)
y_sub_s = scaler_M_sub.fit_transform(M_sub)

model_sub = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh', max_iter=5000, random_state=42)
model_sub.fit(X_sub_s, y_sub_s.ravel())

# --- مدل فراصوتی (Supersonic) ---
scaler_T_sup = StandardScaler()
scaler_M_sup = StandardScaler()
X_sup_s = scaler_T_sup.fit_transform(T_sup)
y_sup_s = scaler_M_sup.fit_transform(M_sup)

model_sup = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh', max_iter=5000, random_state=42)
model_sup.fit(X_sup_s, y_sup_s.ravel())

print("Inverse models trained.")

# ==============================================================================
#  بخش چهارم: رسم نمودارها (Plotting)
# ==============================================================================
print("--- 3. Generating Plots ---")

# داده‌های پیوسته برای رسم تمیز
M_plot = np.linspace(M_min, M_max, 400).reshape(-1, 1)
X_plot_ft = np.hstack([M_plot, M_plot**2, 1./(M_plot+1e-6), 1./(M_plot**2+1e-6)])
y_true_plot = rayleigh_ratios(M_plot.flatten())

# پیش‌بینی مستقیم
y_pred_s = model_fwd.predict(scaler_X_fwd.transform(X_plot_ft))
y_pred_plot = scaler_y_fwd.inverse_transform(y_pred_s)

labels = [r"$T/T^*$", r"$p/p^*$", r"$\rho/\rho^*$", r"$u/u^*$", r"$p_0/p_0^*$"]

# --- نمودار 1: مقایسه تحلیلی و ML ---
plt.figure(figsize=(12, 10))
plt.suptitle(f"1. Forward Solution: Analytical vs Neural Network (R2={r2_fwd:.5f})", fontsize=16)
for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.plot(M_plot, y_true_plot[:, i], 'b-', linewidth=2, label='Analytical')
    plt.plot(M_plot, y_pred_plot[:, i], 'r--', linewidth=2, label='Neural Net')
    plt.ylabel(labels[i], fontsize=12)
    plt.grid(True)
    if i == 0: plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("1_forward_comparison.png", dpi=200)

# --- نمودار 2: خطای نسبی ---
rel_error = 100 * (y_pred_plot - y_true_plot) / (y_true_plot + 1e-9)
plt.figure(figsize=(12, 10))
plt.suptitle("2. Relative Error of Neural Network Predictions (%)", fontsize=16)
for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.plot(M_plot, rel_error[:, i], 'k-', linewidth=1.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.5)
    plt.ylabel(f"Error % {labels[i]}", fontsize=12)
    # زوم هوشمند برای نمایش بهتر خطا
    limit = max(0.5, np.max(np.abs(rel_error[:, i])) * 1.1)
    plt.ylim(-limit, limit)
    plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("2_forward_error.png", dpi=200)

# --- نمودار 3: حل معکوس (Inverse) ---
# تست روی بازه دما
T_test = np.linspace(0.1, 1.0, 300).reshape(-1, 1)
# پیش‌بینی
M_pred_sub = scaler_M_sub.inverse_transform(model_sub.predict(scaler_T_sub.transform(T_test)).reshape(-1, 1))
M_pred_sup = scaler_M_sup.inverse_transform(model_sup.predict(scaler_T_sup.transform(T_test)).reshape(-1, 1))

plt.figure(figsize=(10, 7))
plt.suptitle("3. Inverse Problem: Predicting Mach from Temperature", fontsize=16)
# رسم تحلیلی (مرجع)
plt.plot(T_sub, M_sub, 'k-', alpha=0.2, linewidth=6, label='Analytical Ref')
plt.plot(T_sup, M_sup, 'k-', alpha=0.2, linewidth=6)
# رسم ML
plt.plot(T_test, M_pred_sub, 'r--', linewidth=2, label='ML Subsonic Branch')
plt.plot(T_test, M_pred_sup, 'b--', linewidth=2, label='ML Supersonic Branch')

plt.xlabel(r"Input: Temperature Ratio $T/T^*$", fontsize=12)
plt.ylabel(r"Output: Mach Number $M$", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("3_inverse_solution.png", dpi=200)

print("All plots generated and saved:")
print("1. 1_forward_comparison.png")
print("2. 2_forward_error.png")
print("3. 3_inverse_solution.png")
plt.show()