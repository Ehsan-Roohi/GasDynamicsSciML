import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
#  Professional Graphic Settings
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'], 
    'mathtext.fontset': 'cm',
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 3.5,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'figure.figsize': (18, 11),
    'savefig.bbox': 'tight'
})

# ==============================================================================
#  1. Physics (Exact Anderson Formula)
# ==============================================================================
def calculate_theta_exact(M, beta_deg, gamma=1.4):
    M = np.array(M, dtype=float)
    beta_rad = np.radians(np.array(beta_deg, dtype=float))
    
    # Anderson Eq. 4.17
    numerator = M**2 * np.sin(beta_rad)**2 - 1
    denominator = M**2 * (gamma + np.cos(2 * beta_rad)) + 2
    tan_beta = np.tan(beta_rad)
    tan_theta = 2 * (1.0 / tan_beta) * (numerator / denominator)
    
    return np.degrees(np.arctan(tan_theta))

# ==============================================================================
#  2. Training (Wide Range for Hypersonic Capability)
# ==============================================================================
print("--- Training Hypersonic Capable Model ---")

# استراتژی تولید داده چند مرحله‌ای برای پوشش تمام نواحی
# 1. ناحیه ماخ پایین (حساس‌ترین ناحیه)
M_part1 = np.random.uniform(1.1, 5.0, 25000)
# 2. ناحیه ماخ متوسط
M_part2 = np.random.uniform(5.0, 50.0, 15000)
# 3. ناحیه ماخ بسیار بالا (ابرفراصوتی) برای پوشش M=1000
M_part3 = np.random.uniform(50.0, 1500.0, 10000)

M_train = np.concatenate([M_part1, M_part2, M_part3])

# تولید بتا و تتا
mu_deg = np.degrees(np.arcsin(1/M_train))
beta_train = np.random.uniform(mu_deg, 90.0)
theta_train = calculate_theta_exact(M_train, beta_train)

# فیلتر داده‌های سالم
mask = (theta_train >= 0) & (theta_train < 50)
M_data = M_train[mask].reshape(-1, 1)
beta_data = beta_train[mask].reshape(-1, 1)
theta_data = theta_train[mask].reshape(-1, 1)

# مهندسی ویژگی (Feature Engineering)
beta_rad_data = np.radians(beta_data)
# نکته: ویژگی 1/M برای ماخ‌های بالا حیاتی است چون 1/1000 عدد کوچکی است و شبکه راحت یاد می‌گیرد
X_eng = np.column_stack((
    M_data, 
    beta_data,
    np.sin(beta_rad_data),
    1.0 / np.tan(beta_rad_data), # cot(beta)
    1.0 / M_data                 # ویژگی کلیدی برای ماخ بالا
))

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_s = scaler_X.fit_transform(X_eng)
y_s = scaler_y.fit_transform(theta_data)

# مدل شبکه عصبی
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(256, activation='swish'),
    layers.Dense(256, activation='swish'),
    layers.Dense(128, activation='swish'),
    layers.Dense(1, activation='linear')
])
model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='huber')
# تعداد دورهای آموزش را کمی بیشتر می‌کنیم چون دامنه وسیع شده
model.fit(X_s, y_s, epochs=200, batch_size=512, verbose=0)
print("--- Training Complete ---")

# ==============================================================================
#  3. Plotting Final Diagram (With M=100, M=1000)
# ==============================================================================
print("--- Generating Diagram ---")

# لیست ماخ‌ها شامل مقادیر هایپرسونیک
mach_lines = [1.5, 2.0, 3.0, 5.0, 10.0, 100.0, 1000.0] 
colors = plt.cm.jet(np.linspace(0, 1.0, len(mach_lines))) 

fig, ax = plt.subplots(figsize=(18, 11))

for i, M_val in enumerate(mach_lines):
    mu = np.degrees(np.arcsin(1/M_val))
    beta_sweep = np.linspace(mu + 0.01, 89.99, 800)
    
    # ورودی شبکه
    M_vec = np.full_like(beta_sweep, M_val)
    beta_rad_sweep = np.radians(beta_sweep)
    
    X_test_eng = np.column_stack((
        M_vec, 
        beta_sweep,
        np.sin(beta_rad_sweep),
        1.0 / np.tan(beta_rad_sweep),
        1.0 / M_vec
    ))
    
    # پیش‌بینی و حل دقیق
    theta_pred = scaler_y.inverse_transform(model.predict(scaler_X.transform(X_test_eng), verbose=0)).flatten()
    theta_exact = calculate_theta_exact(M_val, beta_sweep)
    
    # رسم خط مرجع (تحلیلی)
    if i == 0:
        ax.plot(theta_exact, beta_sweep, color='#2C3E50', linewidth=5, alpha=0.25, label='Analytical Ref')
    else:
        ax.plot(theta_exact, beta_sweep, color='#2C3E50', linewidth=5, alpha=0.25)
        
    # رسم خط هوش مصنوعی
    label_txt = f'AI ($M={int(M_val)}$)'
    ax.plot(theta_pred, beta_sweep, '--', color=colors[i], linewidth=3.5, label=label_txt)

# تنظیمات متن و محورها
ax.set_title(r'Oblique Shock $\beta-\theta-M$ Diagram', pad=20)
ax.set_xlabel(r'Deflection Angle, $\theta$ (degrees)')
ax.set_ylabel(r'Shock Wave Angle, $\beta$ (degrees)')

ax.set_xlim(0, 47) # محور افقی تا 45
ax.set_ylim(0, 90)

ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth=1.2, color='gray', alpha=0.3)
ax.grid(which='minor', linestyle=':', linewidth=0.8, color='gray', alpha=0.2)

# متن‌های راهنما (سمت راست)
ax.text(28, 86, 'Strong Shock', fontsize=18, color='#1F618D', fontweight='bold', ha='left')
ax.text(28, 15, 'Weak Shock', fontsize=18, color='#1E8449', fontweight='bold', ha='left')

# رسم مکان هندسی Theta Max
theta_max_x, theta_max_y = [], []
# محاسبه دقیق تا ماخ 1000
for m_curr in np.logspace(np.log10(1.1), np.log10(1000), 400):
    b_scan = np.linspace(0, 90, 600)
    t_scan = calculate_theta_exact(m_curr, b_scan)
    idx = np.argmax(t_scan)
    if t_scan[idx] > 0:
        theta_max_x.append(t_scan[idx])
        theta_max_y.append(b_scan[idx])
ax.plot(theta_max_x, theta_max_y, 'k:', linewidth=2.5)
ax.annotate(r'$\theta_{max}$ Locus', xy=(32, 67), xytext=(35, 75),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16)

# لجند بیرون کادر
ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0, title="Mach Number")

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig("Beta_Theta_Mach_v2.jpg", dpi=300)
print("Diagram saved as 'Beta_Theta_Mach_v2.jpg'")
plt.show()