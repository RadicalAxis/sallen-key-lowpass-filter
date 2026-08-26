"""
Theoretical Bode plot for a 2nd-order Sallen-Key low-pass filter
(Butterworth, Q = 0.707)

Design:
    R1 = R2 = 16 kOhm
    C1 = C2 = 10 nF
    Rf = 5.6 kOhm, Rg = 10 kOhm  -> gain K = 1 + Rf/Rg = 1.56
    fc = 1 / (2*pi*R*C) ~= 995 Hz

Transfer function (standard Sallen-Key LPF form):

    H(s) = K * wc^2 / (s^2 + (wc/Q)*s + wc^2)

wc = 2*pi*fc, Q = 1 / (3 - K)  [equal R/C Sallen-Key]
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Component values ----
R1 = R2 = 16e3      # ohms
C1 = C2 = 10e-9      # farads
Rf = 5.6e3
Rg = 10e3

# ---- Derived design parameters ----
fc = 1 / (2 * np.pi * np.sqrt(R1 * R2 * C1 * C2))
K = 1 + Rf / Rg
Q = 1 / (3 - K)
wc = 2 * np.pi * fc

print(f"Cutoff frequency fc  = {fc:.1f} Hz")
print(f"Gain K               = {K:.3f}")
print(f"Quality factor Q     = {Q:.3f}")

# ---- Frequency sweep ----
f = np.logspace(1, 5, 2000)   # 10 Hz to 100 kHz
w = 2 * np.pi * f
s = 1j * w

H = K * wc**2 / (s**2 + (wc / Q) * s + wc**2)

mag_db = 20 * np.log10(np.abs(H))
phase_deg = np.angle(H, deg=True)

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

ax1.semilogx(f, mag_db, color="tab:blue", linewidth=2)
ax1.axvline(fc, color="gray", linestyle="--", linewidth=1)
ax1.axhline(max(mag_db) - 3, color="gray", linestyle=":", linewidth=1)
ax1.set_ylabel("Magnitude (dB)")
ax1.set_title("Sallen-Key 2nd-Order Low-Pass Filter — Theoretical Bode Plot")
ax1.grid(True, which="both", alpha=0.3)
ax1.annotate(f"fc = {fc:.0f} Hz", xy=(fc, max(mag_db) - 3),
             xytext=(fc * 1.5, max(mag_db) - 15),
             arrowprops=dict(arrowstyle="->"))

ax2.semilogx(f, phase_deg, color="tab:red", linewidth=2)
ax2.axvline(fc, color="gray", linestyle="--", linewidth=1)
ax2.set_ylabel("Phase (degrees)")
ax2.set_xlabel("Frequency (Hz)")
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("bode_plot.png", dpi=150)
print("Saved bode_plot.png")
