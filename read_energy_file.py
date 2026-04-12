import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("energy.csv")

initial_mass         = df["mass"].iloc[0]
df["mass_error_rel"] = np.abs(df["mass"] - initial_mass) / np.abs(initial_mass)

# Figure 1: all three energies log scale
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(df["time"], df["mass_error_rel"].abs() + 1e-20)
ax.set_xlabel("Time $t$")
ax.set_ylabel("Absolute Relative Mass Error + $10^{-20}$")
ax.set_title("Mass Conservation Residual")
plt.tight_layout()
plt.savefig("fig_mass.png", dpi=200)
print("Saved fig_mass.png")