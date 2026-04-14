import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("relative_residual.csv")  # was l2_error.csv

plt.figure(figsize=(8, 5))
plt.semilogy(df["time"], df["relative_residual"])
plt.xlabel("Time")
plt.ylabel("Relative Residual")
plt.title("Relative Residual vs Time - 1D Heat Equation")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("relative_residual_plot.png", dpi=150)
plt.show()