import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("l2_error.csv")

plt.figure(figsize=(8, 5))
plt.semilogy(df["time"], df["l2_error"])
plt.xlabel("Time")
plt.ylabel("L2 Error")
plt.title("L2 Error vs Time - 1D Heat Equation")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("l2_error.png", dpi=150)
plt.show()