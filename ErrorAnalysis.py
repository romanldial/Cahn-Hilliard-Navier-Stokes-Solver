# This python file precisely imports a file and runs a 
# loop for plotting the error vs the time step size. 
# This is done in a log-log plot to show the order of 
# convergence. Note: when using this for a spread, only 
# initalize the dt in the main() function in your cpp file.

import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Read data from cpp file. 
def read_l2_output():
    with open("L2_Results_heat3-LILS/L2outputFile_heat3-LILS.txt", "r") as f:
        return float(f.read().strip())

# Loop over the dt for the plot.
x = []
y = []
dts = np.logspace(-4, -1, 12)  # 12 points from 1e-4 to 1e-1
for dt in dts:
    subprocess.run(["./heat3-LILS", str(dt)], check=True)
    r_l2 = read_l2_output()
    x.append(dt)
    y.append(r_l2)


# Visualize plot and specifycreate log-log space for scale.  
plt.loglog(x, y)
plt.xlabel("Time Step Size (dt)")
plt.ylabel("L2 Norm of Residual")
plt.title("Error Analysis of Heat Equation Solver")
plt.savefig("convergence_heat3-LILS.png", dpi=300)