import socket
import time
import os
import glob
import subprocess

output_dir = "/lustre/isaac24/scratch/rdial/mfem/mfem-4.9/examples/output"
glvis_host = "localhost"
glvis_port = 19916
delay = 0.05  # seconds between frames
snapshots_dir = "/lustre/isaac24/scratch/rdial/mfem/mfem-4.9/examples/snapshots"
os.makedirs(snapshots_dir, exist_ok=True)

# Get all timestep directories in order
dirs = sorted(glob.glob(os.path.join(output_dir, "heat_equation_1d_*[0-9]")))
print(f"Found {len(dirs)} timesteps")

# Connect once and keep connection open
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((glvis_host, glvis_port))
except ConnectionRefusedError:
    print("Could not connect to GLVis. Make sure GLVis is running!")
    exit(1)

print("Connected! Streaming frames...")

# Send first frame
with open(dirs[0]+'/mesh.000000') as f:
    mesh_data = f.read()
with open(dirs[0]+'/temperature.000000') as f:
    temp_data = f.read()

sock.sendall(("solution\n" + mesh_data + temp_data + "valuerange 0 6\n").encode())
time.sleep(1.5)  # wait for window to open

# Send setup commands for nice formatting
sock.sendall(b"keys c\n")       # show colorbar
time.sleep(0.2)
sock.sendall(b"keys a\n")       # show axes
time.sleep(0.2)
sock.sendall(b"keys g\n")       # white background
time.sleep(0.2)
sock.sendall(b"window_size 1200 800\n")  # larger window
time.sleep(0.5)

# Stream all frames and save snapshots
for i, d in enumerate(dirs):
    mesh_file = os.path.join(d, "mesh.000000")
    temp_file = os.path.join(d, "temperature.000000")

    if not os.path.exists(mesh_file) or not os.path.exists(temp_file):
        continue

    with open(mesh_file, 'r') as f:
        mesh_data = f.read()
    with open(temp_file, 'r') as f:
        temp_data = f.read()

    # Extract cycle number from directory name for title
    cycle = os.path.basename(d).split('_')[-1].lstrip('0') or '0'
    t = int(cycle) * 1e-3  # dt = 1e-3

    message = (
        "solution\n" + mesh_data + temp_data +
        "valuerange 0 6\n" +
        f"plot_caption '1D Heat Equation  t = {t:.3f}'\n" +
        f"screenshot {snapshots_dir}/frame_{i:04d}.png\n"
    )
    sock.sendall(message.encode())
    print(f"Frame {i+1}/{len(dirs)}", end='\r')
    time.sleep(delay)

sock.close()
print("\nAll frames saved! Stitching video...")

# Stitch frames into mp4 using ffmpeg
output_video = "/lustre/isaac24/scratch/rdial/mfem/mfem-4.9/examples/heat_equation_1d.mp4"
result = subprocess.run([
    "ffmpeg", "-y",
    "-framerate", "30",
    "-i", f"{snapshots_dir}/frame_%04d.png",
    "-vcodec", "mpeg4",
    "-q:v", "5",
    output_video
], capture_output=True, text=True)

if result.returncode == 0:
    print(f"Video saved to {output_video}")
else:
    print("ffmpeg failed:", result.stderr)
    print("You can manually stitch the PNG frames in snapshots/ directory")