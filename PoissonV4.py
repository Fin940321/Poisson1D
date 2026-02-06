import sys
from MDAnalysis import *
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# get Voltage from command line input
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--voltage", type=float, default=0.0,
                    help="voltage of simulation (default: 0.0 V)")
args, unknown = parser.parse_known_args()  # 忽略未知參數
Vapp = float(args.voltage)

# 動態生成電壓標籤（整數電壓顯示為整數，否則保留小數）
voltage_label = f"{int(Vapp)}V" if Vapp == int(Vapp) else f"{Vapp}V"

folder1 = "./for_poisson_4V/"

pdb_file = folder1 + "start_nodrudes.pdb"
# this file should contain charge density in e/nm^3, but we will convert it to e/Ang^3
electrolyte_charge_file = folder1 + "hist_q_total_4V.dat"
electrode_charge_file = folder1 + "charges_output_4V.dat"

# conversions
ang2bohr = 1.88973
eV2hartree = 0.0367
pi = 3.1415926535
conv = 1/(ang2bohr**3)

print("=" * 70)
print("    Poisson Analysis V4 - Consistent Bins Edition")
print("=" * 70)
print()

# get average electrode charge
electrode_charges = []
with open(electrode_charge_file) as f:
    for line in f:
        electrode_charges.append(float(line))
avg_electrode_charge = np.average(electrode_charges)
print(f"Average electrode charge: {avg_electrode_charge}")

# import pdb file to get Lcell, Lgap, surface Area
u = Universe(pdb_file, pdb_file)
atoms = u.select_atoms('resname grpc')  # should pull both virtual cathode/anode graphene
# get bounds of electrochemical cell
z = atoms.positions[:, 2]
z_left = z.min()
z_right = z.max()
Lcell = z_right - z_left
# get length of vacuum gap
Lgap = u.trajectory[0].triclinic_dimensions[2][2] - Lcell
# get area of electrode
surfaceArea = np.linalg.norm(np.cross(u.trajectory[0].triclinic_dimensions[0], u.trajectory[0].triclinic_dimensions[1]))

print(f"=== Cell Parameters ===")
print(f"  Voltage:      {Vapp} V")
print(f"  Lcell:        {Lcell:.2f} Å")
print(f"  Lgap:         {Lgap:.2f} Å")
print(f"  Surface Area: {surfaceArea:.2f} Å²")
print()

#******* Bins calculation (consistent with charge_density1D_V10.py) **********
print("=== Bins Calculation (consistent with charge_density1D_V10.py) ===")
cell_dist_nm = Lcell / 10.0  # Å -> nm
target_dz = 0.01  # nm (與 charge_density1D_V10.py 一致)
num_bins = int(cell_dist_nm / target_dz)
dz_nm = cell_dist_nm / num_bins
dz = dz_nm * 10.0  # 轉回 Å，用於後續積分

print(f"  Cell distance:   {cell_dist_nm:.4f} nm ({Lcell:.2f} Å)")
print(f"  Target dz:       {target_dz} nm")
print(f"  Number of bins:  {num_bins}")
print(f"  Actual dz:       {dz_nm:.6f} nm ({dz:.4f} Å)")

#******* read in electrolyte charge density **********
ifile = open(electrolyte_charge_file)
lines = ifile.readlines()
ifile.close()

# 驗證：從檔案讀取的 dz 應該與計算值一致
dz_from_file = float(lines[1].split()[0]) - float(lines[0].split()[0])
print(f"  dz from file:    {dz_from_file:.4f} Å")

if abs(dz_from_file - dz) > 0.01:
    print(f"  ⚠ WARNING: dz mismatch! Using dz from file: {dz_from_file:.4f} Å")
    dz = dz_from_file
else:
    print(f"  ✓ dz values match")
print()

z_dist = []
q_avg = []

#***** data may be collected outside electrochemical cell, only store points inside cell
for i in range(len(lines)):
    z = float(lines[i].split()[0])
    if z > z_left and z < z_right:
        z_dist.append(z)
        # charge density in e/nm^3 from charge_density1D_V8.py
        # convert to e/Ang^3 by dividing by 1000 (1 nm^3 = 1000 Ang^3)
        q_avg.append(float(lines[i].split()[1]) / 1000.0)

# ********** Assuming "flat electrode" is on right **************
# reverse arrays, because we want to integrate from right to left
z_dist = np.flip(z_dist)
q_avg = np.flip(q_avg)


#********** Now integrate to get Poisson profile **********
print("=== Computing Poisson Profile ===")

# 1. 計算表面電荷造成的電場
surfQ = avg_electrode_charge
E_surface = 4*pi*surfQ/((surfaceArea)*(ang2bohr**2))
# 2. 計算真空層(Gap)的電場
E_gap = -(Vapp*eV2hartree/(Lgap*ang2bohr))
E_z = []
z1_dist = []

# 3. 疊加得到起始電場
# new code for integrating R to L ...
E_i = E_gap - E_surface

# append initial efield
z1_dist.append(z_right)
E_z.append(E_i)

# only compute to halfway through the cell, as integration out to CNT electrode is unphysical (no x/y symmetry)
max_length = int(len(q_avg) / 2)

# the first integration point is z_right to z_dist[0], this may not equal dz so do this before loop
# 進行帕松方程式的第一次積分:
E_i = E_i - 4*pi*q_avg[0]*conv*abs(z_dist[0] - z_right)*ang2bohr
E_z.append(E_i)
z1_dist.append(z_dist[0])

for z_i in range(1, max_length):
    # new code for integrating R to L ...
    # 從電極的左端到右端做帕松公式的積分所以會是用扣的
    E_i = E_i - 4*pi*q_avg[z_i]*conv*dz*ang2bohr
    E_z.append(E_i)
    z1_dist.append(z_dist[z_i])

print(f"  Integration points: {len(E_z)}")
print(f"  E_field range: [{min(E_z):.6f}, {max(E_z):.6f}] e/bohr²")

# calculate voltage in a.u., store in volts
V_z = []
# assume right electrode is at - Vapp / 2
V_i = -Vapp/2.0 * eV2hartree
V_z.append(-Vapp/2.0)  # store in volts

# the first integration point is z_right to z_dist[0], this may not equal dz so do this before loop
V_i = V_i + E_z[0]*abs(z_dist[0] - z_right)*ang2bohr
# store in volts
V_z.append(V_i/eV2hartree)

for z_i in range(1, max_length):
    # new code for integration R to L
    V_i = V_i + E_z[z_i]*dz*ang2bohr
    # store in volts
    V_z.append(V_i/eV2hartree)

print(f"  Voltage range: [{min(V_z):.4f}, {max(V_z):.4f}] V")
print()

# compute bulk Voltage by averaging v_z[40:60].
# Note this is 2nm from the electrode, which is sufficient for double layer to decay to bulk
V_bulk = np.mean(V_z[40:60])
print("=== Results ===")
print(f"  V_cathode:  {-Vapp/2.0:.4f} V")
print(f"  V_bulk:   {V_bulk:.4f} V")
print(f"  delta_V:  {V_bulk + Vapp/2.0:.4f} V")
print()


#********** Generate Plots **********
print("=== Generating Plots ===")

# 1. Electric Field Plot
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_title(f"Electric Field of Poisson potential under {voltage_label}", fontsize=12, fontweight='bold')
ax1.set_xlabel("Z Position (Å)", fontsize=11)
ax1.set_ylabel("Electric field (e/bohr²)", fontsize=11)
ax1.plot(z1_dist, E_z, color='blue', linewidth=1.5)
ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
# Electrode position markers
#ax1.axvline(x=z_left, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Anode ({z_left:.1f} Å)')
ax1.axvline(x=z_right, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Cathode ({z_right:.1f} Å)')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

efield_filename = f'Electricfield_Poisson_{voltage_label}_3.png'
fig1.savefig(efield_filename, dpi=600, bbox_inches='tight')
print(f"  ✓ {efield_filename}")
plt.close(fig1)

# 2. Voltage Plot
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_title(f"Voltage of Poisson potential under {voltage_label}", fontsize=12, fontweight='bold')
ax2.set_xlabel("Z Position (Å)", fontsize=11)
ax2.set_ylabel("Voltage (V)", fontsize=11)
ax2.plot(z1_dist, V_z, color='red', linewidth=1.5)
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
# Electrode position markers
#ax2.axvline(x=z_left, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Anode ({z_left:.1f} Å)')
ax2.axvline(x=z_right, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Cathode ({z_right:.1f} Å)')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

voltage_filename = f'Voltage_Poisson_{voltage_label}_3.png'
fig2.savefig(voltage_filename, dpi=600, bbox_inches='tight')
print(f"  ✓ {voltage_filename}")
plt.close(fig2)

print()
print("=" * 70)
print("✓ Analysis Complete - Poisson V4")
print()
print("New features in V4:")
print("  • Bins calculation consistent with charge_density1D_V10.py")
print("  • Dynamic output filenames based on voltage parameter")
print("  • Simultaneous Electric Field and Voltage plot generation")
print("  • Enhanced diagnostic output")
print("=" * 70)
