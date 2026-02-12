"""
Poisson Overlay Script
======================
Computes Poisson potential and electric field for multiple voltages,
then overlays them on the same plot for comparison.

Usage:
    python Poisson_overlayV2.py
"""

import sys
from MDAnalysis import *
import numpy as np
import matplotlib.pyplot as plt

# ==================== Configuration ====================
# Define voltages and their corresponding folders/files
VOLTAGE_CONFIGS = [
    {
        'voltage': 2.0,
        'folder': './for_poisson_2V/',
        'charge_file': 'hist_q_total_2V.dat',
        'electrode_file': 'charges_output_2V.dat',
        'color': 'blue',
        'linestyle': '-'
    },
    {
        'voltage': 4.0,
        'folder': './for_poisson_4V/',
        'charge_file': 'hist_q_total_4V.dat',
        'electrode_file': 'charges_output_4V.dat',
        'color': 'red',
        'linestyle': '-'
    }
]

# conversions
ang2bohr = 1.88973
eV2hartree = 0.0367
pi = 3.1415926535
conv = 1/(ang2bohr**3)

# ==================== Functions ====================
def compute_poisson_profile(config):
    """
    Compute Poisson profile for a given voltage configuration.
    Returns: z1_dist, E_z, V_z, z_left, z_right, V_bulk
    """
    Vapp = config['voltage']
    folder = config['folder']
    
    pdb_file = folder + "start_nodrudes.pdb"
    electrolyte_charge_file = folder + config['charge_file']
    electrode_charge_file = folder + config['electrode_file']
    
    print(f"\n--- Processing {Vapp}V ---")
    
    # get average electrode charge
    electrode_charges = []
    with open(electrode_charge_file) as f:
        for line in f:
            electrode_charges.append(float(line))
    avg_electrode_charge = np.average(electrode_charges)
    print(f"  Average electrode charge: {avg_electrode_charge:.6f}")
    
    # import pdb file to get Lcell, Lgap, surface Area
    u = Universe(pdb_file, pdb_file)
    atoms = u.select_atoms('resname grpc')
    z = atoms.positions[:, 2]
    z_left = z.min()
    z_right = z.max()
    Lcell = z_right - z_left
    Lgap = u.trajectory[0].triclinic_dimensions[2][2] - Lcell
    surfaceArea = np.linalg.norm(np.cross(
        u.trajectory[0].triclinic_dimensions[0], 
        u.trajectory[0].triclinic_dimensions[1]
    ))
    
    print(f"  Lcell: {Lcell:.2f} Å, Lgap: {Lgap:.2f} Å, Area: {surfaceArea:.2f} Å²")
    
    # Bins calculation (consistent with charge_density1D_V10.py)
    cell_dist_nm = Lcell / 10.0
    target_dz = 0.01
    num_bins = int(cell_dist_nm / target_dz)
    dz_nm = cell_dist_nm / num_bins
    dz = dz_nm * 10.0
    
    # read electrolyte charge density
    with open(electrolyte_charge_file) as ifile:
        lines = ifile.readlines()
    
    # verify dz
    dz_from_file = float(lines[1].split()[0]) - float(lines[0].split()[0])
    if abs(dz_from_file - dz) > 0.01:
        print(f"  ⚠ WARNING: dz mismatch! Using dz from file: {dz_from_file:.4f} Å")
        dz = dz_from_file
    
    z_dist = []
    q_avg = []
    
    for i in range(len(lines)):
        z_val = float(lines[i].split()[0])
        if z_val > z_left and z_val < z_right:
            z_dist.append(z_val)
            q_avg.append(float(lines[i].split()[1]) / 1000.0)
    
    # reverse arrays for R to L integration
    z_dist = np.flip(z_dist)
    q_avg = np.flip(q_avg)
    
    # Compute Poisson profile
    surfQ = avg_electrode_charge
    E_surface = 4*pi*surfQ/((surfaceArea)*(ang2bohr**2))
    E_gap = -(Vapp*eV2hartree/(Lgap*ang2bohr))
    
    E_z = []
    z1_dist = []
    
    E_i = E_gap - E_surface
    z1_dist.append(z_right)
    E_z.append(E_i)
    
    max_length = int(len(q_avg) / 2)
    
    E_i = E_i - 4*pi*q_avg[0]*conv*abs(z_dist[0] - z_right)*ang2bohr
    E_z.append(E_i)
    z1_dist.append(z_dist[0])
    
    for z_i in range(1, max_length):
        E_i = E_i - 4*pi*q_avg[z_i]*conv*dz*ang2bohr
        E_z.append(E_i)
        z1_dist.append(z_dist[z_i])
    
    # Calculate voltage
    V_z = []
    V_i = -Vapp/2.0 * eV2hartree
    V_z.append(-Vapp/2.0)
    
    V_i = V_i + E_z[0]*abs(z_dist[0] - z_right)*ang2bohr
    V_z.append(V_i/eV2hartree)
    
    for z_i in range(1, max_length):
        V_i = V_i + E_z[z_i]*dz*ang2bohr
        V_z.append(V_i/eV2hartree)
    
    V_bulk = np.mean(V_z[40:60])
    
    print(f"  E_field range: [{min(E_z):.6f}, {max(E_z):.6f}] e/bohr²")
    print(f"  Voltage range: [{min(V_z):.4f}, {max(V_z):.4f}] V")
    print(f"  V_cathode: {-Vapp/2.0:.4f} V, V_bulk: {V_bulk:.4f} V, delta_V: {V_bulk + Vapp/2.0:.4f} V")
    
    return {
        'z1_dist': z1_dist,
        'E_z': E_z,
        'V_z': V_z,
        'z_left': z_left,
        'z_right': z_right,
        'V_bulk': V_bulk,
        'Vapp': Vapp
    }


def main():
    print("=" * 70)
    print("    Poisson Overlay Analysis")
    print("    Comparing multiple voltage profiles")
    print("=" * 70)
    
    # Compute profiles for all voltages
    results = []
    for config in VOLTAGE_CONFIGS:
        result = compute_poisson_profile(config)
        result['color'] = config['color']
        result['linestyle'] = config['linestyle']
        results.append(result)
    
    print("\n" + "=" * 70)
    print("=== Generating Overlay Plots ===")
    
    # Use the first result's electrode positions for reference lines
    z_right_ref = results[0]['z_right']
    
    # ==================== Electric Field Overlay ====================
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_title("Electric Field of Poisson Potential - [BMIM][TFSI] at CNT Electrodes", 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel("Z Position (Å)", fontsize=12)
    ax1.set_ylabel("Electric field (e/bohr²)", fontsize=12)
    
    for res in results:
        voltage_label = f"{int(res['Vapp'])}V" if res['Vapp'] == int(res['Vapp']) else f"{res['Vapp']}V"
        ax1.plot(res['z1_dist'], res['E_z'], 
                 color=res['color'], 
                 linestyle=res['linestyle'],
                 linewidth=1.5, 
                 label=f'{voltage_label}',
                 alpha=0.8)
    
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=z_right_ref, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Cathode ({z_right_ref:.1f} Å)')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    efield_filename = 'Electricfield_Poisson_overlay.png'
    fig1.savefig(efield_filename, dpi=600, bbox_inches='tight')
    print(f"  ✓ {efield_filename}")
    plt.close(fig1)
    
    # ==================== Voltage Overlay ====================
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.set_title("Voltage of Poisson Potential - [BMIM][TFSI] at CNT Electrodes", 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel("Z Position (Å)", fontsize=12)
    ax2.set_ylabel("Voltage (V)", fontsize=12)

    # 定義計算本體電位 (Bulk Potential) 的 Z 軸範圍 (根據您的圖形，70-85A 是平坦的)
    bulk_z_min = 70.0
    bulk_z_max = 90.0
    
    # 設定箭頭的 X 軸位置 (錯開以避免重疊)
    arrow_x_positions = [75, 85]

    for i, res in enumerate(results):
        voltage_label = f"{int(res['Vapp'])}V" if res['Vapp'] == int(res['Vapp']) else f"{res['Vapp']}V"
        
        # 轉換資料為 numpy array 以便進行條件篩選
        z_array = np.array(res['z1_dist'])
        v_array = np.array(res['V_z'])

        # 1. 繪製電位曲線
        ax2.plot(z_array, v_array, 
                 color=res['color'], 
                 linestyle=res['linestyle'],
                 linewidth=1.5, 
                 label=f'{voltage_label}',
                 alpha=0.8)
        
        # 2. 透過取平均計算 V_bulk (Finding V_bulk by averaging in flat region)
        # 篩選出位於 bulk_z_min 和 bulk_z_max 之間的索引
        bulk_mask = (z_array >= bulk_z_min) & (z_array <= bulk_z_max)

        if np.any(bulk_mask):
            v_bulk_avg = np.mean(v_array[bulk_mask])
        else:
            # 如果範圍設定錯誤抓不到點，回退到舊方法或最後幾個點
            print(f"Warning: No points found in range {bulk_z_min}-{bulk_z_max} for {voltage_label}. Using tail average.")
            v_bulk_avg = np.mean(v_array[-20:])

        # 3. 計算 Delta V_negative
        # 陰極電位 (V_cathode) 理論上是 -Vapp/2
        v_cathode = -res['Vapp'] / 2.0
        delta_v = v_bulk_avg - v_cathode
        print(f"  [{voltage_label}] V_bulk (avg 70-85A): {v_bulk_avg:.4f} V, V_cathode: {v_cathode:.2f} V, Delta V: {delta_v:.4f} V")
        
        # 4. 繪製雙箭頭 (Double-headed arrow)
        x_arrow = arrow_x_positions[i] if i < len(arrow_x_positions) else 105
        
        # 箭頭：從 V_cathode 指向 V_bulk
        ax2.annotate(
            '', 
            xy=(x_arrow, v_cathode), 
            xytext=(x_arrow, v_bulk_avg),
            arrowprops=dict(arrowstyle='<->', color=res['color'], lw=1.5, shrinkA=0, shrinkB=0)
        )

        # 5. 加上文字標籤 (Label with calculated value)
        
        # 設定文字偏移邏輯：
        # 第一組數據 (2V, i=0) 文字放左邊
        # 第二組數據 (4V, i=1) 文字放右邊，利用右邊較空的區域
        if i == 0:
            align_h = 'right'   # 文字靠右對齊 (即文字尾端貼著偏移點)
            offset_x = -2.0     # 往左移 2 Å
        else:
            align_h = 'left'    # 文字靠左對齊 (即文字開頭貼著偏移點)
            offset_x = 2.0      # 往右移 2 Å

        label_text = r'$\Delta V_{neg} = ' + f'{delta_v:.2f}V$'
        
        ax2.text(x_arrow + offset_x, (v_cathode + v_bulk_avg) / 2, 
                 label_text, 
                 color=res['color'], 
                 rotation=0,          # <--- 關鍵修改：改為 0 度 (水平)
                 va='center',         # 垂直置中
                 ha=align_h,          # 水平對齊方向 (動態調整)
                 fontweight='bold',
                 fontsize=15)
    
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=z_right_ref, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Cathode ({z_right_ref:.1f} Å)')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    voltage_filename = 'Voltage_Poisson_overlay.png'
    fig2.savefig(voltage_filename, dpi=600, bbox_inches='tight')
    print(f"  ✓ {voltage_filename}")
    plt.close(fig2)
    
    # ==================== Combined Plot (2x1 layout) ====================
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12))
    fig3.suptitle("Poisson Analysis - [BMIM][TFSI] at CNT Electrodes", fontsize=16, fontweight='bold', y=0.995)
    
    # Electric Field
    ax3.set_title("Electric Field", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Z Position (Å)", fontsize=11)
    ax3.set_ylabel("Electric field (e/bohr²)", fontsize=11)
    for res in results:
        voltage_label = f"{int(res['Vapp'])}V" if res['Vapp'] == int(res['Vapp']) else f"{res['Vapp']}V"
        ax3.plot(res['z1_dist'], res['E_z'], 
                 color=res['color'], linestyle=res['linestyle'],
                 linewidth=1.5, label=f'{voltage_label}', alpha=0.8)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=z_right_ref, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Cathode')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Voltage
    ax4.set_title("Voltage", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Z Position (Å)", fontsize=11)
    ax4.set_ylabel("Voltage (V)", fontsize=11)
    for res in results:
        voltage_label = f"{int(res['Vapp'])}V" if res['Vapp'] == int(res['Vapp']) else f"{res['Vapp']}V"
        ax4.plot(res['z1_dist'], res['V_z'], 
                 color=res['color'], linestyle=res['linestyle'],
                 linewidth=1.5, label=f'{voltage_label}', alpha=0.8)
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=z_right_ref, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Cathode')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    combined_filename = 'Poisson_combined_overlay.png'
    fig3.savefig(combined_filename, dpi=600, bbox_inches='tight')
    print(f"  ✓ {combined_filename}")
    plt.close(fig3)
    
    print()
    print("=" * 70)
    print("✓ Overlay Analysis Complete")
    print()
    print("Generated files:")
    print(f"  • {efield_filename}")
    print(f"  • {voltage_filename}")
    print(f"  • {combined_filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()
