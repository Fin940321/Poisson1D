#!/usr/bin/env python3
"""
Unified Poisson Pipeline
========================
整併三個分析腳本為單一 pipeline：
  Step 1: charge_density   — 電荷密度分析 (來自 charge_density1D_V10.py)
  Step 2: search_charges   — 電極電荷提取 (來自 search_charges_all.ipynb)
  Step 3: poisson           — Poisson 積分 (來自 PoissonV4.py 增強版)

用法:
  python unified_poisson_pipeline.py -v 4.0
  python unified_poisson_pipeline.py -v 0 --skip-step search_charges --skip-step poisson
  python unified_poisson_pipeline.py -v 1.5 --top for_openmm.pdb --ffdir ../ffdir/
"""

import sys
import os
import re
import glob
import shutil
import math
import time
import argparse

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis import Universe
from openmm.app import *
from openmm import *
from openmm.unit import nanometer, picosecond, picoseconds, kelvin


# ======================================================================
#  Constants
# ======================================================================
PI = 3.1415926535
ANG2BOHR = 1.88973
EV2HARTREE = 0.0367
CONV = 1.0 / (ANG2BOHR ** 3)

CATION_TYPES = ["BMIM", "BMI", "EMIM", "PMIM", "OMIM"]
ANION_TYPES = ["Tf2N", "trfl", "Tf2", "TRF", "trf", "TFSI", "BF4", "PF6", "Cl"]


# ======================================================================
#  Helper functions
# ======================================================================
def make_voltage_label(voltage):
    """整數電壓 → '4V'，小數 → '1.5V'"""
    return f"{int(voltage)}V" if voltage == int(voltage) else f"{voltage}V"


def compute_charge_histogram(frame_xyz_z, atom_indices, charges_array, reference_z, bins):
    """Compute charge density histogram using NumPy vectorization."""
    if len(atom_indices) == 0:
        return np.zeros(len(bins) - 1)
    positions_z = frame_xyz_z[atom_indices] - reference_z
    charges = charges_array[atom_indices]
    hist, _ = np.histogram(positions_z, bins=bins, weights=charges)
    return hist


def process_frame(frame_xyz_z, cation_idx, anion_idx, solvent_idx, charges, ref_z, bins):
    """Process single frame for all species."""
    return (
        compute_charge_histogram(frame_xyz_z, cation_idx, charges, ref_z, bins),
        compute_charge_histogram(frame_xyz_z, anion_idx, charges, ref_z, bins),
        compute_charge_histogram(frame_xyz_z, solvent_idx, charges, ref_z, bins),
    )


# ======================================================================
#  Step 1: Charge Density Analysis
# ======================================================================
def run_charge_density(traj_file, top_file, electrode_pdb, ffdir, output_dir, voltage_label):
    """
    來自 charge_density1D_V10.py —
    計算陰/陽離子/溶劑的電荷密度直方圖，
    輸出 hist_q_total_{V}.dat 等檔案到 output_dir。
    回傳 (framestart, total_frames) 供 Step 2 使用。
    """
    print()
    print("=" * 70)
    print("    Step 1: Charge Density Analysis (V10)")
    print("=" * 70)
    print()

    temperature = 300

    # --- Load trajectory ---
    print("=== Loading Trajectory ===")
    print(f"  Trajectory: {traj_file}")
    print(f"  Topology:   {top_file}")

    u = mda.Universe(top_file, traj_file)
    total_frames = len(u.trajectory)
    framestart = total_frames // 2
    frameCount = total_frames - framestart
    frameend = framestart + frameCount

    print(f"  ✓ Loaded successfully")
    print(f"    Total frames:    {total_frames}")
    print(f"    Analysis frames: {framestart} to {frameend} ({frameCount} frames)")
    print(f"    Time range:      {u.trajectory[framestart].time:.2f}"
          f" - {u.trajectory[frameend-1].time:.2f} ps")
    print()

    # --- Detect electrode boundaries ---
    print("=== Detecting Electrode Boundaries ===")
    print(f"  Loading electrode structure: {electrode_pdb}")

    u_electrode = mda.Universe(electrode_pdb)
    electrode = u_electrode.select_atoms("resname grpc")

    if len(electrode) == 0:
        raise ValueError(
            f"No electrode atoms found with resname 'grpc'\n"
            f"Available residues: {set(res.resname for res in u_electrode.residues)}"
        )

    z_positions_electrode = electrode.positions[:, 2]
    z_min_angstrom = z_positions_electrode.min()
    z_max_angstrom = z_positions_electrode.max()
    z_range_angstrom = z_max_angstrom - z_min_angstrom

    z_min_nm = z_min_angstrom / 10.0
    z_max_nm = z_max_angstrom / 10.0
    cell_dist = z_range_angstrom / 10.0

    print(f"  ✓ Electrode atoms (grpc): {len(electrode)}")
    print(f"  ✓ Z range (Å):            {z_min_angstrom:.2f} to {z_max_angstrom:.2f} Å")
    print(f"  ✓ Z range (nm):           {z_min_nm:.4f} to {z_max_nm:.4f} nm")
    print(f"  ✓ Electrode spacing:      {cell_dist:.4f} nm")
    print()

    # --- Load OpenMM system for charges ---
    print("=== Loading Force Field ===")
    pdb = PDBFile(electrode_pdb)
    pdb.topology.loadBondDefinitions(ffdir + 'graph_residue_c.xml')
    pdb.topology.loadBondDefinitions(ffdir + 'graph_residue_n.xml')
    pdb.topology.loadBondDefinitions(ffdir + 'sapt_residues.xml')
    pdb.topology.createStandardBonds()

    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField(
        ffdir + 'graph_c_freeze.xml',
        ffdir + 'graph_n_freeze.xml',
        ffdir + 'sapt_noDB_2sheets.xml',
    )
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedCutoff=1.4 * nanometer,
        constraints=None,
        rigidWater=True,
    )

    nbondedForce = [
        f for f in [system.getForce(i) for i in range(system.getNumForces())]
        if type(f) == NonbondedForce
    ][0]

    integ_md = DrudeLangevinIntegrator(
        temperature, 1 / picosecond, 1 * kelvin, 1 / picosecond, 0.001 * picoseconds
    )
    platform = Platform.getPlatformByName('CPU')
    simmd = Simulation(modeller.topology, system, integ_md, platform)

    boxVecs = simmd.topology.getPeriodicBoxVectors()
    crossBox = np.cross(boxVecs[0], boxVecs[1])
    sheet_area = np.dot(crossBox, crossBox) ** 0.5 / nanometer ** 2

    print(f"  ✓ Force field loaded")
    print(f"    Sheet area: {sheet_area:.6f} nm²")
    print()

    # --- Auto-detect ion types ---
    print("=== Detecting Ion Types ===")
    unique_residues = set(res.name for res in simmd.topology.residues())
    print(f"  Residue types: {sorted(unique_residues)}")

    namecat = next((cat for cat in CATION_TYPES if cat in unique_residues), None)
    namean = next((an for an in ANION_TYPES if an in unique_residues), None)

    if namecat is None:
        raise ValueError(f"No cation detected in residues: {sorted(unique_residues)}")
    if namean is None:
        raise ValueError(f"No anion detected in residues: {sorted(unique_residues)}")

    print(f"  ✓ Cation: {namecat}")
    print(f"  ✓ Anion:  {namean}")
    print()

    # --- Extract atom indices ---
    print("=== Extracting Atom Indices ===")
    cation_list, anion_list, solvent_list = [], [], []
    for res in simmd.topology.residues():
        if res.name == namecat:
            cation_list.extend(atom.index for atom in res._atoms)
        elif res.name == namean:
            anion_list.extend(atom.index for atom in res._atoms)

    cation_idx = np.array(cation_list, dtype=np.int32)
    anion_idx = np.array(anion_list, dtype=np.int32)
    solvent_idx = np.array(solvent_list, dtype=np.int32)

    print(f"  Cation atoms:  {len(cation_idx)}")
    print(f"  Anion atoms:   {len(anion_idx)}")
    print(f"  Solvent atoms: {len(solvent_idx)}")
    print()

    assert len(cation_idx) > 0, "No cation atoms found!"
    assert len(anion_idx) > 0, "No anion atoms found!"
    assert np.max(cation_idx) < u.atoms.n_atoms, "Invalid cation index!"
    assert np.max(anion_idx) < u.atoms.n_atoms, "Invalid anion index!"
    if len(solvent_idx) > 0:
        assert np.max(solvent_idx) < u.atoms.n_atoms, "Invalid solvent index!"

    # --- Extract charges ---
    print("Extracting atomic charges...")
    num_particles = nbondedForce.getNumParticles()
    all_charges = np.zeros(num_particles, dtype=np.float64)
    for atom_idx in tqdm(range(num_particles), desc="Charges", ncols=70):
        q, sig, eps = nbondedForce.getParticleParameters(atom_idx)
        all_charges[atom_idx] = q._value
    print(f"  ✓ Extracted {num_particles} charges")
    print()

    # --- Setup histogram bins ---
    print("=== Setting Up Histogram Bins ===")
    target_dz = 0.01  # nm
    num_bins = int(cell_dist / target_dz)
    dz = (z_max_nm - z_min_nm) / num_bins
    bins = np.linspace(0, cell_dist, num_bins + 1)

    print(f"  ✓ Target bin width: {target_dz} nm")
    print(f"  ✓ Actual bin width: {dz:.6f} nm")
    print(f"  ✓ Number of bins:   {num_bins}")
    print(f"  ✓ Z range:          0 to {cell_dist:.4f} nm")
    print()

    # Pre-allocate
    hist_cat_total = np.zeros(num_bins, dtype=np.float64)
    hist_an_total = np.zeros(num_bins, dtype=np.float64)
    hist_solv_total = np.zeros(num_bins, dtype=np.float64)

    print(f"=== Analysis Parameters ===")
    print(f"  Bin width:   {dz:.6f} nm")
    print(f"  Bins:        {num_bins}")
    print(f"  Frames:      {frameCount}")
    print(f"  Sheet area:  {sheet_area:.6f} nm²")
    print(f"  Bin volume:  {sheet_area * dz:.6e} nm³")
    print()

    # --- Main calculation loop ---
    start_time = time.time()
    print("Computing charge density distributions...")
    print()

    for frame_idx in tqdm(range(framestart, frameend), desc="Frames", unit="fr", ncols=70):
        u.trajectory[frame_idx]
        frame_xyz_z = u.atoms.positions[:, 2] / 10.0  # Å -> nm
        reference_z = frame_xyz_z[1]

        hist_cat, hist_an, hist_solv = process_frame(
            frame_xyz_z, cation_idx, anion_idx, solvent_idx,
            all_charges, reference_z, bins
        )
        hist_cat_total += hist_cat
        hist_an_total += hist_an
        hist_solv_total += hist_solv

    elapsed = time.time() - start_time
    print(f"\n✓ Computation complete")
    print(f"  Total time:  {elapsed:.2f} s")
    print(f"  Per frame:   {elapsed / frameCount * 1000:.2f} ms")
    print(f"  Throughput:  {frameCount / elapsed:.1f} frames/s")
    print()

    # --- Calculate densities ---
    avg_charge_cat = hist_cat_total / frameCount
    avg_charge_an = hist_an_total / frameCount
    avg_charge_solv = hist_solv_total / frameCount
    avg_charge_total = avg_charge_cat + avg_charge_an + avg_charge_solv

    bin_volume = sheet_area * dz
    density_cat = avg_charge_cat / bin_volume
    density_an = avg_charge_an / bin_volume
    density_solv = avg_charge_solv / bin_volume
    density_total = avg_charge_total / bin_volume

    z_centers = (bins[:-1] + bins[1:]) / 2
    z_centers_angstrom = (z_centers + z_min_nm) * 10.0

    print("=== Charge Density Statistics ===")
    print(f"  Cation:  [{density_cat.min():.6e}, {density_cat.max():.6e}] C/nm³")
    print(f"  Anion:   [{density_an.min():.6e}, {density_an.max():.6e}] C/nm³")
    print(f"  Solvent: [{density_solv.min():.6e}, {density_solv.max():.6e}] C/nm³")
    print(f"  Total:   [{density_total.min():.6e}, {density_total.max():.6e}] C/nm³")
    print()

    # --- Write output files ---
    print("Writing results...")
    output_files = [
        (f"hist_q_cat_{voltage_label}.dat", density_cat, "cation"),
        (f"hist_q_an_{voltage_label}.dat", density_an, "anion"),
        (f"hist_q_solv_{voltage_label}.dat", density_solv, "solvent"),
        (f"hist_q_total_{voltage_label}.dat", density_total, "total"),
    ]
    for filename, density, species in output_files:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            for z_val, rho in zip(z_centers_angstrom, density):
                print(f'{z_val:5.8f}  {rho:5.8f}', file=f)
        print(f"  ✓ {filepath}")

    # Copy electrode PDB into output_dir for Poisson step
    pdb_dest = os.path.join(output_dir, os.path.basename(electrode_pdb))
    shutil.copy2(electrode_pdb, pdb_dest)
    print(f"  ✓ Copied {electrode_pdb} → {pdb_dest}")
    print()

    # --- Plot charge density ---
    print("Generating charge density plot...")
    sns.set_theme(style="ticks", context="notebook", font_scale=1.1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=300)

    # Individual species
    ax1.plot(z_centers_angstrom, density_cat, label=f'Cation ({namecat})',
             color='#1f77b4', linewidth=2.5)
    ax1.fill_between(z_centers_angstrom, density_cat, alpha=0.2, color='#1f77b4')
    ax1.plot(z_centers_angstrom, density_an, label=f'Anion ({namean})',
             color='#d62728', linewidth=2.5)
    ax1.fill_between(z_centers_angstrom, density_an, alpha=0.2, color='#d62728')
    if len(solvent_idx) > 0 and np.any(density_solv != 0):
        ax1.plot(z_centers_angstrom, density_solv, label='Solvent',
                 color='#9467bd', linewidth=2.5)
        ax1.fill_between(z_centers_angstrom, density_solv, alpha=0.2, color='#9467bd')

    ax1.axvline(x=z_min_angstrom, color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Anode ({z_min_angstrom:.1f} Å)')
    ax1.axvline(x=z_max_angstrom, color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Cathode ({z_max_angstrom:.1f} Å)')
    ax1.set_xlabel('Z Position (Å)', fontweight='bold')
    ax1.set_ylabel('Charge Density (C/nm³)', fontweight='bold')
    ax1.set_xlim(z_min_angstrom - 5, z_max_angstrom + 5)
    ax1.set_ylim(-15, 25)
    ax1.set_title('Individual Species Charge Density', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
    sns.despine(ax=ax1)

    # Total
    ax2.plot(z_centers_angstrom, density_total, label='Total', color='#333333', linewidth=2.5)
    ax2.fill_between(z_centers_angstrom, density_total, alpha=0.2, color='#333333')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.4, linewidth=1.5)
    ax2.axvline(x=z_min_angstrom, color='#2ca02c', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Anode ({z_min_angstrom:.1f} Å)')
    ax2.axvline(x=z_max_angstrom, color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Cathode ({z_max_angstrom:.1f} Å)')
    ax2.set_xlabel('Z Position (Å)', fontweight='bold')
    ax2.set_ylabel('Charge Density (C/nm³)', fontweight='bold')
    ax2.set_xlim(z_min_angstrom - 5, z_max_angstrom + 5)
    ax2.set_ylim(-15, 25)
    ax2.set_title('Total Charge Density', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
    sns.despine(ax=ax2)

    fig.suptitle(f'Charge Density Analysis - {namecat}/{namean}',
                 fontsize=18, fontweight='black', y=0.98)
    plt.tight_layout()
    plot_filename = f"charge_density_{voltage_label}.png"
    fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
    print(f"  ✓ {plot_filename}")
    plt.close(fig)
    print()

    print("✓ Step 1 complete — Charge Density Analysis")
    print()

    return framestart, total_frames


# ======================================================================
#  Step 2: Search Charges from Log
# ======================================================================
def run_search_charges(log_file, framestart, total_frames, output_dir, voltage_label):
    """
    來自 search_charges_all.ipynb —
    從 energy log 中提取電極電荷，
    自動偵測兩種 log 格式（'X iteration' / 'Iteration X/Y'）。
    輸出 charges_output_{V}.dat 到 output_dir。
    """
    print()
    print("=" * 70)
    print("    Step 2: Search Electrode Charges")
    print("=" * 70)
    print()

    # Auto-detect log file if not specified
    if log_file is None:
        log_files = glob.glob("*.log")
        if not log_files:
            raise FileNotFoundError("找不到 .log 檔案，請用 --log 指定路徑")
        log_file = log_files[0]
        print(f"  Auto-detected log file: {log_file}")
    else:
        print(f"  Log file: {log_file}")

    with open(log_file) as f:
        lines = f.readlines()

    # --- Auto-detect log format ---
    # Format A: "X iteration"     (e.g. "0 iteration", "6776 iteration")
    # Format B: "Iteration X/Y"   (e.g. "Iteration 256/512")
    pattern_A = re.compile(r'^(\d+) iteration')
    pattern_B = re.compile(r'^Iteration (\d+)/(\d+)')

    first_iteration = None
    last_iteration = None
    log_format = None

    for line in lines:
        match_A = pattern_A.match(line)
        match_B = pattern_B.match(line)
        if match_A:
            log_format = 'A'
            current_iter = int(match_A.group(1))
            if first_iteration is None:
                first_iteration = current_iter
            last_iteration = current_iter
        elif match_B:
            log_format = 'B'
            current_iter = int(match_B.group(1))
            if first_iteration is None:
                first_iteration = current_iter
            last_iteration = current_iter

    if first_iteration is None or last_iteration is None:
        raise ValueError("在 log 檔案中找不到 iteration 標記")

    total_iterations = last_iteration - first_iteration + 1
    format_name = '"X iteration"' if log_format == 'A' else '"Iteration X/Y"'

    print(f"  ✓ Log format detected:    {format_name}")
    print(f"  ✓ Iteration range:        {first_iteration} ~ {last_iteration}")
    print(f"  ✓ Total iterations:       {total_iterations}")

    # 計算後半段（與 charge_density 的 frame 選擇一致）
    mid_iteration = (first_iteration + last_iteration) // 2
    start_iteration = mid_iteration
    end_iteration = last_iteration

    print(f"  ✓ Analysis range (2nd half): {start_iteration} ~ {end_iteration}")
    print()

    # Build search strings based on format
    if log_format == 'A':
        start_marker = f"{start_iteration} iteration"
        end_marker = f"{end_iteration} iteration"
    else:  # format B
        start_marker = f"Iteration {start_iteration}/"
        end_marker = f"Iteration {end_iteration}/"

    # Find start/end positions
    first_iteration_index = 0
    last_iteration_index = 0

    for idx, line in enumerate(lines):
        if log_format == 'A':
            if line.strip() == start_marker:
                first_iteration_index = idx + 1
                print(f"  起始位置: 行 {first_iteration_index} — {line.strip()}")
                break
        else:
            if start_marker in line:
                first_iteration_index = idx + 1
                print(f"  起始位置: 行 {first_iteration_index} — {line.strip()}")
                break

    if first_iteration_index == 0:
        raise ValueError(f"未找到起始 iteration: {start_iteration}")

    for idx, line in enumerate(lines):
        if log_format == 'A':
            if line.strip() == end_marker:
                last_iteration_index = idx + 1
                print(f"  終止位置: 行 {last_iteration_index} — {line.strip()}")
                break
        else:
            if end_marker in line:
                last_iteration_index = idx + 1
                print(f"  終止位置: 行 {last_iteration_index} — {line.strip()}")
                break

    if last_iteration_index == 0:
        raise ValueError(f"未找到終止 iteration: {end_iteration}")

    # Extract charges
    charges_list = []
    for z in range(first_iteration_index, last_iteration_index):
        if "Q_numeric , Q_analytic charges on  anode" in lines[z]:
            charge_values = lines[z].strip().split()
            if len(charge_values) > 6:
                charges_list.append(charge_values[6])

    output_file = os.path.join(output_dir, f"charges_output_{voltage_label}.dat")
    if charges_list:
        with open(output_file, "w") as f:
            for charge in charges_list:
                f.write(charge + "\n")
        print(f"\n  ✓ 已儲存 {len(charges_list)} 筆 charges → {output_file}")
    else:
        raise ValueError("未在指定範圍內找到電極電荷數據")

    print()
    print("✓ Step 2 complete — Electrode Charges Extracted")
    print()

    return charges_list


# ======================================================================
#  Step 3: Poisson Analysis
# ======================================================================
def run_poisson(output_dir, voltage_label, Vapp, electrode_pdb_name):
    """
    來自 PoissonV4.py 增強版 —
    讀取 hist_q_total 和 charges_output，進行 Poisson 積分。
    包含 half-cell + full-cell 積分及 bulk E-field 校正。
    """
    print()
    print("=" * 70)
    print("    Step 3: Poisson Analysis (V4 Enhanced)")
    print("=" * 70)
    print()

    # Seaborn style setup
    sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1)
    sns.set_palette('deep')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'grid.color': '#dee2e6',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.edgecolor': '#495057',
        'axes.linewidth': 1.2,
        'font.family': 'sans-serif',
    })

    # Input files
    pdb_file = os.path.join(output_dir, electrode_pdb_name)
    electrolyte_charge_file = os.path.join(output_dir, f"hist_q_total_{voltage_label}.dat")
    electrode_charge_file = os.path.join(output_dir, f"charges_output_{voltage_label}.dat")

    print(f"  PDB file:             {pdb_file}")
    print(f"  Electrolyte charges:  {electrolyte_charge_file}")
    print(f"  Electrode charges:    {electrode_charge_file}")
    print()

    # --- Read average electrode charge ---
    electrode_charges = []
    with open(electrode_charge_file) as f:
        for line in f:
            electrode_charges.append(float(line))
    avg_electrode_charge = np.average(electrode_charges)
    print(f"  Average electrode charge: {avg_electrode_charge}")

    # --- Get cell parameters from PDB ---
    u = Universe(pdb_file, pdb_file)
    atoms = u.select_atoms('resname grpc')
    z = atoms.positions[:, 2]
    z_left = z.min()
    z_right = z.max()
    Lcell = z_right - z_left
    Lgap = u.trajectory[0].triclinic_dimensions[2][2] - Lcell
    surfaceArea = np.linalg.norm(
        np.cross(u.trajectory[0].triclinic_dimensions[0],
                 u.trajectory[0].triclinic_dimensions[1])
    )

    print(f"=== Cell Parameters ===")
    print(f"  Voltage:      {Vapp} V")
    print(f"  Lcell:        {Lcell:.2f} Å")
    print(f"  Lgap:         {Lgap:.2f} Å")
    print(f"  Surface Area: {surfaceArea:.2f} Å²")
    print()

    # --- Bins calculation (consistent with charge_density1D_V10.py) ---
    print("=== Bins Calculation ===")
    cell_dist_nm = Lcell / 10.0
    target_dz = 0.01  # nm
    num_bins = int(cell_dist_nm / target_dz)
    dz_nm = cell_dist_nm / num_bins
    dz = dz_nm * 10.0  # 轉回 Å

    print(f"  Cell distance:   {cell_dist_nm:.4f} nm ({Lcell:.2f} Å)")
    print(f"  Number of bins:  {num_bins}")
    print(f"  Actual dz:       {dz_nm:.6f} nm ({dz:.4f} Å)")

    # --- Read electrolyte charge density ---
    with open(electrolyte_charge_file) as f:
        lines = f.readlines()

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
    for i in range(len(lines)):
        z_val = float(lines[i].split()[0])
        if z_val > z_left and z_val < z_right:
            z_dist.append(z_val)
            q_avg.append(float(lines[i].split()[1]) / 1000.0)  # e/nm³ → e/Å³

    # Reverse: integrate from right to left
    z_dist = np.flip(z_dist)
    q_avg = np.flip(q_avg)

    # =============================================
    #  Half-cell integration
    # =============================================
    print("=== Computing Poisson Profile (Half-Cell) ===")

    surfQ = avg_electrode_charge
    E_surface = 4 * PI * surfQ / (surfaceArea * ANG2BOHR ** 2)
    E_gap = -(Vapp * EV2HARTREE / (Lgap * ANG2BOHR))

    E_i = E_gap - E_surface
    E_z = [E_i]
    z1_dist = [z_right]

    max_length = int(len(q_avg) / 2)

    # First integration point
    E_i = E_i - 4 * PI * q_avg[0] * CONV * abs(z_dist[0] - z_right) * ANG2BOHR
    E_z.append(E_i)
    z1_dist.append(z_dist[0])

    for z_i in range(1, max_length):
        E_i = E_i - 4 * PI * q_avg[z_i] * CONV * dz * ANG2BOHR
        E_z.append(E_i)
        z1_dist.append(z_dist[z_i])

    print(f"  Integration points: {len(E_z)}")
    print(f"  E_field range: [{min(E_z):.6f}, {max(E_z):.6f}] e/bohr²")

    # Bulk E-field correction
    bulk_start = max_length // 4
    bulk_end = 3 * max_length // 4
    E_bulk_offset = np.mean(E_z[bulk_start:bulk_end])
    E_z = [e - E_bulk_offset for e in E_z]

    print(f"  Bulk E-field offset: {E_bulk_offset:.8f} e/bohr² (corrected)")
    print(f"  E_field range (corrected): [{min(E_z):.6f}, {max(E_z):.6f}] e/bohr²")

    # Voltage (half-cell)
    V_z = []
    V_i = -Vapp / 2.0 * EV2HARTREE
    V_z.append(-Vapp / 2.0)

    V_i = V_i + E_z[0] * abs(z_dist[0] - z_right) * ANG2BOHR
    V_z.append(V_i / EV2HARTREE)

    for z_i in range(1, max_length):
        V_i = V_i + E_z[z_i] * dz * ANG2BOHR
        V_z.append(V_i / EV2HARTREE)

    print(f"  Voltage range: [{min(V_z):.4f}, {max(V_z):.4f}] V")
    print()

    V_bulk = np.mean(V_z[190:210])
    print("=== Results (Half-Cell) ===")
    print(f"  V_anode (right):  {-Vapp / 2.0:.4f} V")
    print(f"  V_bulk:   {V_bulk:.4f} V")
    print(f"  delta_V:  {V_bulk - (-Vapp / 2.0):.4f} V")
    print()

    # =============================================
    #  Full-cell integration
    # =============================================
    print("=== Computing Full-Cell Poisson Profile ===")

    max_length_full = len(q_avg)

    E_i_full = E_gap - E_surface
    E_z_full = [E_i_full]
    z1_dist_full = [z_right]

    E_i_full = E_i_full - 4 * PI * q_avg[0] * CONV * abs(z_dist[0] - z_right) * ANG2BOHR
    E_z_full.append(E_i_full)
    z1_dist_full.append(z_dist[0])

    for z_i in range(1, max_length_full):
        E_i_full = E_i_full - 4 * PI * q_avg[z_i] * CONV * dz * ANG2BOHR
        E_z_full.append(E_i_full)
        z1_dist_full.append(z_dist[z_i])

    print(f"  Integration points (full): {len(E_z_full)}")
    print(f"  E_field range: [{min(E_z_full):.6f}, {max(E_z_full):.6f}] e/bohr²")

    # Bulk E-field correction for full cell
    bulk_start_full = max_length_full // 4
    bulk_end_full = 3 * max_length_full // 4
    E_bulk_offset_full = np.mean(E_z_full[bulk_start_full:bulk_end_full])
    E_z_full = [e - E_bulk_offset_full for e in E_z_full]

    print(f"  Bulk E-field offset: {E_bulk_offset_full:.8f} e/bohr² (corrected)")
    print(f"  E_field range (corrected): [{min(E_z_full):.6f}, {max(E_z_full):.6f}] e/bohr²")

    # Voltage (full cell)
    V_z_full = []
    V_i_full = -Vapp / 2.0 * EV2HARTREE
    V_z_full.append(-Vapp / 2.0)

    V_i_full = V_i_full + E_z_full[0] * abs(z_dist[0] - z_right) * ANG2BOHR
    V_z_full.append(V_i_full / EV2HARTREE)

    for z_i in range(1, max_length_full):
        V_i_full = V_i_full + E_z_full[z_i] * dz * ANG2BOHR
        V_z_full.append(V_i_full / EV2HARTREE)

    print(f"  Voltage range (full): [{min(V_z_full):.4f}, {max(V_z_full):.4f}] V")

    V_bulk_full = np.mean(V_z_full[190:210])
    print(f"  V_bulk (full):  {V_bulk_full:.4f} V")
    print()

    # =============================================
    #  Generate Plots
    # =============================================
    print("=== Generating Plots ===")

    palette = sns.color_palette('deep')
    accent = sns.color_palette('Set2')

    # 1. Electric Field (Half-Cell)
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.set_title(f"Electric Field of Poisson Potential under {voltage_label} (Half-Cell)",
                  fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel("Z Position (Å)", fontsize=12)
    ax1.set_ylabel("Electric Field (e/bohr²)", fontsize=12)
    ax1.plot(z1_dist, E_z, color=palette[0], linewidth=2, alpha=0.9, label='E-field')
    ax1.fill_between(z1_dist, E_z, alpha=0.15, color=palette[0])
    ax1.axhline(y=0, color='#adb5bd', linestyle='-', linewidth=0.8, alpha=0.7)
    ax1.axvline(x=z_right, color=accent[2], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Cathode ({z_right:.1f} Å)')
    ax1.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    sns.despine(ax=ax1, left=False, bottom=False)
    plt.tight_layout()
    efield_filename = f'Electricfield_Poisson_{voltage_label}_3.png'
    fig1.savefig(efield_filename, dpi=600, bbox_inches='tight')
    print(f"  ✓ {efield_filename}")
    plt.close(fig1)

    # 2. Voltage (Half-Cell)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.set_title(f"Voltage of Poisson Potential under {voltage_label} (Half-Cell)",
                  fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel("Z Position (Å)", fontsize=12)
    ax2.set_ylabel("Voltage (V)", fontsize=12)
    ax2.plot(z1_dist, V_z, color=palette[3], linewidth=2, alpha=0.9, label='Voltage')
    ax2.fill_between(z1_dist, V_z, alpha=0.15, color=palette[3])
    ax2.axhline(y=0, color='#adb5bd', linestyle='-', linewidth=0.8, alpha=0.7)
    ax2.axvline(x=z_right, color=accent[2], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Cathode ({z_right:.1f} Å)')
    ax2.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    sns.despine(ax=ax2, left=False, bottom=False)
    plt.tight_layout()
    voltage_filename = f'Voltage_Poisson_{voltage_label}_3.png'
    fig2.savefig(voltage_filename, dpi=600, bbox_inches='tight')
    print(f"  ✓ {voltage_filename}")
    plt.close(fig2)

    # 3. Electric Field (Full Cell)
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    ax3.set_title(f"Electric Field (Full Cell) under {voltage_label}",
                  fontsize=14, fontweight='bold', pad=12)
    ax3.set_xlabel("Z Position (Å)", fontsize=12)
    ax3.set_ylabel("Electric Field (e/bohr²)", fontsize=12)
    ax3.plot(z1_dist_full, E_z_full, color=palette[0], linewidth=2, alpha=0.9, label='E-field')
    ax3.fill_between(z1_dist_full, E_z_full, alpha=0.15, color=palette[0])
    ax3.axhline(y=0, color='#adb5bd', linestyle='-', linewidth=0.8, alpha=0.7)
    ax3.axvline(x=z_left, color=accent[1], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Anode ({z_left:.1f} Å)')
    ax3.axvline(x=z_right, color=accent[2], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Cathode ({z_right:.1f} Å)')
    ax3.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    sns.despine(ax=ax3, left=False, bottom=False)
    plt.tight_layout()
    efield_full_filename = f'Electricfield_Poisson_{voltage_label}_fullcell.png'
    fig3.savefig(efield_full_filename, dpi=600, bbox_inches='tight')
    print(f"  ✓ {efield_full_filename}")
    plt.close(fig3)

    # 4. Voltage (Full Cell)
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    ax4.set_title(f"Voltage (Full Cell) under {voltage_label}",
                  fontsize=14, fontweight='bold', pad=12)
    ax4.set_xlabel("Z Position (Å)", fontsize=12)
    ax4.set_ylabel("Voltage (V)", fontsize=12)
    ax4.plot(z1_dist_full, V_z_full, color=palette[3], linewidth=2, alpha=0.9, label='Voltage')
    ax4.fill_between(z1_dist_full, V_z_full, alpha=0.15, color=palette[3])
    ax4.axhline(y=0, color='#adb5bd', linestyle='-', linewidth=0.8, alpha=0.7)
    ax4.axvline(x=z_left, color=accent[1], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Anode ({z_left:.1f} Å)')
    ax4.axvline(x=z_right, color=accent[2], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Cathode ({z_right:.1f} Å)')
    ax4.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    sns.despine(ax=ax4, left=False, bottom=False)
    plt.tight_layout()
    voltage_full_filename = f'Voltage_Poisson_{voltage_label}_fullcell.png'
    fig4.savefig(voltage_full_filename, dpi=600, bbox_inches='tight')
    print(f"  ✓ {voltage_full_filename}")
    plt.close(fig4)

    print()
    print("=== Final Results ===")
    print(f"  Half-cell:  V_bulk = {V_bulk:.4f} V,  ΔV = {V_bulk - (-Vapp / 2.0):.4f} V")
    print(f"  Full-cell:  V_bulk = {V_bulk_full:.4f} V")
    print()
    print("✓ Step 3 complete — Poisson Analysis")
    print()


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified Poisson Pipeline: charge_density → search_charges → Poisson",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python unified_poisson_pipeline.py -v 4.0
  python unified_poisson_pipeline.py -v 0 --skip-step search_charges --skip-step poisson
  python unified_poisson_pipeline.py -v 1.5 --ffdir ../ffdir/ --log energy.log
        """,
    )
    parser.add_argument("-v", "--voltage", type=float, required=True,
                        help="模擬電壓 (V)，例如 0, 1.5, 4.0")
    parser.add_argument("--traj", type=str, default="FV_NVT.dcd",
                        help="軌跡檔案 (預設: FV_NVT.dcd)")
    parser.add_argument("--top", type=str, default="for_openmm.pdb",
                        help="含 Drude 的拓撲檔 (預設: for_openmm.pdb)")
    parser.add_argument("--electrode-pdb", type=str, default="for_openmm_nodrudes.pdb",
                        help="無 Drude 的電極 PDB (預設: for_openmm_nodrudes.pdb)")
    parser.add_argument("--ffdir", type=str, default="../ffdir/",
                        help="力場資料夾路徑 (預設: ../ffdir/)")
    parser.add_argument("--log", type=str, default=None,
                        help="energy log 檔案 (預設: 自動搜尋 *.log)")
    parser.add_argument("--skip-step", type=str, action="append", default=[],
                        choices=["charge_density", "search_charges", "poisson"],
                        help="跳過指定步驟 (可多次使用)")

    args = parser.parse_args()

    Vapp = args.voltage
    voltage_label = make_voltage_label(Vapp)
    output_dir = f"./for_poisson_{voltage_label}/"
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  Unified Poisson Pipeline".center(68) + "║")
    print("║" + f"  Voltage: {Vapp} V ({voltage_label})".center(68) + "║")
    print("║" + f"  Output:  {output_dir}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  Steps to run: ", end="")
    steps = []
    if "charge_density" not in args.skip_step:
        steps.append("1:charge_density")
    if "search_charges" not in args.skip_step:
        steps.append("2:search_charges")
    if "poisson" not in args.skip_step:
        steps.append("3:poisson")
    print(" → ".join(steps) if steps else "(none)")
    print()

    framestart = None
    total_frames = None

    # ---- Step 1: Charge Density ----
    if "charge_density" not in args.skip_step:
        framestart, total_frames = run_charge_density(
            traj_file=args.traj,
            top_file=args.top,
            electrode_pdb=args.electrode_pdb,
            ffdir=args.ffdir,
            output_dir=output_dir,
            voltage_label=voltage_label,
        )
    else:
        print("[SKIP] Step 1: charge_density")

    # ---- Step 2: Search Charges ----
    if "search_charges" not in args.skip_step:
        if framestart is None or total_frames is None:
            # 如果跳過 Step 1，嘗試從軌跡檔推斷 frame 資訊
            print("  (Step 1 was skipped — loading trajectory to determine frame range)")
            u_tmp = mda.Universe(args.top, args.traj)
            total_frames = len(u_tmp.trajectory)
            framestart = total_frames // 2
            del u_tmp

        run_search_charges(
            log_file=args.log,
            framestart=framestart,
            total_frames=total_frames,
            output_dir=output_dir,
            voltage_label=voltage_label,
        )
    else:
        print("[SKIP] Step 2: search_charges")

    # ---- Step 3: Poisson ----
    if "poisson" not in args.skip_step:
        run_poisson(
            output_dir=output_dir,
            voltage_label=voltage_label,
            Vapp=Vapp,
            electrode_pdb_name=os.path.basename(args.electrode_pdb),
        )
    else:
        print("[SKIP] Step 3: poisson")

    # ---- Summary ----
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  Pipeline Complete!".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  中間檔案:  {output_dir}")
    print(f"  圖表:      ./charge_density_{voltage_label}.png")
    print(f"             ./Electricfield_Poisson_{voltage_label}_3.png")
    print(f"             ./Voltage_Poisson_{voltage_label}_3.png")
    print(f"             ./Electricfield_Poisson_{voltage_label}_fullcell.png")
    print(f"             ./Voltage_Poisson_{voltage_label}_fullcell.png")
    print()


if __name__ == "__main__":
    main()
