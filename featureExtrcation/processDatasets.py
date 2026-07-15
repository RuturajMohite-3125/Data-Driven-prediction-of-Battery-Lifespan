import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import interp1d
import torch
import gc

import tqdm as tqdm
import glob
import os
import string
from scipy.stats import kurtosis, skew


SOH_HORIZON = int(os.environ.get("SOH_HORIZON", "1500"))


MIN_CUR = 1e-1


DQDV_BINS = 1000

V_WINDOW_MIN = 2.4
V_WINDOW_MAX = 3.6


def _interp_nan(arr):
   
    arr = np.asarray(arr, dtype=np.float32).reshape(-1).copy()
    if arr.size == 0:
        return arr
    nans = ~np.isfinite(arr)
    if not nans.any():
        return arr
    idx = np.arange(arr.size)
    valid = ~nans
    if not valid.any():
        arr[:] = 0.0
        return arr
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr


def _dedupe_interp(x, y, grid):
   
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return np.zeros(grid.size, dtype=np.float32)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size < 2:
        return np.zeros(grid.size, dtype=np.float32)

    y_sum = np.zeros(unique_x.size, dtype=np.float64)
    counts = np.zeros(unique_x.size, dtype=np.float64)
    np.add.at(y_sum, inverse, y)
    np.add.at(counts, inverse, 1.0)
    unique_y = y_sum / np.maximum(counts, 1.0)

    return np.interp(grid, unique_x, unique_y).astype(np.float32)


class HUSTDataProcessor:
    def __init__(self, pkl_path, min_cycles=0):
        self.pkl_path = pkl_path
        self.min_cycles = min_cycles

    def _iter_pkl_cells(self):
        pkl_files = glob.glob(os.path.join(self.pkl_path, "*.pkl"))
        print(f"Found {len(pkl_files)} pkl files.")
        for file in tqdm.tqdm(pkl_files, desc="Processing PKL files"):
            stem = os.path.splitext(os.path.basename(file))[0]
            cell_name = stem.replace("MATR_", "") if stem.startswith("MATR_") else stem
            with open(file, 'rb') as f:
                data = pickle.load(f)
            cycle_data = data.get('cycle_data', None)
            if cycle_data is not None:
                yield cell_name, data, cycle_data
            del data, cycle_data
            gc.collect()

    def _collect_dqdv_grids(self, cell, cycles_data, n_bins=DQDV_BINS):
        """
        Fixed voltage/capacity grid for this cell's dQ/dV and dV/dQ, shared
        across all of its cycles. Voltage is pinned to
        [V_WINDOW_MIN, V_WINDOW_MAX] (rather than the cell's declared
        limits or a percentile estimate) so every cell uses the same
        window. The capacity grid is the 1st/99th percentile of discharge
        capacity within that voltage window, over the cell's first 50
        cycles.
        """
        vmin, vmax = V_WINDOW_MIN, V_WINDOW_MAX

        q_samples = []
        for item in cycles_data[:min(len(cycles_data), 50)]:
            if not isinstance(item, dict):
                continue
            voltage = np.asarray(item.get('voltage_in_V', []))
            current = np.asarray(item.get('current_in_A', []))
            capacity = np.asarray(item.get('discharge_capacity_in_Ah', []))
            n = min(voltage.size, current.size, capacity.size)
            if n < 2:
                continue
            discharge = (np.isfinite(voltage[:n]) & np.isfinite(current[:n])
                         & (current[:n] < -MIN_CUR)
                         & (voltage[:n] >= vmin) & (voltage[:n] <= vmax))
            if discharge.sum() >= 2:
                q_samples.append(capacity[:n][discharge])

        if q_samples:
            all_q = np.concatenate(q_samples)
            qmin = float(np.nanpercentile(all_q, 1))
            qmax = float(np.nanpercentile(all_q, 99))
        else:
            qmin, qmax = 0.0, 1.0
        if not (np.isfinite(qmin) and np.isfinite(qmax) and qmax > qmin):
            qmin, qmax = 0.0, 1.0

        v_grid = np.linspace(float(vmin), float(vmax), n_bins)
        q_grid = np.linspace(qmin, qmax, n_bins)
        return v_grid, q_grid

    def _process_single_cycle(self, item, v_grid=None, q_grid=None):
        voltage = np.asarray(item['voltage_in_V'])
        charge_capacity = np.asarray(item['charge_capacity_in_Ah'])
        discharge_capacity = np.asarray(item['discharge_capacity_in_Ah'])
        current = np.asarray(item['current_in_A'])

        v_window = (voltage >= V_WINDOW_MIN) & (voltage <= V_WINDOW_MAX)
        charge_mask = (current > MIN_CUR) & v_window
        discharge_mask = (current < -MIN_CUR) & v_window

        V_c = voltage[charge_mask]
        Q_c = charge_capacity[charge_mask]
        I_c = current[charge_mask]
        V_d = voltage[discharge_mask]
        Q_d = discharge_capacity[discharge_mask]
        I_d = current[discharge_mask]

        discharge_deriv = self.calculate_dQdV(V_d, Q_d, v_grid=v_grid, q_grid=q_grid)

       
        full_discharge_mask = current < -MIN_CUR
        Q_d_full = discharge_capacity[full_discharge_mask]
        soh_val = float(np.max(Q_d_full)) if len(Q_d_full) > 0 else 0.0

        return {
            'voltage_charge': V_c, 'capacity_charge': Q_c, 'current_charge': I_c,
            'voltage_discharge': V_d, 'capacity_discharge': Q_d, 'current_discharge': I_d,
        }, discharge_deriv, soh_val

    def _extract_cycle_features(self, cycle_data, deriv, deriv_10):
        if deriv is not None:
            dQdV = deriv["dQdV"]
            dQdV_10 = deriv_10["dQdV"] if deriv_10 is not None else np.array([])
            ref_max = float(np.max(dQdV_10)) if len(dQdV_10) > 0 else 0.0
            ref_min = float(np.min(dQdV_10)) if len(dQdV_10) > 0 else 0.0
            ref_var = float(np.var(dQdV_10)) if len(dQdV_10) > 0 else 0.0
            dQdV_max = float(np.max(dQdV)) - ref_max
            dQdV_min = float(np.min(dQdV)) - ref_min
            dQdV_var = float(np.var(dQdV)) - ref_var
        else:
            dQdV_max = dQdV_min = dQdV_var = 0.0

        I_c = cycle_data['current_charge']
        V_c = cycle_data['voltage_charge']
        Q_c = cycle_data['capacity_charge']
        I_d = cycle_data['current_discharge']
        V_d = cycle_data['voltage_discharge']
        Q_d = cycle_data['capacity_discharge']

        if all(len(a) > 0 for a in [I_c, V_c, Q_c, I_d, V_d, Q_d]):
            log_std_I    = float(20.0 * np.log10(np.std(I_c).clip(1e-9)))
            log_std_qc   = float(20.0 * np.log10(np.std(Q_c).clip(1e-9)))
            log_std_V    = float(20.0 * np.log10(np.std(V_c).clip(1e-9)))
            log_std_Id   = float(20.0 * np.log10(np.std(I_d).clip(1e-9)))
            log_std_qd   = float(20.0 * np.log10(np.std(Q_d).clip(1e-9)))
            log_std_Vd   = float(20.0 * np.log10(np.std(V_d).clip(1e-9)))
            max_Q_c      = float(np.max(Q_c))
            max_Q_d      = float(np.max(Q_d))
            coulombic_eff = max_Q_d / max(max_Q_c, 1e-9)
            v_drop_start  = float(V_d[0] - V_d[-1]) if len(V_d) > 1 else 0.0
            return [
                dQdV_max, dQdV_min, dQdV_var,
                log_std_I, log_std_qc, log_std_V,
                log_std_Id, log_std_qd, log_std_Vd,
                float(np.min(I_c)), float(np.max(I_c)),
                float(np.min(V_c)), float(np.max(V_c)),
                float(np.min(I_d)), float(np.max(I_d)),
                float(np.min(V_d)), float(np.max(V_d)),
                max_Q_d, float(kurtosis(V_d)),
                coulombic_eff, v_drop_start,
            ]
        return [0] * 21

    def process_and_extract(self, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if cache and ('soh_traj' not in cache[0] or 'dqdv_curves' not in cache[0]):
                print("Cache missing 'soh_traj'/'dqdv_curves' field — regenerating cache...")
            else:
                print(f"Loaded {len(cache)} cells from cache.")
                return cache

        cells_cache = []

        for cell_name, cell, cycles_data in self._iter_pkl_cells():
            n_cycles = len(cycles_data)

            if n_cycles < self.min_cycles:
                del cell, cycles_data
                continue

            v_grid, q_grid = self._collect_dqdv_grids(cell, cycles_data)
            del cell

            deriv_10 = None
            if n_cycles > 10 and isinstance(cycles_data[10], dict):
                _, deriv_10, _ = self._process_single_cycle(cycles_data[10], v_grid=v_grid, q_grid=q_grid)

            soh_list = []
            cell_features = []
            dqdv_curves = np.full((n_cycles, DQDV_BINS), np.nan, dtype=np.float32)
            dvdq_curves = np.full((n_cycles, DQDV_BINS), np.nan, dtype=np.float32)

            for idx, item in enumerate(cycles_data):
                if not isinstance(item, dict):
                    soh_list.append(0.0)
                    cell_features.append([0.0] * 18)
                    continue

                processed, deriv, soh_val = self._process_single_cycle(item, v_grid=v_grid, q_grid=q_grid)
                soh_list.append(soh_val)

                feats = self._extract_cycle_features(processed, deriv, deriv_10)
                feats.append(soh_val)
                cell_features.append(feats)

                if deriv is not None:
                    dqdv_curves[idx] = deriv["dQdV"]
                    dvdq_curves[idx] = deriv["dVdQ"]

            del cycles_data
            gc.collect()

            soh_array = np.asarray(soh_list, dtype=np.float32)
            eol_threshold = 0.88
            peak_idx = int(np.argmax(soh_array))
            post_peak = soh_array[peak_idx:]
            below = np.where(post_peak < eol_threshold)[0]
            eol_cycle = int(peak_idx + below[0]) if len(below) > 0 else len(soh_array)


            peak_cap = float(np.max(soh_array)) if np.max(soh_array) > 1e-8 else 1.0
            soh_norm = soh_array / peak_cap
            soh_traj = np.full(SOH_HORIZON, -1.0, dtype=np.float32)
            end = min(eol_cycle, SOH_HORIZON)
            soh_traj[:end] = soh_norm[:end]

            cells_cache.append({
                'cell_name': cell_name,
                'features': np.asarray(cell_features, dtype=np.float32),
                'soh': soh_array,
                'soh_traj': soh_traj,  
                'eol': eol_cycle,
                'num_cycles': n_cycles,
                'v_grid': v_grid,               
                'q_grid': q_grid,               
                'dqdv_curves': dqdv_curves,      
                'dvdq_curves': dvdq_curves,      
            })

            del soh_list, soh_array, cell_features, dqdv_curves, dvdq_curves
            gc.collect()

        print(f"Processed {len(cells_cache)} cells.")
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(cells_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cache saved to {cache_path}")

        return cells_cache

    def plot_discharge_capacity_vs_voltage(self, cell_ids, cycle_ids, max_pts=500):
        """
        Plot discharge capacity vs voltage for selected cells and cycles.

        cell_ids  : list of cell names (str) or 0-based indices (int) relative to
                    the sorted list of pkl files in self.pkl_path.
        cycle_ids : list of 0-based cycle indices to plot for every selected cell.
        max_pts   : max points per cycle before downsampling (default 500).

        Each cell gets one subplot; cycles are coloured by index (plasma colormap).
        """
        pkl_files = glob.glob(os.path.join(self.pkl_path, "*.pkl"))
        file_map = {}
        for f in sorted(pkl_files):
            stem = os.path.splitext(os.path.basename(f))[0]
            name = stem.replace("MATR_", "") if stem.startswith("MATR_") else stem
            file_map[name] = f
        all_names = sorted(file_map.keys())

        resolved = []
        for cid in cell_ids:
            if isinstance(cid, int):
                if 0 <= cid < len(all_names):
                    resolved.append(all_names[cid])
                else:
                    print(f"Warning: index {cid} out of range (0..{len(all_names)-1}).")
            else:
                if cid in file_map:
                    resolved.append(cid)
                else:
                    print(f"Warning: cell '{cid}' not found in pkl path.")

        if not resolved:
            print("No valid cells selected.")
            return

        use_legend = len(cycle_ids) <= 8
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=min(cycle_ids), vmax=max(cycle_ids))

        n_cells = len(resolved)
        n_cols = min(4, n_cells)
        n_rows = int(np.ceil(n_cells / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for i, name in enumerate(resolved):
            ax = flat_axes[i]
            print(f"Loading cell '{name}' ...")
            with open(file_map[name], 'rb') as f:
                data = pickle.load(f)
            cycle_data = data.get('cycle_data', [])
            del data

            for cyc_idx in cycle_ids:
                if cyc_idx >= len(cycle_data) or not isinstance(cycle_data[cyc_idx], dict):
                    print(f"  Cycle {cyc_idx} not available for cell '{name}'.")
                    continue

                item = cycle_data[cyc_idx]
                voltage  = np.asarray(item['voltage_in_V'])
                current  = np.asarray(item['current_in_A'])
                cap_d    = np.asarray(item['discharge_capacity_in_Ah'])

                idx = np.where(current < 0)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > max_pts:
                    idx = idx[np.linspace(0, len(idx) - 1, max_pts, dtype=int)]

                color = cmap(norm(cyc_idx))
                label = f"Cycle {cyc_idx}" if use_legend else "_nolegend_"
                ax.plot(voltage[idx], cap_d[idx], color=color, linewidth=1.2,
                        alpha=0.85, label=label)

            del cycle_data
            gc.collect()

            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Discharge Capacity (Ah)")
            ax.set_title(f"Discharge Capacity vs Voltage — Cell {name}")
            ax.grid(True, alpha=0.4)
            if use_legend:
                ax.legend(fontsize=8)

        for j in range(n_cells, len(flat_axes)):
            flat_axes[j].set_visible(False)

        if not use_legend:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=flat_axes[:n_cells].tolist(), label="Cycle Index")

        plt.suptitle("Discharge Capacity vs Voltage — Selected Cells & Cycles", fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_voltage_kurtosis(self, cells_cache, cell_ids):
        """
        Plot discharge-voltage kurtosis vs cycle index in a grid, one subplot
        per selected cell, across all of each cell's cycles.

        cells_cache : list of cell dicts as returned by process_and_extract().
        cell_ids    : list of cell names (str) or 0-based indices (int) into
                      cells_cache.

        Points are coloured by cycle index; a single colorbar (scaled
        0..max EOL among the selected cells) is placed outside the grid on
        the right. Grid has a fixed 4 columns.
        """
        resolved = []
        for cid in cell_ids:
            if isinstance(cid, int):
                if 0 <= cid < len(cells_cache):
                    resolved.append(cells_cache[cid])
                else:
                    print(f"Warning: index {cid} out of range (0..{len(cells_cache)-1}).")
            else:
                cell = next((c for c in cells_cache if c['cell_name'] == cid), None)
                if cell is None:
                    print(f"Warning: cell '{cid}' not found in cells_cache.")
                else:
                    resolved.append(cell)

        if not resolved:
            print("No valid cells selected.")
            return

        kurtosis_idx = 18
        cmap = plt.cm.viridis
        max_eol = max(c['eol'] for c in resolved)
        norm = plt.Normalize(vmin=0, vmax=max(max_eol, 1))

        n_cells = len(resolved)
        n_cols = 4
        n_rows = int(np.ceil(n_cells / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for i, cell in enumerate(resolved):
            ax = flat_axes[i]
            features = cell['features']
            eol = cell['eol']
            name = cell['cell_name']

            n_cycles = min(features.shape[0], eol) if eol > 0 else features.shape[0]
            if n_cycles == 0:
                ax.set_visible(False)
                continue

            cycles = np.arange(n_cycles)
            values = features[:n_cycles, kurtosis_idx]

            ax.scatter(cycles, values, c=cycles, cmap=cmap, norm=norm, s=8, alpha=0.85)
            ax.set_xlabel("Cycle Index")
            ax.set_ylabel("Kurtosis")
            ax.set_title(f"{name} (EOL: {eol})")
            ax.grid(True, alpha=0.4)

        for j in range(n_cells, len(flat_axes)):
            flat_axes[j].set_visible(False)

        plt.suptitle("Discharge Voltage Kurtosis vs Cycle - Selected Cells", fontsize=13)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, label="Cycle Index (cell cycle life)")
        cbar.set_ticks([0, int(norm.vmax)])
        cbar.ax.set_yticklabels([f"{int(t)}" for t in cbar.get_ticks()])

        plt.show()

    def calculate_dQdV(self, voltage, capacity, v_min=None, v_max=None, n_bins=DQDV_BINS,
                        v_grid=None, q_grid=None):
        """
        dQ/dV and dV/dQ via fixed-grid interpolation + np.gradient — same
        method as gen_bml_features.py's _cycle_qdlin/_dedupe_interp +
        np.gradient(qd_curve, voltage_grid), applied symmetrically for dV/dQ.

        v_grid / q_grid : optional precomputed grids shared across a cell's
        cycles (see _collect_dqdv_grids), matching gen_bml_features.py's
        single-grid-per-cell approach. If omitted, falls back to a grid
        built from this call's own voltage/capacity min/max (per-cycle grid).
        """
        voltage = np.asarray(voltage, dtype=np.float64)
        capacity = np.asarray(capacity, dtype=np.float64)

        valid = np.isfinite(voltage) & np.isfinite(capacity)
        voltage = voltage[valid]
        capacity = capacity[valid]

        if v_min is not None or v_max is not None:
            lo = v_min if v_min is not None else -np.inf
            hi = v_max if v_max is not None else np.inf
            vmask = (voltage >= lo) & (voltage <= hi)
            voltage = voltage[vmask]
            capacity = capacity[vmask]

        if (voltage.size < 20 or len(np.unique(voltage)) < 20
                or len(np.unique(capacity)) < 20):
            return None

        
        if v_grid is None:
            v_grid = np.linspace(float(np.min(voltage)), float(np.max(voltage)), n_bins)
        q_on_v = _dedupe_interp(voltage, capacity, v_grid)
        dQdV = _interp_nan(np.gradient(q_on_v, v_grid))

       
        if q_grid is None:
            q_grid = np.linspace(float(np.min(capacity)), float(np.max(capacity)), n_bins)
        v_on_q = _dedupe_interp(capacity, voltage, q_grid)
        dVdQ = _interp_nan(np.gradient(v_on_q, q_grid))

        return {
            "V_dQdV": v_grid,
            "dQdV": dQdV,
            "Q_dVdQ": q_grid,
            "dVdQ": dVdQ,
        }
    
    def plot_dQdV(self, dQdV_discharge):
        cell_index = 3
        plt.figure(figsize=(10, 6))

        cell_data = dQdV_discharge[cell_index]
        if cell_data is None:
            print(f"No valid dQ/dV data for cell index {cell_index}.")
            print(f"Type of cell_data: {type(cell_data)}")
            return
        print(f"Processing dQ/dV data for cell index {cell_index} with {len(cell_data)} cycles.")
        for dQdV in cell_data[10:100]:
            if dQdV is not None:
                voltage = dQdV["V_dQdV"]
                dQdV_values = dQdV["dQdV"]
                plt.plot(voltage, dQdV_values, alpha=0.5)
            else:
                print(f"No valid dQ/dV data for one of the cycles in cell index {cell_index}.")
        plt.xlabel("Voltage (V)")
        plt.ylabel("dQ/dV (Ah/V)")
        plt.title(f"dQ/dV Curves for Cell Index {cell_index}")
        
        plt.grid()
        plt.show()
        

    def plot_number_of_cycles(self, num_cycles):
        fig, ax = plt.subplots(figsize=(8, 5))

        num_cycles = np.sort(num_cycles)

        cell_indices = range(1, len(num_cycles) + 1)

        norm = plt.Normalize(
            vmin=0,
            vmax=max(num_cycles)
        )
        colors = plt.cm.viridis(norm(num_cycles))

        ax.bar(cell_indices, num_cycles, color=colors)

        ax.set_xlabel("Cell Index")
        ax.set_ylabel("Number of Cycles")
        ax.set_title("Number of Cycles per Cell")
        ax.grid(axis='y')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array(num_cycles)
        plt.colorbar(sm, ax=ax, label='Number of Cycles')

        plt.show()

    def plot_SoH(self, SoH, num_cycles):
        fig, ax = plt.subplots(figsize=(10, 6))
        max_cycles = max(num_cycles)

        for i, capacity in enumerate(SoH):

            if len(capacity) < 5:
                continue

            normalized = capacity / capacity[5]
            cycles = range(len(normalized))

            color = plt.cm.viridis(num_cycles[i] / max_cycles)
            ax.plot(cycles, normalized, color=color)

        sm = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=0, vmax=max_cycles)
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Number of Cycles')

        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Normalized Capacity (SoH)")
        ax.set_title("Capacity Trends (SoH)")
        ax.grid()
        plt.show()

    def plot_soh_all(self, cells_cache, normalize=True):
        """
        Plot SoH curves for all cells in cells_cache, coloured by EOL length.

        normalize : divide each cell's capacity by its peak value.
        """
        if not cells_cache:
            print("No cells in cache.")
            return

        eols = [c['eol'] for c in cells_cache]
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(eols), vmax=max(eols))

        fig, ax = plt.subplots(figsize=(12, 6))

        for cell in cells_cache:
            soh = cell['soh'].copy()
            peak = np.max(soh)
            if peak < 1e-8:
                continue
            
            start_window = soh[:10] / peak
            if np.max(start_window) < 0.9:
                continue
            if normalize:
                soh = soh / peak
            
            wl = min(51, len(soh) if len(soh) % 2 == 1 else len(soh) - 1)
            if wl >= 5:
                soh = savgol_filter(soh, window_length=wl, polyorder=3, mode='nearest')
            color = cmap(norm(cell['eol']))
            ax.plot(range(len(soh)), soh, color=color, linewidth=0.8, alpha=0.7)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="EOL Cycle")

        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Normalized Capacity (SoH)" if normalize else "Discharge Capacity (Ah)")
        ax.set_title(f"State of Health — All Cells ({len(cells_cache)} total)")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    def plot_log_dispersion_features(self, cells_cache, cell_id):
        """
        Plot the log-dispersion features (log_std_I, log_std_qc, log_std_V,
        log_std_Id, log_std_qd, log_std_Vd) vs cycle index for a single cell,
        across all of its cycles, laid out in a grid (one subplot per feature).

        cells_cache : list of cell dicts as returned by process_and_extract().
        cell_id     : cell name (str) or 0-based index (int) into cells_cache.

        Points are coloured by cycle index, with a colorbar scaled to the
        cell's own cycle life (0..EOL).
        """
        if isinstance(cell_id, int):
            if not (0 <= cell_id < len(cells_cache)):
                print(f"Warning: index {cell_id} out of range (0..{len(cells_cache)-1}).")
                return
            cell = cells_cache[cell_id]
        else:
            cell = next((c for c in cells_cache if c['cell_name'] == cell_id), None)
            if cell is None:
                print(f"Warning: cell '{cell_id}' not found in cells_cache.")
                return

        features = cell['features']
        eol = cell['eol']
        name = cell['cell_name']

        feature_names = ['log_std_I', 'log_std_qc', 'log_std_V',
                          'log_std_Id', 'log_std_qd', 'log_std_Vd']
        feature_idx = [3, 4, 5, 6, 7, 8]

        n_cycles = min(features.shape[0], eol) if eol > 0 else features.shape[0]
        if n_cycles == 0:
            print(f"No cycles available for cell '{name}'.")
            return
        cycles = np.arange(n_cycles)

        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=max(eol, 1))

        n_cols = 3
        n_rows = int(np.ceil(len(feature_names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        panel_labels = string.ascii_lowercase

        for i, (fname, fidx) in enumerate(zip(feature_names, feature_idx)):
            ax = flat_axes[i]
            values = features[:n_cycles, fidx]
            ax.scatter(cycles, values, c=cycles, cmap=cmap, norm=norm, s=10, alpha=0.85)
            ax.set_xlabel("Cycle Index")
            ax.set_ylabel(f"{fname} (dB)")
            ax.set_title(f"({panel_labels[i]}) {fname}")
            ax.grid(True, alpha=0.4)

        for j in range(len(feature_names), len(flat_axes)):
            flat_axes[j].set_visible(False)

        plt.suptitle(f"Log-Dispersion Features vs Cycle — Cell {name} (EOL: {eol})", fontsize=13)
        plt.tight_layout()
        fig.subplots_adjust(right=0.88)

        cax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, label="Cycle Index (cell cycle life)")
        cmin, cmax = int(cycles.min()), int(cycles.max())
        cbar.set_ticks(sorted({0, cmin, cmax, int(norm.vmax)}))
        cbar.ax.set_yticklabels([f"{int(t)}" for t in cbar.get_ticks()])

        plt.show()

    def plot_feature_eol_correlation(self, cells_cache, early_cycles=100):
        """
        Correlate per-cell engineered features (averaged over each cell's
        first `early_cycles` cycles) with EOL, and plot the resulting
        correlation matrix as a lower-triangular heatmap.

        cells_cache  : list of cell dicts as returned by process_and_extract().
        early_cycles : number of initial cycles averaged per cell to build the
                       per-cell feature vector (default 100).
        """
        feature_names = [
            'dQdV_max', 'dQdV_min', 'dQdV_var',
            'log_std_I', 'log_std_qc', 'log_std_V',
            'log_std_Id', 'log_std_qd', 'log_std_Vd',
            'min_I_c', 'max_I_c', 'min_V_c', 'max_V_c',
            'min_I_d', 'max_I_d', 'min_V_d', 'max_V_d',
            'max_Q_d', 'kurtosis_Vd', 'coulombic_eff', 'v_drop_start',
        ]

        rows = []
        for cell in cells_cache:
            features = cell['features']
            eol = cell['eol']
            n = min(early_cycles, features.shape[0])
            if n == 0:
                continue
            mean_feats = features[:n, :len(feature_names)].mean(axis=0)
            rows.append(list(mean_feats) + [eol])

        if not rows:
            print("No cells available to compute correlation.")
            return

        df = pd.DataFrame(rows, columns=feature_names + ['EOL'])
        corr = df.corr()

        labels = corr.columns.tolist()
        upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        corr_masked = np.ma.array(corr.to_numpy(), mask=upper_mask)

        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(corr_masked, cmap='coolwarm', vmin=-1, vmax=1)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(len(labels)):
            for j in range(i + 1):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6,
                        color='white' if abs(val) > 0.6 else 'black')

        fig.colorbar(im, ax=ax, label="Pearson Correlation", fraction=0.046, pad=0.04)
        ax.set_title(f"Feature Correlation with EOL (mean of first {early_cycles} cycles)",
                     fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_dqdv_selected(self, cell_ids, cycle_ids, v_min=2.4, v_max=3.6):
        """
        Load raw pkl files and plot dQ/dV curves for selected cells and cycles.

        cell_ids  : list of cell names (str) or 0-based indices (int) relative to
                    the sorted list of pkl files in self.pkl_path.
        cycle_ids : list of 0-based cycle indices to plot for every selected cell.
        v_min     : lower voltage cutoff (V) for dQ/dV computation.
        v_max     : upper voltage cutoff (V) for dQ/dV computation.
        """
        pkl_files = glob.glob(os.path.join(self.pkl_path, "*.pkl"))
        file_map = {}
        for f in sorted(pkl_files):
            stem = os.path.splitext(os.path.basename(f))[0]
            name = stem.replace("MATR_", "") if stem.startswith("MATR_") else stem
            file_map[name] = f
        all_names = sorted(file_map.keys())

        resolved = []
        for cid in cell_ids:
            if isinstance(cid, int):
                if 0 <= cid < len(all_names):
                    resolved.append(all_names[cid])
                else:
                    print(f"Warning: index {cid} out of range (0..{len(all_names)-1}).")
            else:
                if cid in file_map:
                    resolved.append(cid)
                else:
                    print(f"Warning: cell '{cid}' not found in pkl path.")

        if not resolved:
            print("No valid cells selected.")
            return

        n_cells = len(resolved)
        use_legend = len(cycle_ids) <= 8
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=min(cycle_ids), vmax=max(cycle_ids))

        n_cols = min(4, n_cells)
        n_rows = int(np.ceil(n_cells / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for i, name in enumerate(resolved):
            ax = flat_axes[i]
            print(f"Loading cell '{name}' ...")
            with open(file_map[name], 'rb') as f:
                data = pickle.load(f)
            cycle_data = data.get('cycle_data', [])
            del data

            for cyc_idx in cycle_ids:
                if cyc_idx >= len(cycle_data) or not isinstance(cycle_data[cyc_idx], dict):
                    print(f"  Cycle {cyc_idx} not available for cell '{name}'.")
                    continue

                item = cycle_data[cyc_idx]
                voltage = np.asarray(item['voltage_in_V'])
                capacity = np.asarray(item['discharge_capacity_in_Ah'])
                current = np.asarray(item['current_in_A'])

                mask = current < 0
                result = self.calculate_dQdV(voltage[mask], capacity[mask], v_min=v_min, v_max=v_max)
                if result is None:
                    print(f"  dQ/dV could not be computed for cycle {cyc_idx} of cell '{name}'.")
                    continue

                color = cmap(norm(cyc_idx))
                label = f"Cycle {cyc_idx}" if use_legend else "_nolegend_"
                ax.plot(result["V_dQdV"], result["dQdV"], color=color, label=label,
                        linewidth=1.2, alpha=0.85)

            del cycle_data
            gc.collect()

            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("dQ/dV (Ah/V)")
            ax.set_title(f"dQ/dV — Cell {name}")
            ax.grid(True, alpha=0.4)
            if use_legend:
                ax.legend(fontsize=8)

        for j in range(n_cells, len(flat_axes)):
            flat_axes[j].set_visible(False)

        if not use_legend:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=flat_axes[:n_cells].tolist(), label="Cycle Index")

        plt.suptitle("dQ/dV Curves — Selected Cells & Cycles", fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_dqdv_from_cache(self, cells_cache, cell_ids, cycle_ids, v_min=None, v_max=None, curve="dqdv"):
        """
        Plot dQ/dV (or dV/dQ) curves for selected cells and cycles straight
        from a process_and_extract() cache — no raw pkl files are touched.
        Requires a cache generated after 'dqdv_curves'/'v_grid' were added
        (process_and_extract regenerates automatically if they're missing).

        cells_cache : list of cell dicts as returned by process_and_extract().
        cell_ids    : list of cell names (str) or 0-based indices (int) into
                      cells_cache.
        cycle_ids   : list of 0-based cycle indices to plot for every selected cell.
        v_min/v_max : optional voltage/capacity window to restrict the x-axis.
        curve       : "dqdv" (default) or "dvdq".
        """
        if curve not in ("dqdv", "dvdq"):
            raise ValueError("curve must be 'dqdv' or 'dvdq'")
        grid_key = "v_grid" if curve == "dqdv" else "q_grid"
        curves_key = "dqdv_curves" if curve == "dqdv" else "dvdq_curves"
        x_label = "Voltage (V)" if curve == "dqdv" else "Capacity (Ah)"
        y_label = "dQ/dV (Ah/V)" if curve == "dqdv" else "dV/dQ (V/Ah)"

        resolved = []
        for cid in cell_ids:
            if isinstance(cid, int):
                if 0 <= cid < len(cells_cache):
                    resolved.append(cells_cache[cid])
                else:
                    print(f"Warning: index {cid} out of range (0..{len(cells_cache)-1}).")
            else:
                cell = next((c for c in cells_cache if c['cell_name'] == cid), None)
                if cell is None:
                    print(f"Warning: cell '{cid}' not found in cells_cache.")
                else:
                    resolved.append(cell)

        if not resolved:
            print("No valid cells selected.")
            return

        missing = [c['cell_name'] for c in resolved if curves_key not in c or grid_key not in c]
        if missing:
            print(f"Warning: cache missing '{curves_key}'/'{grid_key}' for cells {missing} — "
                  f"delete the cache file and rerun process_and_extract() to regenerate.")
            return

        n_cells = len(resolved)
        use_legend = len(cycle_ids) <= 8
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=min(cycle_ids), vmax=max(cycle_ids))

        n_cols = min(4, n_cells)
        n_rows = int(np.ceil(n_cells / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for i, cell in enumerate(resolved):
            ax = flat_axes[i]
            name = cell['cell_name']
            grid = cell[grid_key]
            curves = cell[curves_key]

            xmask = np.ones(grid.shape, dtype=bool)
            if v_min is not None or v_max is not None:
                lo = v_min if v_min is not None else -np.inf
                hi = v_max if v_max is not None else np.inf
                xmask = (grid >= lo) & (grid <= hi)

            for cyc_idx in cycle_ids:
                if cyc_idx >= curves.shape[0]:
                    continue
                cyc_curve = curves[cyc_idx]
                if np.isnan(cyc_curve).all():
                    continue

                color = cmap(norm(cyc_idx))
                label = f"Cycle {cyc_idx}" if use_legend else "_nolegend_"
                ax.plot(grid[xmask], cyc_curve[xmask], color=color,
                        linewidth=1.0, alpha=0.8, label=label)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{curve.upper()} — Cell {name} (from cache)")
            ax.grid(True, alpha=0.4)
            if use_legend:
                ax.legend(fontsize=8)

        for j in range(n_cells, len(flat_axes)):
            flat_axes[j].set_visible(False)

        if not use_legend:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=flat_axes[:n_cells].tolist(), label="Cycle Index")

        plt.suptitle(f"{curve.upper()} Curves — Selected Cells & Cycles (from cache)", fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_current_profile(self, cell_ids, cycle_ids):
        """
        Plot charge and discharge current profiles for selected cells and cycles.

        cell_ids  : list of cell names (str) or 0-based indices (int) relative to
                    the sorted list of pkl files in self.pkl_path.
        cycle_ids : list of 0-based cycle indices to plot for every selected cell.

        Each cell gets one subplot. Within a subplot, cycles are coloured by index
        (plasma colormap). Charge phase (I > 0) is plotted solid, discharge (I < 0)
        is plotted dashed, both in the same cycle colour so phases are visually
        distinguishable but cycle progression is still clear.
        """
        pkl_files = glob.glob(os.path.join(self.pkl_path, "*.pkl"))
        file_map = {}
        for f in sorted(pkl_files):
            stem = os.path.splitext(os.path.basename(f))[0]
            name = stem.replace("MATR_", "") if stem.startswith("MATR_") else stem
            file_map[name] = f
        all_names = sorted(file_map.keys())

        resolved = []
        for cid in cell_ids:
            if isinstance(cid, int):
                if 0 <= cid < len(all_names):
                    resolved.append(all_names[cid])
                else:
                    print(f"Warning: index {cid} out of range (0..{len(all_names)-1}).")
            else:
                if cid in file_map:
                    resolved.append(cid)
                else:
                    print(f"Warning: cell '{cid}' not found in pkl path.")

        if not resolved:
            print("No valid cells selected.")
            return

        use_legend = len(cycle_ids) <= 8
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=min(cycle_ids), vmax=max(cycle_ids))

        n_cells = len(resolved)
        n_cols = min(4, n_cells)
        n_rows = int(np.ceil(n_cells / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for i, name in enumerate(resolved):
            ax = flat_axes[i]
            print(f"Loading cell '{name}' ...")
            with open(file_map[name], 'rb') as f:
                data = pickle.load(f)
            cycle_data = data.get('cycle_data', [])
            del data

            for cyc_idx in cycle_ids:
                if cyc_idx >= len(cycle_data) or not isinstance(cycle_data[cyc_idx], dict):
                    print(f"  Cycle {cyc_idx} not available for cell '{name}'.")
                    continue

                item = cycle_data[cyc_idx]
                current = np.asarray(item['current_in_A'])
                samples = np.arange(len(current))

                color = cmap(norm(cyc_idx))
                label = f"Cycle {cyc_idx}" if use_legend else "_nolegend_"

                charge_mask    = current >= 0
                discharge_mask = current <  0

                if charge_mask.any():
                    ax.plot(samples[charge_mask], current[charge_mask],
                            color=color, linestyle='-', linewidth=1.0,
                            alpha=0.85, label=label)
                    label = "_nolegend_"
                if discharge_mask.any():
                    ax.plot(samples[discharge_mask], current[discharge_mask],
                            color=color, linestyle='--', linewidth=1.0,
                            alpha=0.85, label=label)

            del cycle_data
            gc.collect()

            ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Current (A)")
            ax.set_title(f"Current Profile — Cell {name}")
            ax.grid(True, alpha=0.4)
            if use_legend:
                ax.legend(fontsize=8)

        for j in range(n_cells, len(flat_axes)):
            flat_axes[j].set_visible(False)

        if not use_legend:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=flat_axes[:n_cells].tolist(), label="Cycle Index")

        # shared legend patch for line style
        from matplotlib.lines import Line2D
        style_handles = [
            Line2D([0], [0], color='gray', linestyle='-',  linewidth=1.2, label='Charge (I ≥ 0)'),
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.2, label='Discharge (I < 0)'),
        ]
        fig.legend(handles=style_handles, loc='lower center', ncol=2,
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

        plt.suptitle("Charge / Discharge Current — Selected Cells & Cycles", fontsize=13)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.show()

    def plot_voltage_profile(self, cell_ids, cycle_ids):
        """
        Plot charge and discharge voltage profiles for selected cells and cycles.

        cell_ids  : list of cell names (str) or 0-based indices (int) relative to
                    the sorted list of pkl files in self.pkl_path.
        cycle_ids : list of 0-based cycle indices to plot for every selected cell.

        Each cell gets one subplot. Charge phase (I > 0) is plotted solid,
        discharge (I < 0) dashed, both in the same cycle colour.
        """
        pkl_files = glob.glob(os.path.join(self.pkl_path, "*.pkl"))
        file_map = {}
        for f in sorted(pkl_files):
            stem = os.path.splitext(os.path.basename(f))[0]
            name = stem.replace("MATR_", "") if stem.startswith("MATR_") else stem
            file_map[name] = f
        all_names = sorted(file_map.keys())

        resolved = []
        for cid in cell_ids:
            if isinstance(cid, int):
                if 0 <= cid < len(all_names):
                    resolved.append(all_names[cid])
                else:
                    print(f"Warning: index {cid} out of range (0..{len(all_names)-1}).")
            else:
                if cid in file_map:
                    resolved.append(cid)
                else:
                    print(f"Warning: cell '{cid}' not found in pkl path.")

        if not resolved:
            print("No valid cells selected.")
            return

        use_legend = len(cycle_ids) <= 8
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=min(cycle_ids), vmax=max(cycle_ids))

        n_cells = len(resolved)
        n_cols = min(4, n_cells)
        n_rows = int(np.ceil(n_cells / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for i, name in enumerate(resolved):
            ax = flat_axes[i]
            print(f"Loading cell '{name}' ...")
            with open(file_map[name], 'rb') as f:
                data = pickle.load(f)
            cycle_data = data.get('cycle_data', [])
            del data

            for cyc_idx in cycle_ids:
                if cyc_idx >= len(cycle_data) or not isinstance(cycle_data[cyc_idx], dict):
                    print(f"  Cycle {cyc_idx} not available for cell '{name}'.")
                    continue

                item = cycle_data[cyc_idx]
                voltage = np.asarray(item['voltage_in_V'])
                current = np.asarray(item['current_in_A'])
                samples = np.arange(len(voltage))

                color = cmap(norm(cyc_idx))
                label = f"Cycle {cyc_idx}" if use_legend else "_nolegend_"

                charge_mask    = current > 0
                discharge_mask = current < 0

                if charge_mask.any():
                    ax.plot(samples[charge_mask], voltage[charge_mask],
                            color=color, linestyle='-', linewidth=1.0,
                            alpha=0.85, label=label)
                    label = "_nolegend_"
                if discharge_mask.any():
                    ax.plot(samples[discharge_mask], voltage[discharge_mask],
                            color=color, linestyle='--', linewidth=1.0,
                            alpha=0.85, label=label)

            del cycle_data
            gc.collect()

            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Voltage (V)")
            ax.set_title(f"Voltage Profile — Cell {name}")
            ax.grid(True, alpha=0.4)
            if use_legend:
                ax.legend(fontsize=8)

        for j in range(n_cells, len(flat_axes)):
            flat_axes[j].set_visible(False)

        if not use_legend:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=flat_axes[:n_cells].tolist(), label="Cycle Index")

        from matplotlib.lines import Line2D
        style_handles = [
            Line2D([0], [0], color='gray', linestyle='-',  linewidth=1.2, label='Charge (I > 0)'),
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.2, label='Discharge (I < 0)'),
        ]
        fig.legend(handles=style_handles, loc='lower center', ncol=2,
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

        plt.suptitle("Charge / Discharge Voltage — Selected Cells & Cycles", fontsize=13)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.show()

    def print_charge_discharge_time(self, cell_ids, cycle_ids):
        """
        Print charge/discharge time spans for selected cells and cycles, using
        the same |current| > MIN_CUR filtering as _process_single_cycle (so
        rest-step noise isn't counted as charge/discharge).

        cell_ids  : list of cell names (str) or 0-based indices (int) relative to
                    the sorted list of pkl files in self.pkl_path.
        cycle_ids : list of 0-based cycle indices to print for every selected cell.
        """
        pkl_files = glob.glob(os.path.join(self.pkl_path, "*.pkl"))
        file_map = {}
        for f in sorted(pkl_files):
            stem = os.path.splitext(os.path.basename(f))[0]
            name = stem.replace("MATR_", "") if stem.startswith("MATR_") else stem
            file_map[name] = f
        all_names = sorted(file_map.keys())

        resolved = []
        for cid in cell_ids:
            if isinstance(cid, int):
                if 0 <= cid < len(all_names):
                    resolved.append(all_names[cid])
                else:
                    print(f"Warning: index {cid} out of range (0..{len(all_names)-1}).")
            else:
                if cid in file_map:
                    resolved.append(cid)
                else:
                    print(f"Warning: cell '{cid}' not found in pkl path.")

        if not resolved:
            print("No valid cells selected.")
            return

        for name in resolved:
            print(f"Loading cell '{name}' ...")
            with open(file_map[name], 'rb') as f:
                data = pickle.load(f)
            cycle_data = data.get('cycle_data', [])
            del data

            print(f"\n=== Cell {name} ===")
            for cyc_idx in cycle_ids:
                if cyc_idx >= len(cycle_data) or not isinstance(cycle_data[cyc_idx], dict):
                    print(f"  Cycle {cyc_idx}: not available")
                    continue

                item = cycle_data[cyc_idx]
                current = np.asarray(item['current_in_A'])
                # 'time_in_s' is mislabeled: values are actually in minutes
                # (confirmed against real cycle durations), so convert here.
                time_s  = np.asarray(item['time_in_s']) * 60.0
                Q_c     = np.asarray(item['charge_capacity_in_Ah'])
                Q_d     = np.asarray(item['discharge_capacity_in_Ah'])

                n = min(current.size, time_s.size, Q_c.size, Q_d.size)
                current, time_s, Q_c, Q_d = current[:n], time_s[:n], Q_c[:n], Q_d[:n]

                charge_mask    = current > MIN_CUR
                discharge_mask = current < -MIN_CUR

                t_c = time_s[charge_mask]
                t_d = time_s[discharge_mask]

                charge_time    = float(t_c[-1] - t_c[0]) if t_c.size > 1 else 0.0
                discharge_time = float(t_d[-1] - t_d[0]) if t_d.size > 1 else 0.0

                max_Q_c = float(np.max(Q_c[charge_mask])) if charge_mask.any() else 0.0
                max_Q_d = float(np.max(Q_d[discharge_mask])) if discharge_mask.any() else 0.0

                print(f"  Cycle {cyc_idx:4d} | "
                      f"charge_time={charge_time:8.2f}s  n_pts={int(charge_mask.sum()):4d}  max_Qc={max_Q_c:.4f} | "
                      f"discharge_time={discharge_time:8.2f}s  n_pts={int(discharge_mask.sum()):4d}  max_Qd={max_Q_d:.4f}")

            del cycle_data
            gc.collect()

    def plot_dqdv_diff(self, cell_ids, ref_cycle=10, v_min=2.4, v_max=3.6,
                       cycle_step=10, n_grid=500):
        """
        Plot dQ/dV(cycle) − dQ/dV(ref_cycle) for every selected cell (all cycles).

        cell_ids   : list of cell names (str) or 0-based indices (int).
        ref_cycle  : cycle index used as the reference (default 10).
        v_min/max  : voltage window for dQ/dV computation.
        cycle_step : plot every Nth cycle to reduce overplotting.
        n_grid     : number of points in the shared voltage interpolation grid.
        """
        pkl_files = glob.glob(os.path.join(self.pkl_path, "*.pkl"))
        file_map = {}
        for f in sorted(pkl_files):
            stem = os.path.splitext(os.path.basename(f))[0]
            name = stem.replace("MATR_", "") if stem.startswith("MATR_") else stem
            file_map[name] = f
        all_names = sorted(file_map.keys())

        resolved = []
        for cid in cell_ids:
            if isinstance(cid, int):
                if 0 <= cid < len(all_names):
                    resolved.append(all_names[cid])
                else:
                    print(f"Warning: index {cid} out of range (0..{len(all_names)-1}).")
            else:
                if cid in file_map:
                    resolved.append(cid)
                else:
                    print(f"Warning: cell '{cid}' not found in pkl path.")

        if not resolved:
            print("No valid cells selected.")
            return

        V_grid = np.linspace(v_min, v_max, n_grid)

        n_cells = len(resolved)
        n_cols = min(4, n_cells)
        n_rows = int(np.ceil(n_cells / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for i, name in enumerate(resolved):
            ax = flat_axes[i]
            print(f"Loading cell '{name}' ...")
            with open(file_map[name], 'rb') as f:
                data = pickle.load(f)
            cycle_data = data.get('cycle_data', [])
            del data

            def _dqdv_on_grid(cyc_idx):
                if cyc_idx >= len(cycle_data) or not isinstance(cycle_data[cyc_idx], dict):
                    return None
                item = cycle_data[cyc_idx]
                voltage = np.asarray(item['voltage_in_V'])
                capacity = np.asarray(item['discharge_capacity_in_Ah'])
                current = np.asarray(item['current_in_A'])
                mask = current < 0
                result = self.calculate_dQdV(voltage[mask], capacity[mask], v_min=v_min, v_max=v_max)
                if result is None:
                    return None
                interp = interp1d(result["V_dQdV"], result["dQdV"],
                                  bounds_error=False, fill_value=0.0)
                return interp(V_grid)

            ref = _dqdv_on_grid(ref_cycle)
            if ref is None:
                print(f"  Reference cycle {ref_cycle} unavailable for cell '{name}', skipping.")
                ax.set_visible(False)
                del cycle_data
                continue

            n_total = len(cycle_data)
            cmap = plt.cm.plasma
            plot_norm = plt.Normalize(vmin=0, vmax=n_total)

            for cyc_idx in range(0, n_total, cycle_step):
                if cyc_idx == ref_cycle:
                    continue
                dqdv = _dqdv_on_grid(cyc_idx)
                if dqdv is None:
                    continue
                ax.plot(dqdv - ref, V_grid, color=cmap(plot_norm(cyc_idx)),
                        linewidth=0.8, alpha=0.7)

            del cycle_data
            gc.collect()

            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel("ΔdQ/dV (Ah/V)")
            ax.set_ylabel("Voltage (V)")
            ax.set_title(f"ΔdQ/dV — Cell {name} (ref: cycle {ref_cycle})")
            ax.grid(True, alpha=0.4)

        for j in range(n_cells, len(flat_axes)):
            flat_axes[j].set_visible(False)

        plt.suptitle(f"dQ/dV − dQ/dV(cycle {ref_cycle})", fontsize=13)
        plt.tight_layout()
        fig.subplots_adjust(right=0.88)

        cax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, cax=cax, label="Cycle Index (relative)")

        plt.show()


if __name__ == "__main__":
    pkl_path = "/Users/ruturaj/Master-Thesis/Dataset/MIT"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(repo_root, "Models", "processed_hust_MIT_cache.pkl")
    processor = HUSTDataProcessor(pkl_path)
    cells_cache = processor.process_and_extract(cache_path=cache_path)
    for i, cell in enumerate(cells_cache):
        print(f"Cell {cell['cell_name']}: features {cell['features'].shape}, "
              f"soh len {len(cell['soh'])}, eol {cell['eol']}")

    # --- SoH: all cells ---
    #processor.plot_soh_all(cells_cache, normalize=True)

    # --- dQ/dV: selected cells and cycles ---
    CELL_IDS  = [ "b0c29", "b0c44", "b0c45", "b0c35", "b0c36", "b1c3", "b1c41", "b1c42", "b1c43", "b1c44", "b2c34",
    "b2c30", "b2c31", "b2c17", "b2c18", "b3c40", "b3c26", "b3c27"]          # cell names (str) or 0-based indices (int)
    CYCLE_IDS = list(range(1, 250))  # 0-based cycle indices

    processor.plot_dqdv_selected(cell_ids=CELL_IDS, cycle_ids=CYCLE_IDS)
    #processor.plot_current_profile(cell_ids=CELL_IDS, cycle_ids=CYCLE_IDS)
    #processor.plot_discharge_capacity_vs_voltage(cell_ids=CELL_IDS, cycle_ids=CYCLE_IDS)
    #processor.plot_voltage_kurtosis(cells_cache=cells_cache, cell_ids=CELL_IDS)
    #processor.plot_dqdv_diff(cell_ids=["b2c34"], ref_cycle=10, cycle_step=50)
    processor.plot_log_dispersion_features(cells_cache=cells_cache, cell_id="b2c34")
    #processor.plot_feature_eol_correlation(cells_cache=cells_cache, early_cycles=100)
    #processor.plot_feature_eol_correlation(cells_cache=cells_cache, early_cycles=100)