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
from scipy.stats import kurtosis, skew

# Fixed output length for the SoH trajectory stored in the cache.
# Cells with EOL < SOH_HORIZON are padded with -1 (sentinel = "not reached yet").
SOH_HORIZON = int(os.environ.get("SOH_HORIZON", "1500"))


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
            del data
            if cycle_data is not None:
                yield cell_name, cycle_data
            del cycle_data
            gc.collect()

    def _process_single_cycle(self, item):
        voltage = np.asarray(item['voltage_in_V'])
        charge_capacity = np.asarray(item['charge_capacity_in_Ah'])
        discharge_capacity = np.asarray(item['discharge_capacity_in_Ah'])
        current = np.asarray(item['current_in_A'])

        charge_mask = current > 0
        discharge_mask = current < 0

        V_c = voltage[charge_mask]
        Q_c = charge_capacity[charge_mask]
        I_c = current[charge_mask]
        V_d = voltage[discharge_mask]
        Q_d = discharge_capacity[discharge_mask]
        I_d = current[discharge_mask]

        discharge_deriv = self.calculate_dQdV(V_d, Q_d)
        soh_val = float(np.max(Q_d)) if len(Q_d) > 0 else 0.0

        return {
            'voltage_charge': V_c, 'capacity_charge': Q_c, 'current_charge': I_c,
            'voltage_discharge': V_d, 'capacity_discharge': Q_d, 'current_discharge': I_d,
        }, discharge_deriv, soh_val

    def _extract_cycle_features(self, cycle_data, deriv, deriv_10):
        if deriv is not None:
            dQdV = deriv["dQdV"]
            dQdV_10 = deriv_10["dQdV"] if deriv_10 is not None else np.array([])
            # dQdV_max = np.max(dQdV) - (np.max(dQdV_10) if len(dQdV_10) > 0 else 0)
            dQdV_min = np.min(dQdV) - (np.min(dQdV_10) if len(dQdV_10) > 0 else 0)
            dQdV_var = np.var(dQdV) - (np.var(dQdV_10) if len(dQdV_10) > 0 else 0)
        else:
            dQdV_max = dQdV_min = dQdV_var = 0

        I_c = cycle_data['current_charge']
        V_c = cycle_data['voltage_charge']
        Q_c = cycle_data['capacity_charge']
        I_d = cycle_data['current_discharge']
        V_d = cycle_data['voltage_discharge']
        Q_d = cycle_data['capacity_discharge']

        if all(len(a) > 0 for a in [I_c, V_c, Q_c, I_d, V_d, Q_d]):
            log_std_I = float(20.0 * np.log10(np.std(I_c).clip(1e-9)))
            log_std_qd = float(20.0 * np.log10(np.std(Q_c).clip(1e-9)))
            log_std_V = float(20.0 * np.log10(np.std(V_c).clip(1e-9)))
            log_std_Id = float(20.0 * np.log10(np.std(I_d).clip(1e-9)))
            log_std_qd_d = float(20.0 * np.log10(np.std(Q_d).clip(1e-9)))
            log_std_Vd = float(20.0 * np.log10(np.std(V_d).clip(1e-9)))
            return [
                dQdV_min, dQdV_var,
                log_std_I, log_std_qd, log_std_V,
                log_std_Id, log_std_qd_d, log_std_Vd,
                float(np.min(I_c)), float(np.max(I_c)),
                float(np.min(V_c)), float(np.max(V_c)),
                float(np.min(I_d)), float(np.max(I_d)),
                float(np.min(V_d)), float(np.max(V_d)),
                float(np.max(Q_d)), float(kurtosis(V_d)),
            ]
        return [0] * 18

    def process_and_extract(self, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if cache and 'soh_traj' not in cache[0]:
                print("Cache missing 'soh_traj' field — regenerating cache...")
            else:
                print(f"Loaded {len(cache)} cells from cache.")
                return cache

        cells_cache = []

        for cell_name, cycles_data in self._iter_pkl_cells():
            n_cycles = len(cycles_data)

            if n_cycles < self.min_cycles:
                del cycles_data
                continue

            deriv_10 = None
            if n_cycles > 10 and isinstance(cycles_data[10], dict):
                _, deriv_10, _ = self._process_single_cycle(cycles_data[10])

            soh_list = []
            cell_features = []

            for idx, item in enumerate(cycles_data):
                if not isinstance(item, dict):
                    soh_list.append(0.0)
                    cell_features.append([0.0] * 18)
                    continue

                processed, deriv, soh_val = self._process_single_cycle(item)
                soh_list.append(soh_val)

                feats = self._extract_cycle_features(processed, deriv, deriv_10)
                feats.append(soh_val)
                cell_features.append(feats)

            del cycles_data
            gc.collect()

            soh_array = np.asarray(soh_list, dtype=np.float32)
            eol_threshold = 0.88
            peak_idx = int(np.argmax(soh_array))
            post_peak = soh_array[peak_idx:]
            below = np.where(post_peak < eol_threshold)[0]
            eol_cycle = int(peak_idx + below[0]) if len(below) > 0 else len(soh_array)

            # Normalised SoH trajectory from cycle 0 to EOL, padded to SOH_HORIZON.
            # Cycles beyond EOL are set to -1 (sentinel so the model can mask them).
            peak_cap = float(np.max(soh_array)) if np.max(soh_array) > 1e-8 else 1.0
            soh_norm = soh_array / peak_cap
            soh_traj = np.full(SOH_HORIZON, -1.0, dtype=np.float32)
            end = min(eol_cycle, SOH_HORIZON)
            soh_traj[:end] = soh_norm[:end]

            cells_cache.append({
                'cell_name': cell_name,
                'features': np.asarray(cell_features, dtype=np.float32),
                'soh': soh_array,
                'soh_traj': soh_traj,   # [SOH_HORIZON], normalised; -1 beyond EOL
                'eol': eol_cycle,
                'num_cycles': n_cycles,
            })

            del soh_list, soh_array, cell_features
            gc.collect()

        print(f"Processed {len(cells_cache)} cells.")
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(cells_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cache saved to {cache_path}")

        return cells_cache

    def plot_capacity_vs_voltage(self, processed_cells):
        plt.figure(figsize=(10, 6))
        cell_indices = [2,4]
        cycles_indices = range(0, 10)
        for idx, processed_data in enumerate(processed_cells):
            if idx not in cell_indices:
                continue
            for item in processed_data:
                if isinstance(item, dict):
                    voltage_discharge = item.get('voltage_discharge')
                    capacity_discharge = item.get('capacity_discharge')

                    # voltage_discharge = voltage_discharge[10:100]
                    # capacity_discharge = capacity_discharge[10:100]
                    
                    plt.plot(voltage_discharge, capacity_discharge, alpha=0.5)
                else:
                    print("Expected a dict in cycle_data, but got:", type(item))

    def calculate_dQdV(self, voltage, capacity, v_min=None, v_max=None):
        voltage = np.array(voltage)
        capacity = np.array(capacity)

        if v_min is not None or v_max is not None:
            lo = v_min if v_min is not None else -np.inf
            hi = v_max if v_max is not None else np.inf
            vmask = (voltage >= lo) & (voltage <= hi)
            voltage = voltage[vmask]
            capacity = capacity[vmask]

        idx = np.argsort(voltage)
        voltage = voltage[idx]
        capacity = capacity[idx]

       
        V_unique, unique_idx = np.unique(voltage, return_index=True)
        Q_unique = capacity[unique_idx]

        if len(V_unique) < 20:
            return None
        
        dQ = np.diff(Q_unique)
        dV = np.diff(V_unique)

        dQdV = dQ / dV
        lo, hi = np.percentile(dQdV, [2, 98])
        dQdV = np.clip(dQdV, lo, hi)

        dQdV = savgol_filter(
            dQdV,
            window_length=31,
            polyorder=3,
            mode='nearest'
        )

        idx_q = np.argsort(Q_unique)
        Q_sorted = Q_unique[idx_q]
        V_sorted = V_unique[idx_q]

        Q_unique2, unique_idx2 = np.unique(Q_sorted, return_index=True)
        V_unique2 = V_sorted[unique_idx2]

        if len(Q_unique2) < 20:
            return None

        dV = np.diff(V_unique2)
        dQ = np.diff(Q_unique2)

        dVdQ = dV / dQ
        lo, hi = np.percentile(dVdQ, [2, 98])
        dVdQ = np.clip(dVdQ, lo, hi)

        dVdQ = savgol_filter(
            dVdQ,
            window_length=31,
            polyorder=3,
            mode='nearest'
        )

        return {
            "V_dQdV": V_unique[:-1],
            "dQdV": dQdV,
            "Q_dVdQ": Q_unique2[:-1],
            "dVdQ": dVdQ
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
            if np.max(soh) < 0.88:
                continue
            if normalize and np.max(soh) > 1e-8:
                soh = soh / np.max(soh)
            # Savitzky-Golay: window must be odd and < len(soh)
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
    cache_path = r"processed_hust_MIT_cache.pkl"
    processor = HUSTDataProcessor(pkl_path)
    cells_cache = processor.process_and_extract(cache_path=cache_path)
    for i, cell in enumerate(cells_cache):
        print(f"Cell {cell['cell_name']}: features {cell['features'].shape}, "
              f"soh len {len(cell['soh'])}, eol {cell['eol']}")

    # --- SoH: all cells ---
    # processor.plot_soh_all(cells_cache, normalize=True)

    # --- dQ/dV: selected cells and cycles ---
    CELL_IDS  = [ "b0c29", "b0c44", "b0c45", "b0c35", "b0c36", "b1c3", "b1c41", "b1c42", "b1c43", "b1c44", "b2c34",
    "b2c30", "b2c31", "b2c17", "b2c18", "b3c40", "b3c26", "b3c27"]          # cell names (str) or 0-based indices (int)
    CYCLE_IDS = [1, 10, 50, 100, 150, 200, 250]  # 0-based cycle indices

    processor.plot_dqdv_selected(cell_ids=CELL_IDS, cycle_ids=CYCLE_IDS)
    processor.plot_dqdv_diff(cell_ids=["b2c34"], ref_cycle=10, cycle_step=50)