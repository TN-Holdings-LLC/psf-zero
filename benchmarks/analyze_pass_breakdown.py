import pandas as pd
import glob
import os

def analyze_csv(filepath):
    df = pd.read_csv(filepath)
    
    # Check that the required timing columns are present
    req_cols = ['spare', 'vf2layout_ms', 'vf2postlayout_ms', 'time_ms']
    if not all(c in df.columns for c in req_cols):
        return

    # Restrict to optimization_level=3 (the level the cliff was documented at)
    if 'optimization_level' in df.columns:
        df = df[df['optimization_level'] == 3].copy()
        if df.empty:
            return

    print(f"\n{'='*90}")
    print(f"File: {os.path.basename(filepath)}")
    print(f"{'='*90}")

    # Resolve the occupancy column name (it differs slightly between scripts)
    occ_col = None
    if 'occupancy' in df.columns:
        occ_col = 'occupancy'
    elif 'occupancy_of_device' in df.columns:
        occ_col = 'occupancy_of_device'
    elif 'occupancy_of_matching_capacity' in df.columns:
        occ_col = 'occupancy_of_matching_capacity'

    group_cols = ['spare']
    if occ_col:
        group_cols.append(occ_col)
        
    agg_funcs = {
        'time_ms': 'median',
        'vf2layout_ms': 'median',
        'vf2postlayout_ms': 'median'
    }
    
    # Include stop_reason if the file has it
    if 'vf2_stop_reason' in df.columns:
        agg_funcs['vf2_stop_reason'] = lambda x: '/'.join(sorted(set(str(v) for v in x if pd.notna(v))))

    # Aggregate
    summary = df.groupby(group_cols).agg(agg_funcs).reset_index()
    summary = summary.sort_values('spare')
    
    # Round to 1 decimal place for readability
    summary['time_ms'] = summary['time_ms'].round(1)
    summary['vf2layout_ms'] = summary['vf2layout_ms'].round(1)
    summary['vf2postlayout_ms'] = summary['vf2postlayout_ms'].round(1)
    if occ_col:
        summary[occ_col] = (summary[occ_col] * 100).round(1).astype(str) + '%'

    # Terminal display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    csv_files = sorted(glob.glob("*.csv"))
    if not csv_files:
        print("No matching CSV files found in the current directory.")
    else:
        print("Re-analysing VF2Layout vs VF2PostLayout clearing thresholds from existing CSVs...")
        for f in csv_files:
            analyze_csv(f)