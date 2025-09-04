# runs/combine_per_frame_data.py
import pandas as pd
import json
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent

def extract_metadata_from_path(file_path):
    """Extract experiment metadata from folder structure"""
    parts = file_path.parts
    try:
        idx = parts.index("runs")
        return {
            "exp": parts[idx+1],
            "tracker": parts[idx+2], 
            "seq": parts[idx+3],
            "work_w": int(parts[idx+4][1:]) if parts[idx+4].startswith('w') else None,
            "glare_mode": parts[idx+5].split('_')[1] if 'glare_' in parts[idx+5] else None,
            "delta": int(parts[idx+6][1:]) if parts[idx+6].startswith('d') else None,
            "run": int(parts[idx+7][3:]) if parts[idx+7].startswith('run') else None
        }
    except:
        return {}

def load_meta_json(per_frame_path):
    """Load metadata from meta.json if available"""
    meta_path = per_frame_path.parent / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f)
        except:
            return {}
    return {}

def combine_per_frame_data():
    """Combine all per_frame.csv files into one dataset"""
    
    # Find all per_frame.csv files
    per_frame_files = list(ROOT.rglob("per_frame.csv"))
    print(f"Found {len(per_frame_files)} per_frame.csv files")
    
    if not per_frame_files:
        print("No per_frame.csv files found. Please check your directory structure.")
        return None
    
    combined_frames = []
    
    for file_path in per_frame_files:
        try:
            # Load per-frame data
            df = pd.read_csv(file_path)
            
            # Extract metadata from path
            path_meta = extract_metadata_from_path(file_path)
            
            # Load metadata from meta.json if available
            json_meta = load_meta_json(file_path)
            
            # Combine metadata (json takes precedence)
            metadata = {**path_meta, **json_meta}
            
            # Add metadata columns to dataframe
            for key, value in metadata.items():
                df[key] = value
            
            # Add file path for debugging
            df['source_file'] = str(file_path)
            
            combined_frames.append(df)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not combined_frames:
        print("No valid per_frame.csv files could be loaded")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(combined_frames, ignore_index=True)
    
    # Ensure consistent data types
    numeric_cols = ['frame', 't', 'cx', 'cy', 'ex', 'ey', 'e_norm', 'de_dt', 
                   'speed_px_s', 'inliers', 'klt_pts', 'fps_cb', 'fps_msg', 'fps_ema']
    
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    # Clean up tracker names and ensure consistent ordering
    if 'tracker' in combined_df.columns:
        combined_df['tracker'] = combined_df['tracker'].str.lower().str.strip()
    
    # Save combined dataset
    output_file = ROOT / "combined_per_frame_all_runs.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Combined dataset saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    
    return combined_df

if __name__ == "__main__":
    combine_per_frame_data()
