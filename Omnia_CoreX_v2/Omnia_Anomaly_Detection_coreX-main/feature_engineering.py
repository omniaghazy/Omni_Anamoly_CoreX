import pandas as pd
import numpy as np

def compute_error_signals(df):
    """
    Computes Error Signals: error = actual - target
    Then DROPS target_* columns since they are fully recoverable (target = actual - error).
    KEEPS actual_* columns (robot's real state) + error_* columns (deviation signal).
    """
    print("--- [Feature Engineering] Computing Error Signals (Actual - Target) ---")
    
    # Find all target_* columns and match them with actual_* columns
    target_cols = [c for c in df.columns if c.startswith('target_')]
    errors_added = 0
    targets_to_drop = []
    
    for t_col in target_cols:
        # Convert target_ prefix to actual_ prefix
        a_col = t_col.replace('target_', 'actual_', 1)
        
        if a_col in df.columns:
            error_col = t_col.replace('target_', 'error_', 1)
            df[error_col] = df[a_col] - df[t_col]
            errors_added += 1
            targets_to_drop.append(t_col)
    
    # Drop target columns (redundant: target = actual - error)
    if targets_to_drop:
        df = df.drop(columns=targets_to_drop)
        print(f"[OK] Added {errors_added} error signals, dropped {len(targets_to_drop)} target columns")
    
    return df


def remove_constant_features(df, numeric_cols):
    """
    Removes columns that have zero variance (constant values across all rows),
    as they provide no useful information for the anomaly detection model.
    """
    print("--- [Feature Selection] Checking for constant features ---")
    
    # Check for constant columns only within the numeric columns
    constant_cols = [col for col in numeric_cols if col in df.columns and df[col].nunique() <= 1]
    
    if constant_cols:
        print(f"Dropped {len(constant_cols)} constant features: {constant_cols}")
        df = df.drop(columns=constant_cols)
    else:
        print("No constant features found.")
        
    return df

def remove_correlated_features(df, threshold=0.90):
    """
    Removes features that are highly correlated with others,
    retaining only one to reduce redundancy.
    """
    print(f"--- [Feature Selection] Checking for highly correlated features (>{threshold}) ---")
    
    # Compute correlation matrix for numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop (those with correlation greater than the threshold)
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print("No highly correlated features found.")
        
    # Get remaining numeric columns to pass them forward in the pipeline
    remaining_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, remaining_numeric

def add_temporal_features(df, numeric_cols, window=5):
    """
    Adds Delta (velocity) features ONLY for:
    - error_* columns (how fast deviation is growing)
    - actual_q_* and actual_TCP_pose_* (robot motion)
    Skips: _qd (already velocity), _qdd (already acceleration),
           _current, _moment, joint_temperatures, joint_mode, robot_mode
    """
    print("--- [Feature Engineering] Adding Smart Temporal Features (Delta only) ---")
    
    # Only compute deltas for meaningful columns
    skip_patterns = ['_qd_', '_qdd_', '_current_', '_moment_', 
                     'joint_temperatures', 'joint_mode', 'robot_mode',
                     '_delta', '_mag']
    
    new_features = {}
    skipped = 0
    
    for col in numeric_cols:
        # Check if this column should be skipped
        should_skip = any(pattern in col for pattern in skip_patterns)
        
        if not should_skip:
            new_features[f"{col}_delta"] = df[col].diff()
        else:
            skipped += 1
    
    temp_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, temp_df], axis=1)
    
    # Fill NaNs created by diff
    df = df.fillna(method='bfill').fillna(0)
    
    print(f"[OK] Added {len(new_features)} delta features (skipped {skipped} derivative/utility columns)")
    return df

def add_robotic_magnitudes(df):
    """
    Adds Euclidean magnitude for 3-axis vectors (TCP Pose, etc.)
    Also adds error magnitude for position and rotation errors.
    """
    print("--- [Feature Engineering] Adding Robotic Magnitudes ---")
    
    # Actual TCP Pose (targets are dropped at this point)
    pose_cols = ['actual_TCP_pose']
    
    for base in pose_cols:
        # Position Magnitude (XYZ)
        xyz = [f"{base}_{i}" for i in range(3)]
        if all(c in df.columns for c in xyz):
            df[f"{base}_pos_mag"] = np.sqrt(df[xyz[0]]**2 + df[xyz[1]]**2 + df[xyz[2]]**2)
            print(f"Added Position Magnitude for {base}")
            
        # Rotation Magnitude (Roll/Pitch/Yaw)
        rpy = [f"{base}_{i}" for i in range(3, 6)]
        if all(c in df.columns for c in rpy):
            df[f"{base}_rot_mag"] = np.sqrt(df[rpy[0]]**2 + df[rpy[1]]**2 + df[rpy[2]]**2)
            print(f"Added Rotation Magnitude for {base}")
    
    # Error Magnitudes
    error_pose_bases = ['error_TCP_pose']
    for base in error_pose_bases:
        xyz = [f"{base}_{i}" for i in range(3)]
        if all(c in df.columns for c in xyz):
            df[f"{base}_pos_mag"] = np.sqrt(df[xyz[0]]**2 + df[xyz[1]]**2 + df[xyz[2]]**2)
            print(f"Added Error Position Magnitude for {base}")
            
        rpy = [f"{base}_{i}" for i in range(3, 6)]
        if all(c in df.columns for c in rpy):
            df[f"{base}_rot_mag"] = np.sqrt(df[rpy[0]]**2 + df[rpy[1]]**2 + df[rpy[2]]**2)
            print(f"Added Error Rotation Magnitude for {base}")

    return df
