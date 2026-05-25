## 1. Problem Understanding
* **Task type:** Kinematic State Estimation / Time-Series Global Optimization.
* **Evaluation metric:** Mean Haversine Distance (error in meters between predicted and ground truth latitude/longitude).
* **Key challenges:** Highly noisy smartphone GNSS observations, missing measurements, multipath errors in urban canyons, severe hardware-specific anomalies (clock sync issues, missing data), and inaccurate device-provided uncertainty estimates.

## 2. Data Pipeline (Code-Oriented)
* **`load_data()`**: Parse raw GNSS messages directly from `gnss_log.txt` (do not rely on the pre-processed `device_gnss.csv` as it contains missing pseudoranges). Load satellite broadcast ephemeris and single-base station data (NGS CORS at 30-sec intervals).
* **`preprocess()`**: 
    * Apply base station coordinate offset corrections globally.
    * Align hardware clocks.
    * **Device specific:** If `device == 'XiaomiMi8'`, apply a `+600 ms` temporal offset to Doppler measurements to align with the Android system clock.
* **`feature_engineering()`**:
    * Calculate satellite elevation angles for all observations.
    * Compute Double-Difference (or Single-Difference) pseudoranges using the base station to remove atmospheric and satellite clock biases.
    * Compute ADR (Accumulated Delta Range / carrier phase) time differences between epochs.
    * Track `HardwareClockDiscontinuedCount` flags; if changed, mark an immediate cycle slip and invalidate ADR continuity.
* **`split_folds()`**: Group data by `collectionName` (drive). Optimization is performed per drive, so "folds" represent distinct driving sessions for hyperparameter tuning.

## 3. Model Design
* **`build_model()`**: Construct a Factor Graph using `gtsam` (Georgia Tech Smoothing and Mapping library).
* **Model types:** Factor Graph Optimization (FGO).
* **Graph Nodes:** * Position states ($X_t$)
    * Velocity states ($V_t$)
    * Receiver Clock Bias states ($C_t$)
* **Edges (Factors):**
    * *Prior Factors:* Initial position guess.
    * *Absolute Constraints:* Base-station corrected pseudoranges.
    * *Relative Constraints:* ADR time differences, Doppler-derived velocities.

## 4. Training Strategy (Optimization Execution)
* **`train_one_fold()`** (Adapted for FGO tuning): The "training" phase involves grid-searching FGO hyperparameters over the training set to minimize the Haversine metric.
* **Optimization Engine:** `gtsam.LevenbergMarquardtOptimizer`.
* **Loss function / Error Model:** Huber M-Estimator. Do not use Switchable Constraints (too slow/poor convergence). Use fixed Huber thresholds to downweight large residuals without fully zeroing them out.
* **Two-Stage Execution Flow:**
    * *Stage 1 (Velocity Stage):* Optimize *only* velocity nodes using robust Doppler measurements. Reject velocity outliers. Interpolate missing velocities using Akima interpolation (prevents overshooting). Detect stationary epochs (velocity $\approx 0$).
    * *Stage 2 (Position Stage):* Build the full graph. Inject Stage 1 velocities as loose relative between-state constraints. If stationary epoch detected, drop pseudorange constraints for that timestamp. Add ADR relative constraints (where valid). Add corrected pseudorange absolute constraints. Optimize for position.

## 5. Validation Strategy
* **Cross-validation logic:** Leave-one-drive-out or K-fold across training collections to tune the Huber M-Estimator parameters and elevation-model noise covariances. 
* **OOF generation:** Run the two-stage FGO pipeline sequentially on every training drive. Compare final optimized $X_t$ coordinates against ground truth.

## 6. Inference Pipeline
* **`predict()`**: Iterate through the test dataset grouping by collection and phone. Construct the Stage 1 velocity FGO. Extract interpolated velocities. Construct the Stage 2 position FGO. Extract final coordinates.
* **TTA / ensemble:** Not utilized in this strict mathematical optimization approach. Final FGO output is deterministic and directly submitted.
* **post_process()**: None required. The stop-point logic and FGO smoothing replaces the need for post-processing filters (like Kalman smoothing or snap-to-grid).

## 7. Key Tricks (ACTIONABLE)
* **If `device == 'XiaomiMi8'` $\rightarrow$** `doppler_time += 0.600` (seconds).
* **If `device in ['SamsungGalaxyS20', 'XiaomiMi8']` $\rightarrow$** Discard device-provided uncertainty. Implement `noise_model = f(satellite_elevation_angle)`. Low elevation = high variance.
* **If `HardwareClockDiscontinuedCount` diff $\neq 0$ $\rightarrow$** Set `adr_valid = False` for that transition edge.
* **If `velocity_stage1 < STOP_THRESHOLD` $\rightarrow$** Drop pseudorange factors from the Stage 2 factor graph for that specific timestamp to prevent stationary drift.
* **If missing velocity in Stage 1 $\rightarrow$** Use `scipy.interpolate.Akima1DInterpolator`. Do NOT use standard spline (`scipy.interpolate.CubicSpline`) as it overshoots.

## 8. FINAL SINGLE-FILE CODE STRUCTURE (CRITICAL)

```python
import numpy as np
import pandas as pd
import gtsam
from scipy.interpolate import Akima1DInterpolator

def seed_everything(seed=42):
    # Ensure deterministic behavior where possible
    np.random.seed(seed)

def load_data(data_dir):
    # Parse raw gnss_log.txt directly to recover missing pseudoranges
    # Load NGS CORS base station data
    # Return train_df, test_df, base_station_df
    pass

def preprocess(df, base_station_df):
    # Correct base station coordinate offsets globally
    # Apply +600ms temporal offset to XiaomiMi8 Doppler data
    pass

def feature_engineering(df):
    # Calculate satellite elevation angles
    # Calculate ADR time differences
    # Flag cycle slips using HardwareClockDiscontinuedCount
    pass

def calc_elevation_error_model(elevation_angle, device):
    # Return custom variance for Samsung/Xiaomi based on elevation
    # Return default device uncertainty for other devices
    pass

def build_velocity_factor_graph(drive_df):
    # Stage 1 FGO construction
    # Add Doppler factors with Huber M-estimator
    pass

def build_position_factor_graph(drive_df, stage1_velocities):
    # Stage 2 FGO construction
    # Inject stage1_velocities as relative constraints
    # Add corrected pseudorange factors (filter out if velocity == 0)
    # Add ADR constraints if no cycle slip occurred
    pass

def optimize_graph(graph, initial_estimates):
    # Initialize gtsam.LevenbergMarquardtOptimizer
    # Return optimized state variables
    pass

def run_two_stage_fgo(drive_df):
    # 1. Build and optimize velocity graph
    # 2. Extract, filter outliers, and apply Akima1DInterpolator on velocity
    # 3. Build and optimize position graph using interpolated velocities
    # 4. Return optimized latitude/longitude
    pass

def tune_hyperparameters(train_df):
    # FGO equivalent of train_one_fold
    # Grid search Huber parameters across training collections
    pass

def inference(test_df):
    # Group test_df by collectionName and phoneName
    # Apply run_two_stage_fgo() per drive
    # Aggregate results into submission format
    pass

def main():
    seed_everything()
    
    # 1. Pipeline Execution
    train_df, test_df, base_df = load_data('./data')
    
    # 2. Preprocessing & Feature Engineering
    train_df = preprocess(train_df, base_df)
    train_df = feature_engineering(train_df)
    
    test_df = preprocess(test_df, base_df)
    test_df = feature_engineering(test_df)
    
    # 3. FGO execution (drive by drive)
    # Note: tune_hyperparameters() would be run offline here to find optimal configs
    
    submission_preds = inference(test_df)
    
    # 4. Save
    submission_preds.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
```

## 9. Strategy Priority (IMPORTANT)

1.  **Most impactful techniques:** Two-stage Factor Graph Optimization (separating velocity and position estimation) and utilizing Base Station (CORS) data to correct pseudoranges.
2.  **Secondary improvements:** Filtering pseudorange updates at stationary points (velocity $\approx 0$) and applying Akima interpolation for robust velocity continuity.
3.  **Minor tricks:** Device-specific interventions (Xiaomi 600ms Doppler shift, elevation-based error substitution for Samsung/Xiaomi).