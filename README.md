# Tank Hydrologic Model Calibration via Genetic Algorithm

## Overview

This repository contains code to calibrate a multi‐reservoir (tank) hydrologic model against observed runoff data using a Genetic Algorithm (GA). The calibration process is performed over multiple seasonal periods, optimizing storage parameters, runoff coefficients, side‐outlet heights, and infiltration rates to maximize the coefficient of determination (R²) between simulated and observed runoff.

The main steps are:

1. **Load observed data** (precipitation, actual evapotranspiration, runoff) from a CSV file.
2. **Define seasonal periods** for calibration (February–June, June–September, October–January for years 2009–2012).
3. **Initialize a GA optimizer** (`GAOptimizer`) with parameter ranges and genetic‐algorithm settings.
4. **Run GA** to find the best tank‐model coefficients for each period.
5. **Simulate** the tank model forward using the optimized coefficients.
6. **Visualize** both the simulated vs. observed runoff (via `visualize_runoff`) and the GA‐parameter‐history (`visualize_tuning`) for each period.

## Prerequisites

* Python 3.8 or higher
* The following Python packages (install via `pip`):

  * `numpy`
  * `pandas`
  * `matplotlib`
  * (Any other dependencies required by modules under `src/`, e.g., if additional packages are imported there.)

## Installation

1. **Clone this repository** (or download the ZIP):

   ```bash
   git clone https://github.com/yourusername/tank-hydrologic-ga.git
   cd tank-hydrologic-ga
   ```

2. **Verify the directory structure**:

   ```
   tank-hydrologic-ga/
   ├── data/
   │   └── 3009680_p4.csv
   ├── src/
   │   ├── model.py
   │   ├── opt.py
   │   └── vis.py
   ├── run_calibration.py
   ├── README.md
   └── requirements.txt       (optional)
   ```

   * **`data/3009680_p4.csv`** must contain at least the following columns:

     * `date` (YYYY-MM-DD or parseable by `pd.to_datetime`)
     * `P` (precipitation time series)
     * `Q` (observed runoff time series)
     * `AET` (actual evapotranspiration time series)

## Usage

1. **Place your data file** (e.g., `3009680_p4.csv`) under the `data/` directory.

2. **Edit configuration parameters** at the top of `run_calibration.py` if needed:

   * **`TANK_LEVEL`**: Number of tanks in the cascade (e.g., 4).

   * **`TIMESTEPS`**: Number of timesteps per day (e.g., 60 × 60 × 24 for hourly data).

   * **`AREA`**: Catchment area (e.g., 601.61 km² or consistent units).

   * **`DATA_PATH`** and **`DATA_NAME`**: Path and filename under `data/`.

   * **Parameter ranges** for GA:

     ```python
     STORAGE_RANGE = None
     RUNOFF_RANGE = [
         (0.1, 0.5),    # Tank 1 runoff coefficient
         (0.1, 0.5),    # Tank 2 runoff coefficient
         (0.03, 0.1),   # Tank 3 runoff coefficient
         (0.005, 0.01), # Tank 4 runoff coefficient
         (0.0005, 0.01) # Outlet runoff from last tank
     ]
     SIDE_RANGE = [
         (5, 60),  # Side‐outlet height tank 1 (mm or consistent units)
         (5, 60),  # Side‐outlet height tank 2
         (0, 50),  # Side‐outlet height tank 3
         (0, 30),  # Side‐outlet height tank 4
         (0, 0)    # No side‐outlet for last tank (fixed)
     ]
     INFIL_RANGE = [
         (0.1, 0.5),  # Infiltration rate tank 1
         (0.01, 0.1), # Infiltration rate tank 2
         (0.005, 0.01), # Infiltration rate tank 3
         (0, 0)       # No infiltration for tanks 4+ (fixed zero)
     ]
     ```

   * **GA settings**:

     ```python
     GENERATION = 200         # Number of GA generations per period
     POPULATION = 100         # Population size per generation
     ```

3. **Run the calibration script**:

   This will loop over each defined seasonal period (Feb 1–Jun 14, Jun 15–Sep 30, Oct 1–Jan 31 for 2009–2012). For each period:

   * A new `GAOptimizer` instance is created with the fixed storage (last storage of the previous period).
   * The GA searches for the best set of tank parameters (storage, runoff coefficients, side‐outlet heights, infiltration rates) that maximize the R² between simulated and observed runoff.
   * After convergence, the optimized coefficients are printed.
   * The tank model is run forward for that period, and

     * **`visualize_runoff(...)`** plots observed vs. simulated runoff.
     * **`visualize_tuning(...)`** plots the GA’s parameter‐history (fitness vs. generation, parameter distributions over time).

4. **Results**:

   * Each period’s best coefficients are printed to the console.
   * Time‐series plots of runoff comparison (`visualize_runoff`) and GA tuning history (`visualize_tuning`) appear interactively (or save to disk if `save_path` is provided).
   * The final storage from the last period will be printed (and can be reused if extending beyond 2012).

## Configuration Details

* **Seasonal Periods**
  Defined in `run_calibration.py` as:

  ```python
  years = range(2009, 2013)
  periods = []
  for y in years:
      periods.extend([
          (f"{y}-02-01", f"{y}-06-14"),
          (f"{y}-06-15", f"{y}-09-30"),
          (f"{y}-10-01", f"{y+1}-01-31"),
      ])
  ```

  This covers Feb 1–Jun 14, Jun 15–Sep 30, and Oct 1–Jan 31 for each year from 2009 through 2012 (inclusive). Adjust as needed.

* **Objective Function**
  A custom R² score is defined:

  ```python
  def r2_score(y_true, y_pred):
      y_true = np.array(y_true)
      y_pred = np.array(y_pred)
      y_mean = np.mean(y_true)
      ss_res = np.sum((y_true - y_pred)**2)
      ss_tot = np.sum((y_true - y_mean)**2)
      return 1 - (ss_res / ss_tot)
  ```

  The GA maximizes this metric.

* **Fixed Storage**

  * `final_storage` is initially `None`.
  * After each period’s simulation, `final_storage` is set to the last storage state from the best model.
  * In subsequent periods, `fixed_storage=final_storage` ensures continuity between seasons.

## Project Structure

```
tank-hydrologic-ga/
├── data/                           # Input time‐series data (CSV)
│   └── 3009680_p4.csv              # Must contain columns: date, P, Q, AET
│
├── src/                            # Core Python modules
│   ├── model.py                    # Tank model implementation
│   ├── opt.py                      # Genetic Algorithm optimizer
│   └── vis.py                      # Visualization utilities
│
├── run_calibration.py              # Main script that coordinates:
│                                    #   • data loading
│                                    #   • period splitting
│                                    #   • GA calibration
│                                    #   • simulation and visualization
│
├── README.md                       # Project documentation (this file)
├── requirements.txt                # (Optional) pinned dependencies, e.g.:
│                                    #   numpy==1.25.0
│                                    #   pandas==2.1.0
│                                    #   matplotlib==3.8.0
│
└── results/                        # (Optional) output directory for saved plots/coefficients
```

* **Data Folder (`data/`)**

  * Place your CSV file named exactly `3009680_p4.csv` (or update `DATA_NAME` in `run_calibration.py` if you prefer a different name).
  * The CSV must have:

    * `date` (parsable by `pd.to_datetime`),
    * `P` (precipitation),
    * `Q` (observed runoff),
    * `AET` (actual evapotranspiration).

* **Source Folder (`src/`)**

  * `model.py`:

    * Implements a `Tank` class that encapsulates storage → runoff → infiltration computations per timestep.
    * `make_models(...)` constructs a multi‐tank cascade using the best‐found coefficients.

  * `opt.py`:

    * Implements `GAOptimizer`, including:

      * Parameter encoding/decoding to/from real‐valued vectors.
      * Population initialization, selection (e.g., tournament or roulette), crossover, and mutation operators.
      * Fitness evaluation loop (calls the tank model simulation internally).

  * `vis.py`:

    * Two primary functions: `visualize_runoff` and `visualize_tuning`.
    * By default, uses `matplotlib` to create time‐series and convergence plots.

* **Main Script (`run_calibration.py`)**

  * This is the entry point. It does not depend on any other external scripts (aside from `src/` modules).
  * To adapt this script for different data or parameter ranges, simply edit the top section (constants and ranges).

## Customization and Extending

* **Change Number of Tanks**

  * Modify `TANK_LEVEL` in `run_calibration.py` (e.g., `TANK_LEVEL = 5` for a five‐tank cascade).
  * Ensure you update `RUNOFF_RANGE`, `SIDE_RANGE`, and `INFIL_RANGE` lists to have length ≥ `TANK_LEVEL + 1` where appropriate.

* **Use a Different Objective**

  * Replace the `r2_score` function with any user‐defined metric (e.g., Nash–Sutcliffe Efficiency, other statistical metrics).
  * Pass that function to `GAOptimizer(..., objective=<your_function>, direction="maximize" or "minimize")`.

* **Save Plots to Disk**

  * In each call to `visualize_runoff(...)` or `visualize_tuning(...)`, set `save_path` to a `str` or `pathlib.Path` to write PNG/PDF files automatically:

    ```python
    plot_path = DATA_PATH / "results" / f"runoff_{start.date()}_{end.date()}.png"
    visualize_runoff(dates, runoff_target, total_runoff, show=False, save_path=plot_path)
    ```

* **Parallel GA Fitness Evaluations**

  * If running on a multi‐core machine, consider parallelizing the fitness evaluation in `GAOptimizer` (e.g., using `multiprocessing.Pool`). This can drastically reduce run times for large populations.

Plots appear one by one for each period. Close the figure window (or press “Enter” if using non‐interactive backends) to proceed to the next period’s GA.

## Citations

```
# =============================================================================
# Lim, J. and Yang, S. (2025). *Tank model implementation in Python (Sugawara & Maruyama, 1956)* (v1.0). 
# [Computer software]. Department of Civil and Environmental Engineering, Seoul National University.
# https://doi.org/10.5281/zenodo.15464005
# =============================================================================
```

*Last updated: June 2025*
