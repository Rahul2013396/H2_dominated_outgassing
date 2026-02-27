# H₂-Dominated Outgassing

This repository contains a physically motivated outgassing model developed to quantify volatile release from rocky exoplanet interiors and evaluate its impact on atmospheric composition and detectability. The framework couples melt–gas equilibrium chemistry with volatile solubility, redox state (fO₂), and mantle volatile inventories to compute speciation and fluxes of major gases (e.g., H₂, H₂O, CO₂, CO, SO₂, H₂S).

---

## 🔬 Overview

The model couples:
- **Interior degassing** (COHS species)  
- **Atmospheric escape** (energy-limited hydrodynamic escape)  
- **Mantle redox and volatile solubility**   
- **Tidal or radiogenic heating**   

It predicts which combinations of planetary mass, irradiation, eccentricity, and mantle oxidation allow long-lived H₂ atmospheres.

---

## 📁 Repository Structure

| File | Description |
|------|--------------|
| **`COHS_gc.py`** | Main equilibrium solver that computes volcanic outgassing speciation for the C–O–H–S system. |
| **`grid_plot_COHS_gc.py`** | Generates contour and parameter-space plots for speciaiton of outgassing. |
| **`solubility.py`** | Calculates volatile solubilities, graphite saturation, and degassing limits as functions of volatile composition and redox state. |
| **`subfunctions.py`** | Contains helper functions, constants, and conversion utilities used across the model. |
| **`volatile_inventory.py`** | Calcualtes the volatile invetory timescale for a given set of planet as a function of outgassing efficiency. |

---

## ⚙️ Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/Rahul2013396/H2_dominated_outgassing
cd H2_dominated_outgassing
pip install numpy scipy matplotlib pandas
```

(Optional)  
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

---

## 🚀 Usage

### 1️⃣ Run the main model
Edit physical and chemical inputs at the top of **`COHS_gc.py`** (see next section).  
Then execute:

```bash
python COHS_gc.py
```

This computes outgassing and escape across the defined parameter grid and saves results as `.csv` files.

### 2️⃣ Plot parameter maps
After running the solver:

```bash
python grid_plot_COHS_gc.py
```

This script visualizes where volcanic H₂ outgassing balances atmospheric escape.

---

## 🧩 User Inputs (in `COHS_gc.py`)

Set the parameters that define your grid and physical setup.

| Variable | Description | Typical Range / Units | Purpose |
|-----------|--------------|------------------------|----------|
| **`Temp`** | Temperature array | `np.arange(973, 1873, 100)` K | Thermal state of the melt or mantle |
| **`fo2`** | Oxygen fugacity offset (ΔFMQ) | `np.arange(-5, 5.5, 0.5)` | Controls oxidation and gas speciation |
| **`wh2o`** | Melt H₂O content | `np.logspace(-5, -1, 10)` | Sets available water for degassing |
| **`wco2`** | Melt CO₂ content | `np.logspace(-5, -2, 10)` | Defines carbon reservoir |
| **`ws`** | Melt sulfur content | `np.logspace(-4, -3, 10)` | Defines sulfur reservoir |
| **`outfile`** | Output directory names | e.g., `['Output_with_gc_COHS']` | Where CSVs are stored |
| **`newrun`** | Run flag | `1 = force rerun`, `0 = skip` | Prevents overwriting |
| **`uselast`** | Continuation flag | `1 = use previous solution` | Speeds up grid runs |

Each combination of (`Temp`, `fo2`, `wh2o`, `wco2`, `ws`) triggers a separate equilibrium calculation.

---

## 📊 How to Interpret Outputs

Each run produces a CSV file named after its parameters, e.g.:

```
-2.0_-4.0_-4.0_-3.0.csv == FMQ_log10(H2O)_log10(CO2)_log10(S).csv
```

Each file contains melt and gas composition versus pressure:

| Column | Symbol | Description | Units |
|---------|---------|-------------|--------|
| **P** | P | Total pressure | bar |
| **mmw** | — | Mean molecular weight of gas | g mol⁻¹ |
| **mfco2** | Xₘ(CO₂) | CO₂ mole fraction in melt | — |
| **mfh2o** | Xₘ(H₂O) | H₂O mole fraction in melt | — |
| **mfs** | Xₘ(S) | Sulfur fraction in melt | — |
| **mfh2** | Xₘ(H₂) | Dissolved hydrogen in melt | — |
| **pco2** | pCO₂ | Partial pressure of CO₂ | bar |
| **ph2o** | pH₂O | Partial pressure of H₂O | bar |
| **pch4** | pCH₄ | Partial pressure of CH₄ | bar |
| **pco** | pCO | Partial pressure of CO | bar |
| **ph2** | pH₂ | Partial pressure of H₂ | bar |
| **pso2** | pSO₂ | Partial pressure of SO₂ | bar |
| **ph2s** | pH₂S | Partial pressure of H₂S | bar |
| **ps2** | pS₂ | Partial pressure of S₂ | bar |
| **alphagas** | α_gas | Fraction of volatiles in gas phase | — |

---

## 🧠 Typical Workflow

1. Define temperature, redox, and volatile ranges in `COHS_gc.py`.  
2. Run `python COHS_gc.py` to compute equilibrium grids.  
3. Examine CSV files in the output folder.  
4. Use `grid_plot_COHS_gc.py` to visualize Outgassing-Zone contours.

---

## 📚 Citation

If this code or its outputs contribute to your research, please cite:

> Arora, R. (2025). *H₂-Dominated Outgassing: Interior–Atmosphere Coupling on Tidally Heated Exoplanets.*  
> [https://github.com/Rahul2013396/H2_dominated_outgassing](https://github.com/Rahul2013396/H2_dominated_outgassing)

```bibtex
@misc{arora2025_h2outgassing,
  author       = {Rahul Arora},
  title        = {H2_dominated_outgassing: Interior–Atmosphere Coupling of Volcanic H2 and Escape},
  year         = {2025},
  howpublished = {\url{https://github.com/Rahul2013396/H2_dominated_outgassing}}
}
```

---

## 📄 License

MIT License (recommended — add a `LICENSE` file if not already included).
