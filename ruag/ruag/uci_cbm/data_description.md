# Naval Vessel Gas Turbine (GT) Simulator Dataset

## Dataset Overview
This dataset is generated from a numerical simulator of a frigate with a Gas Turbine (GT) propulsion system. The simulator includes:

- Propeller  
- Hull  
- Gas Turbine (GT)  
- Gear Box  
- Controller  

The simulator models real vessel behavior, including **performance decay of GT components** (compressor and turbine).

**Objective:** Represent different degradation states of the GT propulsion system and related measures for analysis of performance decay and condition-based monitoring.

---

## Features (16 Measurements at Steady State)

1. **Lever position (lp)**  
2. **Ship speed (v)** [knots]  
3. **Gas Turbine shaft torque (GTT)** [kN·m]  
4. **GT rate of revolutions (GTn)** [rpm]  
5. **Gas Generator rate of revolutions (GGn)** [rpm]  
6. **Starboard Propeller Torque (Ts)** [kN]  
7. **Port Propeller Torque (Tp)** [kN]  
8. **High Pressure Turbine exit temperature (T48)** [°C]  
9. **GT Compressor inlet air temperature (T1)** [°C]  
10. **GT Compressor outlet air temperature (T2)** [°C]  
11. **HP Turbine exit pressure (P48)** [bar]  
12. **GT Compressor inlet air pressure (P1)** [bar]  
13. **GT Compressor outlet air pressure (P2)** [bar]  
14. **GT exhaust gas pressure (Pexh)** [bar]  
15. **Turbine Injection Control (TIC)** [%]  
16. **Fuel flow (mf)** [kg/s]  

---

## Target Variables / Degradation States

- **GT Compressor decay state coefficient (kMc)**, domain: `[0.95, 1]`  
- **GT Turbine decay state coefficient (kMt)**, domain: `[0.975, 1]`  

**Parameter Sampling:**  
- `kMc` and `kMt` sampled uniformly with **0.001 granularity**  
- Ship speed sampled from **3 to 27 knots** in steps of 3  

---

## Dataset Format

- Each row contains **18 elements**: 16 features + 2 decay coefficients  
- Features are **not normalized**  
- Files included:  
  - `README.txt` – dataset description  
  - `Features.txt` – list of all features  
  - `data.txt` – dataset  

---

## Research Usage / Applications

This dataset has been used to evaluate **Machine Learning approaches for condition-based maintenance** of naval propulsion systems. Key applications include:

- **Predictive maintenance**: estimating remaining useful life (RUL) of GT components  
- **Degradation modeling**: analyzing how compressor and turbine efficiency decays over time  
- **Fault detection**: identifying abnormal operating conditions via feature-based classification  
- **Simulation-based training**: testing algorithms under controlled, realistic naval vessel scenarios  
- **Performance optimization**: studying operational parameters to improve fuel efficiency and safety  

Representative publication:  
A. Coraddu, L. Oneto, A. Ghio, S. Savio, D. Anguita, M. Figari, *Machine Learning Approaches for Improving Condition-Based Maintenance of Naval Propulsion Plants*, Journal of Engineering for the Maritime Environment, 2014.
