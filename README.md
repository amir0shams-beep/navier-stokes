# navier-stokes
computational sciences project for summer 2025-molecular dynamics of water
# Molecular Dynamics of Water — Phase 1 (Preliminaries)

## Model pieces (what they do)
- **Kinetic energy (T):** energy of motion; makes atoms move.
- **Bond/Angle energy (V_int):** keeps each H–O bond near a fixed length and the H–O–H angle near its target value.
- **Intermolecular energy (V_ext):** Lennard-Jones (short-range repulsion/attraction) + Coulomb (charge–charge interaction) between different molecules.

## Equations of motion (we'll implement later)
- \dot{q_i} = p_i / m_i  
- \dot{p_i} = -∇_{q_i} V(q)  
(Plain words: positions change with velocity; forces come from the energy terms.)

## Computational restrictions (why MD is hard)
- **Stiff bonds/angles:** very fast vibrations → need small time steps for stability.
- **Coulomb interactions:** long-range, many pairs → expensive for many atoms.

## Project structure (course-required files)
- `src/mdwater/` — Python package code
- `setup.py` — install script
- `requirements.txt` — dependencies (start empty)
- `README.md` — how to run + notes (this file)
## Run the LJ demo
pip install -e .
python -m mdwater.examples.lj_demo
## Run the water demo
python3 -m mdwater.examples.water_demo

## Results (examples)
- Water (no PBC): drift ≈ 3.2e-08 over 3000 steps (dt = 5e-4)
- Water + PBC + Ewald: drift ≈ 1.0e-04 over 3000 steps (dt = 5e-4, α=0.25, r_real=6.0, kmax=6)