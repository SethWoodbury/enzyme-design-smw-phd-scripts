# Step03: FastMPNN Design with Rosetta Refinement
"""
Iterative MPNN sequence design with Rosetta relaxation for enzyme active site optimization.

This module takes step02 output (relaxed PDB + metrics JSON) and performs:
1. Residue classification (catalytic, conserved motif, primary/secondary sphere)
2. Iterative MPNN sequence design with configurable protocols
3. Rosetta relaxation (cartesian and torsional) with coordinate constraints
4. Favorable interaction detection and conservation
5. Comprehensive metrics tracking

Usage:
    python fastmpnn_design.py --step02_json <path> --params <file> [options]
"""
