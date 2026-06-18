import os, sys, importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_mod():
    return importlib.import_module("prepare_PDB_structure_into_theozyme")
