#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 2024

@author: Donghyo
"""

import sys, argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd

def calculate_protein_ec_mw(sequence):
    """
    Calculate protein concentration based on absorbance at 280 nm using a Nanodrop instrument.
    
    Parameters:
    sequence (str): Protein amino acid sequence (single-letter code).
    A280 (float): Absorbance at 280 nm measured by Nanodrop.
    
    Returns:
    float: Protein concentration in mol/L.
    """
    # Assume the path length for Nanodrop is around 10 mm (verify for your specific model)
    path_length_nanodrop = 1  # in cm
    
    # Get extinction coefficient and molecular weight
    analysis = ProteinAnalysis(sequence)
    extinction_coefficient = analysis.molar_extinction_coefficient()[1]
    mw = analysis.molecular_weight()    # Unit: kDa
    
    return extinction_coefficient, mw

parser = argparse.ArgumentParser()

parser.add_argument("--input_fasta", required=True, type=str, help="Fasta file with amino acid sequences")
#parser.add_argument("--output_csv", type=str, help="Path of output csv")
args = parser.parse_args()

# Load sequence information
fasta_file = args.input_fasta
fasta_dic = {record.id: str(record.seq).replace("*", "") for record in SeqIO.parse(fasta_file, "fasta")}

#print(f"{('Name'):<{4}} {('Extinction_Coefficient'):>21} {('Molecular_weight'):>21}")
print(f"Name\tExtinction_Coefficient\tMolecular_weight\tseq")

concentration_dic = {}
for i, (name, sequence) in enumerate(fasta_dic.items()):
    sa = ProteinAnalysis(sequence)
    
    extinction_coefficient, mw = calculate_protein_ec_mw(sequence)
    #print(f"{name:<{4}} {extinction_coefficient:>21f} {mw:>21f}")
    print(f"{name:}\t{extinction_coefficient}\t{mw}\t{sequence}")

    
