#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selectors Module - H-Bond Residue Selection

This module provides classes for selecting residues that form hydrogen bonds
to specified target residues. The primary use case is identifying which
residues are forming H-bonds to a ligand or other target residue(s) in a
protein structure, which can then be used to inform design decisions
(e.g., fixing certain residues during sequence design).

Classes:
    SelectHBondsToResidue: Identifies residues forming H-bonds to target residues
"""

import copy

import numpy as np
import pandas as pd
import pyrosetta
import pyrosetta.distributed.io
import pyrosetta.rosetta


# =============================================================================
# H-Bond Residue Selector
# =============================================================================

class SelectHBondsToResidue():
    """
    Selects residues that form hydrogen bonds to specified target residue(s).

    This class analyzes a pose to find all residues that donate or accept
    hydrogen bonds to/from the target residue(s). Options allow filtering
    by backbone involvement and probabilistic selection.
    """

    def __init__(self, name):
        self.__name = name
        self.__target_resnos = None
        self.__target_atoms = None
        self.__accept_prob = 1.0
        self.__accept_energy = 0.0
        self.__include_bb_hbonds = False
        self.__enable_updating = False

    # -------------------------------------------------------------------------
    # Property Getters/Setters
    # -------------------------------------------------------------------------

    def target(self, resno=None):
        """
        Sets or returns the target residue number(s) for H-bond analysis
        """
        if resno is None:
            return self.__target_resnos
        else:
            assert isinstance(resno, (int, list))
            if isinstance(resno, int):
                self.__target_resnos = [resno]
            else:
                self.__target_resnos = [x for x in resno]

    def target_atoms(self, target_atoms=None):
        """
        Currently not used
        Sets or returns the target residue atom names number for H-bond analysis
        """
        if target_atoms is None:
            return self.__target_atoms
        else:
            assert isinstance(target_atoms, list)
            self.__target_atoms = target_atoms

    def name(self):
        return self.__name

    def accept_probability(self, probability=None):
        if probability is None:
            return self.__accept_prob
        else:
            assert isinstance(probability, float)
            assert 0.0 <= probability <= 1.0
            self.__accept_prob = probability

    def accept_energy(self, energy=None):
        """
        Not used currently.
        Would enable selection of H-bonds only if they're below specified energy threshold'
        """
        if energy is None:
            return self.__accept_energy
        else:
            assert isinstance(energy, float)
            self.__accept_energy = energy

    def copy(self):
        return copy.deepcopy(self)

    def include_backbone_hbonds(self, include=None):
        """
        Whether residues with backbones atoms as H-bond donors/acceptors will be fixed.
        default = False

        Parameters
        ----------
        include : bool, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        bool
        """
        if include is None:
            return self.__include_bb_hbonds
        else:
            assert isinstance(include, bool)
            self.__include_bb_hbonds = include

    def allow_updating(self, allow=None):
        if allow is None:
            return self.__enable_updating
        else:
            assert isinstance(allow, bool)
            self.__enable_updating = allow

    # -------------------------------------------------------------------------
    # Main Computation
    # -------------------------------------------------------------------------

    def compute(self, pose):
        """
        Compute residues forming H-bonds to the target residue(s).

        Parameters
        ----------
        pose : pyrosetta.Pose
            The pose to analyze for hydrogen bonds

        Returns
        -------
        list
            List of residue numbers that form H-bonds to the target residue(s)
        """
        # Calculating HBonds in the pose (Sam Pellock implementation)
        hbonds = pose.get_hbonds()

        # Create empty lists to store the data
        donor_residues = []
        acceptor_residues = []
        donor_atoms = []
        acceptor_atoms = []
        hbond_energies = []

        # Loop over each HBond in the HBondSet and extract the information
        for hbond in hbonds.hbonds():
            donor_residues.append(hbond.don_res())
            acceptor_residues.append(hbond.acc_res())
            donor_atoms.append(pose.residue(hbond.don_res()).atom_name(hbond.don_hatm()))
            acceptor_atoms.append(pose.residue(hbond.acc_res()).atom_name(hbond.acc_atm()))
            hbond_energies.append(hbond.energy())

        # Create the pandas DataFrame from the lists
        hbond_df = pd.DataFrame({
            'donor_residue': donor_residues,
            'acceptor_residue': acceptor_residues,
            'donor_atom': donor_atoms,
            'acceptor_atom': acceptor_atoms,
            'energy': hbond_energies
            })

        # Filter the DataFrame to only include hydrogen bonds involving the target
        mask = (hbond_df['donor_residue'].isin(self.target())) | (hbond_df['acceptor_residue'].isin(self.target()))
        target_hbond_df = hbond_df[mask]
        if self.target_atoms() is not None:
            pass

        if len(target_hbond_df) == 0:
            return []

        # Filter out backbone H-bonds if not included
        if self.include_backbone_hbonds() is False:
            mask_not_bb = []
            for idx, row in target_hbond_df.iterrows():
                hbond_kept = True
                if all([row.donor_residue in self.target(), row.acceptor_residue in self.target()]):
                    hbond_kept = False

                elif any([pose.residue(row.donor_residue).is_ligand(), pose.residue(row.acceptor_residue).is_ligand()]):
                    # If the target is ligand then do not select bb-hbonds
                    res1 = pose.residue(row.donor_residue)
                    res2 = pose.residue(row.acceptor_residue)
                    if any([res1.atom_is_backbone(res1.atom_index(row.donor_atom)), res2.atom_is_backbone(res2.atom_index(row.acceptor_atom))]):
                        hbond_kept = False
                else:
                    # Keep residue if the target is backbone and h-bond partner is sidechain
                    if row.donor_residue in self.target():
                        tgt_pair = (row.donor_residue, row.donor_atom)
                        other_pair = (row.acceptor_residue, row.acceptor_atom)
                    else:
                        tgt_pair = (row.acceptor_residue, row.acceptor_atom)
                        other_pair = (row.donor_residue, row.donor_atom)
                    res_tgt = pose.residue(tgt_pair[0])
                    res_other = pose.residue(other_pair[0])
                    if all([res_tgt.atom_is_backbone(res_tgt.atom_index(tgt_pair[1])), res_other.atom_is_backbone(res_other.atom_index(other_pair[1]))]):
                        hbond_kept = False
                    elif res_other.atom_is_backbone(res_other.atom_index(other_pair[1])):
                        hbond_kept = False
                    else:
                        hbond_kept = True
                mask_not_bb.append(hbond_kept)
            target_hbond_df = target_hbond_df[mask_not_bb]

        # Build final selection list
        selection = list(set(target_hbond_df.donor_residue.unique().tolist()+target_hbond_df.acceptor_residue.unique().tolist()))
        selection = [x for x in selection if x not in self.target()]

        # Apply probabilistic filtering if accept_probability < 1.0
        if self.accept_probability() < 1.0:
            selection_prob = []
            for rn in selection:
                if np.random.rand() <= self.accept_probability():
                    selection_prob.append(rn)
            selection = [x for x in selection_prob]

        return selection
