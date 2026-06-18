#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:45:33 2023

@author: ikalvet

Seth Woodbury adapted protocol for FastRelax with CST only
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
import os, sys
import glob
import random
import json
import argparse

sys.path.append("/software/scripts/enzyme_design/FastMPNNDesign")
import FastMPNNdesign
from Selectors import SelectHBondsToResidue
sys.path.append("/software/scripts/enzyme_design/utils")
import design_utils
import scoring_utils


class CSTs():
    def __init__(self, cstfile, scorefxn):
        self.__scorefxn = scorefxn
        self.__addcst_mover = pyrosetta.rosetta.protocols.enzdes.AddOrRemoveMatchCsts()
        self.__chem_manager = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
        self.__residue_type_set = self.__chem_manager.residue_type_set("fa_standard")
        self.__cst_io = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO(self.__residue_type_set)
        self.__cst_io.read_enzyme_cstfile(cstfile)
        pass
    
    def add_cst(self, pose):
        self.__cst_io.add_constraints_to_pose(pose, self.__scorefxn, True)
    
    def cst_io(self):
        return self.__cst_io

    def remove_cst(self, pose):
        self.__cst_io.remove_constraints_from_pose(pose, True, True)
        
    def cst_score(self, pose):
        """
        To be implemented
        """
        return None


def get_2nd_layer_fixed_pos(pose, ligand_resno, heavyatoms, keep_pos):
    dist_bb = 6.0
    dist_sc = 5.0
    motif_label_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(label_str="keep_hbonds_to_ligand_and_catres")
    pocket_positions = keep_pos+list(pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_label_sel.apply(pose)))
    pocket_positions = list(set(pocket_positions))
    SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
        = design_utils.get_layer_selections(pose, repack_only_pos=pocket_positions,
                                            design_pos=[], ref_resno=ligand_seqpos, heavyatoms=heavyatoms,
                                            cuts=[dist_bb, dist_bb+2.0, dist_bb+4.0, dist_bb+6.0], design_GP=True)

    # Need to somehow pick pocket residues that have SC atoms close to the ligand.
    # Exclude from design: residues[0] and those that have SC atoms very close.
    close_ones = design_utils.get_residues_with_close_sc(pose, heavyatoms, residues[1]+residues[2], exclude_residues=pocket_positions, cutoff=dist_sc)
    pocket_positions += residues[0] + close_ones
    pocket_positions = list(set(pocket_positions))
    design_residues = [x for x in residues[0]+residues[1]+residues[2]+residues[3] if x not in pocket_positions]

    # Also including all Alanines that are not in the pocket.
    ala_positons = [res.seqpos() for res in pose.residues if res.seqpos() not in pocket_positions+design_residues and res.name3() == "ALA"]
    print("ALA positions", '+'.join([str(x) for x in ala_positons]))
    design_residues += ala_positons
    fixed_residues = [f"{pose.pdb_info().chain(r.seqpos())}{pose.pdb_info().number(r.seqpos())}" for r in pose.residues if r.seqpos() not in design_residues and r.is_protein()]
    return fixed_residues


parser = argparse.ArgumentParser()

parser.add_argument("--pdb", type=str, required=True, help="Input PDB file, containing a ligand and matcher CST lines in header.")
parser.add_argument("--nstruct", type=int, default=1, help="How many design iterations?")
parser.add_argument("--suffix", type=str, help="Suffix to be added to the end the output filename")

parser.add_argument("--params", type=str, nargs="+", help="Ligand and NCAA params file(s)")
parser.add_argument("--cstfile", type=str, help="Matcher/enzdes CSTfile")

parser.add_argument("--design_pos", type=int, nargs="+", help="Positions that will be redesigned.")
parser.add_argument("--keep_pos", type=int, nargs="+", help="Positions that will be kept fixed. Repack is allowed.")
parser.add_argument("--detect_pocket", action="store_true", default=False, help="Figure out designable positions around the ligand algorithmically.")
parser.add_argument("--protocol", type=str, help="Text file defining the FastMPNNDesign protocol that will be applied.")
parser.add_argument("--scoring", type=str, help="Scoring script that calculates custom scores for a design, and implements filtering.")

parser.add_argument("--position_bias", type=float, default=-1.0, help="(default = -1.0) Bias that will be applied to polar AAs at positions selected with distance from --bias_atoms.")
parser.add_argument("--bias_atoms", nargs="+", type=str, help="Ligand atom names for which the surrounding residues will receive a bias towards/agains KREDYQWSTH aa's")
parser.add_argument("--bias_AAs", type=str, default="KREDYQWSTH", help="(default = KREDYQWSTH) AA1 letters of amino acids that should be biased near atoms defined with --bias_atoms with a bias defined with --position_bias")
parser.add_argument("--filter", action="store_true", default=False, help="Only dump outputs that meet filtering criteria set in scoring script")
parser.add_argument("--mpnn", action="store_true", default=False, help="Performs additional 2nd layer MPNN on successful outputs")


args = parser.parse_args()

if args.scoring is None:
    args.scoring = os.path.dirname(__file__) + "/esterase_scoring.py"

## Loading the user-provided scoring module
sys.path.append(os.path.dirname(args.scoring))
scoring = __import__(os.path.basename(args.scoring.replace(".py", "")))
assert hasattr(scoring, "score_design")
assert hasattr(scoring, "filter_scores")
assert hasattr(scoring, "filters")

pdbname = os.path.basename(args.pdb).replace(".pdb", "")


if args.mpnn is True:
    os.makedirs("seqs/", exist_ok=True)

suffix = ""
if args.suffix is not None:
    suffix = "_" + args.suffix

## Saving scorefiles separately
os.makedirs("scores/", exist_ok=True)
scorefilename = f"scores/{pdbname}{suffix}.sc"


"""
Rosetta stuff
"""
if args.params is None:
    args.params = ["/home/ikalvet/projects/Organophosphate/Esterase/theozyme/ZRE/ZRE.params"]

extra_res_fa = "-extra_res_fa "+" ".join(args.params)


NPROC = os.cpu_count()
if "OMP_NUM_THREADS" in os.environ:
    NPROC = os.environ["OMP_NUM_THREADS"]
if "SLURM_CPUS_ON_NODE" in os.environ:
    NPROC = os.environ["SLURM_CPUS_ON_NODE"]

multithreading = ""
if int(NPROC) > 1:
    multithreading = f"-multithreading true -multithreading:total_threads {NPROC} -multithreading:interaction_graph_threads {NPROC}"

DAB = "/net/software/lab/scripts/enzyme_design/DAlphaBall.gcc"
pyr.init(f"{extra_res_fa} -dalphaball {DAB} -beta_nov16 -run:preserve_header {multithreading}")


sfx = pyr.get_fa_scorefxn()

if args.cstfile is not None:
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 1.0)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), 1.0)
    sfx.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), 1.0)     
    cst_mover = CSTs(args.cstfile, sfx)





pose = pyr.pose_from_file(args.pdb)

ligand_seqpos = pose.size()
assert pose.residue(ligand_seqpos).is_ligand()
ligand = pose.residue(ligand_seqpos)


"""
Setting up design/repack layers
"""

if args.design_pos is None:
    design_pos = [] 
else:
    design_pos = args.design_pos

keep_pos = []
if args.keep_pos is not None:
    keep_pos = args.keep_pos

catres = design_utils.get_matcher_residues(pose)  # catalytic residues defined with REMARK 666

keep_pos += list(catres.keys())

## Identifying any motif residues based on pdbinfo reslabel
motif_label_sel = pyrosetta.rosetta.core.select.residue_selector.ResiduePDBInfoHasLabelSelector(label_str="motif")
keep_pos += list(pyrosetta.rosetta.core.select.get_residue_set_from_subset(motif_label_sel.apply(pose)))
keep_pos = list(set(keep_pos))

heavyatoms = design_utils.get_ligand_heavyatoms(pose)

# Finding out what residues belong to what layer, based on the CA distance
# from ligand heavyatoms.
SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
    = design_utils.get_layer_selections(pose, keep_pos,
                                        design_pos, ligand_seqpos, heavyatoms)

if args.detect_pocket is False:
    if args.design_pos is not None:
        design_residues = design_pos
    else:
         ## Designing all residues that are not meant to stay fixed
        design_residues = [res.seqpos() for res in pose.residues if not res.is_ligand() and res.seqpos() not in keep_pos]
        repack_residues = list(set(keep_pos + [res.seqpos() for res in pose.residues if res.seqpos() not in design_residues]))
        do_not_touch_residues = []  # not really relevant with mpnn-packing. Was used formerly to not repack certain sidechains at all.
else:
    substrate_atoms_ref = [ligand.atom_name(n).strip() for n in range(1, ligand.natoms()+1) if ligand.atom_name(n).strip() not in ["ZN1", "O1"]]
    __a, __b, __c, residues_substrate\
        = design_utils.get_layer_selections(pose, keep_pos,
                                            design_pos, ligand_seqpos, substrate_atoms_ref, cuts=[7.0, 9.0, 11.0, 13.0])
    design_residues = [x for x in residues_substrate[0]+residues_substrate[1]]
    design_residues += design_utils.get_residues_with_close_sc(pose, substrate_atoms_ref, residues_substrate[2]+residues_substrate[3], keep_pos, 8.0)
    design_residues = list(set(design_residues))



repack_residues = residues[2] + residues[3] + residues[4]+ [ligand_seqpos]
do_not_touch_residues = []

for res in residues[0]+residues[1]:
    if res not in design_residues:
        repack_residues.append(res)

repack_residues = [x for x in repack_residues if x not in design_residues]

unclassified_residues = [res.seqpos() for res in pose.residues if res.seqpos() not in design_residues+repack_residues+do_not_touch_residues]
assert len(unclassified_residues) == 0, f"Some residues have not been layered: {unclassified_residues}"

print("Design positions: ", "+".join([str(x) for x in design_residues]))


"""
Pre-relax with constraints
"""

## Performing quick constrained cartesian FastRelax on the input structure to get the ligand placement correct
## Mutating clashing non-motif residues to ALA
cartesian_prerlx = True
print(f"Performing quick constrained cartesian={cartesian_prerlx} FastRelax on the input structure to get the ligand placement correct")
sfx_cart = sfx.clone()
if cartesian_prerlx:
    sfx_cart.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("cart_bonded"), 0.5)
    sfx_cart.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("pro_close"), 0.0)
fastRelax = design_utils.setup_fastrelax(sfx_cart, crude=True, disable_min_resons=[ligand_seqpos])
fastRelax.cartesian(cartesian_prerlx)
ligand = pose.residue(ligand_seqpos)
clashes = design_utils.find_clashes_between_target_and_sidechains(pose, pose.size(),
                                                                  target_atoms=[n for n in range(1, ligand.natoms()+1) if not ligand.atom_is_hydrogen(n)],
                                                                  residues=[n for n in range(1, pose.size()) if n not in keep_pos])
clashes = [x for x in clashes if pose.residue(x).name3() not in ["ALA", "GLY", "PRO"] and x not in keep_pos]
_pose2 = design_utils.mutate_residues(pose, clashes, "ALA")
if args.cstfile is not None:
    cst_mover.cst_io().add_constraints_to_pose(_pose2, sfx_cart, True)
# _pose3 = _pose2.clone()
fastRelax.apply(_pose2)
sfx_cart(_pose2)
if args.cstfile is not None:
    cst_score = sum([_pose2.scores[s] for s in _pose2.scores if "constraint" in s])
    print(f"CST score after ALA FastRelax: {cst_score:.2f}")

# _pose2.dump_pdb(f"{pdbname}_prerelax3_{suffix}.pdb")



## Seting up per-position bias on positions close to user-defined atoms
bias_positions_dict = {}
if args.bias_atoms is not None:
    assert all([_pose2.residue(ligand_seqpos).has(a) for a in args.bias_atoms]), "Some --bias_atoms atom names are invalid."
    __a, __b, __c, residues_bias\
        = design_utils.get_layer_selections(_pose2, keep_pos, design_pos, ligand_seqpos, args.bias_atoms, cuts=[5.0, 7.0, 9.0, 11.0])
    bias_positions = [x for x in residues_bias[0]+residues_bias[1]]
    # bias_positions += design_utils.get_residues_with_close_sc(_pose2, args.bias_atoms, residues_bias[2]+residues_bias[3], keep_pos, 4.0)
    bias_positions = list(set(bias_positions))

    # Excluding positions that are closer to ligand H1 than to O1
    if "O1" not in args.bias_atoms and "H1" in args.bias_atoms:
        bias_positions2 = []
        for pos in bias_positions:
            _dist_nbr = (_pose2.residue(pos).xyz("CA") - _pose2.residue(ligand_seqpos).xyz("O1")).norm()
            _dists = [(_pose2.residue(pos).xyz("CA") - _pose2.residue(ligand_seqpos).xyz(a)).norm() for a in args.bias_atoms]
            if min(_dists) < _dist_nbr:
                bias_positions2.append(pos)
        bias_positions = bias_positions2


    print(f"Bias positions for {args.bias_AAs}: ", "+".join([str(x) for x in bias_positions]))
    bias_positions_dict = {}
    for pos in bias_positions:
        bias_positions_dict[f"{_pose2.pdb_info().chain(pos)}{pos}"] = {a: args.position_bias for a in args.bias_AAs}


## Figuring out positions near the H2O for placing a GLU/ASP
## The water atoms are called O1 H1
# water_activator_pos = []
# for res in _pose2.residues:
#     if not res.is_protein():
#         continue
#     if (res.xyz("CA") - _pose2.residue(ligand_seqpos).xyz("H1")).norm() > 8.0:
#         continue
#     if (res.xyz("CA") - _pose2.residue(ligand_seqpos).xyz("H1")).norm() > (res.xyz("CA") - _pose2.residue(ligand_seqpos).xyz("O1")).norm():
#         continue
#     water_activator_pos.append(res.seqpos())
# print(f"Water activator positions: {water_activator_pos}")


## Defining the design protocol for FastMPNNDesign
protocol = """
scale:coordinate_constraint 1.0
scale:fa_rep 0.150
repack
scale:fa_rep 0.200
min 0.01
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.5
scale:fa_rep 0.365
scale:fa_rep 0.480
min 0.01
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.0
scale:fa_rep 0.659
scale:fa_rep 0.750
min 0.01
task_operation keep_hbonds_to_ligand_and_catres
scale:coordinate_constraint 0.0
scale:fa_rep 1
min 0.00001
"""

#scale:coordinate_constraint 1.0
#scale:fa_rep 0.150
#repack
#scale:fa_rep 0.200
#min 0.01
#task_operation keep_hbonds_to_ligand_and_catres
#scale:coordinate_constraint 0.5
#scale:fa_rep 0.365
#repack
#scale:fa_rep 0.480
#min 0.01
#task_operation keep_hbonds_to_ligand_and_catres
#scale:coordinate_constraint 0.0
#scale:fa_rep 0.659
#repack
#scale:fa_rep 0.750
#min 0.01
#task_operation keep_hbonds_to_ligand_and_catres
#scale:coordinate_constraint 0.0
#scale:fa_rep 1
#repack
#min 0.00001


# filter_dict = {"L_SASA": [0.30, "<="],
#            "substrate_SASA": [1.0, ">="],
#            "H2O_hbond": [0.0, "="],
#            "oxy_hbond": [0.0, "="],
#            "cms_per_atom": [2.0, ">="],
#            "corrected_ddg": [-20.0, "<="],
#            "nlr_totrms": [1.2, "<="]}

if args.protocol is not None:
    protocol = args.protocol


## Defining a method that keeps H-bond contacts to ligand and motif fixed
ligand_and_catres_hbond_keeper = SelectHBondsToResidue(name="keep_hbonds_to_ligand_and_catres")
ligand_and_catres_hbond_keeper.target([ligand_seqpos]+list(catres))  # ligand and catalytic residues
ligand_and_catres_hbond_keeper.allow_updating(True)  # allowing updating the target set based on found H-bond contacts to these
ligand_and_catres_hbond_keeper.accept_probability(0.75)  # 75% chance of keeping an identified H-bond contact

if args.mpnn is True:
    mpnnrunner = FastMPNNdesign.fusedmpnn.MPNNRunner("ligand_mpnn", verbose=True, pack_sc=False, ligand_mpnn_use_side_chain_context=True)

for N_iter in range(args.nstruct):
    if len(glob.glob(f"{pdbname}{suffix}_{N_iter}_*.pdb")) > 0:
        print(f"{pdbname}{suffix}_{N_iter}_* outputs already exist")
        continue

    pose2 = _pose2.clone()

    if args.cstfile is not None:
        cst_mover.remove_cst(pose2)
        cst_mover.add_cst(pose2)

    fmd = FastMPNNdesign.FastMPNNdesign(model_type="enhanced_mpnn", params=args.params, scorefxn=sfx, script_file=protocol,
                                        design_positions=design_residues,
                                        repack_positions=repack_residues, do_not_repack_positions=do_not_touch_residues,
                                        cst_io=cst_mover.cst_io(), omit_AA="CM")

    if args.bias_atoms is not None:
        if len(bias_positions_dict) != 0:
            fmd.set_mpnn_bias_per_residue(bias_positions_dict)

    fmd.add_task_operation(ligand_and_catres_hbond_keeper)

    poses = fmd.apply(pose2)

    for i, p in enumerate(poses):
        cst_mover.add_cst(p)
        scores_df = scoring.score_design(p, pyr.get_fa_scorefxn(), list(catres.keys()))

        if args.filter is True and len(scoring.filter_scores(scores_df)) == 0:
            print(f"BAD design: {pdbname}{suffix}_{N_iter}_{i}")
            continue

        scores_df.at[0, "description"] = f"{pdbname}{suffix}_{N_iter}_{i}"
        scoring_utils.dump_scorefile(scores_df, scorefilename)
        p.dump_pdb(f"{pdbname}{suffix}_{N_iter}_{i}.pdb")

        ## Performing 2nd layer MPNN on successful outputs
        if args.mpnn is True:
            fixed_residues = get_2nd_layer_fixed_pos(p, ligand_seqpos, heavyatoms, keep_pos)
            pdbstr = pyrosetta.distributed.io.to_pdbstring(p)
            for T in [0.1, 0.2]:
                mpnn_input = mpnnrunner.MPNN_Input()
                mpnn_input.fixed_residues = fixed_residues
                mpnn_input.omit_AA = ["C", "M"]
                mpnn_input.pdb = pdbstr
                mpnn_input.name = f"{pdbname}{suffix}_{N_iter}_{i}"
                mpnn_input.number_of_batches = 1
                mpnn_input.batch_size = 5
                mpnn_input.temperature = T
                mpnn_out = mpnnrunner.run(mpnn_input)

                ## Threading the new MPNN sequences onto the designed backbone,
                ## and repacking the sidechains again. Not going to do full relax.
                for seqid, seq in enumerate(mpnn_out["generated_sequences"]):
                    _pose_threaded = design_utils.thread_seq_to_pose(p, seq)
                    _pose_threaded = design_utils.fix_catalytic_residue_rotamers(_pose_threaded, p, catres)
                    if args.cstfile is not None:
                        cst_mover.add_cst(_pose_threaded)
                    _pose_threaded_packed = design_utils.repack(_pose_threaded, sfx)
                    sfx(_pose_threaded_packed)
                    _pose_threaded_packed.dump_pdb(f"seqs/{pdbname}{suffix}_{N_iter}_{i}_T{T}_{seqid}.pdb")


                # with open(f"seqs/{pdbname}{suffix}_{N_iter}_{i}.fasta", "a") as file:
                #     file.write(f">{pdbname}{suffix}_{N_iter}_{i}_native\n")
                #     file.write(mpnn_out["native_sequence"]+"\n")
                #     for j, s in enumerate(mpnn_out["generated_sequences"]):
                #         file.write(f">{pdbname}{suffix}_{N_iter}_{i}_T{T}_s0_{j}\n")
                #         file.write(s+"\n")
