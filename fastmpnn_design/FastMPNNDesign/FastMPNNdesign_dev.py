#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:23:42 2023

@author: ikalvet
"""
import pyrosetta
import pyrosetta.rosetta
import pyrosetta.distributed.io
import os, sys
import io
import copy
import time
import pandas as pd
proteinmpnn_dir = '/net/software/lab/scripts/enzyme_design/fused_mpnn_api'
sys.path.append(proteinmpnn_dir)
import fusedmpnn
SCRIPT_DIR = os.path.dirname(__file__)

class FastMPNNdesign():
    def __init__(self, model_type=None, N_seq=5, params=None, name=None,
                 scorefxn=None, min_type="lbfgs_armijo_nonmonotone", script_file=f"{SCRIPT_DIR}/fastmpnndesign_protocol.txt", taskfactory=None,
                 cartesian=False,
                 design_positions=None, repack_positions=None, do_not_repack_positions=None, omit_AA=None, cst_io=None, debug=False,
                 mpnn_pack_sc=True, ligand_mpnn_use_side_chain_context=True):

        #  Attributes user can set
        self.__mpnnrunner = fusedmpnn.MPNNRunner(model_type, verbose=True, pack_sc=mpnn_pack_sc,
                                                 ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context)
        self.__num_sequences = N_seq
        self.__num_sequences_original = N_seq
        self.__params_files = params
        self.__min_type = min_type
        self.__script = self._setup_schedule(script_file)
        self.__cst_io = cst_io

        self.__minimizer_rmsd_cutoff = 3.0

        if scorefxn is None:
            self.__scorefxn = pyrosetta.get_fa_scorefxn()
        else:
            self.__scorefxn = scorefxn

        self.__tf = taskfactory
        self.__cartesian = cartesian
        self.__design_positions = design_positions
        self.__repack_positions = repack_positions
        self.__do_not_repack_positions = do_not_repack_positions
        self.__name = name
        if name is None:
            self.__name = "pose_0000"
        self.__MPNN_pack_sc = mpnn_pack_sc
        self.__debug = debug
        self.__bias_AAs = None
        self.__bias_AAs_per_residue = None
        self.__omit_AA = omit_AA

        self.__task_operations = {}

        # Information stored at runtime and not meant to be settable
        self.__input_pose = None
        self.__not_design_pos_list = None
        self.__design_pos_list = None
        self.__movers = {}
        self.__movemap = None
        self.__mpnn_input = None
        
        self.__mpnn_N_seq_after_first = 1

        pass

    def mpnnrunner(self):
        return self.__mpnnrunner

    def set_mpnn_N_seq_after_first(self, N):
        """
        How many sequences will be designed with MPNN after the first application of MPNN
        """
        self.__mpnn_N_seq_after_first = N

    def minimizer_rmsd_cutoff(self, cutoff=None):
        """
        Sets or returns the minimizer rmsd cutoff
        """
        if cutoff is None:
            return self.__minimizer_rmsd_cutoff
        else:
            assert isinstance(cutoff, float)
            self.__minimizer_rmsd_cutoff = cutoff

    def set_minimizer_movemap(self, movemap):
        assert isinstance(movemap, pyrosetta.rosetta.core.kinematics.MoveMap)
        self.__movemap = movemap

    def MPNN_pack_sc(self, enable_pack=None):
        """
        if enable_pack is None then returns the stored value for MPNN_pack_sc
        if enable_pack is bool then sets MPNN_pack_sc to that value
        """
        assert isinstance(enable_pack, (bool, type(None)))
        if enable_pack is None:
            return self.__MPNN_pack_sc
        else:
            self.__MPNN_pack_sc = enable_pack

    def add_task_operation(self, taskop):
        """
        Adds a taskoperation instance to the method.
        The TaskOperation can be any arbitrary Python object that has the
        following methods implemented: 
            compute(pose) -> list
            target() -> list
            target(list) :: sets a new value as target
            allow_updating() -> bool
            name() -> str
            copy() -> obj
        The method 'compute' takes 'pose' as argument and returns a list of
        residue numbers that would then be used to update the list of residues for MPNN
        """
        assert hasattr(taskop, "target")
        assert hasattr(taskop, "compute")
        assert hasattr(taskop, "allow_updating")
        assert hasattr(taskop, "name")
        assert hasattr(taskop, "copy")
        self.__task_operations[taskop.name()] = taskop.copy()


    # to utils
    def _build_pose_from_str_and_append_stuff(self, pdb_str, append_pose=None, append_pose_resnos=None, prepend_lines=None, append_lines=None, ref_pose=None):

        if prepend_lines is None:
            prepend_lines = []
        if append_lines is None:
            append_lines = []
    
        _pdb = "\n".join(prepend_lines) + pdb_str + "\n".join(append_lines)
        _pose = pyrosetta.rosetta.core.pose.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(_pose, _pdb)

        # Adjusting residue PDB numbering based on a reference pose
        if ref_pose is not None:
            for res in _pose.residues:
                _pose.pdb_info().number(res.seqpos(), ref_pose.pdb_info().number(res.seqpos()))

        if append_pose is not None and append_pose_resnos is not None:
            for append_pose_resno in append_pose_resnos:
                pyrosetta.rosetta.core.pose.append_subpose_to_pose(_pose, append_pose, append_pose_resno, append_pose_resno, 1)

        # Apply user-provided constraint mover
        if self.__cst_io is not None:
            if isinstance(self.__cst_io, pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO):
                self.__cst_io.add_constraints_to_pose(_pose, self.scorefxn(), True)
                constrained_residues = self.__cst_io.ordered_constrained_positions(_pose)
                self.__cst_io.remove_constraints_from_pose(_pose, True, True)
                _pose.constraint_set().clear()
                _pose.constraint_set().clear_sequence_constraints()
                # Re-adjusting the rotamers of constrained residues because MPNN-packer can mess them up
                for resno in constrained_residues:
                    if _pose.residue(resno).is_ligand():
                        continue
                    # Making sure it's the same HIS tautomer
                    if _pose.residue(resno).name3() == "HIS" and (_pose.residue(resno).name() != ref_pose.residue(resno).name()):
                        print(f"Mutating residue {resno} from {_pose.residue(resno).name()} to {ref_pose.residue(resno).name()}")
                        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                        mutres.set_res_name(ref_pose.residue(resno).name())
                        mutres.set_target(resno)
                        mutres.apply(_pose)
                        print(f"Mutated residue {resno} to {_pose.residue(resno).name()}")
    
                    for chino in range(1, _pose.residue(resno).nchi()+1):
                        print(f"Changing chi {chino} from {_pose.residue(resno).chi(chino)} to {ref_pose.residue(resno).chi(chino)}")
                        _pose.residue(resno).set_chi(chino, ref_pose.residue(resno).chi(chino))
                self.__cst_io.add_constraints_to_pose(_pose, self.scorefxn(), True)  # This changes HIS tautomer???

        return _pose

    # to utils
    def _setup_schedule(self, script_file):
        script = None
        script_list = []
        if isinstance(script_file, str):
            if os.path.exists(script_file):
                print(f"Reading design script from {script_file}")
                script = open(script_file, "r").readlines()
            else:
                script = script_file.split("\n")
            print("###### Parsed protocol: ####")
            print("\n".join(script))
            print("############################")
            for l in script:
                if len(l) == 0:
                    continue
                if len(l.split()) > 1:
                    script_list.append([l.split()[0].strip()])
                    for x in l.split()[1:]:
                        if x[0].isalpha():
                            script_list[-1].append(x)
                        else:
                            script_list[-1].append(float(x))
                else:
                    script_list.append([l.strip()])
        elif isinstance(script_file, list):
            script_list = script_file
        return script_list


    # TODO: add more setters and getters

    def scorefxn(self):
        return self.__scorefxn

    def script(self):
        return self.__script
    
    def mpnn_input(self):
        return self.__mpnn_input

    def set_mpnn_bias(self, bias_dict):
        """
        dict of per-AA bias
        """
        self.__bias_AAs = bias_dict

    def set_mpnn_bias_per_residue(self, bias_dict):
        """
        dict of per-AA bias per position
        Keys must be in format {chain}{resno}
        """
        for k in bias_dict.keys():
            assert isinstance(k, str)
            assert not k[0].isnumeric()
        self.__bias_AAs_per_residue = bias_dict

    def do_minimize(self, pose, tolerance, movemap, min_type):
        min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(movemap, self.scorefxn, min_type, tolerance, True)
        min_mover.apply(pose)

    def do_mpnn(self, pose, mpnn_input, temperature, num_sequences):
        # parsed_pdb = ligandmpnn_api.parser_tools.parse_pose(pose, self.__name, self.__params_files)
        pdbstr = pyrosetta.distributed.io.to_pdbstring(pose)
        remarks = [l for l in pdbstr.split("\n") if "REMARK 666" in l]

        ligands = [res.seqpos() for res in pose if res.is_ligand()]

        # Making a new instance of mpnn input for this run call
        # it inherits attributes that were globally set
        mpnn_input = mpnn_input.copy()

        mpnn_input.pdb = pdbstr
        mpnn_input.name = self.__name
        mpnn_input.temperature = temperature
        num_sequences = int(num_sequences)
        if num_sequences <= 15:
            mpnn_input.batch_size = num_sequences
            mpnn_input.number_of_batches = 1
        else:
            ## Finding the largest batch size up to 15 that would allow generating
            ## the number of sequences that was requested
            batch_size_num = []
            for n in range(15, 0, -1):
                if num_sequences % n == 0:
                    batch_size_num = [n, num_sequences // n]
                    break
            if max(batch_size_num) <= 15:
                batch_size = max(batch_size_num)
                num_batches = min(batch_size_num)
            else:
                batch_size = min(batch_size_num)
                num_batches = max(batch_size_num)
            mpnn_input.batch_size = batch_size
            mpnn_input.number_of_batches = num_batches

        print(f"Generating {num_sequences} sequences with ligandMPNN")
        mpnn_out = self.__mpnnrunner.run(mpnn_input, pack_sc=self.__MPNN_pack_sc)  # TODO: enable user control

        _df = pd.DataFrame()
        for i, s in enumerate(mpnn_out["generated_sequences"]):
            _df.at[i, "seq"] = s
        print(f"{len(_df.seq.unique())} / {len(_df)} unique sequences.")

        for seq in _df.seq.unique():
            self._report_seqs(pose.sequence(), seq)

        #########################################################################################
        ### Finding which of the MPNN-packed structures has the least clashes with the ligand ###
        #########################################################################################
        if self.__MPNN_pack_sc is True:
            print("Creating poses from MPNN-packed structures")
            poses = []
            for n in mpnn_out["packed"].keys():
                if _df.seq.duplicated(keep="first")[n] == True:  # not making poses out of duplicate sequences
                    print(f"Duplicate sequence {n}: {_df.iloc[n].seq}")
                    continue
                poses.append(self._build_pose_from_str_and_append_stuff(pdb_str=mpnn_out["packed"][n][0],
                                                                        append_pose=None, append_pose_resnos=None, prepend_lines=remarks, ref_pose=pose))
            print(len(poses), "poses created")
        else:
            # Thread sequences to pose
            print("Threading MPNN sequences to input pose")
            poses = [self._thread_seq_to_pose(pose, seq) for seq in _df.seq.unique()]
        
        # Adding any residue labels to the new generated poses
        for i,p in enumerate(poses):
            for r in pose.residues:
                for label in pose.pdb_info().get_reslabels(r.seqpos()):
                    poses[i].pdb_info().add_reslabel(r.seqpos(), label)
        return poses

    # to utils
    def _report_seqs(self, seq1, seq2):
        str0 = "  Resno: 1  "
        str1 = " Before: "
        str2 = "Mutated: "
        str3 = "  After: "
        n = 1
        print("Sequences before and after MPNN design")
        for i, (r1, r2) in enumerate(zip(seq1, seq2)):
            if n == 81 or i == len(seq2)-1:
                n_spaces = n-5
                print(str0 + " "* n_spaces + f"{i:<3}")
                print(str1)
                print(str2)
                print(str3 + "\n")
                n = 1
                str0 = f"  Resno: {(i+1):<3}"
                str1 = " Before: "
                str2 = "Mutated: "
                str3 = "  After: "

            str1 += r1
            str3 += r2
            if r1 == r2:
                str2 += " "
            else:
                str2 += "*"
            n += 1
            
    # to utils
    def _thread_seq_to_pose(self, pose, sequence):
        pose2 = pose.clone()
        for i, r in enumerate(sequence):
            if r not in "ACDEFKRYPGLIVMHNWQST":
                continue
            if pose2.residue(i+1).name1() == r:
                continue
            mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
            mutres.set_target(i+1)
            mutres.set_res_name(fusedmpnn.restype_1to3[r])
            mutres.apply(pose2)
        return pose2



    def setup_mpnn(self, pose, design_positions, repack_positions, do_not_repack_positions):
        def _figure_out_ch_resno(pose, resno):
            ### TODO: consider making residue numbers PDB-based not pose-based
            if isinstance(resno, int):
                chain_let = pose.pdb_info().chain(resno)
            if isinstance(resno, str):
                if resno[0].isalpha():
                    chain_let = resno[0]
                    resno = int(resno[1:])
                else:
                    resno = int(resno)
                    chain_let = pose.pdb_info().chain(resno)
            resno = pose.pdb_info().number(resno)
            return f"{chain_let}{resno}"

        design_pos_list = []
        not_design_pos_list = []

        if repack_positions is not None:
            for resno in repack_positions:
                if isinstance(resno, int) and pose.residue(resno).is_ligand():
                    continue
                not_design_pos_list.append(_figure_out_ch_resno(pose, resno))

        if do_not_repack_positions is not None:
            for resno in do_not_repack_positions:
                if isinstance(resno, int) and pose.residue(resno).is_ligand():
                    continue
                not_design_pos_list.append(_figure_out_ch_resno(pose, resno))
        
        if design_positions is not None:
            for resno in design_positions:
                design_pos_list.append(_figure_out_ch_resno(pose, resno))

        if design_positions is not None and len(not_design_pos_list) == 0:
            for res in pose.residues:
                if res.is_protein() is True:
                    chain_let = pose.pdb_info().chain(res.seqpos())
                    if f"{chain_let}{resno}" in design_pos_list:
                        continue
                    not_design_pos_list.append(f"{chain_let}{res.seqpos()}")  # can this fail with multichain/symmetric poses?

        mpnn_input = self.__mpnnrunner.MPNN_Input()
        mpnn_input.fixed_residues = copy.deepcopy(not_design_pos_list)
        # self.__mpnn_input.designed_chains = list(set([x[0] for x in self.__design_pos_list]))
        # self.__mpnn_input.fixed_chains = list(set([k[0] for k in self.__not_design_pos_list if k[0] not in self.__mpnn_input.designed_chains]))

        if self.__omit_AA is not None:
            mpnn_input.omit_AA = [x for x in self.__omit_AA]
        else:
            mpnn_input.omit_AA = ["C"]

        if self.__bias_AAs is not None:
            mpnn_input.bias_AA = self.__bias_AAs
            
        if self.__bias_AAs_per_residue is not None:
            mpnn_input.bias_AA_per_residue = self.__bias_AAs_per_residue

        mpnn_input.number_of_batches = 1
        return mpnn_input


    def setup_minimizer(self):
        
        if self.__movemap is None:
            mm = pyrosetta.rosetta.core.kinematics.MoveMap()
            mm.set_chi(True)
            mm.set_bb(True)
            mm.set_jump(True)
            self.__movemap = mm.clone()

        min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
        min_mover.set_type(self.__min_type)
        min_mover.cartesian(self.__cartesian)
        min_mover.set_movemap(self.__movemap)
        min_mover.score_function(self.scorefxn())
        min_mover.nb_list(True)
        return min_mover


    def setup_packer(self):
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
        if self.__tf is None:
            self.__tf = self.setup_taskfactory()
        packer.task_factory(self.__tf)
        return packer


    def setup_taskfactory(self):
        # Now lets see how to set up task operations
        # The task factory accepts all the task operations
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        
        # These three are pretty standard
        taskops = [pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline(),
                   pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent(),
                   pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides(),
                   pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking()]

        for to in taskops:
            tf.push_back(to)
        return tf


    def setup_packer_positions(self, design_resnos=None, repack_only_resnos=None, do_not_repack_resnos=None):
        
        if self.__tf is None:
            self.__tf = self.setup_taskfactory()

        ## Design positions applies only to MPNN
        if design_resnos is None:
            self.__design_positions = []
            for res in self.__input_pose.residues:
                if res.seqpos() not in repack_only_resnos+do_not_repack_resnos:
                    self.__design_positions.append(res.seqpos())

        if repack_only_resnos is not None and len(repack_only_resnos) > 0:
            repack_only_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
            for resno in repack_only_resnos:
                repack_only_selector.append_index(resno)
            print(f"Adding RestrictToRepackingRLT for positions {repack_only_resnos} to taskfactory")
            self.__tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
                pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), repack_only_selector))

        if do_not_repack_resnos is not None and len(do_not_repack_resnos) > 0:
            do_not_repack_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
            for resno in do_not_repack_resnos:
                do_not_repack_selector.append_index(resno)
            print(f"Adding PreventRepackingRLT for positions {do_not_repack_resnos} to taskfactory")
            self.__tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
                pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), do_not_repack_selector))


    def setup_movers(self):
        setup_methods = {"repack": self.setup_packer, "min": self.setup_minimizer, "mpnn": self.setup_mpnn}
        for k in ["min", "repack"]:
            if k in [l[0] for l in self.script()]:
                self.__movers[k] = setup_methods[k]()


    def apply(self, pose):
        """
        Performs FastDesign-like protein sequence design, following
        instructions set in a design script. Uses proteinMPNN (as implemented in fused-mpnn)
        to do sequence design.
        
        Protocol/script keywords that are respected:
            min, repack, mpnn, scale:{scorefunction_name}, task_operation, keep_best
        Some keywords accept a number of values/arguments:
            min <float, minimizer tolerance>
            mpnn <float, mpnn temperature> <int num_sequences>
            scale:{scorefunction} <float, weight>
            task_operation <str, taskop name>
            keep_best <int, N best to keep>  :: best N poses will be kept based on Rosetta total_score
        
        Example protocol/script:
        scale:coordinate_constraint 1.0
        scale:fa_rep 0.150
        mpnn 0.4
        repack
        scale:fa_rep 0.200
        min 0.01
        scale:coordinate_constraint 0.5
        scale:fa_rep 0.365
        mpnn 0.2
        repack
        scale:fa_rep 0.480
        min 0.01
        scale:coordinate_constraint 0.0
        scale:fa_rep 0.659
        mpnn 0.1
        repack
        scale:fa_rep 0.750
        min 0.01
        scale:coordinate_constraint 0.0
        scale:fa_rep 1
        mpnn 0.1
        repack
        min 0.00001
        """
        start_time = time.time()

        self.__input_pose = pose.clone()

        self.setup_packer_positions(self.__design_positions, self.__repack_positions, self.__do_not_repack_positions)
        self.setup_movers()
        self.__mpnn_input = self.setup_mpnn(pose, self.__design_positions, self.__repack_positions, self.__do_not_repack_positions)

        poses = {0: {"pose": pose.clone(), "mpnn_input": self.__mpnn_input.copy()}}

        mpnn_iterations = 0

        for i, cmd in enumerate(self.script()):
            command = cmd[0]
            val = None
            if len(cmd) > 1:
                val = cmd[1:]
            print(f"######### Performing step {i}: {command}, {val} ###############")

            if command == "min":
                self.__movers[command].tolerance(*val)

            elif command == "mpnn":
                assert len(val) >= 1, "Need to provide mpnn temperature and optionally num_Seq"
                ## Expecting 1 or 2 values as `val`
                if len(val) == 1:
                    val.append(self.__num_sequences)
                poses_designed = []
                mpnn_inputs = []
                for pi in poses:
                    poses_designed += self.do_mpnn(pose=poses[pi]["pose"], mpnn_input=poses[pi]["mpnn_input"], temperature=val[0], num_sequences=val[1])
                    mpnn_inputs += [poses[pi]["mpnn_input"]]*int(val[1])
                poses = {pi: {"pose": p.clone(), "mpnn_input": mpnn_inputs[pi]} for pi, p in enumerate(poses_designed)}

                # Setting the number of MPNN sequences to 1, if more than 1 were designed in current round. Idea from GRL
                # This is ignored if the design protocol specifies the number of sequences for a given step
                if len(poses) > 1:
                    self.__num_sequences = self.__mpnn_N_seq_after_first  
                mpnn_iterations += 1

            elif command[:6] == "scale:":
                _scoreterm = command.split(":")[1]
                self.__scorefxn.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name(_scoreterm), *val)

            ## TODO: consider also keeping best based on MPNN score?
            ## or even some custom metric?
            elif command == "keep_best":
                N_keep = 1
                if val is not None:
                    N_keep = int(*val)
                if len(poses) < N_keep:
                    print(f"{i}: {command}, requested N_keep = {N_keep} is more than the number of poses ({len(poses)}). Keeping all of them.")
                    N_keep = len(poses)
                scores = {j: self.__scorefxn(p["pose"]) for j, p in poses.items()}
                print("Scored poses:")
                for k, _score in scores.items():
                    print(k, _score)
                scores_keys_sorted = sorted(scores, key=scores.get)
                _tmp_dict = copy.deepcopy(poses)
                poses = {}
                for _n in range(N_keep):
                    poses[_n] = {"pose": _tmp_dict[scores_keys_sorted[_n]]["pose"], "mpnn_input": _tmp_dict[scores_keys_sorted[_n]]["mpnn_input"]}
                _tmp_dict = None

                # Setting the number of MPNN sequences to original value if only 1 best design is kept
                # This is ignored if the design protocol specifies the number of sequences for a given step
                if N_keep == 1:
                    self.__num_sequences = self.__num_sequences_original


            ## Applying packer or minimizer
            if command in ["repack", "min"]:
                self.__movers[command].score_function(self.scorefxn())
                poses_moved = []
                for pi in poses:
                    _p = poses[pi]["pose"].clone()
                    print(command, " pose has constraints: ", _p.constraint_set().has_constraints())
                    self.__movers[command].apply(_p)

                    ## Checking how much the backbone has moved during minimization
                    if command == "min":
                        overlay_pos = pyrosetta.rosetta.utility.vector1_unsigned_long()
                        for n in range(1, poses[pi]["pose"].size()):
                            overlay_pos.append(n)
                        rmse = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(poses[pi]["pose"], _p, overlay_pos, 0)
                        if rmse > self.__minimizer_rmsd_cutoff:
                            print(f"Backbone moved more than allowed (rmsd > {self.__minimizer_rmsd_cutoff}) during minimization! rmsd = {rmse:.3f}")
                            continue
                    poses_moved.append(_p.clone())
                if len(poses_moved) == 0:
                    print(f"No poses with backbone movement rmsd < {self.__minimizer_rmsd_cutoff} remained after {command} {val}")
                    sys.exit(1)

                poses = {pi: {"pose": p.clone(), "mpnn_input": poses[pi]["mpnn_input"]} for pi, p in enumerate(poses_moved)}

            ### This is supposed to enable fixing additional residues for MPNN
            ### if they are deemed to be good based on some logic.
            ### TODO: maybe also consider movers that would set fixed residues as designable again?
            if command in ["task_operation"]:
                taskop_name = val[0]
                assert taskop_name in self.__task_operations.keys()
                for j,p in poses.items():
                    _taskop = self.__task_operations[taskop_name].copy()
                    if _taskop.target() is None or len(_taskop.target()) == 0:
                        print(f"No target for {taskop_name}: skipping...")
                        continue

                    # Fetching the original set of target residues, if the pose has none
                    if all([p["pose"].pdb_info().res_haslabel(r.seqpos(), f"{taskop_name}_target") == False for r in p["pose"].residues]):
                        for r in _taskop.target():
                            poses[j]["pose"].pdb_info().add_reslabel(res=r, label=f"{taskop_name}_target")

                    # Updating the task operation target list based on any target residues the pose has
                    old_targets = [r.seqpos() for r in p["pose"].residues if p["pose"].pdb_info().res_haslabel(r.seqpos(), f"{taskop_name}_target")]
                    if _taskop.allow_updating() is True:
                        _taskop.target(old_targets)

                    # Finding new residues based on the task operation logic
                    selection = _taskop.compute(p["pose"])

                    ## Setting any found residues as fixed for MPNN by creating a new pose-associated mpnn input object
                    if len(selection) > 0 and len([x for x in selection if x not in old_targets]) > 0:
                        try:
                            _old_fixed = [p["pose"].pdb_rsd((r[0], int(r[1:]))).seqpos() for r in p["mpnn_input"].fixed_residues]
                        except AttributeError:
                            print(p["mpnn_input"].fixed_residues)
                            print(p["pose"])
                            for r in p["mpnn_input"].fixed_residues:
                                if p["pose"].pdb_rsd( (r[0], int(r[1:])) ) is None:
                                    print(r, "pdb_rsd is None !!")
                            sys.exit(1)
                        new_fixed_res = sorted(list(set(selection + _old_fixed)))
                        print(f"{i}: TaskOperation {taskop_name}: Updated fixed residues for pose {j}: {[x for x in selection if x not in old_targets]}")
                        new_design = [x for x in self.__design_positions if x not in new_fixed_res]
                        poses[j]["mpnn_input"] = self.setup_mpnn(p["pose"], new_design, new_fixed_res, self.__do_not_repack_positions)

                        # Updating the set of target residues on the pose
                        if _taskop.allow_updating() is True:
                            new_targets = sorted(list(set(selection+old_targets)))
                            for r in new_targets:
                                poses[j]["pose"].pdb_info().add_reslabel(res=r, label=f"{taskop_name}_target")
                    else:
                        print(f"{i}: TaskOperation {taskop_name}: Did not find any additional residues to select for pose {j}")


            ## Dumping PDBs if in debug mode
            if self.__debug is True and command in ["min", "repack", "mpnn"]:
                for j, p in enumerate(poses):
                    p["pose"].dump_pdb(self.__name + f"_{command}_{i}.{j}.pdb")



        for i, p in poses.items():
            print(f"################# Scoring final pose {i} ####################")
            self.__scorefxn(p["pose"])
            for k, val in p["pose"].scores.items():
                print(f"{k:>40}: {val:>20.3f}")

        print(f"Finished running the FastMPNNdesign protocol in {(time.time() - start_time):.3f} seconds.")
        return [p["pose"].clone() for i,p in poses.items()]

