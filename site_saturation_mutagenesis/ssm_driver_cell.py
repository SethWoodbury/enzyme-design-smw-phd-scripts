############################
### SETUP SSM — Site Saturation Mutagenesis ###
############################

### INPUTS ###
pdb_path   = f"{input_design_dir}zapp_i1_p1D1__lig_YYE.pdb"
output_dir = f"{WORKING_DIR}ssm_output/"
residues   = "A78-81, A132, A119-120"

### OPTIONAL SETTINGS ###
omit_aas                 = "MC"     # 1-letter codes of amino acids to omit (e.g. "MC" for Met/Cys); "" for none
seed                     = 42
sc_num_denoising_steps   = 3        # denoising steps for MPNN sidechain packing
sc_num_samples           = 16       # sidechain samples evaluated per denoising step
num_processes            = 0        # CPU threads for PyTorch (0 = auto-detect all available)
parse_these_chains_only  = ""       # restrict PDB parsing to these chains (e.g. "AB"); "" for all

### CONSTANTS ###
python     = "/home/woodbuse/.conda/envs/rfd3/bin/python"
ssm_script = f"{special_scripts_dir}site_saturation_mutagenesis/ssm.py"

### BUILD COMMAND ###
os.makedirs(output_dir, exist_ok=True)

cmd = (
    f"{python} {ssm_script}"
    f" --pdb_path {pdb_path}"
    f" --residues '{residues}'"
    f" --output_dir {output_dir}"
)

if omit_aas:
    cmd += f" --omit '{omit_aas}'"
if seed != 42:
    cmd += f" --seed {seed}"
if sc_num_denoising_steps != 3:
    cmd += f" --sc_num_denoising_steps {sc_num_denoising_steps}"
if sc_num_samples != 16:
    cmd += f" --sc_num_samples {sc_num_samples}"
if num_processes != 0:
    cmd += f" --num_processes {num_processes}"
if parse_these_chains_only:
    cmd += f" --parse_these_chains_only '{parse_these_chains_only}'"

print("### SSM Command ###")
print(cmd)
