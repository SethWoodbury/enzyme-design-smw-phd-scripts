"""
Basile Wicky -- 200401

Basically Cycler.py (thanks to whoever wrote that one!), but simplified, and in TINDER (Through Inspection Never Discard Excellent pRoteins) MODE

Utility commands to cycle through a directory of PDBs one at at time, tinder-style.
good/bad the PDBs you like by using the left/right key. Left makes a symlink to a 'bad' subdirectory, Right makes a symlink to a 'good' directory

Adds cycler types to pymol:
    tinder                - Cycles through all pdbs in a given directory, loading one object at a time. Go to the next one by 'swiping' left/right

Adds cycler commands:
    set_tinder_command    - Sets command run on each tinder iteration.
"""

import logging
logger = logging.getLogger("Cycler")

from pymol import cmd,viewing
import os,re

from glob import glob
from os import path


# ashworth
# minimal general classes to support convenient "list mode" behavior in pymol (left/right arrows cycle through list)
# the 'Lite' classes use the LoadDeleteCycler instead of the EnableCycler, in order that only a single pdb from the list is loaded into memory at any given time. These 'Lite' versions are preferable for large numbers of pdbs that would exceed system memory if loaded all at once.

####################################################################################################
# relates paths to object names in a way that matches the result of cmd.load
def objname(objpath):
    return re.sub( r'(\.pdb|\.pdb.gz)$', '', path.basename(objpath))


# base class cycler for load/delete behavior (to be employed when there are too many pdbs to hold in memory all at once)
class LoadDeleteCycler(object):
    def __init__(self):
        self.auto_zoom = False
        self.onload_command = None

    def iter(self,by=1):
        loaded = cmd.get_names('objects')[0]
        choices = self.choices()
        l = len(choices)
        next_file = 0
        for i in range(l):
            if objname(choices[i]) == loaded:
                print(i)
                next_file = choices[ (i+by) % l ]
                break
        cmd.delete('all')
        if not os.path.exists(next_file):
            raise ValueError("Can not locate file: %s" % next_file)
        cmd.load(next_file)
        if self.auto_zoom:
            cmd.zoom()

        if self.onload_command:
            logging.debug("onload_command: %s", self.onload_command)
            cmd.do(self.onload_command)

        cmd.replace_wizard('message',next_file)

    def choices(self):
        raise NotImplementedError("EnableCycler.choices")

####################################################################################################
# cycler over all pdbs in directory
class PDBDirCyclerLite(LoadDeleteCycler):
    def __init__(self,target_dir='.'):
        super(PDBDirCyclerLite, self).__init__()
        self.pdbs = [ f for f in os.listdir(target_dir) if re.search('.pdb.*$',f) ]
        pdb = self.pdbs[0]
        cmd.load(pdb)
    def choices(self):
        return self.pdbs

####################################################################################################
# tinder through your proteins
def good_pdb(): # Swipe Right!
    print('What a beautiful protein!')
    current_pdb=cmd.get_object_list()[0]
    os.system(f'ln -s ../{current_pdb}.pdb good/')
    viewing.cycler.iter(1)

def bad_pdb(): # Swipe Left!
    print('Maybe another time...')
    current_pdb=cmd.get_object_list()[0]
    os.system(f'ln -s ../{current_pdb}.pdb bad/')
    viewing.cycler.iter(1)

def spawnPDBDirCycler(target_dir='.'):

    viewing.cycler = PDBDirCyclerLite(target_dir)

    current_dir=os.getcwd()
    try:
        os.makedirs(f'{current_dir}/good/')
    except:
        pass
    try:
        os.makedirs(f'{current_dir}/bad/')
    except:
        pass

    cmd.set_key('left',bad_pdb)
    cmd.set_key('right',good_pdb)


def setCyclerOnloadCommand(command_string):
    if command_string[0] == '"' or command_string[0] == "'":
        command_string = command_string[1:-1]
    if viewing.cycler:
        logging.debug("Setting cycler onload_command: %s", command_string)
        viewing.cycler.onload_command = command_string

cmd.extend( 'tinder', lambda dir='.': spawnPDBDirCycler('.') )

cmd.extend( 'set_tinder_command', setCyclerOnloadCommand )

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
