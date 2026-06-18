"""
show_termini — mark protein N- and C-termini with colored spheres.

Usage (PyMOL command line):
    show_termini              # all objects
    show_termini objectName   # specific object
    hide_termini              # remove the markers
"""

from pymol import cmd


def show_termini(selection="all", _self=cmd):
    """Show blue sphere on N-terminal N and red sphere on C-terminal O for
    each protein chain in *selection*."""
    _self.delete("_termini_*")

    n_atoms = []
    c_atoms = []

    objects = _self.get_object_list(selection)
    for obj in objects:
        chains = _self.get_chains("(%s) and (%s) and polymer.protein" % (obj, selection))
        for ch in chains:
            chain_sel = "(%s) and chain %s and polymer.protein" % (obj, ch)

            # N-terminus: first residue's N atom
            stored_resi = []
            _self.iterate(
                "%s and name N" % chain_sel,
                "stored_resi.append((model, chain, resi, resn, index))",
                space={"stored_resi": stored_resi},
            )
            if stored_resi:
                # sort by residue number to find the true N-terminal residue
                stored_resi.sort(key=lambda x: int("".join(c for c in x[2] if c.isdigit() or c == "-") or "0"))
                first = stored_resi[0]
                n_atoms.append("(model %s and index %d)" % (first[0], first[4]))

            # C-terminus: last residue's carboxylate oxygen
            # Prefer OXT (terminal oxygen) then O
            stored_resi_c = []
            _self.iterate(
                "%s and (name OXT or name O)" % chain_sel,
                "stored_resi_c.append((model, chain, resi, resn, name, index))",
                space={"stored_resi_c": stored_resi_c},
            )
            if stored_resi_c:
                stored_resi_c.sort(key=lambda x: int("".join(c for c in x[2] if c.isdigit() or c == "-") or "0"))
                last_resi = stored_resi_c[-1][2]
                # among atoms in the last residue, prefer OXT over O
                last_atoms = [a for a in stored_resi_c if a[2] == last_resi]
                oxt = [a for a in last_atoms if a[4] == "OXT"]
                picked = oxt[0] if oxt else last_atoms[-1]
                c_atoms.append("(model %s and index %d)" % (picked[0], picked[5]))

    if not n_atoms and not c_atoms:
        print("show_termini: no protein chains found in '%s'." % selection)
        return

    if n_atoms:
        n_sel = " or ".join(n_atoms)
        _self.select("_termini_N", n_sel)
        _self.show("spheres", "_termini_N")
        _self.set("sphere_scale", 1.0, "_termini_N")
        _self.color("tv_blue", "_termini_N")

    if c_atoms:
        c_sel = " or ".join(c_atoms)
        _self.select("_termini_C", c_sel)
        _self.show("spheres", "_termini_C")
        _self.set("sphere_scale", 1.0, "_termini_C")
        _self.color("tv_red", "_termini_C")

    _self.deselect()
    print("show_termini: marked %d N-termini (blue) and %d C-termini (red)." % (len(n_atoms), len(c_atoms)))


def hide_termini(_self=cmd):
    """Remove termini markers created by show_termini."""
    _self.hide("spheres", "_termini_N")
    _self.hide("spheres", "_termini_C")
    _self.delete("_termini_N")
    _self.delete("_termini_C")
    print("hide_termini: done.")


cmd.extend("show_termini", show_termini)
cmd.extend("hide_termini", hide_termini)
