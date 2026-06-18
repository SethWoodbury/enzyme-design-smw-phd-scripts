#!/usr/bin/env python3
"""
Script: systematic_name_shortening.py
Author: Seth Woodbury (woodbuse@uw.edu)
Date:   2026-05-27

Description
-----------
    Systematically inspect a directory (or glob) of files (typically PDBs but works
    on anything) whose basenames are joined with '_' separators, identify long
    common token-sequences shared across the set, and rename the files so those
    redundant sequences are shortened, replaced or removed.

    Three operating modes:
      1) --suggest      (default)  Just print common shared sub-strings the script
                                   thinks could safely be removed / condensed.
      2) --interactive             Walk through each candidate one-by-one and let
                                   the user keep / remove / rename it on the spot.
                                   Verification step at the end with retry/edit/exit.
      3) --rename OLD=NEW [...]    Apply a fixed set of substitutions specified on
                                   the command line, no interaction.

    Renames are computed at the token level (filenames are split on '_'), so a
    multi-token target like 'pte_kcx_hbond' is treated as ONE unit rather than
    three independent tokens.  Two operating "write" modes:
      * --out_dir DIR (default)    Copy each matched file to DIR with its new name.
      * --in_place                 Rename files where they live (DANGEROUS).

    Collisions in the renamed set are detected up-front and abort the run before
    any filesystem changes happen.  Use --dry_run to preview without writing.


Driver cell (paste into a notebook / driver script and edit)
------------------------------------------------------------
    #####################################################################
    ### RUN SYSTEMATIC NAME SHORTENING                                ###
    #####################################################################
    import shlex

    ### CONFIGURATIONS ###
    INPUT_DIR_OR_GLOB = "/net/scratch/woodbuse/organophosphatase/i4_design_260515/predesign_out/test/"
    ext               = ".pdb"        # extension to filter (when INPUT is a directory). Use "" to match everything.
    out_dir           = "./renamed/"  # output dir for copy-rename. Set to None and use in_place=True for in-place.
    in_place          = False         # True -> rename files where they sit (dangerous). False -> copy to out_dir.
    dry_run           = True          # True -> print plan but do not touch disk.

    ### MODE (pick ONE) ###
    mode              = "suggest"     # "suggest" | "interactive" | "rename"

    # Only used when mode == "suggest" / "interactive":
    threshold         = 1.0           # n-gram must appear in this fraction of files (1.0 = all of them)
    min_ngram         = 1             # smallest n-gram length (in tokens) to surface
    max_suggestions   = 50            # cap on number of common patterns to list

    # Only used when mode == "rename":
    #   OLD=NEW pairs.  Empty NEW removes the token sequence.  Multi-token OLD ok ("pte_kcx_hbond").
    rename_rules = [
        ("pte_kcx_hbond", ""),
        ("idealized",     ""),
        ("FFUFU_OOAOO",   "scaff"),
    ]

    ### NON-INTERACTIVE TOGGLES ###
    yes               = False         # skip the final "apply?" confirmation in rename / interactive mode
    collapse_inner    = False         # also collapse runs of '_' inside the result (default only strips outer)
    extra_args        = ""            # any extra raw CLI flags

    ### CONSTANTS ###
    from pathlib import Path
    _HERE = Path(__file__).resolve().parent
    script = str(_HERE / "systematic_name_shortening.py")

    ### GENERATE & PRINT COMMAND ###
    cmd_parts = ["python3", script, shlex.quote(INPUT_DIR_OR_GLOB)]
    if ext is not None:           cmd_parts += ["--ext", shlex.quote(ext)]
    if in_place:                  cmd_parts += ["--in_place"]
    elif out_dir is not None:     cmd_parts += ["--out_dir", shlex.quote(out_dir)]
    if dry_run:                   cmd_parts += ["--dry_run"]
    if yes:                       cmd_parts += ["--yes"]
    if collapse_inner:            cmd_parts += ["--collapse_inner_underscores"]

    if mode == "suggest":
        cmd_parts += ["--suggest",
                      "--threshold", str(threshold),
                      "--min_ngram", str(min_ngram),
                      "--max_suggestions", str(max_suggestions)]
    elif mode == "interactive":
        cmd_parts += ["--interactive",
                      "--threshold", str(threshold),
                      "--min_ngram", str(min_ngram),
                      "--max_suggestions", str(max_suggestions)]
    elif mode == "rename":
        cmd_parts += ["--rename"] + [shlex.quote(f"{old}={new}") for old, new in rename_rules]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if extra_args:
        cmd_parts += [extra_args]

    print(" ".join(cmd_parts), "\n")
"""

import argparse
import glob
import os
import re
import shutil
import sys
import random
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}", flush=True)


def step(msg: str) -> None:
    print(f"\n[STEP] ===== {msg} =====", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Path & tokenization helpers
# ──────────────────────────────────────────────────────────────────────────────

def split_path(path: str, ext_override: str = None) -> tuple:
    """Return (dirname, basename_no_ext, extension).

    If ext_override is given and the filename ends with it, use that as the ext.
    Otherwise fall back to splitting at the FIRST dot in the basename so that
    multi-dot extensions like '.cif.gz' are kept together.
    """
    dirname = os.path.dirname(path)
    name = os.path.basename(path)
    if ext_override:
        if name.endswith(ext_override):
            return dirname, name[:-len(ext_override)], ext_override
    if "." in name:
        i = name.index(".")
        return dirname, name[:i], name[i:]
    return dirname, name, ""


def tokenize(basename_no_ext: str) -> tuple:
    """Split a basename on '_', preserving empty tokens caused by runs of '_'."""
    return tuple(basename_no_ext.split("_"))


def detokenize(tokens) -> str:
    return "_".join(tokens)


# ──────────────────────────────────────────────────────────────────────────────
# N-gram analysis
# ──────────────────────────────────────────────────────────────────────────────

def get_all_ngrams(tokens: tuple, max_n: int) -> set:
    """All contiguous n-grams (as tuples) of length 1..max_n inside tokens."""
    out = set()
    n_max = min(max_n, len(tokens))
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            out.add(tuple(tokens[i:i + n]))
    return out


def find_common_ngrams(token_lists, threshold: float, max_n: int):
    """Return list of (ngram, count) sorted by length desc, then count desc.

    Only includes n-grams whose presence count is >= threshold * n_files.
    """
    if not token_lists:
        return []
    counter = Counter()
    for tl in token_lists:
        for ng in get_all_ngrams(tl, max_n):
            counter[ng] += 1
    n_files = len(token_lists)
    min_count = max(1, int(round(threshold * n_files)))
    common = [(ng, c) for ng, c in counter.items() if c >= min_count]
    common.sort(key=lambda x: (-len(x[0]), -x[1]))
    return common


def is_contiguous_sub(sub: tuple, sup: tuple) -> bool:
    if len(sub) > len(sup):
        return False
    for i in range(len(sup) - len(sub) + 1):
        if sup[i:i + len(sub)] == sub:
            return True
    return False


def filter_maximal(common_ngrams, min_ngram: int):
    """Keep only n-grams not contained in a longer accepted n-gram and >= min_ngram tokens."""
    maximal = []
    for ng, count in common_ngrams:
        if len(ng) < min_ngram:
            continue
        skip = False
        for m_ng, _ in maximal:
            if is_contiguous_sub(ng, m_ng):
                skip = True
                break
        if not skip:
            maximal.append((ng, count))
    return maximal


# ──────────────────────────────────────────────────────────────────────────────
# Rule application
# ──────────────────────────────────────────────────────────────────────────────

def parse_rule(rule_str: str) -> tuple:
    """Parse 'OLD=NEW' into (old_tokens, new_tokens).  NEW may be empty."""
    if "=" not in rule_str:
        raise ValueError(f"Bad --rename rule (expected OLD=NEW): {rule_str!r}")
    old, new = rule_str.split("=", 1)
    if old == "":
        raise ValueError(f"--rename rule has empty OLD: {rule_str!r}")
    old_tokens = tuple(old.split("_"))
    new_tokens = tuple(new.split("_")) if new != "" else tuple()
    return old_tokens, new_tokens


def _replace_subseq(tokens, old: tuple, new: tuple):
    """Replace non-overlapping occurrences of old (tuple) in tokens (list) with new (tuple)."""
    if not old:
        return tokens
    result = []
    i = 0
    n_old = len(old)
    n_tok = len(tokens)
    while i < n_tok:
        if i + n_old <= n_tok and tuple(tokens[i:i + n_old]) == old:
            result.extend(new)
            i += n_old
        else:
            result.append(tokens[i])
            i += 1
    return result


def apply_rules(tokens: tuple, rules) -> tuple:
    """Apply rules longest-first so 'pte_kcx_hbond' wins over 'pte'."""
    sorted_rules = sorted(rules, key=lambda r: -len(r[0]))
    out = list(tokens)
    for old, new in sorted_rules:
        out = _replace_subseq(out, old, new)
    return tuple(out)


def cleanup_name(tokens: tuple, collapse_inner: bool) -> str:
    joined = "_".join(tokens)
    if collapse_inner:
        joined = re.sub(r"_+", "_", joined)
    return joined.strip("_")


# ──────────────────────────────────────────────────────────────────────────────
# Rename planning
# ──────────────────────────────────────────────────────────────────────────────

def compute_plan(files, rules, ext_override, collapse_inner):
    """Build list of (src_path, new_basename_with_ext).  No filesystem touch."""
    plan = []
    for f in files:
        dirname, base, ext = split_path(f, ext_override)
        new_tokens = apply_rules(tokenize(base), rules)
        new_base = cleanup_name(new_tokens, collapse_inner)
        if not new_base:
            log(f"file basename collapsed to empty after rules — skipping: {f}", "WARN")
            continue
        new_name = f"{new_base}{ext}"
        plan.append((f, new_name))
    return plan


def check_collisions(plan, out_dir, in_place):
    """Return list of (new_full_path, [src_paths]) for any colliding destinations."""
    dest_to_srcs = {}
    for src, new_name in plan:
        if in_place:
            dest = os.path.join(os.path.dirname(src), new_name)
        else:
            dest = os.path.join(out_dir, new_name)
        dest_to_srcs.setdefault(dest, []).append(src)
    return [(d, s) for d, s in dest_to_srcs.items() if len(s) > 1]


def perform_rename(plan, in_place: bool, out_dir: str, dry_run: bool):
    n_done = 0
    n_skip = 0
    if not in_place:
        if not dry_run:
            os.makedirs(out_dir, exist_ok=True)
        log(f"output directory: {out_dir} (will copy with new names)")
    else:
        log("operating IN PLACE", "WARN")

    for src, new_name in plan:
        if in_place:
            dest = os.path.join(os.path.dirname(src), new_name)
            action = "rename"
        else:
            dest = os.path.join(out_dir, new_name)
            action = "copy"

        if src == dest:
            n_skip += 1
            continue

        if dry_run:
            log(f"[DRY] {action}: {src}  ->  {dest}")
            n_done += 1
            continue

        if os.path.exists(dest):
            log(f"destination already exists, skipping: {dest}", "WARN")
            n_skip += 1
            continue

        if in_place:
            os.rename(src, dest)
        else:
            shutil.copy2(src, dest)
        n_done += 1

    log(f"done.  {action}d {n_done} files, skipped {n_skip}.")


# ──────────────────────────────────────────────────────────────────────────────
# File collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_files(input_spec: str, ext: str):
    """Resolve input_spec (directory or glob) into a sorted list of files.

    - If input_spec is an existing directory, glob 'dir/*<ext>' (recursive=False).
    - Otherwise treat input_spec as a glob pattern (passed to glob.glob).
    """
    if os.path.isdir(input_spec):
        pattern = os.path.join(input_spec, f"*{ext}") if ext else os.path.join(input_spec, "*")
        log(f"input is a directory; globbing {pattern!r}")
        files = sorted(glob.glob(pattern))
    else:
        log(f"input treated as glob pattern: {input_spec!r}")
        files = sorted(glob.glob(input_spec))
    files = [f for f in files if os.path.isfile(f)]
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def print_ngram_table(ngrams, n_files):
    print()
    print(f"{'#':>4}  {'tokens':>6}  {'count':>7}  {'%':>5}  pattern")
    print("-" * 80)
    for i, (ng, count) in enumerate(ngrams, 1):
        pat = detokenize(ng)
        pct = 100.0 * count / max(1, n_files)
        print(f"{i:>4}  {len(ng):>6}  {count:>7}  {pct:>5.1f}  {pat!r}")
    print()


def print_example_renames(files, rules, ext_override, collapse_inner, n_examples=3):
    if not files:
        return
    sample = files[:1] + random.sample(files, min(n_examples - 1, max(0, len(files) - 1)))
    seen = set()
    print()
    print("EXAMPLE BEFORE -> AFTER:")
    for f in sample:
        if f in seen:
            continue
        seen.add(f)
        _, base, ext = split_path(f, ext_override)
        new_tokens = apply_rules(tokenize(base), rules)
        new_base = cleanup_name(new_tokens, collapse_inner)
        print(f"   {base}{ext}")
        print(f"-> {new_base}{ext}")
        print()


def rules_summary(rules):
    if not rules:
        return "(no rules)"
    lines = []
    for i, (old, new) in enumerate(rules, 1):
        old_s = detokenize(old)
        new_s = detokenize(new) if new else "<removed>"
        lines.append(f"  [{i}] {old_s!r:35s} -> {new_s!r}")
    return "\n".join(lines)


def rules_and_kept_summary(rules, kept):
    """Single numbered view of active rules followed by kept-as-is entries.

    Numbering is contiguous so the user can address either kind with one int.
    """
    if not rules and not kept:
        return "(no rules, no kept entries)"
    lines = []
    n_rules = len(rules)
    if rules:
        lines.append("  RULES (will be applied, longest-first):")
        for i, (old, new) in enumerate(rules, 1):
            old_s = detokenize(old)
            new_s = detokenize(new) if new else "<removed>"
            lines.append(f"    [{i}] {old_s!r:35s} -> {new_s!r}")
    else:
        lines.append("  RULES: (none)")
    if kept:
        lines.append("  KEPT (no action — pass through unchanged):")
        for j, old in enumerate(kept, n_rules + 1):
            old_s = detokenize(old)
            lines.append(f"    [{j}] {old_s!r:35s}    (kept as-is)")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1: suggest
# ──────────────────────────────────────────────────────────────────────────────

def mode_suggest(token_lists, threshold, min_ngram, max_suggestions, max_n):
    step("SUGGEST: scanning for common token n-grams")
    common = find_common_ngrams(token_lists, threshold, max_n)
    log(f"{len(common)} raw common n-grams found (length >= 1) at threshold={threshold}.")
    maximal = filter_maximal(common, min_ngram)
    log(f"{len(maximal)} maximal common n-grams after collapse / min_ngram={min_ngram} filter.")

    shown = maximal[:max_suggestions]
    print_ngram_table(shown, len(token_lists))
    if len(maximal) > max_suggestions:
        log(f"(showing first {max_suggestions} of {len(maximal)}; bump --max_suggestions to see more)")

    print("Suggested actions:")
    print("  - The longest n-grams are usually the safest to remove or condense.")
    print("  - Re-run with --interactive to step through these one at a time,")
    print("  - or with --rename OLD=NEW [...] to apply a fixed set non-interactively.")
    return maximal


# ──────────────────────────────────────────────────────────────────────────────
# MODE 2: interactive
# ──────────────────────────────────────────────────────────────────────────────

def _prompt(msg, valid=None):
    while True:
        try:
            ans = input(msg).strip()
        except EOFError:
            return None
        if valid is None or ans.lower() in valid:
            return ans.lower() if valid is not None else ans
        print(f"  invalid choice {ans!r}, expected one of {sorted(valid)}")


def _collect_rules_interactively(maximal_ngrams):
    """Walk the user through each candidate.

    Returns (rules, kept):
      rules : list of (old_tokens, new_tokens) — will be applied
      kept  : list of old_tokens — explicitly kept; surfaced in the verification
              menu so the user can promote them to rules later without restart.
      Candidates the user 's'kipped or 'q'uit past are not tracked.
    """
    rules = []
    kept = []
    print()
    print("Walking through candidates (longest first).  For each one:")
    print("  k = keep as-is  (recorded; you can still promote it in the edit menu)")
    print("  r = remove      (replace with empty)")
    print("  n = rename      (specify new text)")
    print("  s = skip        (do NOT track — vanishes from the verification view)")
    print("  q = stop walking, accept current decisions")
    print()
    for i, (ng, count) in enumerate(maximal_ngrams, 1):
        pat = detokenize(ng)
        print(f"--- candidate {i}/{len(maximal_ngrams)}: {pat!r}  ({len(ng)} tokens, {count} files)")
        choice = _prompt("    [k]eep / [r]emove / re[n]ame / [s]kip / [q]uit -> ", valid={"k", "r", "n", "s", "q"})
        if choice is None or choice == "q":
            break
        if choice == "s":
            log(f"    skipped (not tracked): {pat!r}")
            continue
        if choice == "k":
            kept.append(ng)
            log(f"    kept as-is: {pat!r}")
            continue
        if choice == "r":
            rules.append((ng, tuple()))
            log(f"    rule added: {pat!r} -> <removed>")
        elif choice == "n":
            new_text = _prompt(f"    new text for {pat!r} (use '_' to separate tokens, empty = remove): ")
            new_tokens = tuple(new_text.split("_")) if new_text else tuple()
            rules.append((ng, new_tokens))
            log(f"    rule added: {pat!r} -> {new_text!r}")
    return rules, kept


def _modify_rules_menu(rules, kept):
    """Edit rules and kept-as-is entries in one numbered list.

    Numbering: [1..len(rules)] are rules, [len(rules)+1 .. len(rules)+len(kept)] are kept.
      - '<N>'  -> edit entry N.
                 For a rule: prompt for new replacement text (empty = remove the token).
                 For a kept: prompt for new text — entry is promoted to a rule.
      - 'd<N>' -> for a rule: demote it to kept (the n-gram passes through unchanged).
                  for a kept entry: warn (already not being applied).
      - blank  -> cancel and return to the verification step.
    """
    if not rules and not kept:
        log("nothing to modify (no rules, no kept entries)", "WARN")
        return rules, kept
    print()
    print("Current decisions:")
    print(rules_and_kept_summary(rules, kept))
    n_rules = len(rules)
    n_total = n_rules + len(kept)
    idx_str = _prompt("Enter entry number to edit (or 'd<N>' to demote a rule to kept, blank to cancel): ")
    if not idx_str:
        return rules, kept

    if idx_str.startswith("d"):
        try:
            idx = int(idx_str[1:])
        except ValueError:
            log("could not parse delete index", "WARN")
            return rules, kept
        if 1 <= idx <= n_rules:
            removed_old, removed_new = rules.pop(idx - 1)
            kept.append(removed_old)
            new_s = detokenize(removed_new) if removed_new else "<removed>"
            log(f"demoted rule [{idx}] ({detokenize(removed_old)!r} -> {new_s}) to KEPT")
        elif n_rules < idx <= n_total:
            log(f"entry [{idx}] is already KEPT — nothing to demote", "WARN")
        else:
            log(f"index {idx} out of range (1..{n_total})", "WARN")
        return rules, kept

    try:
        idx = int(idx_str)
    except ValueError:
        log("could not parse index", "WARN")
        return rules, kept
    if not (1 <= idx <= n_total):
        log(f"index {idx} out of range (1..{n_total})", "WARN")
        return rules, kept

    if idx <= n_rules:
        old, _ = rules[idx - 1]
        new_text = _prompt(f"New replacement text for {detokenize(old)!r} (empty = remove the token): ")
        new_tokens = tuple(new_text.split("_")) if new_text else tuple()
        rules[idx - 1] = (old, new_tokens)
        log(f"updated rule [{idx}]: {detokenize(old)!r} -> {new_text or '<removed>'}")
    else:
        kept_i = idx - n_rules - 1
        old = kept[kept_i]
        new_text = _prompt(f"Promote kept entry {detokenize(old)!r} to a rule.  "
                           f"New text (empty = remove the token): ")
        new_tokens = tuple(new_text.split("_")) if new_text else tuple()
        kept.pop(kept_i)
        rules.append((old, new_tokens))
        log(f"promoted to rule: {detokenize(old)!r} -> {new_text or '<removed>'}")

    return rules, kept


def mode_interactive(files, token_lists, threshold, min_ngram, max_suggestions, max_n,
                     ext_override, collapse_inner):
    step("INTERACTIVE: scanning for common n-grams")
    common = find_common_ngrams(token_lists, threshold, max_n)
    maximal = filter_maximal(common, min_ngram)
    shown = maximal[:max_suggestions]
    log(f"{len(maximal)} maximal common n-grams found.  Showing first {len(shown)}.")
    print_ngram_table(shown, len(token_lists))

    while True:
        rules, kept = _collect_rules_interactively(shown)
        if not rules and not kept:
            log("no decisions were recorded.  Exiting.", "WARN")
            return None

        # verification loop
        while True:
            step("VERIFICATION")
            print("Current decisions:")
            print(rules_and_kept_summary(rules, kept))
            if rules:
                print_example_renames(files, rules, ext_override, collapse_inner, n_examples=3)
            else:
                log("no active RULES — nothing would be renamed.  "
                    "Use 'g' to promote a kept entry to a rule.", "WARN")
            choice = _prompt("[a]pply / [e]xit / [s]tart over / [g]o edit (rule or kept) -> ",
                             valid={"a", "e", "s", "g"})
            if choice == "a":
                if not rules:
                    log("cannot apply — there are no active rules.  Pick something else.", "WARN")
                    continue
                return rules
            if choice == "e":
                log("exiting without applying", "WARN")
                return None
            if choice == "s":
                log("restarting decision collection from scratch")
                break  # break inner loop, re-enter rule collection
            if choice == "g":
                rules, kept = _modify_rules_menu(rules, kept)
                # stay in verification loop


# ──────────────────────────────────────────────────────────────────────────────
# Arg parsing & main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Systematically shorten redundant filename tokens across a set of files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Directory OR glob pattern of files to inspect / rename.")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--suggest", action="store_true",
                      help="(default) Just print common shared token n-grams.")
    mode.add_argument("--interactive", action="store_true",
                      help="Walk through each candidate interactively and collect rename rules.")
    mode.add_argument("--rename", nargs="+", metavar="OLD=NEW",
                      help="Apply fixed rules.  Empty NEW removes the token.  "
                           "Multi-token OLD ok (e.g. 'pte_kcx_hbond=pte').")

    p.add_argument("--ext", default=".pdb",
                   help="Extension filter when INPUT is a directory (default '.pdb').  "
                        "Set to '' (empty) to match every file.")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="An n-gram must appear in at least this fraction of files to be 'common' "
                        "(default 1.0 = must be in every file).")
    p.add_argument("--min_ngram", type=int, default=1,
                   help="Only surface n-grams of at least this many tokens (default 1).")
    p.add_argument("--max_suggestions", type=int, default=50,
                   help="Cap on the number of common n-grams to display / iterate over.")
    p.add_argument("--max_ngram_search", type=int, default=None,
                   help="Cap on n-gram length explored.  Default = length of longest filename.")

    p.add_argument("--in_place", action="store_true",
                   help="Rename files where they live (DANGEROUS).  "
                        "Default is to copy to --out_dir instead.")
    p.add_argument("--out_dir", default=None,
                   help="Output directory for copy-rename (default: <input_dir>/renamed/).")

    p.add_argument("--dry_run", action="store_true",
                   help="Show what would happen but do not touch the filesystem.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the final confirmation prompt (rename / interactive modes).")
    p.add_argument("--collapse_inner_underscores", action="store_true",
                   help="After applying rules, collapse runs of '_' inside the result to a single '_'.  "
                        "Default behaviour only strips leading / trailing '_'.")

    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for picking example basenames in previews (default 0).")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # ---- Mode resolution --------------------------------------------------
    mode = "suggest"
    if args.interactive:
        mode = "interactive"
    elif args.rename:
        mode = "rename"
    elif args.suggest:
        mode = "suggest"
    log(f"mode: {mode}")

    if args.in_place and args.out_dir:
        log("--in_place and --out_dir are mutually exclusive; ignoring --out_dir", "WARN")

    # ---- Collect files ----------------------------------------------------
    step("COLLECTING FILES")
    files = collect_files(args.input, args.ext)
    log(f"matched {len(files)} files.")
    if not files:
        log("no files matched; nothing to do.", "ERROR")
        sys.exit(1)

    # default out_dir
    inferred_dir = os.path.dirname(files[0]) or "."
    if not args.in_place and args.out_dir is None:
        args.out_dir = os.path.join(inferred_dir, "renamed")
        log(f"--out_dir not given; defaulting to {args.out_dir!r}")

    # ---- Tokenize ---------------------------------------------------------
    step("TOKENIZING BASENAMES")
    token_lists = []
    for f in files:
        _, base, _ = split_path(f, args.ext if args.ext else None)
        token_lists.append(tokenize(base))
    lens = [len(t) for t in token_lists]
    log(f"token counts per file -> min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.1f}")
    if min(lens) != max(lens):
        log("files have varying token counts; n-gram matching is content-based, not position-based.")

    max_n = args.max_ngram_search if args.max_ngram_search is not None else max(lens)

    # ---- Dispatch ---------------------------------------------------------
    rules = None
    if mode == "suggest":
        mode_suggest(token_lists, args.threshold, args.min_ngram, args.max_suggestions, max_n)
        return

    if mode == "interactive":
        rules = mode_interactive(
            files, token_lists,
            threshold=args.threshold,
            min_ngram=args.min_ngram,
            max_suggestions=args.max_suggestions,
            max_n=max_n,
            ext_override=args.ext if args.ext else None,
            collapse_inner=args.collapse_inner_underscores,
        )
        if rules is None:
            return

    if mode == "rename":
        step("PARSING --rename RULES")
        rules = [parse_rule(r) for r in args.rename]
        log(f"{len(rules)} rules parsed.")
        print(rules_summary(rules))
        print_example_renames(files, rules,
                              ext_override=args.ext if args.ext else None,
                              collapse_inner=args.collapse_inner_underscores,
                              n_examples=3)

    # ---- Build plan, check collisions, confirm, execute ------------------
    step("COMPUTING RENAME PLAN")
    plan = compute_plan(files, rules,
                        ext_override=args.ext if args.ext else None,
                        collapse_inner=args.collapse_inner_underscores)
    log(f"plan covers {len(plan)} files.")

    no_op = [src for src, new_name in plan if os.path.basename(src) == new_name]
    if no_op:
        log(f"{len(no_op)} files would not change name (no-op).")

    step("CHECKING FOR COLLISIONS")
    collisions = check_collisions(plan, args.out_dir or "", args.in_place)
    if collisions:
        log(f"{len(collisions)} destination collisions detected; aborting BEFORE filesystem touch:", "ERROR")
        for dest, srcs in collisions[:20]:
            print(f"   {dest}")
            for s in srcs:
                print(f"      <- {s}")
        if len(collisions) > 20:
            print(f"   ... and {len(collisions) - 20} more")
        sys.exit(2)
    log("no collisions.")

    if not args.yes and not args.dry_run:
        ans = _prompt(f"\nProceed with {'IN-PLACE rename' if args.in_place else 'copy-rename'} "
                      f"of {len(plan)} files? [y/N] ", valid={"y", "n", ""})
        if ans != "y":
            log("aborted by user.")
            return

    step("EXECUTING")
    perform_rename(plan, in_place=args.in_place, out_dir=args.out_dir or "", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
