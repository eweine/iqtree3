#!/usr/bin/env python3

"""
Simulate a PETracer dataset with known fixed edit rates, run IQ-TREE tree search
with those rates fixed, and compare the inferred tree to the true tree visually
and numerically.

Main outputs
------------
In ./petracer_demo/
    alignment.nex
    true_tree.nwk
    true_tree.png
    infer_fixed_rates.treefile      (from IQ-TREE)
    inferred_tree.png
    comparison.txt
    score_true.iqtree               (optional fixed-tree scoring run)

Assumptions
-----------
- State 0 is unedited.
- States 1..K are absorbing edited states.
- Edit rates are known and fixed during inference.
- IQ-TREE executable is at ./build/iqtree3 relative to the directory from which
  you run this script.
"""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Tree classes for simulation and comparison
# ---------------------------------------------------------------------

@dataclass
class SimNode:
    name: str | None
    length: float
    children: list["SimNode"]

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class ParsedNode:
    name: str | None = None
    length: float = 0.0
    children: list["ParsedNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------
# PETracer simulation
# ---------------------------------------------------------------------

def simulate_child_state(
    parent_state: int,
    branch_length: float,
    edit_rates: list[float],
    rng: random.Random,
) -> int:
    """
    If parent is already edited, remain in that edited absorbing state.
    If parent is 0, remain 0 with probability exp(-sum(rates) * t),
    otherwise jump to edited state k with probability proportional to rate_k.
    """
    if parent_state != 0:
        return parent_state

    total_rate = sum(edit_rates)
    stay_unedited = math.exp(-total_rate * branch_length)
    if rng.random() < stay_unedited:
        return 0

    draw = rng.random() * total_rate
    cumulative = 0.0
    for state_index, rate in enumerate(edit_rates, start=1):
        cumulative += rate
        if draw <= cumulative:
            return state_index

    return len(edit_rates)


def simulate_site(
    node: SimNode,
    parent_state: int,
    edit_rates: list[float],
    rng: random.Random,
    states_by_taxon: dict[str, list[int]],
) -> None:
    state = simulate_child_state(parent_state, node.length, edit_rates, rng)

    if node.is_leaf():
        assert node.name is not None
        states_by_taxon[node.name].append(state)
        return

    for child in node.children:
        simulate_site(child, state, edit_rates, rng, states_by_taxon)


# ---------------------------------------------------------------------
# Balanced tree construction
# ---------------------------------------------------------------------

def build_balanced_binary_tree(
    depth: int,
    internal_lengths: list[float],
    terminal_length: float,
    leaf_prefix: str = "cell",
) -> SimNode:
    """
    Build a balanced rooted binary tree with 2^depth leaves.

    depth=4 -> 16 leaves.
    internal_lengths must have length depth-1.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if len(internal_lengths) != max(depth - 1, 0):
        raise ValueError("internal_lengths must have length depth - 1")

    leaf_counter = 0

    def _build(level: int, incoming_length: float) -> SimNode:
        nonlocal leaf_counter

        if level == 0:
            leaf_counter += 1
            return SimNode(
                name=f"{leaf_prefix}_{leaf_counter:02d}",
                length=incoming_length,
                children=[],
            )

        if level == 1:
            return SimNode(
                name=None,
                length=incoming_length,
                children=[
                    _build(0, terminal_length),
                    _build(0, terminal_length),
                ],
            )

        idx = depth - level
        next_length = internal_lengths[idx]
        return SimNode(
            name=None,
            length=incoming_length,
            children=[
                _build(level - 1, next_length),
                _build(level - 1, next_length),
            ],
        )

    first_length = internal_lengths[0] if depth > 1 else terminal_length
    root = SimNode(
        name=None,
        length=0.0,
        children=[
            _build(depth - 1, first_length),
            _build(depth - 1, first_length),
        ],
    )
    return root


def collect_leaves_sim(node: SimNode) -> list[SimNode]:
    if node.is_leaf():
        return [node]
    out = []
    for child in node.children:
        out.extend(collect_leaves_sim(child))
    return out


def count_nodes_sim(node: SimNode) -> int:
    return 1 + sum(count_nodes_sim(child) for child in node.children)


def max_root_to_tip_distance_sim(node: SimNode, dist_so_far: float = 0.0) -> float:
    current = dist_so_far + node.length
    if node.is_leaf():
        return current
    return max(max_root_to_tip_distance_sim(child, current) for child in node.children)


# ---------------------------------------------------------------------
# Newick writing
# ---------------------------------------------------------------------

def sim_to_newick(node: SimNode) -> str:
    if node.is_leaf():
        assert node.name is not None
        return f"{node.name}:{node.length:.6f}"
    children = ",".join(sim_to_newick(child) for child in node.children)
    return f"({children}):{node.length:.6f}"


def write_tree_newick(path: Path, root: SimNode) -> None:
    rooted_tree = f"({','.join(sim_to_newick(child) for child in root.children)});"
    path.write_text(rooted_tree + "\n", encoding="ascii")


def write_alignment_nexus(
    path: Path,
    states_by_taxon: dict[str, list[int]],
    n_states: int,
) -> None:
    symbols = "".join(str(i) for i in range(n_states))
    ntax = len(states_by_taxon)
    nchar = len(next(iter(states_by_taxon.values())))

    lines = [
        "#NEXUS",
        "",
        "Begin data;",
        f"Dimensions ntax={ntax} nchar={nchar};",
        f'Format datatype=STANDARD symbols="{symbols}" gap=- missing=?;',
        "Matrix",
    ]

    for taxon, states in states_by_taxon.items():
        lines.append(f"{taxon}    {''.join(str(state) for state in states)}")

    lines.extend([";", "End;", ""])
    path.write_text("\n".join(lines), encoding="ascii")


# ---------------------------------------------------------------------
# Simple Newick parser for comparison / plotting inferred trees
# ---------------------------------------------------------------------

class NewickParser:
    def __init__(self, text: str):
        self.text = text.strip()
        self.i = 0

    def parse(self) -> ParsedNode:
        node = self._parse_subtree()
        self._skip_ws()
        if self.i < len(self.text) and self.text[self.i] == ";":
            self.i += 1
        self._skip_ws()
        if self.i != len(self.text):
            raise ValueError(f"Unexpected trailing text at position {self.i}")
        return node

    def _skip_ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def _peek(self) -> str | None:
        self._skip_ws()
        if self.i >= len(self.text):
            return None
        return self.text[self.i]

    def _consume(self, ch: str) -> None:
        self._skip_ws()
        if self.i >= len(self.text) or self.text[self.i] != ch:
            raise ValueError(f"Expected '{ch}' at position {self.i}")
        self.i += 1

    def _parse_subtree(self) -> ParsedNode:
        self._skip_ws()
        ch = self._peek()
        if ch == "(":
            return self._parse_internal()
        return self._parse_leaf()

    def _parse_internal(self) -> ParsedNode:
        self._consume("(")
        children = []
        while True:
            children.append(self._parse_subtree())
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.i += 1
                continue
            elif ch == ")":
                self.i += 1
                break
            else:
                raise ValueError(f"Expected ',' or ')' at position {self.i}")

        name = self._parse_optional_name()
        length = self._parse_optional_length()
        return ParsedNode(name=name, length=length, children=children)

    def _parse_leaf(self) -> ParsedNode:
        name = self._parse_name()
        length = self._parse_optional_length()
        return ParsedNode(name=name, length=length, children=[])

    def _parse_name(self) -> str:
        self._skip_ws()
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in [":", ",", ")", ";"]:
            self.i += 1
        name = self.text[start:self.i].strip()
        if not name:
            raise ValueError(f"Expected node name at position {start}")
        return name

    def _parse_optional_name(self) -> str | None:
        self._skip_ws()
        if self.i >= len(self.text):
            return None
        if self.text[self.i] in [":", ",", ")", ";"]:
            return None
        return self._parse_name()

    def _parse_optional_length(self) -> float:
        self._skip_ws()
        if self.i < len(self.text) and self.text[self.i] == ":":
            self.i += 1
            self._skip_ws()
            start = self.i
            while self.i < len(self.text) and self.text[self.i] not in [",", ")", ";"]:
                self.i += 1
            s = self.text[start:self.i].strip()
            try:
                return float(s)
            except ValueError:
                return 0.0
        return 0.0


def read_newick_tree(path: Path) -> ParsedNode:
    text = path.read_text(encoding="ascii").strip()
    return NewickParser(text).parse()


# ---------------------------------------------------------------------
# Tree utilities for plotting and comparison
# ---------------------------------------------------------------------

def collect_leaf_names(node: ParsedNode) -> set[str]:
    if node.is_leaf():
        assert node.name is not None
        return {node.name}
    out = set()
    for child in node.children:
        out |= collect_leaf_names(child)
    return out


def collect_leaves_parsed(node: ParsedNode) -> list[ParsedNode]:
    if node.is_leaf():
        return [node]
    out = []
    for child in node.children:
        out.extend(collect_leaves_parsed(child))
    return out


def assign_tree_coordinates_parsed(root: ParsedNode) -> dict[int, tuple[float, float]]:
    coords: dict[int, tuple[float, float]] = {}
    next_y = 0

    def _walk(node: ParsedNode, parent_x: float) -> float:
        nonlocal next_y
        x = parent_x + node.length

        if node.is_leaf():
            y = float(next_y)
            next_y += 1
            coords[id(node)] = (x, y)
            return y

        child_ys = []
        for child in node.children:
            child_ys.append(_walk(child, x))
        y = sum(child_ys) / len(child_ys)
        coords[id(node)] = (x, y)
        return y

    coords[id(root)] = (0.0, 0.0)
    child_ys = []
    for child in root.children:
        child_ys.append(_walk(child, 0.0))
    coords[id(root)] = (0.0, sum(child_ys) / len(child_ys))
    return coords


def plot_parsed_tree(root: ParsedNode, out_png: Path, title: str) -> None:
    coords = assign_tree_coordinates_parsed(root)
    leaves = collect_leaves_parsed(root)

    fig_width = 10
    fig_height = max(5, 0.45 * len(leaves))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    def _draw(node: ParsedNode) -> None:
        x_node, y_node = coords[id(node)]
        for child in node.children:
            x_child, y_child = coords[id(child)]
            ax.plot([x_node, x_node], [y_node, y_child], linewidth=1.5)
            ax.plot([x_node, x_child], [y_child, y_child], linewidth=1.5)
            _draw(child)

    _draw(root)

    for leaf in leaves:
        x, y = coords[id(leaf)]
        label = leaf.name if leaf.name is not None else "?"
        ax.text(x + 0.01, y, label, va="center", fontsize=9)

    max_x = max(x for x, _ in coords.values())
    ax.set_xlim(-0.02, max_x + 0.20)
    ax.set_ylim(-1, len(leaves))
    ax.set_xlabel("Root-to-node distance")
    ax.set_ylabel("Taxa")
    ax.set_title(title)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def sim_to_parsed(root: SimNode) -> ParsedNode:
    return read_newick_string(f"({','.join(sim_to_newick(c) for c in root.children)});")


def read_newick_string(text: str) -> ParsedNode:
    return NewickParser(text).parse()


def internal_clades_rooted(node: ParsedNode) -> set[frozenset[str]]:
    """
    Rooted internal clades as descendant leaf sets, excluding trivial leaves and full tree.
    """
    all_leaves = collect_leaf_names(node)
    clades: set[frozenset[str]] = set()

    def _walk(cur: ParsedNode) -> set[str]:
        if cur.is_leaf():
            assert cur.name is not None
            return {cur.name}

        desc = set()
        for child in cur.children:
            desc |= _walk(child)

        if 1 < len(desc) < len(all_leaves):
            clades.add(frozenset(desc))
        return desc

    _walk(node)
    return clades


def bipartitions_unrooted(node: ParsedNode) -> set[frozenset[str]]:
    """
    Unrooted bipartitions represented canonically by the smaller side.
    Trivial splits are excluded.
    """
    all_leaves = collect_leaf_names(node)
    splits: set[frozenset[str]] = set()

    def _walk(cur: ParsedNode) -> set[str]:
        if cur.is_leaf():
            assert cur.name is not None
            return {cur.name}

        desc = set()
        for child in cur.children:
            child_desc = _walk(child)
            desc |= child_desc

            if 1 < len(child_desc) < len(all_leaves) - 1:
                side = frozenset(child_desc)
                other = frozenset(all_leaves - child_desc)
                canonical = side if len(side) < len(other) else other
                if len(canonical) > 1:
                    splits.add(canonical)

        return desc

    _walk(node)
    return splits


def sister_pairs(node: ParsedNode) -> set[frozenset[str]]:
    """
    Collect leaf sister pairs for internal nodes whose two children are both leaves.
    """
    out: set[frozenset[str]] = set()

    def _walk(cur: ParsedNode) -> None:
        if cur.is_leaf():
            return
        if len(cur.children) == 2 and all(ch.is_leaf() for ch in cur.children):
            a = cur.children[0].name
            b = cur.children[1].name
            assert a is not None and b is not None
            out.add(frozenset([a, b]))
        for ch in cur.children:
            _walk(ch)

    _walk(node)
    return out


def compare_trees(true_root: ParsedNode, inferred_root: ParsedNode) -> str:
    true_leaves = collect_leaf_names(true_root)
    inf_leaves = collect_leaf_names(inferred_root)

    if true_leaves != inf_leaves:
        missing_from_inf = sorted(true_leaves - inf_leaves)
        extra_in_inf = sorted(inf_leaves - true_leaves)
        lines = [
            "Leaf-set mismatch between true and inferred trees.",
            f"Missing from inferred: {missing_from_inf}",
            f"Extra in inferred: {extra_in_inf}",
        ]
        return "\n".join(lines)

    true_clades = internal_clades_rooted(true_root)
    inf_clades = internal_clades_rooted(inferred_root)

    true_splits = bipartitions_unrooted(true_root)
    inf_splits = bipartitions_unrooted(inferred_root)

    true_pairs = sister_pairs(true_root)
    inf_pairs = sister_pairs(inferred_root)

    rooted_shared = len(true_clades & inf_clades)
    rooted_union = len(true_clades | inf_clades)
    rooted_precision = rooted_shared / len(inf_clades) if inf_clades else 1.0
    rooted_recall = rooted_shared / len(true_clades) if true_clades else 1.0

    rf_distance = len(true_splits - inf_splits) + len(inf_splits - true_splits)
    max_rf = len(true_splits) + len(inf_splits)
    normalized_rf = rf_distance / max_rf if max_rf > 0 else 0.0

    shared_pairs = len(true_pairs & inf_pairs)

    lines = [
        "Tree comparison summary",
        "=======================",
        "",
        f"Number of taxa: {len(true_leaves)}",
        "",
        "Rooted clade comparison",
        "-----------------------",
        f"True internal clades:     {len(true_clades)}",
        f"Inferred internal clades: {len(inf_clades)}",
        f"Shared internal clades:   {rooted_shared}",
        f"Rooted clade precision:   {rooted_precision:.4f}",
        f"Rooted clade recall:      {rooted_recall:.4f}",
        "",
        "Unrooted split comparison",
        "-------------------------",
        f"True nontrivial splits:     {len(true_splits)}",
        f"Inferred nontrivial splits: {len(inf_splits)}",
        f"RF distance:                {rf_distance}",
        f"Normalized RF distance:     {normalized_rf:.4f}",
        "",
        "Cherry / sister-pair comparison",
        "-------------------------------",
        f"True sister pairs:      {len(true_pairs)}",
        f"Inferred sister pairs:  {len(inf_pairs)}",
        f"Shared sister pairs:    {shared_pairs}",
        "",
        "Missed rooted clades",
        "--------------------",
    ]

    missed = sorted(
        [sorted(list(x)) for x in (true_clades - inf_clades)],
        key=lambda z: (len(z), z),
    )
    for clade in missed[:20]:
        lines.append("  " + ",".join(clade))
    if len(missed) > 20:
        lines.append(f"  ... and {len(missed) - 20} more")

    lines.extend([
        "",
        "Extra rooted clades in inferred tree",
        "------------------------------------",
    ])

    extra = sorted(
        [sorted(list(x)) for x in (inf_clades - true_clades)],
        key=lambda z: (len(z), z),
    )
    for clade in extra[:20]:
        lines.append("  " + ",".join(clade))
    if len(extra) > 20:
        lines.append(f"  ... and {len(extra) - 20} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Command running
# ---------------------------------------------------------------------

def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print()
    print("Running:")
    print("  " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


# ---------------------------------------------------------------------
# Diagnostics / summaries
# ---------------------------------------------------------------------

def summarize_simulation(
    root: SimNode,
    edit_rates: list[float],
    n_sites: int,
    terminal_length: float,
) -> str:
    leaves = collect_leaves_sim(root)
    total_rate = sum(edit_rates)
    p_edit_terminal = 1.0 - math.exp(-total_rate * terminal_length)
    expected_terminal_edits = n_sites * p_edit_terminal

    lines = [
        "Simulation summary",
        "------------------",
        f"Number of taxa: {len(leaves)}",
        f"Number of nodes: {count_nodes_sim(root)}",
        f"Number of sites: {n_sites}",
        f"Edit rates: {edit_rates}  (sum = {total_rate:.4f})",
        f"Terminal branch length: {terminal_length:.4f}",
        f"Approx. edit probability per site on a terminal branch: {p_edit_terminal:.4f}",
        f"Approx. expected edited sites per terminal branch: {expected_terminal_edits:.2f}",
        f"Max root-to-tip distance: {max_root_to_tip_distance_sim(root):.4f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    out_dir = Path(__file__).resolve().parent / "petracer_demo"
    out_dir.mkdir(exist_ok=True)

    rng = random.Random(7)

    # Fixed known PETracer rates
    edit_rates = [0.20, 0.10, 0.05]
    rates_str = ",".join(str(x) for x in edit_rates)

    # 16 taxa = 2^4 leaves
    depth = 4

    # Keep edits between cells modest.
    # With sum(edit_rates)=0.35 and terminal_length=0.08,
    # expected edited sites on a terminal branch is about:
    # n_sites * (1 - exp(-0.35 * 0.08))
    internal_lengths = [0.08, 0.08, 0.08]
    terminal_length = 0.08

    # Main lever for reconstruction difficulty.
    # Increase this if the tree is not recoverable enough for debugging.
    n_sites = 200

    # Build true tree
    root = build_balanced_binary_tree(
        depth=depth,
        internal_lengths=internal_lengths,
        terminal_length=terminal_length,
        leaf_prefix="cell",
    )

    taxa = [leaf.name for leaf in collect_leaves_sim(root)]
    taxa = [t for t in taxa if t is not None]
    states_by_taxon = {taxon: [] for taxon in taxa}

    # Simulate alignment
    for _ in range(n_sites):
        simulate_site(root, 0, edit_rates, rng, states_by_taxon)

    # Write files
    alignment_path = out_dir / "alignment.nex"
    true_tree_path = out_dir / "true_tree.nwk"
    true_tree_png = out_dir / "true_tree.png"
    inferred_tree_png = out_dir / "inferred_tree.png"
    comparison_path = out_dir / "comparison.txt"

    write_alignment_nexus(alignment_path, states_by_taxon, n_states=len(edit_rates) + 1)
    write_tree_newick(true_tree_path, root)

    true_root_parsed = read_newick_tree(true_tree_path)
    plot_parsed_tree(true_root_parsed, true_tree_png, "True simulated tree")

    summary_text = summarize_simulation(
        root=root,
        edit_rates=edit_rates,
        n_sites=n_sites,
        terminal_length=terminal_length,
    )
    print(summary_text)
    print()
    print(f"Wrote alignment:   {alignment_path}")
    print(f"Wrote true tree:   {true_tree_path}")
    print(f"Wrote tree plot:   {true_tree_png}")

    iqtree_exe = Path("/Users/eweine/Documents/iqtree3/build/iqtree3")
    if not iqtree_exe.exists():
        raise FileNotFoundError(
            f"Could not find IQ-TREE executable at {iqtree_exe.resolve()}"
        )

    # Optional: score the true tree under the true fixed rates.
    # This is a good sanity check for the PETRACER likelihood on a fixed tree.
    score_true_cmd = [
        str(iqtree_exe),
        "-s", str(alignment_path),
        "-te", str(true_tree_path),
        "-st", "MORPH",
        "-m", f"PETRACER{{{rates_str}}}",
        "-safe",
        "-redo",
        "-nt", "1",
        "-pre", str(out_dir / "score_true"),
    ]

    # Main target: reconstruct the tree with rates fixed.
    infer_cmd = [
        str(iqtree_exe),
        "-s", str(alignment_path),
        "-st", "MORPH",
        "-m", f"PETRACER{{{rates_str}}}",
        "-safe",
        "-t", "PARS",
        "--ninit", "2",
        "--nbest", "1",
        "-redo",
        "-nt", "1",
        "-pre", str(out_dir / "infer_fixed_rates"),
    ]

    print()
    print("=" * 80)
    print("Scoring the true tree under the fixed true rates")
    print("=" * 80)
    run_command(score_true_cmd)

    print()
    print("=" * 80)
    print("Reconstructing the tree with PETracer rates fixed")
    print("=" * 80)
    run_command(infer_cmd)

    inferred_tree_path = out_dir / "infer_fixed_rates.treefile"
    if not inferred_tree_path.exists():
        raise FileNotFoundError(
            f"Expected inferred tree at {inferred_tree_path}, but it was not created."
        )

    inferred_root = read_newick_tree(inferred_tree_path)
    plot_parsed_tree(inferred_root, inferred_tree_png, "IQ-TREE inferred tree")

    comparison_text = compare_trees(true_root_parsed, inferred_root)
    comparison_path.write_text(
        summary_text + "\n\n" + comparison_text + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("Tree reconstruction comparison")
    print("=" * 80)
    print(comparison_text)

    print()
    print("Key output files")
    print("----------------")
    print(f"Alignment:         {alignment_path}")
    print(f"True tree:         {true_tree_path}")
    print(f"True tree plot:    {true_tree_png}")
    print(f"Inferred tree:     {inferred_tree_path}")
    print(f"Inferred tree plot:{inferred_tree_png}")
    print(f"Comparison text:   {comparison_path}")
    print(f"True-tree score:   {out_dir / 'score_true.iqtree'}")
    print(f"Inference log:     {out_dir / 'infer_fixed_rates.iqtree'}")


if __name__ == "__main__":
    main()
