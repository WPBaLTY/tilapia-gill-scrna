"""Preranked pathway evidence with explicit annotation and identity profiles.

All ranks use tested genes and the original Wald statistics. Exact tied scores
receive deterministic nextafter perturbations in alphabetical feature order.
Matched KEGG freezes the same rank and catalogue for both annotation routes;
the recovered profile additionally requires globally unique feature/GeneID and
bidirectionally unique GeneID/Ensembl identities. Resources are explicit files,
so running an analysis cannot silently query a newer annotation database.
"""
from __future__ import annotations
from .runtime import read_table, source_member, iter_sources, source_provenance, source_size, source_exists, source_label, source_text, copy_source

from collections import defaultdict
from functools import lru_cache
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from .runtime import sha256 as sha, write_table as write_tsv

SEED = 20260823
PERMUTATIONS = 1000
MIN_SIZE = 15
MAX_SIZE = 500
GO_MIN_SIZE = 20
GO_MAX_SIZE = 400
MIN_DEPTH = 4
REDUNDANCY_JACCARD = 0.50


def split_terms(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return sorted({x.strip() for x in str(value).split(",") if re.fullmatch(r"ko\d{5}", x.strip())})

def make_total_order(frame: pd.DataFrame) -> pd.DataFrame:
    rank = (
        frame[["gene", "stat"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .drop_duplicates("gene")
        .sort_values(["stat", "gene"], ascending=[False, True], kind="mergesort")
    )
    rank["rank_score"] = rank["stat"].astype(float)
    for _, tied in rank.groupby("stat", sort=False).groups.items():
        tied = list(tied)
        if len(tied) < 2:
            continue
        value = float(rank.loc[tied[0], "stat"])
        adjusted = []
        for _ in tied:
            value = np.nextafter(value, np.inf)
            adjusted.append(value)
        rank.loc[tied, "rank_score"] = adjusted[::-1]
    return rank

def build_kegg_sets(bridge_path: Path) -> dict[str, list[str]]:
    bridge = read_table(bridge_path, sep="\t", low_memory=False)
    bridge = bridge.loc[bridge["in_emapper"].fillna(False).astype(bool)]
    sets: dict[str, set[str]] = defaultdict(set)
    for row in bridge.itertuples(index=False):
        for term in split_terms(row.KEGG_Pathway_list):
            sets[term].add(str(row.feature_symbol))
    return {
        term: sorted(genes)
        for term, genes in sets.items()
        if MIN_SIZE <= len(genes) <= MAX_SIZE
    }

def term_names(names_path: Path | None) -> dict[str, str]:
    if names_path is None or not names_path.exists():
        return {}
    names = read_table(names_path, sep="\t", usecols=["term", "term_name"]).dropna()
    return names.groupby("term")["term_name"].agg(lambda x: x.astype(str).value_counts().index[0]).to_dict()

def parse_obo(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    terms: dict[str, dict[str, object]] = {}
    alt_to_primary: dict[str, str] = {}
    current: dict[str, object] | None = None

    def commit(block: dict[str, object] | None) -> None:
        if not block or "id" not in block or block.get("is_obsolete"):
            return
        term_id = str(block["id"])
        terms[term_id] = block
        for alt in block.get("alt_id", []):
            alt_to_primary[str(alt)] = term_id

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                commit(current)
                current = {"parents": [], "alt_id": []}
                continue
            if line.startswith("["):
                commit(current)
                current = None
                continue
            if current is None or not line or line.startswith("!"):
                continue
            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:].strip()
            elif line.startswith("alt_id: "):
                current["alt_id"].append(line[8:].strip())
            elif line.startswith("is_a: "):
                current["parents"].append(line[6:].split()[0])
            elif line.startswith("relationship: part_of "):
                current["parents"].append(line.split()[2])
            elif line == "is_obsolete: true":
                current["is_obsolete"] = True
        commit(current)
    return terms, alt_to_primary

def build_go_bp_sets(bridge_path: Path, obo_path: Path) -> tuple[dict[str, list[str]], pd.DataFrame, dict[str, object]]:
    terms, alt = parse_obo(obo_path)
    root = "GO:0008150"

    @lru_cache(maxsize=None)
    def depth(term_id: str) -> int | None:
        if term_id == root:
            return 0
        term = terms.get(term_id)
        if not term or term.get("namespace") != "biological_process":
            return None
        parent_depths = [depth(str(parent)) for parent in term.get("parents", [])]
        parent_depths = [value for value in parent_depths if value is not None]
        return None if not parent_depths else min(parent_depths) + 1

    bridge = read_table(bridge_path, sep="\t", low_memory=False)
    bridge = bridge.loc[bridge["in_emapper"].fillna(False).astype(bool)].copy()
    sets: dict[str, set[str]] = defaultdict(set)
    mapped_annotations = 0
    obsolete_or_missing = 0
    for row in bridge.itertuples(index=False):
        symbol = str(row.feature_symbol)
        value = "" if pd.isna(row.GO_list) else str(row.GO_list)
        for raw_id in value.split(","):
            raw_id = raw_id.strip()
            if not re.fullmatch(r"GO:\d{7}", raw_id):
                continue
            term_id = alt.get(raw_id, raw_id)
            term = terms.get(term_id)
            if not term:
                obsolete_or_missing += 1
                continue
            if term.get("namespace") != "biological_process":
                continue
            term_depth = depth(term_id)
            if term_depth is None or term_depth < MIN_DEPTH:
                continue
            sets[term_id].add(symbol)
            mapped_annotations += 1

    records = []
    eligible: dict[str, list[str]] = {}
    for term_id, genes in sets.items():
        record = {
            "go_id": term_id,
            "term_name": str(terms[term_id].get("name", term_id)),
            "namespace": str(terms[term_id].get("namespace", "")),
            "minimum_depth": depth(term_id),
            "n_genes": len(genes),
            "eligible": GO_MIN_SIZE <= len(genes) <= GO_MAX_SIZE,
        }
        records.append(record)
        if record["eligible"]:
            eligible[term_id] = sorted(genes)
    catalog = pd.DataFrame(records).sort_values(["eligible", "n_genes", "go_id"], ascending=[False, False, True])
    metadata = {
        "ontology_terms": len(terms),
        "ontology_biological_process_terms": sum(term.get("namespace") == "biological_process" for term in terms.values()),
        "eligible_go_bp_terms": len(eligible),
        "mapped_gene_term_annotations_after_filters": mapped_annotations,
        "obsolete_or_missing_annotation_instances": obsolete_or_missing,
        "matrix_symbols_with_emapper_annotation": int(bridge["feature_symbol"].astype(str).nunique()),
    }
    return eligible, catalog, metadata

def total_order(frame: pd.DataFrame) -> pd.DataFrame:
    """GO uses the same numerical ordering with a reset integer index."""
    return make_total_order(frame).reset_index(drop=True)


def split_leading_edge(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {item for item in re.split(r"[;,]", str(value)) if item}

def reduce_redundancy(significant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import networkx as nx
    representatives: list[dict[str, object]] = []
    membership: list[dict[str, object]] = []
    group_cols = ["analysis_level", "group", "contrast", "direction"]
    for keys, frame in significant.groupby(group_cols, observed=True, sort=True):
        frame = frame.copy().reset_index(drop=True)
        graph = nx.Graph()
        graph.add_nodes_from(frame.index)
        leading = {idx: split_leading_edge(frame.loc[idx, "Lead_genes"]) for idx in frame.index}
        for i in frame.index:
            for j in frame.index:
                if j <= i:
                    continue
                union = leading[i] | leading[j]
                similarity = 0.0 if not union else len(leading[i] & leading[j]) / len(union)
                if similarity >= REDUNDANCY_JACCARD:
                    graph.add_edge(i, j, leading_edge_jaccard=similarity)
        for cluster_id, nodes in enumerate(nx.connected_components(graph), start=1):
            subset = frame.loc[sorted(nodes)].copy()
            subset = subset.sort_values(["FDR_q", "abs_NES", "Term"], ascending=[True, False, True])
            best = subset.iloc[0]
            representative_id = str(best["Term"])
            representatives.append(
                {
                    **dict(zip(group_cols, keys)),
                    "redundancy_cluster": cluster_id,
                    "representative_go_id": representative_id,
                    "representative_term": best["term_name"],
                    "NES": best["NES"],
                    "FDR_q": best["FDR_q"],
                    "leading_edge_size": len(split_leading_edge(best["Lead_genes"])),
                    "cluster_term_count": len(subset),
                    "cluster_members": " | ".join(subset["term_name"].astype(str)),
                    "representative_leading_genes": best["Lead_genes"],
                }
            )
            for _, row in subset.iterrows():
                membership.append(
                    {
                        **dict(zip(group_cols, keys)),
                        "redundancy_cluster": cluster_id,
                        "representative_go_id": representative_id,
                        "go_id": row["Term"],
                        "term_name": row["term_name"],
                        "NES": row["NES"],
                        "FDR_q": row["FDR_q"],
                    }
                )
    reps = pd.DataFrame(representatives)
    members = pd.DataFrame(membership)
    if not reps.empty:
        reps = reps.sort_values(group_cols + ["FDR_q", "abs_NES"] if "abs_NES" in reps else group_cols + ["FDR_q"])
    return reps, members

PRIORITY_TERMS = {
    "Oxidative phosphorylation",
    "Citrate cycle (TCA cycle)",
    "Tight junction",
    "Regulation of actin cytoskeleton",
    "Focal adhesion",
    "Cell cycle",
    "DNA replication",
    "Notch signaling pathway",
    "Glutathione metabolism",
    "Ferroptosis",
    "Antigen processing and presentation",
    "Adherens junction",
    "MAPK signaling pathway",
}

KEGG_DISPLAY = {
    "ko04660": "T cell receptor signalling",
    "ko04064": "NF-kappa B signalling",
    "ko04062": "Chemokine signalling",
    "ko04612": "Antigen processing and presentation",
    "ko04623": "Cytosolic DNA sensing",
    "ko04145": "Phagosome",
    "ko03050": "Proteasome",
    "ko00190": "Oxidative phosphorylation",
    "ko03030": "DNA replication",
    "ko04110": "Cell cycle",
}

GO_SELECTION = [
    ("T cell", "96_vs_24", "GO:0045582", "T cell differentiation"),
    ("T cell", "96_vs_24", "GO:0050870", "T cell activation"),
    ("T cell", "96_vs_24", "GO:0035455", "Response to interferon-alpha"),
    ("macrophage", "24_vs_0", "GO:1902622", "Neutrophil migration"),
    ("macrophage", "24_vs_0", "GO:0002526", "Acute inflammatory response"),
    ("macrophage", "96_vs_0", "GO:0006119", "Oxidative phosphorylation"),
    ("macrophage", "96_vs_24", "GO:0006260", "DNA replication"),
]

LEADING_EDGE_SELECTION = [
    ("T cell", "24_vs_0", "ko04660", "Early TCR signalling"),
    ("T cell", "24_vs_0", "ko04064", "Early NF-kappa B signalling"),
    ("T cell", "24_vs_0", "ko04612", "Early antigen presentation"),
    ("T cell", "96_vs_24", "ko04660", "Late TCR signalling"),
    ("T cell", "96_vs_24", "ko04062", "Late chemokine signalling"),
    ("T cell", "96_vs_24", "ko04612", "Late antigen presentation"),
    ("T cell", "96_vs_24", "GO:0050870", "Late T cell activation"),
    ("T cell", "96_vs_24", "GO:0035455", "Late interferon-alpha response"),
    ("macrophage", "24_vs_0", "GO:0002526", "Early acute inflammation"),
    ("macrophage", "24_vs_0", "GO:1902622", "Early neutrophil migration"),
    ("macrophage", "24_vs_0", "ko04612", "Early antigen presentation"),
    ("macrophage", "24_vs_0", "ko04145", "Early phagosome"),
    ("macrophage", "96_vs_24", "ko04612", "Late antigen presentation"),
    ("macrophage", "96_vs_24", "ko03030", "Late DNA replication"),
    ("macrophage", "96_vs_24", "ko04110", "Late cell cycle"),
    ("macrophage", "96_vs_24", "ko03050", "Late proteasome"),
]

def full_immune_results(immune_dir: Path) -> pd.DataFrame:
    frames = []
    for group_token in ["T_cell", "macrophage"]:
        for contrast in ["24_vs_0", "96_vs_0", "96_vs_24"]:
            for library_token in ["GO_BP", "KEGG_eggNOG_KO"]:
                path = immune_dir / f"broad__{group_token}__{contrast}__{library_token}__GSEA.tsv.gz"
                frames.append(read_table(path, sep="\t"))
    return pd.concat(frames, ignore_index=True)

def build_immune_tables(all_results: pd.DataFrame, de_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kegg = all_results.loc[
        all_results["library"].eq("KEGG_eggNOG_KO") & all_results["Term"].isin(KEGG_DISPLAY)
    ].copy()
    kegg["display_term"] = kegg["Term"].map(KEGG_DISPLAY)
    kegg["display_contrast"] = kegg["contrast"].map(
        {"24_vs_0": "24 h vs baseline", "96_vs_0": "96 h vs baseline", "96_vs_24": "96 h vs 24 h"}
    )
    kegg["significant_fdr05"] = kegg["FDR_q"].lt(0.05)
    kegg = kegg[
        ["group", "contrast", "display_contrast", "Term", "display_term", "NES", "FDR_q", "significant_fdr05", "Lead_genes"]
    ].sort_values(["group", "display_term", "contrast"])

    go_rows = []
    for group, contrast, term, display in GO_SELECTION:
        hit = all_results.loc[
            all_results["group"].eq(group)
            & all_results["contrast"].eq(contrast)
            & all_results["Term"].eq(term)
        ]
        if len(hit) != 1:
            raise ValueError(f"Expected one GO row for {(group, contrast, term)}, found {len(hit)}")
        row = hit.iloc[0].copy()
        row["display_term"] = display
        go_rows.append(row)
    go = pd.DataFrame(go_rows)
    go["significant_fdr05"] = go["FDR_q"].lt(0.05)
    go = go[
        ["group", "contrast", "Term", "term_name", "display_term", "NES", "FDR_q", "significant_fdr05", "Lead_genes"]
    ]

    audit_rows = []
    for group, contrast, term, display in LEADING_EDGE_SELECTION:
        hit = all_results.loc[
            all_results["group"].eq(group)
            & all_results["contrast"].eq(contrast)
            & all_results["Term"].eq(term)
        ]
        if len(hit) != 1:
            raise ValueError(f"Expected one leading-edge row for {(group, contrast, term)}, found {len(hit)}")
        row = hit.iloc[0]
        genes = [gene for gene in str(row["Lead_genes"]).split(";") if gene]
        de_path = source_member(de_dir, f"broad__{group.replace(' ', '_')}__{contrast}.tsv.gz")
        de = read_table(de_path, sep="\t").set_index("gene").reindex(genes)
        de = de.dropna(subset=["paired_log2cpm_diff_batch_1", "paired_log2cpm_diff_batch_2"])
        expected_sign = 1 if float(row["NES"]) > 0 else -1
        same = (
            de["paired_log2cpm_diff_batch_1"].mul(expected_sign).gt(0)
            & de["paired_log2cpm_diff_batch_2"].mul(expected_sign).gt(0)
        )
        audit_rows.append(
            {
                "group": group,
                "contrast": contrast,
                "Term": term,
                "display_term": display,
                "NES": float(row["NES"]),
                "FDR_q": float(row["FDR_q"]),
                "leading_edge_genes": len(de),
                "leading_edge_genes_same_direction_both_batches": int(same.sum()),
                "leading_edge_batch_direction_fraction": float(same.mean()),
                "leading_edge_gene_list": ";".join(de.index.astype(str)),
            }
        )
    audit = pd.DataFrame(audit_rows)
    return kegg, go, audit

def prerank_result(de: pd.DataFrame, gene_sets: dict[str, list[str]], names: dict[str, str], *,
                   library: str, min_size: int, max_size: int, threads: int,
                   depth_map: dict | None = None, immune: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    import gseapy as gp
    rank = total_order(de) if library == "GO:BP" else make_total_order(de)
    fit = gp.prerank(rnk=rank[["gene", "rank_score"]], gene_sets=gene_sets,
        min_size=min_size, max_size=max_size, permutation_num=PERMUTATIONS,
        weight=1.0, seed=SEED, threads=threads, no_plot=True, outdir=None, verbose=False)
    result = fit.res2d.copy()
    result["NES"] = pd.to_numeric(result["NES"], errors="coerce")
    result["FDR_q"] = pd.to_numeric(result["FDR q-val"], errors="coerce")
    result["NOM_p"] = pd.to_numeric(result["NOM p-val"], errors="coerce")
    result["term_name"] = result["Term"].astype(str).map(names).fillna(result["Term"].astype(str))
    if depth_map is not None:
        result["minimum_depth"] = result["Term"].astype(str).map(depth_map)
    for position, key in enumerate(["analysis_level", "group", "contrast"]):
        result.insert(position, key, str(de[key].iloc[0]))
    if immune:
        result.insert(0, "library", library)
        result["residual_df"] = int(de["residual_df"].iloc[0])
        result["n_ranked_genes"] = len(rank)
    result["direction"] = np.where(result["NES"] >= 0, "positive", "negative")
    if library == "GO:BP" or immune:
        result["abs_NES"] = result["NES"].abs()
    return result, rank


def select_prespecified_de_files(de_dir: Path, task: str) -> list[Path]:
    """Require the complete frozen dataset/contrast family before any calculation.

    Other broad cell types may coexist in the shared DE directory. Within each
    selected family, missing or extra contrasts/states are an input error.
    """
    contrasts = ("24_vs_0", "96_vs_0", "96_vs_24")
    broad = ("T_cell", "macrophage") if task == "immune" else ("pavement_cell", "ionocyte")
    if task not in {"kegg", "go-bp", "immune"}:
        raise ValueError(f"Unknown DE family: {task}")
    prefixes = [f"broad__{group}__" for group in broad]
    expected = {prefix + contrast + ".tsv.gz" for prefix in prefixes for contrast in contrasts}
    if task == "kegg":
        states = ("SOX4_MYB_ATP1A1_progenitor-like", "FOXI1_NDRG1_ESRRG_transport",
                  "ATP1A1_paralogue_mitochondria-rich")
        expected.update(f"ionocyte_state__{state}__{contrast}.tsv.gz"
                        for state in states for contrast in contrasts)
        prefixes.append("ionocyte_state__")
    selected = [path for prefix in prefixes for path in iter_sources(de_dir, prefix + "*.tsv.gz")]
    actual = {path.name for path in selected}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise FileNotFoundError(f"Incomplete prespecified {task} DE family; missing={missing}; unexpected={extra}")
    return selected


def run_standard_enrichment(ctx, task: str) -> dict[str, str]:
    de_dir = ctx.input("de_dir")
    selected = select_prespecified_de_files(de_dir, task)
    bridge = ctx.input("emapper_bridge")
    is_immune = task == "immune"
    libraries = ["GO:BP", "KEGG_eggNOG_KO"] if is_immune else ["GO:BP" if task == "go-bp" else "KEGG_eggNOG_KO"]
    specifications = {}
    directory = ctx.output(f"enrichment/{task}/manifest.json").parent
    for library in libraries:
        if library == "GO:BP":
            sets, catalog, metadata = build_go_bp_sets(bridge, ctx.input("go_obo"))
            names = catalog.set_index("go_id")["term_name"].to_dict()
            depth = None if is_immune else catalog.set_index("go_id")["minimum_depth"].to_dict()
            specifications[library] = (sets, names, GO_MIN_SIZE, GO_MAX_SIZE, 8, depth)
            catalog.to_csv(directory / "go_bp_gene_set_catalog.tsv.gz", sep="\t", index=False)
        else:
            specifications[library] = (build_kegg_sets(bridge), term_names(ctx.input("kegg_term_names")), MIN_SIZE, MAX_SIZE, 8 if is_immune else 4, None)
    summaries, significant, all_results, errors = [], [], [], []
    for path in selected:
        de = read_table(path, sep="\t")
        for library, (sets, names, lower, upper, threads, depth) in specifications.items():
            try:
                result, rank = prerank_result(de, sets, names, library=library, min_size=lower,
                    max_size=upper, threads=threads, depth_map=depth, immune=is_immune)
                token = path.name.removesuffix(".tsv.gz")
                suffix = library.replace(":", "_")+"__GSEA" if is_immune else ("GO_BP" if library == "GO:BP" else "KEGG_pathway")
                output = directory / f"{token}__{suffix}.tsv.gz"
                result.to_csv(output, sep="\t", index=False)
                sig = result.loc[result["FDR_q"] < .05].copy()
                significant.append(sig); all_results.append(result)
                summary_row = {"analysis_level": str(de.analysis_level.iloc[0]),
                    "group": str(de.group.iloc[0]), "contrast": str(de.contrast.iloc[0]),
                    "residual_df": int(de.residual_df.iloc[0]), "n_ranked_genes": len(rank),
                    "n_gene_sets": len(result), "n_fdr_lt_0_05": len(sig),
                    "n_fdr_lt_0_25": int(result.FDR_q.lt(.25).sum()), "output": str(output)}
                if is_immune:
                    summary_row = {"library": library, **{key: value for key, value in summary_row.items() if key != "analysis_level"}}
                    summary_row["n_gene_sets_tested"] = summary_row.pop("n_gene_sets")
                    summary_row = {key: summary_row[key] for key in ["library", "group", "contrast", "residual_df", "n_ranked_genes", "n_gene_sets_tested", "n_fdr_lt_0_05", "n_fdr_lt_0_25", "output"]}
                summaries.append(summary_row)
            except Exception as exc:
                if is_immune:
                    raise
                errors.append({"file": str(path), "library": library, "error": repr(exc)})
    sig = pd.concat(significant, ignore_index=True) if significant else pd.DataFrame()
    summary_name = "immune_gsea_summary.tsv" if is_immune else "go_bp_gsea_summary.tsv" if task == "go-bp" else "gsea_summary.tsv"
    sig_name = "immune_all_significant_gsea_fdr05.tsv.gz" if is_immune else "all_significant_go_bp_fdr05.tsv.gz" if task == "go-bp" else "all_significant_terms_fdr05.tsv.gz"
    pd.DataFrame(summaries).to_csv(directory / summary_name, sep="\t", index=False)
    sig.to_csv(directory / sig_name, sep="\t", index=False)
    if task == "kegg" and not sig.empty:
        sig.loc[sig.term_name.isin(PRIORITY_TERMS), ["analysis_level", "group", "contrast", "term_name", "NES", "FDR_q"]].sort_values(
            ["analysis_level", "group", "contrast", "term_name"]).to_csv(directory / "story_priority_terms.tsv", sep="\t", index=False)
    if task in ["go-bp", "immune"]:
        go_sig = sig.loc[sig.library.eq("GO:BP")] if is_immune and not sig.empty else sig
        representatives, members = reduce_redundancy(go_sig) if not go_sig.empty else (pd.DataFrame(), pd.DataFrame())
        representatives.to_csv(directory / ("immune_go_bp_representatives.tsv" if is_immune else "representative_go_bp_terms.tsv"), sep="\t", index=False)
        members.to_csv(directory / ("immune_go_bp_redundancy_membership.tsv.gz" if is_immune else "go_bp_redundancy_membership.tsv.gz"), sep="\t", index=False)
    if is_immune:
        kegg, go, audit = build_immune_tables(pd.concat(all_results, ignore_index=True), de_dir)
        for filename, table in [("immune_figure_selected_kegg.tsv", kegg),
                                ("immune_figure_selected_go.tsv", go), ("immune_leading_edge_direction_audit.tsv", audit)]:
            table.to_csv(directory / filename, sep="\t", index=False)
    manifest = {"method": "GSEApy preranked GSEA", "task": task,
        "gene_set_source": str(bridge), "gene_set_source_sha256": sha(bridge),
        "ranking_metric": "PyDESeq2 Wald statistic from ~ sequencing_batch + time_h",
        "permutations": PERMUTATIONS, "seed": SEED, "weight": 1.0,
        "libraries": {key: {"min_size": value[2], "max_size": value[3], "threads": value[4], "eligible_sets": len(value[0])} for key, value in specifications.items()},
        "errors": errors, "inference_scope": "n=2 biological pools per stage; residual_df=2; process support, not independent replication"}
    if "GO:BP" in libraries:
        manifest.update(ontology_file=str(ctx.input("go_obo")), ontology_sha256=sha(ctx.input("go_obo")),
            minimum_ontology_depth=MIN_DEPTH, redundancy_jaccard=REDUNDANCY_JACCARD, ontology_metadata=metadata)
    save_json(directory / "manifest.json", manifest)
    save_json(directory / ("immune_gsea_manifest.json" if is_immune else "go_bp_manifest.json" if task == "go-bp" else "gsea_manifest.json"), manifest)
    if errors:
        raise RuntimeError(f"{len(errors)} enrichment runs failed; see {directory / 'manifest.json'}")
    return {"enrichment_dir": str(directory), "summary": str(directory / summary_name)}



def now():
    return datetime.now(timezone.utc).isoformat()

def save_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        Path(path).write_text(text + "\n", encoding="utf-8")

def safe(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")

def make_rank(frame, common_genes):
    rank = frame.loc[frame.gene.isin(common_genes), ["gene", "stat"]].copy()
    rank["stat"] = pd.to_numeric(rank.stat, errors="coerce")
    rank = rank.loc[np.isfinite(rank.stat)].sort_values(["stat", "gene"], ascending=[False, True], kind="mergesort")
    assert rank.gene.is_unique
    rank["rank_score"] = rank.stat.astype(float)
    for _, group in rank.groupby("stat", sort=False).groups.items():
        ids = list(group)
        if len(ids) > 1:
            value = float(rank.loc[ids[0], "stat"])
            scores = []
            for _ in ids:
                value = np.nextafter(value, np.inf)
                scores.append(value)
            rank.loc[ids, "rank_score"] = scores[::-1]
    return rank.sort_values("rank_score", ascending=False, kind="mergesort")

def gmt(path, sets, names):
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        for key in sorted(sets):
            fh.write("\t".join([key, names.get(key, key), *sorted(sets[key])]) + "\n")

def load_matched_inputs(de_dir: Path, include_states: bool = True):
    files = select_prespecified_de_files(de_dir, "kegg" if include_states else "go-bp")
    frames = {}
    union_rows = []
    for path in files:
        name = path.name.removesuffix(".tsv.gz")
        frame = read_table(path, sep="\t")
        assert not frame.gene.duplicated().any(), path
        frames[name] = frame
        union_rows.append(frame[["gene", "baseMean"]])
    union = pd.concat(union_rows).groupby("gene", as_index=False)["baseMean"].max()
    union["symbol_upper"] = union.gene.astype(str).str.upper()
    return files, frames, union

def exact_symbol_mapping(annotation_path: Path, union: pd.DataFrame, out: Path):
    ann = read_table(annotation_path, sep="\t", dtype=str)
    ann["symbol_upper"] = ann.gene_symbol.astype(str).str.upper()
    ann = ann.loc[ann.gene_id.notna() & ann.gene_symbol.notna()].copy()
    ann["gene_id"] = ann.gene_id.str.replace(r"\.0$", "", regex=True)
    # Require unambiguous exact symbol correspondence. No best-guess assignment.
    valid = ann.loc[ann.symbol_mapping_status.eq("unique")].copy()
    counts = valid.groupby("symbol_upper").gene_id.nunique()
    valid = valid.loc[valid.symbol_upper.isin(counts[counts.eq(1)].index)]
    valid = valid.sort_values(["symbol_upper", "gene_id"]).drop_duplicates("symbol_upper")
    mapping = union.merge(valid[["symbol_upper", "gene_symbol", "gene_id"]], on="symbol_upper", how="left")
    mapping["mapping_status"] = np.where(mapping.gene_id.notna(), "exact_unique_ncbi_geneid", "unmapped_or_ambiguous_symbol")
    # One original feature per GeneID, selected without using the direction or P value.
    duplicate = mapping.loc[mapping.gene_id.notna()].sort_values(
        ["gene_id", "baseMean", "gene"], ascending=[True, False, True]
    ).duplicated("gene_id", keep="first")
    mapping.loc[duplicate.index[duplicate], "mapping_status"] = "duplicate_geneid_secondary_feature"
    write_tsv(mapping, out / "gene_union_mapping_audit.tsv")
    chosen = mapping.loc[mapping.mapping_status.eq("exact_unique_ncbi_geneid")].copy()
    assert chosen.gene_id.is_unique and chosen.gene.is_unique
    return mapping, chosen, annotation_path

def matched_eggnog_sets(path: Path):
    bridge = read_table(path, sep="\t", low_memory=False)
    bridge = bridge.loc[bridge.in_emapper.astype(str).str.lower().eq("true")]
    sets = defaultdict(set)
    for row in bridge.itertuples(index=False):
        for term in str(row.KEGG_Pathway_list).split(","):
            if re.fullmatch(r"ko\d{5}", term.strip()):
                sets["KEGG:" + term.strip()[2:]].add(str(row.feature_symbol))
    return dict(sets), set(bridge.feature_symbol.astype(str)), path

def recovered_geneid_mapping(annotation_path: Path, bridge_path: Path, matrix_features, loc_path: Path, conflicts_path: Path, union: pd.DataFrame, out: Path):
    ann = read_table(annotation_path, sep="\t", dtype=str)
    ann["symbol_upper"] = ann.gene_symbol.astype(str).str.upper()
    ann = ann.loc[ann.gene_id.notna() & ann.gene_symbol.notna()].copy()
    ann["gene_id"] = ann.gene_id.str.replace(r"\.0$", "", regex=True)
    valid = ann.loc[ann.symbol_mapping_status.eq("unique")].copy()
    counts = valid.groupby("symbol_upper").gene_id.nunique()
    valid = valid.loc[valid.symbol_upper.isin(counts[counts.eq(1)].index)]
    valid = valid.sort_values(["symbol_upper", "gene_id"]).drop_duplicates("symbol_upper")
    features = pd.DataFrame({"gene": pd.Index(matrix_features).astype(str)})
    assert features.gene.is_unique
    features["symbol_upper"] = features.gene.str.upper()
    mapping = features.merge(valid[["symbol_upper", "gene_symbol", "gene_id"]], on="symbol_upper", how="left", validate="many_to_one")
    mapping["initial_mapping_status"] = np.where(mapping.gene_id.notna(), "exact_unique_ncbi_geneid", "unmapped_or_ambiguous_symbol")
    loc = read_table(loc_path, sep="\t", dtype=str).set_index("matrix_feature")
    recovered = loc.loc[loc.status.eq("safe_candidate_numeric_gene_id_recovery")]
    selected = mapping.gene.isin(recovered.index)
    # Numeric recovery is accepted only if the independent all-feature audit
    # recorded a unique current-assembly record and no other feature claim.
    assert recovered.assembly_records.astype(int).eq(1).all()
    assert recovered.safe_bridge_candidate.str.lower().eq("true").all()
    mapping.loc[selected, "gene_id"] = mapping.loc[selected, "gene"].map(recovered.literal_gene_id)
    mapping.loc[selected, "gene_symbol"] = mapping.loc[selected, "gene"].map(recovered.current_assembly_symbol)
    mapping.loc[selected, "initial_mapping_status"] = "safe_numeric_LOC_geneid_recovery"
    mapping["mapping_status"] = mapping.initial_mapping_status
    # Collisions are removed for ALL claimants, even when a claimant is not in
    # the DE union. No highest-expression representative is chosen.
    external = read_table(conflicts_path, sep="\t", dtype=str)
    conflict_ids = set(external.literal_gene_id.dropna())
    conflict_ids |= set(mapping.loc[mapping.gene_id.notna() & mapping.gene_id.duplicated(keep=False), "gene_id"])
    external_features = set(external.matrix_feature)
    for s in external.other_feature_columns_claiming_same_gene_id.dropna():
        external_features.update(str(s).split(";"))
    collision = mapping.gene_id.isin(conflict_ids) | mapping.gene.isin(external_features)
    mapping.loc[collision, "mapping_status"] = "excluded_multifeature_geneid_collision"
    mapping["input_feature_geneid_unique"] = mapping.gene_id.notna() & ~collision
    bridge = read_table(bridge_path, sep="\t", dtype=str).fillna("")
    bridge["gene_id"] = bridge.gene_id.str.replace(r"\.0$", "", regex=True)
    idsets = bridge.groupby("feature_symbol").gene_id.agg(lambda x: sorted(set(v for v in x if v)))
    available = set(bridge.loc[bridge.in_emapper.str.lower().eq("true"), "feature_symbol"])
    mapping["eggnog_bridge_geneids"] = mapping.gene.map(idsets.map(lambda v: ";".join(v))).fillna("")
    mapping["eggnog_bridge_n_geneids"] = mapping.gene.map(idsets.map(len)).fillna(0).astype(int)
    mapping["eggnog_annotation_available"] = mapping.gene.isin(available)
    mapping["eggnog_geneid_status"] = np.select([
        ~mapping.eggnog_annotation_available,
        mapping.eggnog_bridge_n_geneids.ne(1),
        mapping.gene_id.isna(),
        mapping.gene_id.eq(mapping.eggnog_bridge_geneids),
    ], ["annotation_unavailable", "missing_or_ambiguous_eggnog_geneid", "ncbi_mapping_unresolved", "exact_geneid_agreement"], default="ncbi_eggnog_geneid_disagreement")
    mapping["pre_gprofiler_eligible"] = mapping.input_feature_geneid_unique & mapping.eggnog_geneid_status.eq("exact_geneid_agreement")
    mapping["in_tested_union"] = mapping.gene.isin(union.gene)
    write_tsv(mapping, out / "all_feature_bidirectional_geneid_audit.tsv")
    m = union.merge(mapping.drop(columns=["symbol_upper", "in_tested_union"]), on="gene", how="left", validate="one_to_one")
    write_tsv(m, out / "gene_union_mapping_audit.tsv")
    write_tsv(m.loc[m.eggnog_geneid_status.eq("ncbi_eggnog_geneid_disagreement")], out / "ncbi_eggnog_geneid_disagreements.tsv")
    summary = {
        "n_all_matrix_features": len(mapping), "n_tested_union": len(union),
        "n_safe_numeric_recoveries_all_features": int(selected.sum()),
        "n_safe_numeric_recoveries_tested_union": int(m.initial_mapping_status.eq("safe_numeric_LOC_geneid_recovery").sum()),
        "n_union_multifeature_collisions": int(m.mapping_status.eq("excluded_multifeature_geneid_collision").sum()),
        "n_union_exact_geneid_agreement": int(m.eggnog_geneid_status.eq("exact_geneid_agreement").sum()),
        "n_union_geneid_disagreement": int(m.eggnog_geneid_status.eq("ncbi_eggnog_geneid_disagreement").sum()),
        "n_union_eggnog_geneid_missing_or_ambiguous": int(m.eggnog_geneid_status.eq("missing_or_ambiguous_eggnog_geneid").sum()),
        "n_pre_gprofiler_eligible": int(m.pre_gprofiler_eligible.sum()),
        "geneid_status_counts_union": m.eggnog_geneid_status.value_counts().to_dict(),
        "changed_specification": "Added verified numeric LOC recovery; all multi-feature GeneID conflicts excluded; required native candidate GeneID to equal the eggNOG bridge GeneID. No representative feature selection.",
    }
    save_json(out / "prequery_mapping_impact.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    chosen = m.loc[m.pre_gprofiler_eligible].copy()
    assert chosen.gene_id.is_unique and chosen.gene.is_unique
    return m, chosen, annotation_path

def profile_memberships(chosen: pd.DataFrame, response: dict, out: Path):
    meta_query = response["meta"]["genes_metadata"]["query"]["query_1"]
    gp_mapping = meta_query["mapping"]
    mapping = chosen.copy()
    mapping["gprofiler_ensembl_ids"] = mapping.gene_id.map(lambda x: ";".join(gp_mapping.get(str(x), [])))
    mapping["gprofiler_n_mappings"] = mapping.gene_id.map(lambda x: len(gp_mapping.get(str(x), [])))
    mapping["gprofiler_mapping_usable"] = mapping.gprofiler_n_mappings.eq(1)
    one = mapping.loc[mapping.gprofiler_mapping_usable]
    duplicate_ensembl = set(one.gprofiler_ensembl_ids[one.gprofiler_ensembl_ids.duplicated(keep=False)])
    mapping.loc[mapping.gprofiler_ensembl_ids.isin(duplicate_ensembl), "gprofiler_mapping_usable"] = False
    write_tsv(mapping, out / "gene_union_gprofiler_mapping.tsv")
    usable = mapping.loc[mapping.gprofiler_mapping_usable]
    ensembl_to_gene = usable.set_index("gprofiler_ensembl_ids").gene.to_dict()
    ensembl_order = meta_query["ensgs"]
    sets, names, rows = {}, {}, []
    for record in response["result"]:
        if record["source"] != "KEGG":
            continue
        key = record["native"]
        assert len(record["intersections"]) == len(ensembl_order)
        genes = sorted({ensembl_to_gene[e] for e, evidence in zip(ensembl_order, record["intersections"])
                        if evidence and e in ensembl_to_gene})
        sets[key], names[key] = set(genes), record["name"]
        rows.append({"pathway": key, "term_name": record["name"],
                     "database_term_size": record["term_size"], "mapped_union_size": len(genes),
                     "mapped_symbols": ";".join(genes)})
    write_tsv(pd.DataFrame(rows).sort_values("pathway"), out / "native_kegg_union_catalog.tsv")
    return sets, names, set(usable.gene), response["meta"].get("version"), mapping

def convert_audited_memberships(chosen: pd.DataFrame, response: dict, converted: dict, profile_map: pd.DataFrame, out: Path):
    version = response["meta"].get("version")
    table=pd.DataFrame(converted["result"])
    write_tsv(table,out / "gconvert_all_input_results.tsv.gz")
    assert set(table.incoming.astype(str)) == set(chosen.gene_id), "g:Convert did not account for every query ID"
    table=table.loc[table.converted.astype(str).str.startswith("ENSONIG")].copy()
    table["incoming"]=table.incoming.astype(str)
    forward=table.groupby("incoming").converted.agg(lambda x: sorted(set(x)))
    reverse=table.groupby("converted").incoming.agg(lambda x: sorted(set(x)))
    reverse_collision=set(reverse.loc[reverse.map(len).gt(1)].index)
    m=chosen.copy()
    m["gprofiler_ensembl_ids"]=m.gene_id.map(forward.map(lambda x:";".join(x))).fillna("")
    m["gprofiler_n_mappings"]=m.gene_id.map(forward.map(len)).fillna(0).astype(int)
    m["gprofiler_reverse_collision"]=m.gene_id.map(lambda x:bool(set(forward.get(x,[])) & reverse_collision))
    m["gprofiler_mapping_usable"]=m.gprofiler_n_mappings.eq(1) & ~m.gprofiler_reverse_collision
    m["gprofiler_identity_status"]=np.select([m.gprofiler_n_mappings.eq(0),m.gprofiler_n_mappings.gt(1),m.gprofiler_reverse_collision],["unmapped","multiple_ensembl_targets","shared_ensembl_target_across_geneids"],default="bidirectionally_unique")
    write_tsv(profile_map,out / "profile_only_mapping_provisional.tsv")
    write_tsv(m,out / "gene_union_gprofiler_mapping.tsv")
    collision_rows=[]
    for ensembl in sorted(reverse_collision):
        for geneid in reverse[ensembl]:
            collision_rows.append({"ensembl_gene_id":ensembl,"ncbi_gene_id":geneid,"n_ncbi_claimants":len(reverse[ensembl]),"all_ncbi_claimants":";".join(reverse[ensembl])})
    write_tsv(pd.DataFrame(collision_rows),out / "gprofiler_reverse_collisions.tsv")
    meta=response["meta"]["genes_metadata"]["query"]["query_1"]
    usable=m.loc[m.gprofiler_mapping_usable]
    assert len(usable) > 10000, "Mapping unexpectedly sparse; stop rather than run an empty/incomplete catalog"
    assert usable.gprofiler_ensembl_ids.is_unique
    lookup=usable.set_index("gprofiler_ensembl_ids").gene.to_dict()
    missing=set(lookup)-set(meta["ensgs"])
    assert not missing, f"Profile membership response omitted {len(missing)} uniquely resolved Ensembl IDs"
    sets,names,rows={}, {}, []
    for record in response["result"]:
        if record["source"]!="KEGG":continue
        assert len(record["intersections"])==len(meta["ensgs"])
        genes=sorted({lookup[e] for e,ev in zip(meta["ensgs"],record["intersections"]) if ev and e in lookup})
        key=record["native"]
        sets[key],names[key]=set(genes),record["name"]
        rows.append({"pathway":key,"term_name":record["name"],"database_term_size":record["term_size"],"mapped_union_size":len(genes),"mapped_symbols":";".join(genes)})
    write_tsv(pd.DataFrame(rows).sort_values("pathway"),out / "native_kegg_union_catalog.tsv")
    save_json(out / "gconvert_identity_summary.json",{"profile_version":version,"convert_version":converted.get("meta",{}).get("version"),"n_candidates":len(m),"identity_status_counts":m.gprofiler_identity_status.value_counts().to_dict(),"n_reverse_collision_ensembl_ids":len(reverse_collision)})
    print("g:Convert strict identity: "+str(m.gprofiler_identity_status.value_counts().to_dict()),flush=True)
    return sets,names,set(usable.gene),version,m

def build_matched_catalog(frames, native, eggnog, names, common_genes, out: Path):
    ranks = {key: make_rank(frame, common_genes) for key, frame in frames.items()}
    coverage, set_audit, family_keys = [], [], defaultdict(list)
    for key, frame in frames.items():
        family = "broad" if frame.analysis_level.iloc[0] == "broad" else "ionocyte_state_exploratory"
        family_keys[family].append(key)
        rank = ranks[key]
        write_tsv(rank, out / "ranks" / f"{key}.tsv.gz")
        coverage.append({"dataset": key, "analysis_family": family, "analysis_level": frame.analysis_level.iloc[0],
                         "group": frame.group.iloc[0], "contrast": frame.contrast.iloc[0],
                         "original_tested_genes": len(frame), "finite_wald_genes": int(np.isfinite(frame.stat).sum()),
                         "common_rank_genes": len(rank), "common_rank_fraction": len(rank)/len(frame)})
        rgenes = set(rank.gene)
        for pathway in sorted(set(native) | set(eggnog)):
            nset, eset = native.get(pathway, set()) & rgenes, eggnog.get(pathway, set()) & rgenes
            set_audit.append({"dataset": key, "analysis_family": family, "pathway": pathway,
                              "term_name": names.get(pathway, pathway), "native_overlap": len(nset),
                              "eggnog_overlap": len(eset), "intersection": len(nset & eset),
                              "jaccard": len(nset & eset)/len(nset | eset) if (nset | eset) else np.nan,
                              "eligible_both_here": MIN_SIZE <= len(nset) <= MAX_SIZE and MIN_SIZE <= len(eset) <= MAX_SIZE})
    coverage = pd.DataFrame(coverage)
    set_audit = pd.DataFrame(set_audit)
    # Fixed matched catalogue within the six broad comparisons; a separate fixed
    # catalogue for the optional nine low-cell-count exploratory state comparisons.
    catalogs = {}
    for family, keys in family_keys.items():
        part = set_audit.loc[set_audit.analysis_family.eq(family)]
        eligible = part.groupby("pathway").eligible_both_here.agg(["all", "size"])
        catalogs[family] = sorted(eligible.loc[eligible["all"] & eligible["size"].eq(len(keys))].index)
    set_audit["in_family_matched_catalog"] = [r.pathway in catalogs[r.analysis_family] for r in set_audit.itertuples()]
    write_tsv(coverage, out / "rank_coverage.tsv")
    write_tsv(set_audit, out / "gene_set_overlap_audit.tsv.gz")
    save_json(out / "matched_catalogs.json", catalogs)
    return ranks, coverage, catalogs


def run_matched_preranks(frames, ranks, coverage, catalogs, native, eggnog, names, out: Path, *, threads: int = 2):
    import gseapy as gp
    status = {"status": "running", "started_utc": now(), "completed_runs": 0}
    results = []
    for key, frame in frames.items():
        family = coverage.set_index("dataset").loc[key, "analysis_family"]
        rank = ranks[key]
        rgenes = set(rank.gene)
        for route, source_sets in [("gprofiler", native), ("eggnog", eggnog)]:
            sets = {term: source_sets[term] & rgenes for term in catalogs[family]}
            assert all(MIN_SIZE <= len(x) <= MAX_SIZE for x in sets.values())
            gmt(out / "gene_sets" / f"{key}__{route}.gmt", sets, names)
            print(f"RUN {key} {route}: {len(rank)} genes; {len(sets)} sets; permutations={PERMUTATIONS}", flush=True)
            run = gp.prerank(rnk=rank[["gene", "rank_score"]], gene_sets={k: sorted(v) for k,v in sets.items()},
                             min_size=MIN_SIZE, max_size=MAX_SIZE, permutation_num=PERMUTATIONS,
                             weight=1.0, seed=SEED, threads=threads, no_plot=True, outdir=None, verbose=False)
            result = run.res2d.copy()
            assert set(result.Term) == set(sets), "GSEA unexpectedly changed matched catalogue"
            for c in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
                result[c] = pd.to_numeric(result[c], errors="raise")
            result.insert(0, "dataset", key)
            result.insert(1, "analysis_family", family)
            result.insert(2, "analysis_level", frame.analysis_level.iloc[0])
            result.insert(3, "group", frame.group.iloc[0])
            result.insert(4, "contrast", frame.contrast.iloc[0])
            result.insert(5, "route", route)
            result["term_name"] = result.Term.map(names)
            result["gene_set_size"] = result.Term.map(lambda x: len(sets[x]))
            result["leading_edge_count"] = result.Lead_genes.map(lambda x: len(str(x).split(";")) if pd.notna(x) and str(x) else 0)
            write_tsv(result, out / "results" / f"{key}__{route}.tsv.gz")
            edge_rows = []
            lookup = frame.set_index("gene")
            for row in result.itertuples(index=False):
                for gene in str(row.Lead_genes).split(";"):
                    if gene and gene != "nan":
                        entry = {"dataset": key, "route": route, "pathway": row.Term, "gene": gene}
                        for col in ["stat", "log2FoldChange_shrunken", "padj", "paired_log2cpm_diff_batch_1", "paired_log2cpm_diff_batch_2", "paired_direction_matches_model"]:
                            entry[col] = lookup.loc[gene, col]
                        edge_rows.append(entry)
            write_tsv(pd.DataFrame(edge_rows), out / "leading_edges" / f"{key}__{route}.tsv.gz")
            results.append(result)
            status.update(completed_runs=len(results), last_completed=f"{key}__{route}", updated_utc=now())
            save_json(out / "status.json", status)
    all_results = pd.concat(results, ignore_index=True)
    write_tsv(all_results, out / "all_matched_results.tsv.gz")
    return all_results


def compare_routes(all_results: pd.DataFrame, out: Path) -> pd.DataFrame:
    ids = ["dataset", "analysis_family", "analysis_level", "group", "contrast", "Term", "term_name"]
    fields = ["NES", "NOM p-val", "FDR q-val", "gene_set_size", "leading_edge_count", "Lead_genes"]
    comparison = all_results.loc[all_results.route.eq("gprofiler"), ids+fields].merge(
        all_results.loc[all_results.route.eq("eggnog"), ids+fields], on=ids, suffixes=("_gprofiler", "_eggnog"), validate="one_to_one")
    comparison["same_NES_direction"] = np.sign(comparison.NES_gprofiler).eq(np.sign(comparison.NES_eggnog))
    comparison["dual_same_direction_fdr05"] = comparison.same_NES_direction & comparison["FDR q-val_gprofiler"].lt(.05) & comparison["FDR q-val_eggnog"].lt(.05)
    comparison["leading_edge_jaccard"] = [len(set(str(a).split(";")) & set(str(b).split(";")))/len(set(str(a).split(";")) | set(str(b).split(";"))) for a,b in zip(comparison.Lead_genes_gprofiler, comparison.Lead_genes_eggnog)]
    write_tsv(comparison, out / "matched_route_comparison.tsv.gz")
    return comparison


def validate_recovered_results(out: Path, de_dir: Path):
    def write(frame, filename):
        write_tsv(frame, out / filename)
    allres = read_table(out / "all_matched_results.tsv.gz", sep="\t")
    paired = read_table(out / "matched_route_comparison.tsv.gz", sep="\t")
    catalogs = json.loads((out / "matched_catalogs.json").read_text(encoding="utf-8"))
    current_map = read_table(out / "common_gene_universe_audit.tsv", sep="\t", dtype={"gene_id":str,"eggnog_bridge_geneids":str})
    gpmap = read_table(out / "gene_union_gprofiler_mapping.tsv", sep="\t")
    if "gprofiler_identity_status" not in current_map:
        current_map = current_map.merge(gpmap[["gene","gprofiler_ensembl_ids","gprofiler_reverse_collision","gprofiler_identity_status"]], on="gene",how="left",validate="one_to_one")
        write(current_map,"common_gene_universe_audit.tsv")
    domain = current_map.loc[current_map.common_mapping_eligible].copy()
    assert domain.gene.is_unique and domain.gene_id.is_unique
    assert domain.gene_id.eq(domain.eggnog_bridge_geneids).all()
    assert domain.input_feature_geneid_unique.all()
    assert domain.gprofiler_mapping_usable.all()
    assert domain.gprofiler_identity_status.eq("bidirectionally_unique").all()
    assert domain.gprofiler_ensembl_ids.is_unique
    checks = []
    for dataset, results in allres.groupby("dataset"):
        family = results.analysis_family.iloc[0]
        native = results.loc[results.route.eq("gprofiler")]
        egg = results.loc[results.route.eq("eggnog")]
        rank = read_table(out / "ranks" / f"{dataset}.tsv.gz", sep="\t")
        assert set(native.Term) == set(egg.Term) == set(catalogs[family])
        assert native.Term.is_unique and egg.Term.is_unique
        assert results.gene_set_size.between(15,500).all()
        assert np.isfinite(results.NES).all() and results["FDR q-val"].between(0,1).all()
        assert rank.gene.is_unique and np.isfinite(rank.rank_score).all() and rank.rank_score.is_monotonic_decreasing
        assert set(rank.gene) <= set(domain.gene)
        for route in ["gprofiler", "eggnog"]:
            with (out / "gene_sets" / f"{dataset}__{route}.gmt").open(encoding="utf-8") as f:
                sets = {p[0]:set(p[2:]) for line in f if (p := line.rstrip("\n").split("\t"))}
            assert set(sets) == set(catalogs[family])
            res = results.loc[results.route.eq(route)].set_index("Term")
            for key,s in sets.items():
                assert 15 <= len(s) <= 500 and s <= set(rank.gene)
                assert set(str(res.loc[key,"Lead_genes"]).split(";")) <= s
        checks.append({"dataset":dataset,"rank_size":len(rank),"n_matched_pathways":len(native),"passed":True})
    assert len(checks)==15 and len(allres)==2*len(paired)
    write(pd.DataFrame(checks), "validation_checks.tsv")
    # This is an explicitly predeclared pathway panel, not selection on new q.
    wanted={"KEGG:00190","KEGG:00020","KEGG:00071","KEGG:04330","KEGG:04530","KEGG:00100"}
    edges=[]
    edge_summ=[]
    for row in paired.loc[paired.Term.isin(wanted)].itertuples(index=False):
        shared=set(str(row.Lead_genes_gprofiler).split(";")) & set(str(row.Lead_genes_eggnog).split(";"))
        de=read_table(source_member(de_dir, f"{row.dataset}.tsv.gz"),sep="\t").set_index("gene")
        direction=np.sign(row.NES_gprofiler)
        nmatch=0
        for gene in sorted(shared):
            d=de.loc[gene]
            match=bool(np.sign(d.paired_log2cpm_diff_batch_1)==direction and np.sign(d.paired_log2cpm_diff_batch_2)==direction)
            nmatch+=match
            edges.append({"dataset":row.dataset,"analysis_level":row.analysis_level,"group":row.group,"contrast":row.contrast,"pathway":row.Term,"gene":gene,"wald_stat":d.stat,"shrunken_log2FC":d.log2FoldChange_shrunken,"gene_FDR":d.padj,"batch1_log2CPM_difference":d.paired_log2cpm_diff_batch_1,"batch2_log2CPM_difference":d.paired_log2cpm_diff_batch_2,"both_batches_match_NES_direction":match})
        edge_summ.append({"dataset":row.dataset,"pathway":row.Term,"shared_leading_edge_count":len(shared),"both_batches_match_count":nmatch})
    write(pd.DataFrame(edges),"headline_shared_leading_edges.tsv")
    write(pd.DataFrame(edge_summ),"headline_shared_leading_edge_summary.tsv")
    broad = paired.loc[paired.analysis_family.eq("broad")]
    state = paired.loc[paired.analysis_family.ne("broad")]
    summary={
      "all_validation_checks_passed":True,"n_runs":30,"n_results":len(allres),"n_paired_pathway_comparisons":len(paired),
      "domain_recovered":len(domain),
      "broad":{"paired":len(broad),"same_direction":int(broad.same_NES_direction.sum()),"dual_support":int(broad.dual_same_direction_fdr05.sum())},
      "state":{"paired":len(state),"same_direction":int(state.same_NES_direction.sum()),"dual_support":int(state.dual_same_direction_fdr05.sum())},
      "gprofiler_database_version":json.loads((out/"manifest.json").read_text(encoding="utf-8"))["gprofiler_database_version"],
    }
    (out/"validation_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    manifest["gconvert_identity_summary"]=json.loads((out/"gconvert_identity_summary.json").read_text(encoding="utf-8"))
    manifest["final_validation_summary"]=summary
    manifest["status"]="complete"
    manifest["is_final_strict_convert_audited_run"]=True
    manifest["superseded_intermediate_directory"]="interim_profile_only_do_not_use"
    manifest["gconvert_numeric_namespace"]="ENTREZGENE_ACC"
    manifest["gconvert_response_sha256"]=hashlib.sha256((out/"gconvert_entrezgene_acc_response.json.gz").read_bytes()).hexdigest()
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    status=json.loads((out/"status.json").read_text(encoding="utf-8"))
    status.update(status="complete", final_domain=len(domain), strict_bidirectional_convert_audit=True, independent_validation_passed=True)
    (out/"status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    hashes=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in {"output_manifest.tsv","status.json"}:
            hashes.append({"file":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    write(pd.DataFrame(hashes),"output_manifest.tsv")
    return summary


def load_json(path: Path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frozen_profile_query(chosen: pd.DataFrame) -> dict:
    return {"organism": "oniloticus", "query": sorted(chosen.gene_id.tolist()),
        "sources": ["KEGG"], "all_results": True, "no_evidences": False,
        "ordered": False, "combined": False, "numeric_ns": "ENTREZGENE",
        "user_threshold": 1.0, "domain_scope": "annotated", "significance_threshold_method": "g_SCS"}


def run_matched_enrichment(ctx, *, recovered: bool) -> dict[str, str]:
    """Replay frozen memberships with the exact original or recovered domain."""
    task = "recovered-kegg" if recovered else "matched-kegg"
    ctx.require(INPUTS_BY_TASK[task])
    start = time.time()
    out = ctx.output("enrichment/"+task+"/manifest.json").parent
    for subdir in ["ranks", "gene_sets", "results", "leading_edges"]:
        (out / subdir).mkdir(exist_ok=True)
    files, frames, union = load_matched_inputs(ctx.input("de_dir"), include_states=True)
    annotation_path, bridge_path = ctx.input("gene_annotation"), ctx.input("emapper_bridge")
    if recovered:
        import h5py
        from anndata.io import read_elem
        with h5py.File(ctx.input("atlas_h5ad"), "r") as handle:
            matrix_features = read_elem(handle["var"]).index
        mapping_audit, chosen, _ = recovered_geneid_mapping(annotation_path, bridge_path, matrix_features,
            ctx.input("loc_geneid_audit"), ctx.input("loc_geneid_conflicts"), union, out)
    else:
        mapping_audit, chosen, _ = exact_symbol_mapping(annotation_path, union, out)
    query = frozen_profile_query(chosen)
    if load_json(ctx.input("gprofiler_profile_query")) != query:
        raise ValueError("Frozen g:Profiler query does not match the selected mapping profile")
    response = load_json(ctx.input("gprofiler_profile_response"))
    native, names, native_mapped, version, gp_map = profile_memberships(chosen, response, out)
    save_json(out / "gprofiler_union_query.json", query)
    save_json(out / "gprofiler_union_response.json.gz", response)
    if recovered:
        convert_query = {"organism": "oniloticus", "query": sorted(chosen.gene_id.tolist()),
                         "target": "ENSG", "numeric_ns": "ENTREZGENE_ACC"}
        if load_json(ctx.input("gconvert_query")) != convert_query:
            raise ValueError("Frozen g:Convert query does not account for the exact candidate GeneIDs")
        converted = load_json(ctx.input("gconvert_response"))
        native, names, native_mapped, version, gp_map = convert_audited_memberships(chosen, response, converted, gp_map, out)
        save_json(out / "gconvert_entrezgene_acc_query.json", convert_query)
        save_json(out / "gconvert_entrezgene_acc_response.json.gz", converted)
    eggnog, eggnog_mapped, _ = matched_eggnog_sets(bridge_path)
    common_genes = native_mapped & eggnog_mapped
    union_audit = mapping_audit.merge(gp_map[["gene", "gprofiler_n_mappings", "gprofiler_mapping_usable"]], on="gene", how="left")
    union_audit["eggnog_annotation_available"] = union_audit.gene.isin(eggnog_mapped)
    union_audit["common_mapping_eligible"] = union_audit.gene.isin(common_genes)
    write_tsv(union_audit, out / "common_gene_universe_audit.tsv")
    ranks, coverage, catalogs = build_matched_catalog(frames, native, eggnog, names, common_genes, out)
    results = run_matched_preranks(frames, ranks, coverage, catalogs, native, eggnog, names, out, threads=2)
    comparison = compare_routes(results, out)
    summary = comparison.groupby(["analysis_family", "group", "contrast"], as_index=False).agg(
        n_pathways=("Term", "size"), n_same_direction=("same_NES_direction", "sum"), n_dual_support=("dual_same_direction_fdr05", "sum"))
    write_tsv(summary, out / "matched_summary.tsv")
    manifest = {"status": "awaiting_identity_validation" if recovered else "complete", "profile": task, "finished_utc": now(),
        "elapsed_seconds": time.time()-start, "gprofiler_database_version": version,
        "source_files": [{"path": str(p), "sha256": sha(p)} for p in files+[annotation_path, bridge_path,
            ctx.input("gprofiler_profile_query"), ctx.input("gprofiler_profile_response")]],
        "n_union_features": len(union), "n_unique_ncbi_mapped_features": len(chosen),
        "n_gprofiler_unambiguously_mapped": len(native_mapped), "n_common_mapping_eligible": len(common_genes),
        "catalogue_counts": {key: len(value) for key, value in catalogs.items()},
        "min_size": MIN_SIZE, "max_size": MAX_SIZE, "permutations": PERMUTATIONS, "seed": SEED,
        "weight": 1.0, "threads": 2,
        "rank_universe": "Finite-Wald tested genes intersecting the selected unambiguous mapping profile and eggNOG availability; KEGG membership is not required; identical ordered ranks for both routes.",
        "catalogue_rule": "All shared pathway IDs with 15-500 rank-overlap genes in BOTH routes in EVERY contrast within the analysis family; separate six-broad and nine-state catalogues; no result selection.",
        "tie_rule": "Alphabetical gene order with deterministic nextafter perturbations of exactly tied Wald statistics.",
        "multiplicity": "Empirical GSEA FDR separately per route/run over an identical eligible catalogue; not BH of nominal P values.",
        "biological_limitations": "n=2 independent pools per stage; residual_df=2; annotation sensitivity is not independent biological validation.",
        "mapping_specification_changed_from_original_matched_run": recovered}
    if recovered:
        manifest["rank_universe"] = "Finite-Wald tested features with exact-unique symbol or verified numeric LOC NCBI mapping, no multi-feature GeneID conflict anywhere in the input matrix, identical exact GeneID in the available eggNOG bridge, and bidirectionally unique NCBI-to-Ensembl mapping audited for ALL query IDs via g:Convert; same ordered ranks for both routes. KEGG membership is not required."
        manifest["extra_mapping_input_files"] = [{"path": str(ctx.input(key)), "sha256": sha(ctx.input(key))} for key in ["loc_geneid_audit", "loc_geneid_conflicts", "gconvert_query", "gconvert_response"]]
        manifest["prequery_mapping_impact"] = load_json(out / "prequery_mapping_impact.json")
    save_json(out / "manifest.json", manifest)
    save_json(out / "status.json", {"status": "awaiting_identity_validation" if recovered else "complete", "finished_utc": now(), "completed_runs": 30})
    if recovered:
        validate_recovered_results(out, ctx.input("de_dir"))
    else:
        write_tsv(pd.DataFrame([{"file": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"output_manifest.tsv", "status.json"}]), out / "output_manifest.tsv")
    return {"matched_dir": str(out), "comparison": str(out / "matched_route_comparison.tsv.gz")}


INPUTS_BY_TASK = {'go-bp': ['de_dir', 'emapper_bridge', 'go_obo'], 'immune': ['de_dir', 'emapper_bridge', 'go_obo', 'kegg_term_names'], 'recovered-kegg': ['de_dir', 'gene_annotation', 'emapper_bridge', 'gprofiler_profile_query', 'gprofiler_profile_response', 'atlas_h5ad', 'loc_geneid_audit', 'loc_geneid_conflicts', 'gconvert_query', 'gconvert_response']}


def run(ctx, *, task="kegg") -> dict[str, str]:
    if task not in INPUTS_BY_TASK:
        raise ValueError(f"Unknown enrichment task: {task}")
    ctx.require(INPUTS_BY_TASK[task])
    if task in ["kegg", "go-bp", "immune"]:
        return run_standard_enrichment(ctx, task)
    if task == "immune-review":
        return run_immune_review(ctx)
    if task in ["matched-kegg", "recovered-kegg"]:
        return run_matched_enrichment(ctx, recovered=task == "recovered-kegg")
    if task == "immune-tables":
        tables = build_immune_tables(full_immune_results(ctx.input("immune_results_dir")), ctx.input("de_dir"))
        outputs = {}
        for filename, table in zip(["immune_figure_selected_kegg.tsv", "immune_figure_selected_go.tsv", "immune_leading_edge_direction_audit.tsv"], tables):
            path = ctx.output("enrichment/immune/"+filename)
            table.to_csv(path, sep="\t", index=False)
            outputs[filename] = str(path)
        return outputs
    # Validation needs to augment its own fresh product set. Read-only inputs
    # are copied explicitly into the new run root before audits are added.
    import shutil
    source = ctx.input("recovered_matched_dir")
    out = ctx.output("enrichment/recovered-validation/manifest.json").parent
    for path in source.rglob("*"):
        if path.is_file():
            destination = ctx.output("enrichment/recovered-validation/"+path.relative_to(source).as_posix())
            shutil.copy2(path, destination)
    validate_recovered_results(out, ctx.input("de_dir"))
    return {"validation_summary": str(out / "validation_summary.json")}



def review_immune_evidence(composition: pd.DataFrame, manifest: pd.DataFrame,
                           de_frames: dict, gsea_frames: dict, prior_audit: pd.DataFrame):
    """Resummarize complete frozen immune evidence; no new tests or FDR changes.

    source_file retains the historical logical resource identifier used by the
    submitted tables. The execution manifest resolves those IDs to actual files.
    The 68 named sentinels are a post hoc descriptive review, not new validation.
    """
    tables = {}
    def read(logical):
        if logical == "Data/01_QC_Metadata/composition_broad_by_pool.tsv":
            frame = composition.copy()
        elif logical == "Data/01_QC_Metadata/sample_manifest_verified.tsv":
            frame = manifest.copy()
        elif logical == "Data/03_Enrichment/immune/immune_leading_edge_direction_audit.tsv":
            frame = prior_audit.copy()
        elif logical.startswith("Data/02_DE/"):
            frame = de_frames[Path(logical).name].copy()
        else:
            frame = gsea_frames[Path(logical).name].copy()
        frame["source_file"] = logical
        frame["source_line"] = np.arange(len(frame)) + 2
        return frame

    def save(frame, filename):
        tables[filename] = frame
    groups = ['T cell', 'macrophage']
    contrasts = ['24_vs_0', '96_vs_0', '96_vs_24']
    comp = read('Data/01_QC_Metadata/composition_broad_by_pool.tsv')
    manifest = read('Data/01_QC_Metadata/sample_manifest_verified.tsv')
    assert int(comp.n_cells.sum()) == 84643
    assert comp.cell_type_v5.nunique() == 14
    assert manifest.biological_replicate.nunique() == 6 and len(manifest) == 12
    pool = comp.groupby('biological_replicate').n_cells.sum().rename('all_recovered_cells')
    immune = comp[comp.cell_type_v5.isin(groups)].copy().merge(pool, on='biological_replicate')
    names = manifest[['biological_replicate','company_sample','tank_id']].drop_duplicates()
    immune = immune.merge(names, on='biological_replicate')
    immune['pct_of_recovered_pool'] = 100*immune.n_cells/immune.all_recovered_cells
    assert np.allclose(immune.pct_of_recovered_pool, immune.fraction_of_pool*100)
    save(immune, 'immune_cell_counts_and_pool_proportions.tsv')
    stage = immune.groupby(['cell_type_v5','time_h']).agg(n_pools=('n_cells','size'), total_recovered_cells=('n_cells','sum'), mean_pool_percent=('pct_of_recovered_pool','mean'), min_pool_percent=('pct_of_recovered_pool','min'), max_pool_percent=('pct_of_recovered_pool','max')).reset_index()
    save(stage, 'immune_stage_composition_descriptive.tsv')
    
    de = {}
    de_summaries=[]
    for group in groups:
        for contrast in contrasts:
            rel=f'Data/02_DE/broad__{group.replace(" ", "_")}__{contrast}.tsv.gz'
            d=read(rel)
            assert d.residual_df.eq(2).all()
            d['fdr_lt_005']=d.padj.lt(.05)
            d['both_batch_same_model_sign']=(np.sign(d.paired_log2cpm_diff_batch_1)==np.sign(d.log2FoldChange_shrunken)) & (np.sign(d.paired_log2cpm_diff_batch_2)==np.sign(d.log2FoldChange_shrunken))
            d['strict_display_DE']=d.fdr_lt_005 & d.log2FoldChange_shrunken.abs().ge(.5) & d.both_batch_same_model_sign
            de[(group,contrast)]=d.set_index('gene',drop=False)
            de_summaries.append({'group':group,'contrast':contrast,'all_rows':len(d),'nonmissing_FDR':int(d.padj.notna().sum()),'FDR_lt005':int(d.fdr_lt_005.sum()),'strict_display_DE':int(d.strict_display_DE.sum()),'strict_positive':int((d.strict_display_DE & d.log2FoldChange_shrunken.gt(0)).sum()),'strict_negative':int((d.strict_display_DE & d.log2FoldChange_shrunken.lt(0)).sum()),'FDR_lt005_but_batch_sign_not_matching_model':int((d.fdr_lt_005 & ~d.both_batch_same_model_sign).sum())})
    save(pd.concat([d.reset_index(drop=True) for d in de.values()],ignore_index=True), 'all_immune_DE_frozen_with_audit.tsv.gz')
    save(pd.DataFrame(de_summaries), 'immune_DE_summary.tsv')
    
    all_gsea=[]
    audits=[]
    for group in groups:
        for contrast in contrasts:
            for lib in ['KEGG_eggNOG_KO','GO_BP']:
                rel=f'Data/03_Enrichment/immune/broad__{group.replace(" ", "_")}__{contrast}__{lib}__GSEA.tsv.gz'
                d=read(rel)
                all_gsea.append(d)
                genes=de[(group,contrast)]
                for row in d.itertuples(index=False):
                    lead=str(row.Lead_genes).split(';') if pd.notna(row.Lead_genes) else []
                    evaluated=genes.reindex(lead)
                    valid=evaluated[['paired_log2cpm_diff_batch_1','paired_log2cpm_diff_batch_2']].notna().all(axis=1)
                    s=np.sign(row.NES)
                    match=(np.sign(evaluated.paired_log2cpm_diff_batch_1)==s) & (np.sign(evaluated.paired_log2cpm_diff_batch_2)==s) & valid
                    opposite=(np.sign(evaluated.paired_log2cpm_diff_batch_1)==-s) & (np.sign(evaluated.paired_log2cpm_diff_batch_2)==-s) & valid
                    audit={'group':group,'contrast':contrast,'library':lib,'Term':row.Term,'term_name':row.term_name,'NES':row.NES,'FDR_q':row.FDR_q,'n_leading_edge':len(lead),'n_evaluable':int(valid.sum()),'n_NES_concordant_both_batches':int(match.sum()),'fraction_NES_concordant_both_batches':float(match.sum()/valid.sum()) if valid.sum() else np.nan,'n_opposite_both_batches':int(opposite.sum()),'n_discordant_or_zero_between_batches':int((valid & ~match & ~opposite).sum()),'concordant_gene_list':';'.join(evaluated.loc[match,'gene'].astype(str)),'opposite_gene_list':';'.join(evaluated.loc[opposite,'gene'].astype(str)),'discordant_gene_list':';'.join(evaluated.loc[valid & ~match & ~opposite,'gene'].astype(str)),'source_file':row.source_file,'source_line':row.source_line}
                    audits.append(audit)
    gsea=pd.concat(all_gsea,ignore_index=True)
    audit=pd.DataFrame(audits)
    save(gsea,'all_immune_GSEA_frozen.tsv.gz')
    save(audit,'all_immune_GSEA_leading_edge_audit.tsv.gz')
    gs=gsea.assign(significant=lambda x:x.FDR_q.lt(.05),positive=lambda x:x.FDR_q.lt(.05)&x.NES.gt(0),negative=lambda x:x.FDR_q.lt(.05)&x.NES.lt(0)).groupby(['group','contrast','library']).agg(tested=('Term','size'),significant=('significant','sum'),positive=('positive','sum'),negative=('negative','sum')).reset_index()
    save(gs,'immune_GSEA_summary.tsv')
    
    # Retain all previously displayed modules at all three contrasts, including nonsignificant results.
    old=read('Data/03_Enrichment/immune/immune_leading_edge_direction_audit.tsv')
    keys=old[['group','Term']].drop_duplicates()
    modules=audit.merge(keys,on=['group','Term']).sort_values(['group','Term','contrast'])
    save(modules,'prior_immune_claim_modules_all_three_contrasts.tsv')
    chk=old.merge(audit,on=['group','contrast','Term'],suffixes=('_old','_new'))
    assert (chk.leading_edge_genes_same_direction_both_batches==chk.n_NES_concordant_both_batches).all()
    assert np.allclose(chk.leading_edge_batch_direction_fraction,chk.fraction_NES_concordant_both_batches)
    save(chk[['group','contrast','Term','leading_edge_genes','n_evaluable','leading_edge_genes_same_direction_both_batches','n_NES_concordant_both_batches','leading_edge_batch_direction_fraction','fraction_NES_concordant_both_batches']],'existing_16_leading_edge_checks_reproduced.tsv')
    
    # All genes in the existing 16 claim modules, rejoined to every frozen contrast.
    gene_keys=[]
    for r in old.itertuples():
        for gene in str(r.leading_edge_gene_list).split(';'):
            gene_keys.append({'group':r.group,'gene':gene,'prior_claim_terms':r.Term})
    gene_keys=pd.DataFrame(gene_keys).groupby(['group','gene']).prior_claim_terms.agg(lambda x:';'.join(sorted(set(x)))).reset_index()
    baseline=[]
    for g in groups:
        k=gene_keys[gene_keys.group.eq(g)]
        for c in contrasts:
            baseline.append(de[(g,c)].reset_index(drop=True).merge(k,on=['group','gene']))
    save(pd.concat(baseline,ignore_index=True),'prior_claim_leading_edge_genes_all_contrasts.tsv')
    
    # Named sentinels selected post hoc for descriptive review, not an independent validation.
    sentinels=['lck','fyn','cd247','prkcq','itk','nfatc2','plcg1','plcg2','nfkbia','nfkbie','nfkb2','rela','ikbkg','ikbkb','traf2','traf3','myd88','il-8','il1b','il-1b','il6','il-6','tnf','tnfa','tgfb1','tgfb2','il17a','il-17a','rora','stat3','cd74','tap1','tap2','tapbpl','ciita','pdia3','psme1','psme2','canx','ctss','ctsb','lamp1','lamp2','mrc1','mpeg1','csf1r','c1qa','c1qb','c1qc','ccr7','cxcr5','irf7','ifih1','mavs','tbk1','mcm2','mcm3','mcm4','mcm5','mcm6','mcm7','pcna','cdk1','ccnb1','il10','foxp3','gata3','runx3']
    sent=[]
    for (g,c),d in de.items():
        for gene in sentinels:
            if gene in d.index:
                item=d.loc[gene].to_dict(); item['present_exact_label']=True
            else:
                item={'group':g,'contrast':c,'gene':gene,'present_exact_label':False}
            sent.append(item)
    save(pd.DataFrame(sent),'named_immune_gene_sentinels_all_results.tsv')
    focus=['nfatc2','traf2','pdia3','cd74','myd88','adam8','il-8','ciita','tap1','stat3','rora','pcna']
    focuscols=['group','contrast','gene','log2FoldChange_shrunken','padj','paired_log2cpm_diff_batch_1','paired_log2cpm_diff_batch_2','source_file','source_line']
    save(pd.concat([d.reset_index(drop=True) for d in de.values()],ignore_index=True).loc[lambda d:d.gene.isin(focus),focuscols],'manuscript_gene_evidence.tsv')
    
    # Added context metabolic modules, no de novo scoring or new inferential tests.
    metabolic=['ko00010','ko00020','ko00071','ko00190','ko03010','ko04150','ko04066','ko04152']
    save(audit[audit.Term.isin(metabolic)].sort_values(['group','Term','contrast']),'immune_metabolic_pathway_context.tsv')
    summary = {"analysis": "frozen-data resummary only; no model fitting, no new significance tests",
        "counts": {"cells": 84643, "broad_types": 14, "independent_pools": 6,
            "technical_libraries": 12, "DE_rows": sum(len(d) for d in de.values()),
            "GSEA_rows": len(gsea), "prior_claim_gene_contrast_rows": sum(len(d) for d in baseline),
            "existing_leading_edge_checks_reproduced": len(chk)},
        "annotation_scope": "immune original eggNOG KO/GO; not the epithelial 16575-feature matched KEGG domain"}
    return tables, summary


def run_immune_review(ctx):
    de_dir = ctx.input("de_dir")
    immune_dir = ctx.input("immune_results_dir")
    de_files = [source_member(de_dir, f"broad__{group}__{contrast}.tsv.gz") for group in ["T_cell", "macrophage"] for contrast in ["24_vs_0", "96_vs_0", "96_vs_24"]]
    gsea_files = [immune_dir / f"broad__{group}__{contrast}__{library}__GSEA.tsv.gz" for group in ["T_cell", "macrophage"] for contrast in ["24_vs_0", "96_vs_0", "96_vs_24"] for library in ["KEGG_eggNOG_KO", "GO_BP"]]
    prior = ctx.input("immune_display_audit")
    all_paths = [ctx.input("composition_broad_by_pool"), ctx.input("sample_manifest"), *de_files, *gsea_files, prior]
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Immune review input files missing: " + "; ".join(missing))
    tables, summary = review_immune_evidence(read_table(all_paths[0], sep="\t"),
        read_table(all_paths[1], sep="\t"), {path.name: read_table(path, sep="\t") for path in de_files},
        {path.name: read_table(path, sep="\t") for path in gsea_files}, read_table(prior, sep="\t"))
    out = ctx.output("enrichment/immune-review/resummary_manifest.json").parent
    for filename, frame in tables.items():
        write_tsv(frame, out / filename)
    write_tsv(pd.DataFrame([{"path": str(path), "sha256": sha(path)} for path in all_paths]), out / "input_manifest.tsv")
    summary["logical_source_files"] = "Historical Data/... resource identifiers preserved in table source_file; input_manifest.tsv provides actual input paths and hashes."
    save_json(out / "resummary_manifest.json", summary)
    return {"immune_review_dir": str(out), "summary": str(out / "resummary_manifest.json")}
