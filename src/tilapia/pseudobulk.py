"""Six-pool pseudobulk models and matched-batch direction audits.

The six biological pools are the analysis units. Fits have two residual degrees
of freedom; significance and direction gates retain their original exploratory
meaning. The paired CPM denominator is the tested model-gene universe.
"""
from __future__ import annotations
from .runtime import read_table, source_member, iter_sources, source_provenance, source_size, source_exists, source_label, source_text, copy_source

import importlib.metadata
import json
import re
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from .runtime import as_csr, sha256, write_json

POOLS = ["BR1", "BR2", "BR3", "BR4", "BR5", "BR6"]
CONTRASTS = [("24", "0", "24_vs_0"), ("96", "0", "96_vs_0"), ("96", "24", "96_vs_24")]
TIME_TO_BATCH_POOL = {"0": {"batch_1": "BR1", "batch_2": "BR2"},
                     "24": {"batch_1": "BR4", "batch_2": "BR3"},
                     "96": {"batch_1": "BR5", "batch_2": "BR6"}}
MIN_CELLS_EXPLORATORY = 30
MIN_CELLS_MAIN = 100


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")

def direct_coefficient(dds, target: str) -> str:
    columns = [str(c) for c in dds.varm["LFC"].columns]
    hits = [c for c in columns if "time_h" in c and target in c and c.lower() != "intercept"]
    if len(hits) != 1:
        raise KeyError(f"Cannot identify time coefficient {target}: {columns}")
    return hits[0]

def paired_direction_columns(raw_counts: pd.DataFrame, target: str, baseline: str) -> pd.DataFrame:
    cpm = raw_counts.div(raw_counts.sum(axis=0).clip(lower=1), axis=1) * 1e6
    log2cpm = np.log2(cpm + 0.5)
    out = pd.DataFrame(index=raw_counts.index)
    diffs = []
    for batch in ["batch_1", "batch_2"]:
        tpool = TIME_TO_BATCH_POOL[target][batch]
        bpool = TIME_TO_BATCH_POOL[baseline][batch]
        diff = log2cpm[tpool] - log2cpm[bpool]
        out[f"paired_log2cpm_diff_{batch}"] = diff
        diffs.append(diff)
    out["paired_effect_mean"] = (diffs[0] + diffs[1]) / 2
    out["paired_effect_min_abs"] = np.minimum(np.abs(diffs[0]), np.abs(diffs[1]))
    out["paired_batches_same_direction"] = np.sign(diffs[0]).eq(np.sign(diffs[1])) & np.sign(diffs[0]).ne(0)
    return out

def aggregate(counts, obs: pd.DataFrame, mask: np.ndarray, var_names=None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    columns = {}
    meta = []
    cell_counts = {}
    for pool in POOLS:
        idx = np.flatnonzero(mask & obs["biological_replicate"].astype(str).eq(pool).to_numpy())
        cell_counts[pool] = int(len(idx))
        columns[pool] = np.asarray(counts[idx].sum(axis=0)).ravel().astype(np.int64)
        row = obs.loc[obs["biological_replicate"].astype(str).eq(pool)].iloc[0]
        meta.append({"pool": pool, "time_h": str(row["time_h"]), "sequencing_batch": str(row["sequencing_batch"]), "n_cells": int(len(idx))})
    return pd.DataFrame(columns, index=obs.attrs["var_names"] if var_names is None else var_names), pd.DataFrame(meta).set_index("pool"), cell_counts

def fit_group(label: str, kind: str, raw_counts: pd.DataFrame, meta: pd.DataFrame, cell_counts: dict[str, int], bad_gene: np.ndarray, output_dir: Path, *, dataset_class=None, stats_class=None) -> list[dict]:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dataset_class = DeseqDataSet if dataset_class is None else dataset_class
    stats_class = DeseqStats if stats_class is None else stats_class
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_gene = (
        raw_counts.sum(axis=1).ge(10)
        & raw_counts.gt(0).sum(axis=1).ge(2)
        & ~pd.Series(bad_gene, index=raw_counts.index)
    )
    fit_counts = raw_counts.loc[eligible_gene]
    summaries = []
    print(f"[fit] {kind}: {label}; cells={cell_counts}", flush=True)
    contrasts_by_reference = {
        "0": [("24", "0", "24_vs_0"), ("96", "0", "96_vs_0")],
        "24": [("96", "24", "96_vs_24")],
    }
    for reference, reference_contrasts in contrasts_by_reference.items():
        times = [reference] + [x for x in ["0", "24", "96"] if x != reference]
        model_meta = meta.copy()
        model_meta["time_h"] = pd.Categorical(model_meta["time_h"], categories=times, ordered=True)
        model_meta["sequencing_batch"] = pd.Categorical(model_meta["sequencing_batch"], categories=["batch_1", "batch_2"], ordered=True)
        dds = dataset_class(
            counts=fit_counts.T,
            metadata=model_meta[["sequencing_batch", "time_h"]],
            design="~ sequencing_batch + time_h",
            refit_cooks=True,
            size_factors_fit_type="ratio",
            n_cpus=2,
            quiet=True,
        )
        dds.deseq2()
        for target, baseline, contrast_label in reference_contrasts:
            stats = stats_class(dds, contrast=["time_h", target, baseline], n_cpus=2, quiet=True)
            stats.summary()
            unshrunk = stats.results_df.copy()
            coeff = direct_coefficient(dds, target)
            stats.lfc_shrink(coeff=coeff)
            shrunk = stats.results_df.copy()
            result = unshrunk.rename(columns={"log2FoldChange": "log2FoldChange_unshrunk", "lfcSE": "lfcSE_unshrunk"})
            result["log2FoldChange_shrunken"] = shrunk["log2FoldChange"]
            result["lfcSE_shrunken"] = shrunk["lfcSE"]
            result = result.join(paired_direction_columns(raw_counts.loc[result.index], target, baseline))
            result["paired_direction_matches_model"] = (
                result["paired_batches_same_direction"]
                & np.sign(result["paired_effect_mean"]).eq(np.sign(result["log2FoldChange_shrunken"]))
            )
            result["direction_robust"] = (
                result["padj"].lt(0.05)
                & result["log2FoldChange_shrunken"].abs().ge(0.5)
                & result["paired_direction_matches_model"]
            )
            result.insert(0, "gene", result.index.astype(str))
            result.insert(1, "analysis_level", kind)
            result.insert(2, "group", label)
            result.insert(3, "contrast", contrast_label)
            result.insert(4, "model", "~ sequencing_batch + time_h")
            result.insert(5, "residual_df", 2)
            result.reset_index(drop=True, inplace=True)
            path = output_dir / f"{kind}__{safe(label)}__{contrast_label}.tsv.gz"
            result.to_csv(path, sep="\t", index=False, compression="gzip")
            summaries.append({
                "analysis_level": kind, "group": label, "contrast": contrast_label,
                "n_pools": 6, "residual_df": 2, "min_cells_any_pool": min(cell_counts.values()),
                "evidence_tier": "main_eligible" if min(cell_counts.values()) >= MIN_CELLS_MAIN else "exploratory_low_cell_count",
                "n_genes_tested": int(len(result)),
                "n_padj05_lfc05": int((result["padj"].lt(0.05) & result["log2FoldChange_shrunken"].abs().ge(0.5)).sum()),
                "n_direction_robust": int(result["direction_robust"].sum()),
                "direction_robust_fraction_of_padj05_lfc05": float(result["direction_robust"].sum() / max(1, (result["padj"].lt(0.05) & result["log2FoldChange_shrunken"].abs().ge(0.5)).sum())),
                "result_file": str(path),
            })
    return summaries

def excluded_gene_mask(var_names, annotation: pd.DataFrame) -> np.ndarray:
    bad_symbols = set(annotation.loc[
        annotation["biotype"].eq("rRNA") | annotation["is_mitochondrial"].str.lower().eq("true"),
        "gene_symbol"].dropna())
    names = pd.Index(var_names).astype(str)
    return names.isin(bad_symbols) | names.str.startswith("KEH09_")


def analysis_data(adata):
    """Retain primary cells and the raw count layer without changing gene order."""
    keep = ~adata.obs["exclude_from_biology_v5"].astype(bool).to_numpy()
    obs = adata.obs.loc[keep].copy()
    obs.attrs["var_names"] = adata.var_names.astype(str)
    counts = adata.layers["counts"][keep]
    counts = as_csr(counts)
    return counts, obs


def group_specs(obs: pd.DataFrame):
    specs = []
    for label in sorted({"pavement cell", "ionocyte", "T cell", "macrophage"}):
        specs.append(("broad", label, obs["cell_type_v5"].astype(str).eq(label).to_numpy()))
    for state in sorted(obs.loc[obs["cell_type_v5"].astype(str).eq("ionocyte"), "cell_state_v5"].astype(str).unique()):
        specs.append(("ionocyte_state", state, obs["cell_state_v5"].astype(str).eq(state).to_numpy()))
    return specs


def fit_atlas_groups(adata, annotation: pd.DataFrame, output_dir: Path):
    counts, obs = analysis_data(adata)
    bad = excluded_gene_mask(adata.var_names, annotation)
    all_summaries, eligibility, errors = [], [], []
    for kind, label, mask in group_specs(obs):
        raw, meta, cell_counts = aggregate(counts, obs, mask)
        minimum = min(cell_counts.values())
        minimum_required = MIN_CELLS_MAIN if kind == "broad" else MIN_CELLS_EXPLORATORY
        tier = "main_eligible" if minimum >= MIN_CELLS_MAIN else ("exploratory" if minimum >= minimum_required else "not_fitted")
        eligibility.append({"analysis_level": kind, "group": label,
            **{f"n_{pool}": cell_counts[pool] for pool in POOLS}, "min_cells_any_pool": minimum, "tier": tier})
        if tier == "not_fitted":
            continue
        try:
            all_summaries.extend(fit_group(label, kind, raw, meta, cell_counts, bad, output_dir))
        except Exception as exc:
            errors.append({"analysis_level": kind, "group": label, "error": repr(exc)})
    return pd.DataFrame(all_summaries), pd.DataFrame(eligibility), errors


def fit_whole_gill(adata, annotation: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    counts, obs = analysis_data(adata)
    raw, meta, cell_counts = aggregate(counts, obs, np.ones(len(obs), dtype=bool))
    return pd.DataFrame(fit_group("whole_gill", "whole_atlas", raw, meta, cell_counts,
        excluded_gene_mask(adata.var_names, annotation), output_dir))


def audited_model_classes(output_dir: Path):
    """Create fit-local instrumentation; no module monkey-patching or global logs."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from scipy.stats import f
    diagnostics = []

    class AuditedDataSet(DeseqDataSet):
        def deseq2(self, *args, **kwargs):
            start = time.time()
            result = super().deseq2(*args, **kwargs)
            reference = str(self.obs.time_h.cat.categories[0])
            design = self.obsm["design_matrix"]
            rank = int(np.linalg.matrix_rank(design.values))
            if rank != 4 or self.n_obs-rank != 2:
                raise ValueError("Frozen APC model must have design rank 4 and residual df 2")
            self.var.to_csv(output_dir / f"fit_ref{reference}_gene_diagnostics.tsv.gz", sep="\t")
            self.obs.to_csv(output_dir / f"fit_ref{reference}_sample_metadata.tsv", sep="\t")
            design.to_csv(output_dir / f"fit_ref{reference}_design_matrix.tsv", sep="\t")
            self.varm["LFC"].to_csv(output_dir / f"fit_ref{reference}_coefficients_natural_log.tsv.gz", sep="\t")
            cooks = np.max(self.layers["cooks"], axis=0)
            threshold = f.ppf(.99, 4, 2)
            pd.DataFrame({"gene": self.var_names, "max_Cooks_distance": cooks,
                "exceeds_nominal_F_99": cooks > threshold}).to_csv(
                    output_dir / f"fit_ref{reference}_Cooks_diagnostic_only.tsv.gz", sep="\t", index=False)
            np.savez_compressed(output_dir / f"fit_ref{reference}_normalized_counts.npz",
                counts=self.X / self.obs.size_factors.to_numpy()[:, None],
                genes=self.var_names.to_numpy(dtype=str), pools=self.obs_names.to_numpy(dtype=str))
            flags = {key: self.var[key].astype(str).value_counts(dropna=False).to_dict()
                for key in self.var if any(token in key.lower() for token in ["converg", "replace", "refit", "outlier"])}
            diagnostics.append({"reference": reference, "n_samples": self.n_obs, "n_genes": self.n_vars,
                "design_rank": rank, "residual_df": 2, "design_columns": list(design.columns),
                "flags": flags, "cooks_nominal_F99": float(threshold),
                "n_exceeds_nominal_cooks": int((cooks > threshold).sum()), "elapsed_seconds": time.time()-start})
            (output_dir / "model_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str)+"\n", encoding="utf-8")
            return result

    class AuditedStats(DeseqStats):
        def summary(self, *args, **kwargs):
            result = super().summary(*args, **kwargs)
            self._audit_contrast = "_".join(map(str, self.contrast))
            return result

        def lfc_shrink(self, *args, **kwargs):
            result = super().lfc_shrink(*args, **kwargs)
            self._LFC_shrink_converged.rename("shrink_converged").to_csv(
                output_dir / f"{self._audit_contrast}_shrink_convergence.tsv.gz", sep="\t")
            return result
    return AuditedDataSet, AuditedStats, diagnostics


def fit_frozen_apc(arrays_path: Path, metadata_path: Path, annotation: pd.DataFrame, output_dir: Path):
    if importlib.metadata.version("pydeseq2") != "0.5.4":
        raise RuntimeError("The frozen APC extension requires PyDESeq2 0.5.4")
    arrays = np.load(arrays_path)
    metadata = read_table(metadata_path, sep="\t")
    idx = np.flatnonzero(metadata.cell_type.eq("dendritic/APC-like"))
    group = metadata.iloc[idx].copy()
    if group.pool.tolist() != POOLS:
        raise ValueError("Frozen APC pool order differs from BR1..BR6")
    raw = pd.DataFrame(arrays["counts"][idx].T, index=arrays["genes"], columns=group.pool)
    meta = group[["pool", "stage", "sequencing_batch", "n_cells"]].rename(columns={"stage": "time_h"}).set_index("pool")
    meta["time_h"] = meta.time_h.astype(str)
    bad = excluded_gene_mask(raw.index, annotation)
    eligible = raw.sum(axis=1).ge(10) & raw.gt(0).sum(axis=1).ge(2) & ~bad
    pd.DataFrame({"gene": raw.index, "total_count": raw.sum(axis=1),
        "positive_pools": raw.gt(0).sum(axis=1), "rRNA_mito_or_KEH_excluded": bad,
        "eligible": eligible}).to_csv(output_dir / "gene_filter.tsv.gz", sep="\t", index=False)
    raw.to_csv(output_dir / "APC_raw_counts_6_pools.tsv.gz", sep="\t")
    meta.to_csv(output_dir / "APC_sample_metadata.tsv", sep="\t")
    write_json(output_dir / "run_manifest.json", {
        "implementation": "tilapia.pseudobulk.fit_group", "pydeseq2": "0.5.4",
        "input_counts": str(arrays_path), "input_counts_sha256": sha256(arrays_path),
        "input_metadata": str(metadata_path), "input_metadata_sha256": sha256(metadata_path),
        "extension": "exploratory; APC-like selected after stage1 review",
        "all_parameters_from_shared_fit_group": True,
        "CPM_direction_denominator": "sum over group model-eligible genes, matching frozen code",
        "raw_input_genes": len(raw), "eligible_genes": int(eligible.sum()),
        "model": "~ sequencing_batch + time_h", "reference_refits": ["0", "24"],
        "refit_cooks": True, "default_min_replicates_replacement": 7,
        "outlier_filter_requires_at_least_3_replicates_for_identical_design_row": True,
        "technical_replicates_not_used_as_n": True})
    dataset_class, stats_class, diagnostics = audited_model_classes(output_dir)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        summary = fit_group("dendritic/APC-like", "broad", raw, meta, meta.n_cells.to_dict(), bad,
            output_dir, dataset_class=dataset_class, stats_class=stats_class)
    (output_dir / "warnings.json").write_text(json.dumps([
        {"category": item.category.__name__, "message": str(item.message),
         "filename": item.filename, "line": item.lineno} for item in seen], indent=2)+"\n", encoding="utf-8")
    return pd.DataFrame(summary), diagnostics


INPUTS_BY_TASK = {'fit': ['atlas_h5ad', 'gene_annotation'], 'whole-gill': ['atlas_h5ad', 'gene_annotation'], 'frozen-apc': ['frozen_apc_counts', 'frozen_apc_metadata', 'gene_annotation'], 'aggregate': ['atlas_h5ad', 'composition_broad_by_pool']}


def run(ctx, *, task="fit") -> dict[str, str]:
    if task not in INPUTS_BY_TASK:
        raise ValueError(f"Unknown pseudobulk task: {task}")
    ctx.require(INPUTS_BY_TASK[task])
    if task == "aggregate":
        return aggregate_current_atlas(ctx.input("atlas_h5ad"),
            read_table(ctx.input("composition_broad_by_pool"), sep="\t"),
            ctx.output("pseudobulk/aggregation_manifest.json").parent)
    annotation = read_table(ctx.input("gene_annotation"), sep="\t", dtype=str)
    directory = ctx.output("pseudobulk/de_summary_with_direction_audit.tsv").parent
    if task == "frozen-apc":
        summary, _ = fit_frozen_apc(ctx.input("frozen_apc_counts"), ctx.input("frozen_apc_metadata"), annotation, directory)
        path = directory / "de_summary.tsv"
    else:
        import anndata as ad
        adata = ad.read_h5ad(ctx.input("atlas_h5ad"))
        if task == "whole-gill":
            summary = fit_whole_gill(adata, annotation, directory)
            path = directory / "whole_gill_de_summary.tsv"
        else:
            summary, eligibility, errors = fit_atlas_groups(adata, annotation, directory)
            eligibility.to_csv(ctx.output("pseudobulk/eligibility.tsv"), sep="\t", index=False)
            ctx.output("pseudobulk/errors.json").write_text(json.dumps(errors, indent=2)+"\n", encoding="utf-8")
            path = directory / "de_summary_with_direction_audit.tsv"
            summary.to_csv(path, sep="\t", index=False)
            if errors:
                raise RuntimeError(f"{len(errors)} pseudobulk group fits failed; see {directory / 'errors.json'}")
    summary.to_csv(path, sep="\t", index=False)
    return {"de_dir": str(directory), "summary": str(path)}


def h5_column(group, key):
    """Read the categorical/string columns used by the frozen backed atlas."""
    import h5py
    value = group[key]
    if isinstance(value, h5py.Group):
        categories = value["categories"][:]
        categories = np.array([v.decode() if isinstance(v, bytes) else v for v in categories], dtype=object)
        return np.array([categories[k] if k >= 0 else None for k in value["codes"][:]])
    array = value[:]
    return np.array([v.decode() if isinstance(v, bytes) else v for v in array]) if array.dtype.kind in ["S", "O"] else array


def aggregate_current_atlas(atlas_path: Path, expected: pd.DataFrame, output_dir: Path):
    """Stream the 14 cell types x six pools with all-feature UMI denominators.

    This exact-atlas aggregation requires 84,643 primary cells (43 review-flagged
    cells excluded from the 84,686 QC-retained atlas). Counts and detection counts
    are calculated together, using 1,000-cell sparse blocks as in the source.
    """
    import h5py
    start = time.time()
    with h5py.File(atlas_path, "r") as handle:
        obs = handle["obs"]
        genes = h5_column(handle["var"], handle["var"].attrs["_index"])
        cell_type = h5_column(obs, "cell_type_v5")
        pool = h5_column(obs, "biological_replicate")
        exclude = h5_column(obs, "exclude_from_biology_v5")
        keep = ~exclude.astype(bool)
        if int(keep.sum()) != 84643:
            raise ValueError("Current-atlas aggregation requires exactly 84,643 primary cells")
        metadata = pd.DataFrame({key: h5_column(obs, key) for key in ["biological_replicate",
            "company_sample", "tank_id", "time_h", "sequencing_batch", "fish_id", "fish_pool", "sex"]})
        metadata.drop_duplicates().to_csv(output_dir / "atlas_sample_metadata.tsv", sep="\t", index=False)
        groups = [(label, p) for label in sorted(set(cell_type[keep])) for p in POOLS]
        lookup = {group: i for i, group in enumerate(groups)}
        labels = np.array([lookup[(label, p)] for label, p in zip(cell_type, pool)])
        counts = np.zeros((len(groups), len(genes)), dtype=np.int64)
        detected = np.zeros_like(counts)
        n_cells = np.bincount(labels[keep], minlength=len(groups))
        for i, (label, p) in enumerate(groups):
            if n_cells[i] != expected[(expected.cell_type_v5 == label) & (expected.biological_replicate == p)].n_cells.iloc[0]:
                raise ValueError(f"Frozen composition mismatch for {label}/{p}")
        matrix = handle["layers/counts"]
        pointer = matrix["indptr"][:]
        for low in range(0, len(pool), 1000):
            high = min(low+1000, len(pool))
            a, b = pointer[low], pointer[high]
            block = sparse.csr_matrix((matrix["data"][a:b], matrix["indices"][a:b], pointer[low:high+1]-a), shape=(high-low, len(genes)))
            present = np.flatnonzero(keep[low:high])
            indicator = sparse.csr_matrix((np.ones(len(present), dtype=np.int64),
                (labels[low:high][present], present)), shape=(len(groups), high-low))
            counts += (indicator @ block).toarray()
            block.data = np.ones_like(block.data)
            detected += (indicator @ block).toarray()
        if counts.min() < 0:
            raise ValueError("Negative aggregated raw counts")
        group_meta = pd.DataFrame(groups, columns=["cell_type", "pool"])
        group_meta["n_cells"] = n_cells
        group_meta["total_UMI"] = counts.sum(axis=1)
        group_meta["stage"] = [0 if p in ["BR1", "BR2"] else 24 if p in ["BR3", "BR4"] else 96 for _, p in groups]
        group_meta["sequencing_batch"] = ["batch_1" if p in ["BR1", "BR4", "BR5"] else "batch_2" for _, p in groups]
        counts_path = output_dir / "current_atlas_84_group_raw_counts.npz"
        metadata_path = output_dir / "current_atlas_84_group_metadata.tsv"
        group_meta.to_csv(metadata_path, sep="\t", index=False)
        np.savez_compressed(counts_path, counts=counts, detected_cells=detected, genes=genes.astype(str))
        rows = []
        for assay in ["LOC102081935"]:
            feature = assay
            hits = np.flatnonzero(genes == feature)
            if len(hits) != 1:
                raise ValueError(f"Assay {assay} has {len(hits)} matrix features")
            gene_index = hits[0]
            for k, row in group_meta.loc[group_meta.cell_type.eq("dendritic/APC-like")].iterrows():
                rows.append({"assay": assay, "matrix_feature": feature, **row.to_dict(),
                    "raw_UMI": counts[k, gene_index],
                    "CPM_all_features_denominator": counts[k, gene_index] / row.total_UMI * 1e6,
                    "detected_cells": detected[k, gene_index],
                    "fraction_detected": detected[k, gene_index] / row.n_cells})
        pd.DataFrame(rows).to_csv(output_dir / "apc_candidate_pool_expression.tsv", sep="\t", index=False)
        manifest = {"source": str(atlas_path), "source_bytes": atlas_path.stat().st_size,
            "source_mtime_ns": atlas_path.stat().st_mtime_ns,
            "raw_dimensions": list(matrix.attrs["shape"].astype(int).tolist()),
            "primary_cells": int(keep.sum()), "excluded_cells": int(exclude.sum()),
            "aggregation": "raw counts; original cell_type_v5; original biological_replicate; exclude_from_biology_v5=False",
            "n_groups": len(groups), "group_cell_counts_equal_frozen_table": True,
            "normalization": "sum raw UMI per gene / total raw UMI over all 37484 features in celltype-pool * 1e6; no DE refit",
            "elapsed_seconds": round(time.time()-start, 1)}
    (output_dir / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return {"frozen_apc_counts": str(counts_path), "frozen_apc_metadata": str(metadata_path),
            "apc_candidate_pool_expression": str(output_dir / "apc_candidate_pool_expression.tsv")}

