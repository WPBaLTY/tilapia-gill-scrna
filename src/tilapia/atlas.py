"""Full-feature atlas construction, annotation and pool-level diagnostics.

The full feature matrix is retained. Annotated rRNA, mitochondrial and KEH09
features are excluded only from embedding. Compatibility column names ending
in ``v5`` are retained because they identify the locked scientific data schema.
Functions receive data or paths explicitly; importing this module does no I/O.
"""
from __future__ import annotations
from .runtime import read_table

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .runtime import as_csr

SEED = 20260823
N_HVG = 4000
N_PCS = 50
N_NEIGHBORS = 15
CLUSTER_KEY = "leiden_v5_full_harmony_r0_5"
RRNA_REVIEW_THRESHOLD_PERCENT = 10.0


MAPPING = {
    "0": ("granulocyte", "MMP9_MMP1_LTB4R", "high"),
    "1": ("chondrocyte", "ACAN_CLEC3A_S100B", "high"),
    "2": ("pavement cell", "ABCA12_EVPL_barrier", "moderate"),
    "3": ("pavement cell", "SCEL_MUC15_HOPX", "moderate"),
    "4": ("basal epithelial", "LAMA3_LAMC2_ITGB4", "high"),
    "5": ("pavement cell", "CEBPD_transitional", "high"),
    "6": ("pavement cell", "EPPK1_remodelling", "high"),
    "7": ("fibroblast", "COL1A1_COL1A2_FSTL1", "high"),
    "8": ("dendritic/APC-like", "CD74_CD83_CSF1R_MPEG1", "high"),
    "9": ("pavement cell", "PTGS2_stress", "moderate"),
    "10": ("cycling non-immune", "MKI67_UBE2C", "moderate"),
    "11": ("B cell", "CD79A_immunoglobulin", "high"),
    "12": ("pavement cell", "PKP3_PERP_desmosomal", "moderate"),
    "13": ("pavement cell", "LGALS4_epithelial", "moderate"),
    "14": ("macrophage", "SPIC_GPX3_MHCII", "moderate"),
    "15": ("ionocyte", "SOX4_MYB_ATP1A1_progenitor-like", "moderate"),
    "16": ("cycling T cell", "CD3_TCR_MKI67", "high"),
    "17": ("T cell", "GIMAP7_IL7R", "high"),
    "18": ("macrophage", "C1Q_MAFB_DAB2", "high"),
    "19": ("mucous/goblet", "MUC5B_AGR2_SPDEF", "high"),
    "20": ("granulocyte", "CLEC6A_NFE2_lectin-rich", "moderate"),
    "21": ("T cell", "GATA3_enriched_T", "moderate"),
    "22": ("dendritic/APC-like", "FCER1G_CD74_APC-like", "moderate"),
    "23": ("vascular-associated", "PLPP3_APLN_MMRN2_GP1BB_mixed", "moderate"),
    "24": ("ionocyte", "FOXI1_NDRG1_ESRRG_transport", "high"),
    "25": ("ionocyte", "ATP1A1_paralogue_mitochondria-rich", "high"),
}

MARKER_PANELS = {
    "tilapia_HR_published": [
        "LOC100692482", "LOC100698977", "LOC100705383", "LOC100696861",
        "LOC109204971", "LOC100705611", "LOC109196493", "LOC109199919",
    ],
    "tilapia_NaR_published": [
        "atp2b1", "LOC100696627", "LOC100706765", "LOC100704157",
        "LOC100695649", "LOC100699133",
    ],
    "tilapia_pavement_published": ["LOC100691306", "LOC100693008"],
    "ionocyte_general": [
        "LOC100699477", "LOC112841652", "LOC100694365", "LOC100706474",
        "foxi2", "clcn2", "LOC100699530", "cftr",
    ],
    "acid_base_HR_support": [
        "atp6v1a", "LOC100710754", "LOC100710668", "atp6v1c1",
        "LOC100703061", "LOC100691637", "LOC100692172", "LOC100708352",
        "LOC100706627", "LOC100706367", "foxi3",
    ],
    "calcium_NaR_support": ["LOC100696021", "atp2b1", "LOC100696627", "LOC100706765"],
    "seawater_secretory_support": ["cftr", "LOC100704048", "LOC100694941", "clcn2"],
    "epithelial_lineage": ["epcam", "perp", "pkp3", "LOC100698188", "LOC100697919"],
    "T_lineage": ["cd3e", "LOC102080392", "LOC100696704", "LOC100702154"],
    "cell_cycle": ["mki67", "ube2c", "smc2", "smc4", "top2a"],
}


def group_sum(x: sparse.csr_matrix, groups: pd.Series) -> tuple[np.ndarray, list[str]]:
    levels = sorted(groups.astype(str).unique())
    codes = pd.Categorical(groups.astype(str), categories=levels).codes
    indicator = sparse.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
        shape=(len(levels), len(codes)),
    )
    return np.asarray((indicator @ x).todense()), levels

def refresh_manifest(adata, manifest_path: Path) -> pd.DataFrame:
    manifest = read_table(manifest_path, sep="\t", dtype=str)
    if manifest["library_id"].duplicated().any():
        raise RuntimeError("sample_manifest.tsv has duplicate library_id rows")
    lookup = manifest.set_index("library_id")
    libs = adata.obs["library_id"].astype(str)
    if not set(libs.unique()).issubset(lookup.index):
        raise RuntimeError("AnnData contains libraries absent from sample_manifest.tsv")
    fields = [
        "company_sample", "fish_id", "fish_pool", "tank_id", "biological_replicate",
        "technical_replicate", "capture_id", "sequencing_date", "sequencing_batch",
        "time_h", "condition", "salinity_psu", "body_length_cm", "body_mass_g",
        "sex", "pooled", "cell_viability_pct",
    ]
    for field in fields:
        adata.obs[field] = libs.map(lookup[field]).to_numpy()
    adata.obs["time_h"] = pd.to_numeric(adata.obs["time_h"], errors="raise").astype(int)
    adata.obs["salinity_psu"] = pd.to_numeric(adata.obs["salinity_psu"], errors="raise")
    adata.obs["pooled"] = pd.to_numeric(adata.obs["pooled"], errors="raise").astype(int)
    return manifest

def compute_rrna_qc(adata, counts: sparse.csr_matrix, annotation: pd.DataFrame) -> pd.DataFrame:
    rrna_symbols = set(annotation.loc[annotation["biotype"].eq("rRNA"), "gene_symbol"].dropna())
    rrna_mask = adata.var_names.astype(str).isin(rrna_symbols)
    rrna_counts = np.asarray(counts[:, rrna_mask].sum(axis=1)).ravel()
    total_counts = np.asarray(counts.sum(axis=1)).ravel()
    pct = np.divide(rrna_counts * 100.0, total_counts, out=np.zeros_like(rrna_counts, dtype=float), where=total_counts > 0)
    adata.obs["pct_counts_rrna_annotated"] = pct
    adata.obs["qc_rrna_review_flag"] = pct > RRNA_REVIEW_THRESHOLD_PERCENT
    return (
        adata.obs.groupby([CLUSTER_KEY, "qc_rrna_review_flag"], observed=True)
        .size().rename("n_cells").reset_index()
    )

def apply_annotation(adata) -> pd.DataFrame:
    clusters = adata.obs[CLUSTER_KEY].astype(str)
    if set(clusters.unique()) != set(MAPPING):
        raise RuntimeError("Explicit mapping does not match observed full-feature clusters")
    adata.obs["cell_type_v5"] = pd.Categorical(clusters.map({k: v[0] for k, v in MAPPING.items()}))
    adata.obs["cell_state_v5"] = pd.Categorical(clusters.map({k: v[1] for k, v in MAPPING.items()}))
    adata.obs["annotation_confidence_v5"] = pd.Categorical(
        clusters.map({k: v[2] for k, v in MAPPING.items()}),
        categories=["high", "moderate", "low"], ordered=True,
    )
    adata.obs["exclude_from_biology_v5"] = adata.obs["qc_rrna_review_flag"].astype(bool)
    rows = []
    for cluster in sorted(MAPPING, key=int):
        mask = clusters.eq(cluster)
        cell_type, state, confidence = MAPPING[cluster]
        rows.append({
            "cluster": int(cluster), "n_cells": int(mask.sum()), "cell_type_v5": cell_type,
            "cell_state_v5": state, "confidence": confidence,
            "n_rrna_review_flag": int(adata.obs.loc[mask, "qc_rrna_review_flag"].sum()),
            "median_pct_rrna": float(adata.obs.loc[mask, "pct_counts_rrna_annotated"].median()),
            "mean_pct_mt": float(adata.obs.loc[mask, "pct_counts_mt"].mean()),
            "median_n_genes": float(adata.obs.loc[mask, "n_genes_by_counts"].median()),
        })
    return pd.DataFrame(rows)

def marker_panel_table(adata) -> pd.DataFrame:
    clusters = adata.obs[CLUSTER_KEY].astype(str)
    levels = sorted(clusters.unique(), key=int)
    codes = pd.Categorical(clusters, categories=levels).codes
    x = as_csr(adata.X)
    rows = []
    for panel, genes in MARKER_PANELS.items():
        for gene in genes:
            if gene not in adata.var_names:
                rows.append({"panel": panel, "gene": gene, "cluster": "NOT_IN_MATRIX", "mean_log1p_cp10k": np.nan, "fraction_detected": np.nan})
                continue
            values = x[:, adata.var_names.get_loc(gene)].toarray().ravel()
            for i, cluster in enumerate(levels):
                mask = codes == i
                rows.append({
                    "panel": panel, "gene": gene, "cluster": cluster,
                    "mean_log1p_cp10k": float(values[mask].mean()),
                    "fraction_detected": float((values[mask] > 0).mean()),
                })
    return pd.DataFrame(rows)

def technical_replicate_audit(adata, counts: sparse.csr_matrix) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = ~adata.obs["exclude_from_biology_v5"].to_numpy(dtype=bool)
    obs = adata.obs.loc[keep].copy()
    x = counts[keep]
    lib_counts, libs = group_sum(x, obs["library_id"])
    lib_meta = obs[["library_id", "biological_replicate", "technical_replicate", "time_h", "sequencing_batch"]].drop_duplicates().set_index("library_id").loc[libs]
    gene_keep = lib_counts.sum(axis=0) >= 20
    cpm = lib_counts[:, gene_keep] / np.maximum(lib_counts[:, gene_keep].sum(axis=1, keepdims=True), 1) * 1e6
    logcpm = np.log1p(cpm)
    rows = []
    for pool, frame in lib_meta.reset_index().groupby("biological_replicate"):
        if len(frame) != 2:
            raise RuntimeError(f"Expected two libraries for {pool}")
        a, b = [libs.index(v) for v in frame["library_id"]]
        pr = pearsonr(logcpm[a], logcpm[b]).statistic
        sr = spearmanr(logcpm[a], logcpm[b]).statistic
        rows.append({
            "fish_pool": pool, "time_h": int(frame["time_h"].iloc[0]),
            "sequencing_batch": frame["sequencing_batch"].iloc[0],
            "library_1": frame["library_id"].iloc[0], "library_2": frame["library_id"].iloc[1],
            "n_genes_compared": int(gene_keep.sum()), "pearson_log1p_cpm": float(pr),
            "spearman_log1p_cpm": float(sr),
        })

    state_counts = (
        obs.groupby(["library_id", "biological_replicate", "cell_state_v5"], observed=True)
        .size().rename("n").reset_index()
    )
    state_counts["fraction"] = state_counts["n"] / state_counts.groupby("library_id")["n"].transform("sum")
    wide = state_counts.pivot_table(index=["biological_replicate", "library_id"], columns="cell_state_v5", values="fraction", fill_value=0)
    comp_rows = []
    for pool in sorted(wide.index.get_level_values(0).unique()):
        part = wide.loc[pool]
        corr = pearsonr(part.iloc[0].to_numpy(), part.iloc[1].to_numpy()).statistic
        comp_rows.append({"fish_pool": pool, "composition_pearson_26_states": float(corr), "library_1": part.index[0], "library_2": part.index[1]})
    return pd.DataFrame(rows), pd.DataFrame(comp_rows)

def composition_tables(adata) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = adata.obs.loc[~adata.obs["exclude_from_biology_v5"].astype(bool)].copy()
    state = obs.groupby(["biological_replicate", "time_h", "sequencing_batch", "cell_type_v5", "cell_state_v5"], observed=True).size().rename("n_cells").reset_index()
    state["fraction_of_pool"] = state["n_cells"] / state.groupby("biological_replicate")["n_cells"].transform("sum")
    broad = state.groupby(["biological_replicate", "time_h", "sequencing_batch", "cell_type_v5"], observed=True).agg(n_cells=("n_cells", "sum")).reset_index()
    broad["fraction_of_pool"] = broad["n_cells"] / broad.groupby("biological_replicate")["n_cells"].transform("sum")
    ion = state.loc[state["cell_type_v5"].astype(str).eq("ionocyte")].copy()
    ion["fraction_within_ionocytes"] = ion["n_cells"] / ion.groupby("biological_replicate")["n_cells"].transform("sum")
    return state, broad, ion

def cluster_purity(labels: pd.Series, truth: pd.Series) -> tuple[float, float, float, int]:
    frame = pd.DataFrame({"cluster": labels.astype(str), "truth": truth.astype(str)})
    tab = pd.crosstab(frame["cluster"], frame["truth"])
    purity = tab.max(axis=1) / tab.sum(axis=1)
    weighted = float((purity * tab.sum(axis=1)).sum() / tab.to_numpy().sum())
    return weighted, float(purity.median()), float(purity.min()), int((purity < 0.8).sum())

def build_embedding(adata):
    """Mutate a count-backed AnnData using the frozen full-feature settings."""
    import scanpy as sc
    is_rrna = adata.var["biotype"].astype(str).eq("rRNA").to_numpy()
    is_mt_annotated = adata.var["is_mitochondrial"].astype(bool).to_numpy()
    is_keh_mt = adata.var_names.astype(str).str.startswith("KEH09_")
    excluded = is_rrna | is_mt_annotated | is_keh_mt
    reason = np.full(adata.n_vars, "included", dtype=object)
    reason[is_rrna] = "annotated_rRNA"
    reason[is_mt_annotated] = "annotated_mitochondrial"
    reason[is_keh_mt] = "provider_KEH09_mitochondrial"
    adata.var["excluded_from_v5_full_embedding"] = excluded
    adata.var["v5_full_embedding_exclusion_reason"] = reason

    adata.X = as_csr(adata.layers["counts"])
    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        flavor="seurat_v3",
        n_top_genes=4500,
        batch_key="sequencing_batch",
        subset=False,
        inplace=True,
    )
    ranks = adata.var["highly_variable_rank"].to_numpy(dtype=float)
    eligible = (~excluded) & np.isfinite(ranks)
    order = np.flatnonzero(eligible)[np.argsort(ranks[eligible])]
    if len(order) < N_HVG:
        raise RuntimeError(f"Only {len(order)} eligible HVGs")
    hvg_mask = np.zeros(adata.n_vars, dtype=bool)
    hvg_mask[order[:N_HVG]] = True
    adata.var["highly_variable"] = hvg_mask

    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    adata.X = as_csr(adata.X)
    hvg = adata[:, hvg_mask].copy()
    if "counts" in hvg.layers:
        del hvg.layers["counts"]
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(hvg, n_comps=N_PCS, svd_solver="arpack", random_state=SEED)
    adata.obsm["X_pca_v5_full"] = hvg.obsm["X_pca"].astype(np.float32, copy=False)
    del hvg

    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        use_rep="X_pca_v5_full",
        key_added="neighbors_v5_full_unintegrated",
        random_state=SEED,
    )
    sc.tl.leiden(
        adata,
        resolution=0.5,
        neighbors_key="neighbors_v5_full_unintegrated",
        key_added="leiden_v5_full_unintegrated_r0_5",
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=SEED,
    )
    sc.tl.umap(
        adata,
        neighbors_key="neighbors_v5_full_unintegrated",
        key_added="X_umap_v5_full_unintegrated",
        random_state=SEED,
    )

    sc.external.pp.harmony_integrate(
        adata,
        key="sequencing_batch",
        basis="X_pca_v5_full",
        adjusted_basis="X_pca_v5_full_harmony",
        max_iter_harmony=20,
        random_state=SEED,
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=N_PCS,
        use_rep="X_pca_v5_full_harmony",
        key_added="neighbors_v5_full_harmony",
        random_state=SEED,
    )
    sc.tl.umap(
        adata,
        neighbors_key="neighbors_v5_full_harmony",
        key_added="X_umap_v5_full_harmony",
        random_state=SEED,
    )
    for resolution in [0.3, 0.5, 0.8]:
        key = f"leiden_v5_full_harmony_r{str(resolution).replace('.', '_')}"
        sc.tl.leiden(
            adata,
            resolution=resolution,
            neighbors_key="neighbors_v5_full_harmony",
            key_added=key,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=SEED,
        )

    return adata




def rank_full_feature_markers(adata) -> pd.DataFrame:
    import scanpy as sc
    sc.tl.rank_genes_groups(adata, groupby=CLUSTER_KEY, method="wilcoxon", pts=True,
                           key_added="markers_v5_full")
    markers = sc.get.rank_genes_groups_df(adata, group=None, key="markers_v5_full")
    return markers.groupby("group", observed=True, group_keys=False).head(100)


def resolution_audit(obs: pd.DataFrame) -> pd.DataFrame:
    x = obs.loc[~obs["exclude_from_biology_v5"].astype(bool)].copy()
    columns = ["leiden_v5_full_harmony_r0_3", CLUSTER_KEY,
               "leiden_v5_full_harmony_r0_8", "leiden_v5_full_unintegrated_r0_5"]
    ref = x[CLUSTER_KEY].astype(str)
    rows = []
    for column in columns:
        labels = x[column].astype(str)
        broad = cluster_purity(labels, x["cell_type_v5"])
        state = cluster_purity(labels, x["cell_state_v5"])
        rows.append({"clustering": column.replace("v5", "v6"),
            "n_clusters": int(labels.nunique()),
            "ARI_vs_primary_r0_5": float(adjusted_rand_score(ref, labels)),
            "NMI_vs_primary_r0_5": float(normalized_mutual_info_score(ref, labels)),
            "weighted_broad_type_purity": broad[0], "median_cluster_broad_type_purity": broad[1],
            "minimum_cluster_broad_type_purity": broad[2], "clusters_below_0_8_broad_type_purity": broad[3],
            "weighted_state_purity": state[0], "median_cluster_state_purity": state[1],
            "minimum_cluster_state_purity": state[2], "clusters_below_0_8_state_purity": state[3],
            "primary_selection": column == CLUSTER_KEY})
    return pd.DataFrame(rows)


def submission_composition_tables(adata, mapping: pd.DataFrame) -> dict[str, pd.DataFrame]:
    state, broad, _ = composition_tables(adata)
    broad = broad.rename(columns={"cell_type_v5": "cell_type_v6",
                                  "fraction_of_pool": "fraction_of_recovered_primary_cells"})
    state = state.rename(columns={"cell_type_v5": "cell_type_v6", "cell_state_v5": "cell_state_v6",
                                  "fraction_of_pool": "fraction_of_recovered_primary_cells"})
    state["fraction_within_broad_type"] = state["n_cells"] / state.groupby(
        ["biological_replicate", "cell_type_v6"], observed=True)["n_cells"].transform("sum")
    return {"cluster_annotations.tsv": mapping.rename(columns={"cell_type_v5": "cell_type_v6", "cell_state_v5": "cell_state_v6"}),
            "broad_composition_by_pool.tsv": broad,
            "state_composition_by_pool.tsv": state,
            "resolution_and_integration_sensitivity.tsv": resolution_audit(adata.obs)}


def annotate(adata, manifest_path: Path, annotation: pd.DataFrame):
    """Apply the 26-cluster map and return all attributable annotation audits."""
    manifest = refresh_manifest(adata, manifest_path)
    counts = as_csr(adata.layers["counts"])
    rrna = compute_rrna_qc(adata, counts, annotation)
    mapping = apply_annotation(adata)
    tech_gene, tech_comp = technical_replicate_audit(adata, counts)
    state, broad, ion = composition_tables(adata)
    tables = {"annotation/cluster_mapping_v5.tsv": mapping,
              "annotation/published_marker_panel_by_cluster.tsv.gz": marker_panel_table(adata),
              "audit/rrna_review_by_cluster.tsv": rrna,
              "audit/technical_replicate_gene_concordance.tsv": tech_gene,
              "audit/technical_replicate_composition_concordance.tsv": tech_comp,
              "audit/composition_state_by_pool.tsv": state,
              "audit/composition_broad_by_pool.tsv": broad,
              "audit/ionocyte_states_by_pool.tsv": ion,
              "audit/sample_manifest_verified.tsv": manifest,
              "audit/resolution_and_integration_sensitivity.tsv": resolution_audit(adata.obs)}
    tables.update({"tables/"+key: frame for key, frame in submission_composition_tables(adata, mapping).items()})
    summary = {"n_cells_input": int(adata.n_obs),
        "n_rrna_review_flag": int(adata.obs["qc_rrna_review_flag"].sum()),
        "rrna_review_threshold_percent": RRNA_REVIEW_THRESHOLD_PERCENT,
        "n_cells_primary_biology": int((~adata.obs["exclude_from_biology_v5"]).sum()),
        "n_clusters": int(adata.obs[CLUSTER_KEY].nunique()),
        "n_broad_types": int(adata.obs["cell_type_v5"].nunique()),
        "technical_replicate_median_pearson": float(tech_gene["pearson_log1p_cpm"].median()),
        "technical_replicate_min_pearson": float(tech_gene["pearson_log1p_cpm"].min()),
        "composition_replicate_median_pearson": float(tech_comp["composition_pearson_26_states"].median()),
        "integration_rule": "Harmony on sequencing_batch only; library_id not corrected",
        "annotation_rule": "Explicit marker-supported mapping on the 26-cluster full-feature batch-Harmony graph.",
        "analysis_version": "v6 full-feature primary analysis", "input_features": int(adata.n_vars),
        "pavement_state_rule": "Seven conservative full-feature clusters; superseded nine-state v5 subdivisions not used.",
        "composition_metric": "Pearson correlation across 26 transcriptional states"}
    adata.uns["v5_final_annotation"] = summary
    return tables, summary


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out

def build_broad_markers(adata, primary: np.ndarray) -> pd.DataFrame:
    labels = adata.obs.loc[primary, "cell_type_v5"].astype(str).to_numpy()
    groups = sorted(pd.unique(labels))
    matrix = adata.X[primary]
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    n_total = matrix.shape[0]
    total_sum = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    total_sq = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel().astype(float)
    detected_by_group: dict[str, np.ndarray] = {}
    for group in groups:
        subset = matrix[labels == group]
        detected_by_group[group] = np.asarray((subset > 0).sum(axis=0)).ravel().astype(float) / subset.shape[0]

    rows: list[pd.DataFrame] = []
    genes = np.asarray(adata.var_names.astype(str))
    tiny = np.finfo(float).tiny
    for group in groups:
        mask = labels == group
        subset = matrix[mask]
        n_group = subset.shape[0]
        n_rest = n_total - n_group
        sum_group = np.asarray(subset.sum(axis=0)).ravel().astype(float)
        sq_group = np.asarray(subset.multiply(subset).sum(axis=0)).ravel().astype(float)
        sum_rest = total_sum - sum_group
        sq_rest = total_sq - sq_group
        mean_group = sum_group / n_group
        mean_rest = sum_rest / n_rest
        var_group = np.maximum((sq_group - sum_group * sum_group / n_group) / max(n_group - 1, 1), 0.0)
        var_rest = np.maximum((sq_rest - sum_rest * sum_rest / n_rest) / max(n_rest - 1, 1), 0.0)
        se2 = var_group / n_group + var_rest / n_rest
        t_stat = np.divide(mean_group - mean_rest, np.sqrt(se2), out=np.zeros_like(mean_group), where=se2 > 0)
        df = np.divide(
            se2 * se2,
            (var_group / n_group) ** 2 / max(n_group - 1, 1) + (var_rest / n_rest) ** 2 / max(n_rest - 1, 1),
            out=np.full_like(se2, np.inf),
            where=se2 > 0,
        )
        pvalue = 2.0 * stats.t.sf(np.abs(t_stat), df)
        fdr = bh_adjust(pvalue)
        log2fc = np.log2((np.expm1(mean_group) + 1e-9) / (np.expm1(mean_rest) + 1e-9))
        pct_group = detected_by_group[group]
        pct_max_other = np.max(np.vstack([detected_by_group[g] for g in groups if g != group]), axis=0)
        table = pd.DataFrame(
            {
                "cell_type": group,
                "gene": genes,
                "log2FC_vs_all_other_cells": log2fc,
                "p_value_welch_t_on_log1p_cp10k": np.maximum(pvalue, tiny),
                "FDR_BH": fdr,
                "fraction_detected_in_cell_type": pct_group,
                "maximum_fraction_detected_in_any_other_cell_type": pct_max_other,
                "n_cells_in_cell_type": n_group,
                "n_cells_outside_cell_type": n_rest,
            }
        )
        table = table.loc[
            table["FDR_BH"].lt(0.05)
            & table["log2FC_vs_all_other_cells"].gt(0.25)
            & table["fraction_detected_in_cell_type"].ge(0.10)
        ].sort_values(["FDR_BH", "log2FC_vs_all_other_cells"], ascending=[True, False])
        table.insert(2, "marker_rank_within_cell_type", np.arange(1, len(table) + 1))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)

def build_state_markers(adata) -> pd.DataFrame:
    marker = adata.uns["markers_v5_full"]
    mapping = (
        adata.obs[["leiden_v5_full_harmony_r0_5", "cell_type_v5", "cell_state_v5"]]
        .drop_duplicates()
        .assign(leiden=lambda d: d["leiden_v5_full_harmony_r0_5"].astype(str))
        .set_index("leiden")
    )
    rows: list[pd.DataFrame] = []
    for cluster in marker["names"].dtype.names:
        genes = marker["names"][cluster].astype(str)
        table = pd.DataFrame(
            {
                "leiden_cluster": cluster,
                "cell_type": str(mapping.loc[cluster, "cell_type_v5"]),
                "cell_state": str(mapping.loc[cluster, "cell_state_v5"]),
                "gene": genes,
                "wilcoxon_score": marker["scores"][cluster].astype(float),
                "log2FC_vs_all_other_states": marker["logfoldchanges"][cluster].astype(float),
                "p_value": marker["pvals"][cluster].astype(float),
                "FDR_BH": marker["pvals_adj"][cluster].astype(float),
                "fraction_detected_in_state": marker["pts"].reindex(genes)[cluster].to_numpy(float),
                "fraction_detected_outside_state": marker["pts_rest"].reindex(genes)[cluster].to_numpy(float),
            }
        )
        table = table.loc[
            table["FDR_BH"].lt(0.05)
            & table["log2FC_vs_all_other_states"].gt(0.25)
            & table["fraction_detected_in_state"].ge(0.10)
        ].sort_values(["FDR_BH", "log2FC_vs_all_other_states"], ascending=[True, False])
        table.insert(4, "marker_rank_within_state", np.arange(1, len(table) + 1))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


INPUTS_BY_TASK = {'build': ['filtered_counts_h5ad', 'sample_manifest', 'gene_annotation'], 'markers': ['atlas_h5ad']}


def run(ctx, *, task="build") -> dict[str, str]:
    """Run one explicit stage in a new output root; preflight before heavy imports."""
    if task not in INPUTS_BY_TASK:
        raise ValueError(f"Unknown atlas task: {task}")
    ctx.require(INPUTS_BY_TASK[task])
    import anndata as ad
    if task == "markers":
        adata = ad.read_h5ad(ctx.input("atlas_h5ad"))
        primary = ~adata.obs["exclude_from_biology_v5"].astype(bool).to_numpy()
        broad = build_broad_markers(adata, primary)
        states = build_state_markers(adata)
        products = {}
        for name, table in [("broad_cell_markers", broad), ("state_markers", states)]:
            path = ctx.output("atlas/markers/" + name + ".tsv.gz")
            table.to_csv(path, sep="\t", index=False, compression="gzip")
            products[name] = str(path)
        return products
    if task == "audit":
        adata = ad.read_h5ad(ctx.input("atlas_h5ad"), backed="r")
        try:
            table = resolution_audit(adata.obs)
        finally:
            adata.file.close()
        path = ctx.output("atlas/audit/resolution_and_integration_sensitivity.tsv")
        table.to_csv(path, sep="\t", index=False)
        return {"resolution_audit": str(path)}
    input_key = "filtered_counts_h5ad" if task == "build" else "atlas_candidate_h5ad"
    adata = ad.read_h5ad(ctx.input(input_key))
    if task == "build":
        # Current atlas products are computed from the filtered counts supplied here.
        # A prior-atlas comparison is a separate internal audit, not a build input.
        refresh_manifest(adata, ctx.input("sample_manifest"))
        build_embedding(adata)
        rank_full_feature_markers(adata).to_csv(ctx.output("atlas/audit/cluster_markers_v5_full_top100.tsv.gz"), sep="\t", index=False)
        adata.write_h5ad(ctx.output("atlas/full_feature_candidate.h5ad"), compression="gzip")
    tables, summary = annotate(adata, ctx.input("sample_manifest"), read_table(ctx.input("gene_annotation"), sep="\t", dtype=str))
    for name, frame in tables.items():
        frame.to_csv(ctx.output("atlas/"+name), sep="\t", index=False)
    summary_path = ctx.output("atlas/audit/reannotation_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    output = ctx.output("atlas/full_feature_annotated.h5ad")
    adata.write_h5ad(output, compression="gzip")
    return {"atlas_h5ad": str(output), "annotation_summary": str(summary_path)}

