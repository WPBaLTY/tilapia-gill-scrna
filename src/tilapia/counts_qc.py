"""Exact-reference import, per-library QC and the conservative filtering strategy.

QC records all robust/mitochondrial/doublet flags before filtering. The primary
filter is conservative_mt10; strict robust filtering is sensitivity only. The
historical 84,686 retained QC cells and later 84,643 analysis cells are distinct
checkpoints: this module does not introduce the later 43-cell exclusion.
"""
from __future__ import annotations
from .runtime import source_exists, source_label

import gc
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from .runtime import read_table, write_table, write_json, sha256, to_builtin

INPUTS_BY_TASK = {'reference': ['reference_gff'], 'import': ['matrix_root', 'sample_manifest', 'reference_map', 'reference_summary'], 'qc': ['raw_counts', 'reference_map'], 'filter': ['qc_counts']}

MITOCHONDRIAL_ACCESSION = 'NC_013663.1'
GENE_FEATURE_TYPES = {'gene', 'pseudogene'}
REQUIRED_FILES = {'barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz'}
QC_METRICS = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
QC_FLAG_COLUMNS = [
    'qc_low_genes', 'qc_high_genes', 'qc_low_counts', 'qc_high_counts',
    'qc_high_mt_robust', 'qc_high_mt_5pct', 'qc_high_mt_10pct',
    'qc_high_mt_20pct', 'qc_outlier_robust', 'predicted_doublet',
    'qc_keep_robust_no_doublet',
]



def parse_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for item in text.split(";"):
        key, separator, value = item.partition("=")
        if separator:
            attributes[unquote(key)] = unquote(value)
    return attributes


def extract_gene_id(attributes: dict[str, str]) -> str:
    for dbxref in attributes.get("Dbxref", "").split(","):
        if dbxref.startswith("GeneID:"):
            return dbxref.removeprefix("GeneID:")
    return attributes.get("ID", "").removeprefix("gene-")


def parse_gff(gff_path: Path) -> pd.DataFrame:
    records: list[dict[str, str | int | bool]] = []
    with gzip.open(gff_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in GENE_FEATURE_TYPES:
                continue
            attributes = parse_attributes(fields[8])
            gene_symbol = attributes.get("gene") or attributes.get("Name")
            if not gene_symbol:
                continue
            records.append(
                {
                    "gene_symbol": gene_symbol,
                    "gff_feature_type": fields[2],
                    "gene_id": extract_gene_id(attributes),
                    "gff_gene_id": attributes.get("ID", ""),
                    "chromosome": fields[0],
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "strand": fields[6],
                    "biotype": attributes.get("gene_biotype", ""),
                    "description": attributes.get("description", ""),
                    "gene_synonym": attributes.get("gene_synonym", ""),
                    "is_mitochondrial": fields[0] == MITOCHONDRIAL_ACCESSION,
                }
            )

    if not records:
        raise RuntimeError(f"No gene records were found in {gff_path}")
    annotation = pd.DataFrame.from_records(records)
    annotation = annotation.sort_values(
        ["gene_symbol", "chromosome", "start", "gff_gene_id"],
        kind="stable",
    ).reset_index(drop=True)
    annotation["symbol_match_count"] = annotation.groupby("gene_symbol")["gene_symbol"].transform("size")
    annotation["symbol_mapping_status"] = annotation["symbol_match_count"].map(
        lambda count: "unique" if count == 1 else "ambiguous"
    )
    return annotation


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def discover_libraries(matrix_root: Path) -> dict[str, Path]:
    libraries: dict[str, Path] = {}
    for directory, _, files in os.walk(matrix_root):
        if REQUIRED_FILES.issubset(files):
            path = Path(directory)
            library_id = path.name
            if library_id in libraries:
                raise RuntimeError(f"Duplicate library directory name: {library_id}")
            libraries[library_id] = path
    if not libraries:
        raise RuntimeError(f"No matrix directories found in {matrix_root}")
    return dict(sorted(libraries.items()))


def validate_manifest(manifest: pd.DataFrame, library_ids: set[str]) -> pd.DataFrame:
    if "library_id" not in manifest.columns:
        raise ValueError("Manifest must contain a library_id column")
    manifest["library_id"] = manifest["library_id"].astype(str)
    if manifest["library_id"].duplicated().any():
        duplicates = manifest.loc[manifest["library_id"].duplicated(), "library_id"].tolist()
        raise ValueError(f"Manifest has duplicate library_id values: {duplicates}")
    manifest_ids = set(manifest["library_id"])
    if manifest_ids != library_ids:
        missing = sorted(library_ids - manifest_ids)
        extra = sorted(manifest_ids - library_ids)
        raise ValueError(f"Manifest/library mismatch. Missing={missing}; extra={extra}")
    return manifest.set_index("library_id", verify_integrity=True)


def make_reference_lookup(reference: pd.DataFrame) -> pd.DataFrame:
    required = {
        "gene_symbol",
        "gene_id",
        "gff_gene_id",
        "chromosome",
        "biotype",
        "is_mitochondrial",
        "symbol_match_count",
        "symbol_mapping_status",
    }
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"Reference annotation is missing columns: {sorted(missing)}")
    reference = reference.copy()
    reference["gene_symbol"] = reference["gene_symbol"].astype(str)
    unique = reference.loc[reference["symbol_mapping_status"] == "unique"].copy()
    unique = unique.set_index("gene_symbol", verify_integrity=True)
    ambiguous = reference.loc[reference["symbol_mapping_status"] == "ambiguous", "gene_symbol"].unique()
    ambiguous_lookup = pd.DataFrame(index=pd.Index(ambiguous, name="gene_symbol"))
    ambiguous_lookup["symbol_mapping_status"] = "ambiguous"
    return unique, ambiguous_lookup


def load_library(
    library_id: str,
    library_dir: Path,
    sample_metadata: pd.Series,
    unique_reference: pd.DataFrame,
    ambiguous_reference: pd.DataFrame,
) -> tuple[ad.AnnData, dict[str, object]]:
    import anndata as ad
    features = read_gzip_lines(library_dir / "features.tsv.gz")
    barcodes = read_gzip_lines(library_dir / "barcodes.tsv.gz")
    if not features or not barcodes:
        raise ValueError(f"{library_id}: feature or barcode file is empty")
    field_counts = Counter(line.count("\t") + 1 for line in features)
    if set(field_counts) != {1}:
        raise ValueError(
            f"{library_id}: expected a one-column feature file, found field counts {dict(field_counts)}"
        )
    if len(set(features)) != len(features):
        duplicates = [gene for gene, count in Counter(features).items() if count > 1][:10]
        raise ValueError(f"{library_id}: duplicate feature symbols, e.g. {duplicates}")
    if len(set(barcodes)) != len(barcodes):
        raise ValueError(f"{library_id}: duplicated cell barcodes inside library")

    matrix = mmread(library_dir / "matrix.mtx.gz").tocsr().transpose().tocsr()
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"{library_id}: matrix shape {matrix.shape} does not match "
            f"{len(barcodes)} barcodes x {len(features)} features"
        )
    if matrix.data.size and (matrix.data.min() < 0 or not np.equal(matrix.data, np.floor(matrix.data)).all()):
        raise ValueError(f"{library_id}: matrix has non-integer or negative values")
    if matrix.data.size and matrix.data.max() > np.iinfo(np.int32).max:
        raise ValueError(f"{library_id}: count value exceeds int32 range")
    matrix = matrix.astype(np.int32, copy=False)

    obs = pd.DataFrame(index=pd.Index([f"{library_id}#{barcode}" for barcode in barcodes], name="cell_id"))
    obs["barcode"] = barcodes
    obs["library_id"] = library_id
    obs["matrix_directory"] = str(library_dir.resolve())
    for column, value in sample_metadata.items():
        if column != "library_id":
            obs[column] = value

    var = pd.DataFrame(index=pd.Index(features, name="gene_symbol"))
    var["gene_symbol"] = var.index
    var = var.join(unique_reference, how="left")
    ambiguous = var.index.isin(ambiguous_reference.index)
    var.loc[ambiguous, "symbol_mapping_status"] = "ambiguous"
    var["symbol_mapping_status"] = var["symbol_mapping_status"].fillna("unmatched")
    for column in ["gene_id", "gff_gene_id", "chromosome", "biotype", "is_mitochondrial", "symbol_match_count"]:
        if column not in var:
            var[column] = pd.NA
    var["is_mitochondrial"] = var["is_mitochondrial"].fillna(False).astype(bool)

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    library_summary = {
        "library_id": library_id,
        "matrix_directory": str(library_dir.resolve()),
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "total_umis": int(adata.X.sum()),
        "max_umi_per_entry": int(adata.X.data.max()) if adata.X.nnz else 0,
        "annotation_status_counts": {
            str(key): int(value) for key, value in var["symbol_mapping_status"].value_counts().items()
        },
        "mitochondrial_features": int(var["is_mitochondrial"].sum()),
    }
    return adata, library_summary


def build_union_var(
    var_names: pd.Index,
    unique_reference: pd.DataFrame,
    ambiguous_reference: pd.DataFrame,
) -> pd.DataFrame:
    union_var = pd.DataFrame(index=pd.Index(var_names, name="gene_symbol"))
    union_var["gene_symbol"] = union_var.index
    union_var = union_var.join(unique_reference, how="left")
    ambiguous = union_var.index.isin(ambiguous_reference.index)
    union_var.loc[ambiguous, "symbol_mapping_status"] = "ambiguous"
    union_var["symbol_mapping_status"] = union_var["symbol_mapping_status"].fillna("unmatched")
    union_var["is_mitochondrial"] = union_var["is_mitochondrial"].fillna(False).astype(bool)
    return union_var


def median_and_mad(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad


def refresh_reference_annotations(adata: ad.AnnData, reference_map: Path) -> None:
    reference = read_table(
        reference_map,
        dtype={"gene_id": str, "gff_gene_id": str},
    )
    counts = reference["gene_symbol"].astype(str).value_counts()
    unique_symbols = counts.index[counts.eq(1)]
    ambiguous_symbols = counts.index[counts.gt(1)]
    unique_reference = reference.loc[reference["gene_symbol"].isin(unique_symbols)].copy()
    unique_reference = unique_reference.set_index("gene_symbol", verify_integrity=True)
    var = pd.DataFrame(index=pd.Index(adata.var_names, name="gene_symbol"))
    var["gene_symbol"] = var.index
    var = var.join(unique_reference, how="left")
    ambiguous = var.index.isin(ambiguous_symbols)
    var.loc[ambiguous, "symbol_mapping_status"] = "ambiguous"
    var["symbol_mapping_status"] = var["symbol_mapping_status"].fillna("unmatched")
    var["is_mitochondrial"] = var["is_mitochondrial"].fillna(False).astype(bool)
    adata.var = var


def robust_thresholds(frame: pd.DataFrame, min_genes: int, nmads: float) -> dict[str, float]:
    log_genes = np.log1p(frame["n_genes_by_counts"].to_numpy(dtype=float))
    log_counts = np.log1p(frame["total_counts"].to_numpy(dtype=float))
    mt = frame["pct_counts_mt"].to_numpy(dtype=float)
    median_log_genes, mad_log_genes = median_and_mad(log_genes)
    median_log_counts, mad_log_counts = median_and_mad(log_counts)
    median_mt, mad_mt = median_and_mad(mt)
    return {
        "lower_n_genes": float(max(min_genes, np.expm1(median_log_genes - nmads * mad_log_genes))),
        "upper_n_genes": float(np.expm1(median_log_genes + nmads * mad_log_genes)),
        "lower_total_counts": float(max(1, np.expm1(median_log_counts - nmads * mad_log_counts))),
        "upper_total_counts": float(np.expm1(median_log_counts + nmads * mad_log_counts)),
        "upper_pct_counts_mt": float(median_mt + nmads * mad_mt),
        "median_n_genes": float(np.median(frame["n_genes_by_counts"])),
        "median_total_counts": float(np.median(frame["total_counts"])),
        "median_pct_counts_mt": median_mt,
        "mad_log1p_n_genes": mad_log_genes,
        "mad_log1p_total_counts": mad_log_counts,
        "mad_pct_counts_mt": mad_mt,
    }


def run_scrublet(
    adata: ad.AnnData,
    expected_doublet_rate: float,
    random_seed: int,
) -> dict[str, dict[str, float | int]]:
    import scanpy as sc
    adata.obs["doublet_score"] = np.nan
    adata.obs["predicted_doublet"] = False
    summaries: dict[str, dict[str, float | int]] = {}
    library_values = adata.obs["library_id"].astype(str)
    for offset, library_id in enumerate(sorted(library_values.unique())):
        mask = (library_values == library_id).to_numpy()
        subset = adata[mask].copy()
        subset.X = subset.layers["counts"]
        del subset.layers["counts"]
        expressed = np.asarray(subset.X.getnnz(axis=0)).ravel() > 0
        subset = subset[:, expressed].copy()
        sc.pp.scrublet(
            subset,
            expected_doublet_rate=expected_doublet_rate,
            random_state=random_seed + offset,
        )
        adata.obs.loc[mask, "doublet_score"] = subset.obs["doublet_score"].to_numpy(dtype=np.float32)
        adata.obs.loc[mask, "predicted_doublet"] = subset.obs["predicted_doublet"].to_numpy(dtype=bool)
        scores = subset.obs["doublet_score"].to_numpy(dtype=float)
        predictions = subset.obs["predicted_doublet"].to_numpy(dtype=bool)
        summaries[library_id] = {
            "n_cells": int(subset.n_obs),
            "n_expressed_features": int(subset.n_vars),
            "predicted_doublets": int(predictions.sum()),
            "predicted_doublet_fraction": float(predictions.mean()),
            "median_doublet_score": float(np.median(scores)),
            "max_doublet_score": float(np.max(scores)),
        }
        print(
            f"Scrublet {library_id}: {int(predictions.sum())}/{subset.n_obs} predicted doublets",
            flush=True,
        )
        del subset
        gc.collect()
    adata.obs["doublet_score"] = adata.obs["doublet_score"].astype(np.float32)
    adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype(bool)
    return summaries


def calculate_flags(
    adata: ad.AnnData,
    thresholds: dict[str, dict[str, float]],
) -> None:
    for column in [
        "qc_low_genes",
        "qc_high_genes",
        "qc_low_counts",
        "qc_high_counts",
        "qc_high_mt_robust",
    ]:
        adata.obs[column] = False
    library_values = adata.obs["library_id"].astype(str)
    for library_id, values in thresholds.items():
        mask = library_values == library_id
        adata.obs.loc[mask, "qc_low_genes"] = (
            adata.obs.loc[mask, "n_genes_by_counts"] < values["lower_n_genes"]
        )
        adata.obs.loc[mask, "qc_high_genes"] = (
            adata.obs.loc[mask, "n_genes_by_counts"] > values["upper_n_genes"]
        )
        adata.obs.loc[mask, "qc_low_counts"] = (
            adata.obs.loc[mask, "total_counts"] < values["lower_total_counts"]
        )
        adata.obs.loc[mask, "qc_high_counts"] = (
            adata.obs.loc[mask, "total_counts"] > values["upper_total_counts"]
        )
        adata.obs.loc[mask, "qc_high_mt_robust"] = (
            adata.obs.loc[mask, "pct_counts_mt"] > values["upper_pct_counts_mt"]
        )
    for cutoff in [5, 10, 20]:
        adata.obs[f"qc_high_mt_{cutoff}pct"] = adata.obs["pct_counts_mt"] > cutoff
    robust_columns = [
        "qc_low_genes",
        "qc_high_genes",
        "qc_low_counts",
        "qc_high_counts",
        "qc_high_mt_robust",
    ]
    adata.obs["qc_outlier_robust"] = adata.obs[robust_columns].any(axis=1)
    adata.obs["qc_keep_robust_no_doublet"] = ~(
        adata.obs["qc_outlier_robust"] | adata.obs["predicted_doublet"]
    )


def make_quantile_table(adata: ad.AnnData) -> pd.DataFrame:
    table = (
        adata.obs.groupby("library_id", observed=True)[QC_METRICS]
        .quantile(QUANTILES)
        .rename_axis(index=["library_id", "quantile"])
        .reset_index()
    )
    return table




def strategy_masks(obs: pd.DataFrame) -> dict[str, pd.Series]:
    provider_all = pd.Series(True, index=obs.index)
    common = obs["qc_low_genes"] | obs["qc_high_genes"] | obs["qc_high_counts"] | obs["predicted_doublet"]
    return {
        "provider_all": provider_all,
        "conservative_mt10": ~(common | obs["qc_high_mt_10pct"]),
        "moderate_mt5": ~(common | obs["qc_high_mt_5pct"]),
        "strict_robust": obs["qc_keep_robust_no_doublet"].astype(bool),
    }


def _prepare_outputs(*paths):
    for value in paths:
        path = Path(value)
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_counts(adata, output):
    output = Path(output)
    temporary = output.with_suffix(output.suffix + '.partial')
    if output.exists() or temporary.exists():
        raise FileExistsError(f'Output or unfinished write already exists: {output}')
    adata.write_h5ad(temporary, compression='gzip')
    os.replace(temporary, output)


def run_reference(gff_path, output, summary_path):
    """Build a symbol/GeneID table from the explicitly supplied gzip GFF."""
    gff_path, output, summary_path = map(Path, (gff_path, output, summary_path))
    if not gff_path.is_file():
        raise FileNotFoundError(gff_path)
    _prepare_outputs(output, summary_path)
    annotation = parse_gff(gff_path)
    write_table(annotation, output)
    symbol_counts = Counter(annotation['gene_symbol'])
    summary = {
        'gff_path': str(gff_path.resolve()), 'gff_sha256': sha256(gff_path),
        'mitochondrial_accession': MITOCHONDRIAL_ACCESSION,
        'gene_records': int(len(annotation)),
        'gff_feature_type_counts': {str(k): int(v) for k, v in annotation['gff_feature_type'].value_counts().items()},
        'unique_gene_symbols': int(annotation['gene_symbol'].nunique()),
        'ambiguous_gene_symbols': int(sum(count > 1 for count in symbol_counts.values())),
        'mitochondrial_gene_records': int(annotation['is_mitochondrial'].sum()),
        'mitochondrial_gene_symbols': int(annotation.loc[annotation['is_mitochondrial'], 'gene_symbol'].nunique()),
        'biotype_counts': {str(k): int(v) for k, v in annotation['biotype'].fillna('').value_counts().items()},
    }
    write_json(summary_path, summary)
    return summary


def run_import(matrix_root, manifest_path, reference_map, reference_summary_path, output, summary_path):
    """Import provider one-column matrices using an outer feature union."""
    import anndata as ad
    matrix_root, reference_map, reference_summary_path, output, summary_path = map(
        Path, (matrix_root, reference_map, reference_summary_path, output, summary_path))
    if not matrix_root.is_dir():
        raise NotADirectoryError(matrix_root)
    for path in (manifest_path, reference_map, reference_summary_path):
        if not source_exists(path):
            raise FileNotFoundError(path)
    _prepare_outputs(output, summary_path)
    libraries = discover_libraries(matrix_root)
    manifest = validate_manifest(read_table(manifest_path, dtype=str, keep_default_na=False), set(libraries))
    reference = read_table(reference_map, dtype={'gene_id': str, 'gff_gene_id': str})
    unique_reference, ambiguous_reference = make_reference_lookup(reference)
    loaded, library_summaries = [], []
    for library_id, library_dir in libraries.items():
        adata, summary = load_library(library_id, library_dir, manifest.loc[library_id], unique_reference, ambiguous_reference)
        loaded.append(adata)
        library_summaries.append(summary)
        print(f'Loaded {library_id}: {adata.n_obs} cells x {adata.n_vars} features')
    combined = ad.concat(loaded, axis=0, join='outer', merge=None, fill_value=0, index_unique=None)
    if not combined.obs_names.is_unique:
        raise RuntimeError('Combined cell IDs are not unique')
    if not sparse.isspmatrix_csr(combined.X):
        combined.X = combined.X.tocsr()
    combined.X = combined.X.astype(np.int32, copy=False)
    combined.var = build_union_var(combined.var_names, unique_reference, ambiguous_reference)
    combined.layers['counts'] = combined.X
    combined.obs['library_id'] = combined.obs['library_id'].astype('category')
    for column in ['company_sample', 'condition', 'sequencing_date', 'time_h']:
        if column in combined.obs:
            combined.obs[column] = combined.obs[column].astype('category')
    combined.strings_to_categoricals()
    reference_summary = json.loads(reference_summary_path.read_text(encoding='utf-8'))
    combined.uns['provenance'] = {
        'reference_accession': 'GCF_001858045.2',
        'reference_gff_sha256': reference_summary['gff_sha256'],
        'reference_mitochondrial_accession': reference_summary['mitochondrial_accession'],
        'feature_format': 'one-column gene symbols supplied by provider',
        'feature_join': 'outer with zero fill', 'manifest_path': source_label(manifest_path),
        'matrix_root': str(matrix_root.resolve()),
    }
    combined.uns['raw_import'] = {'library_summaries': {str(item['library_id']): item for item in library_summaries}}
    _write_counts(combined, output)
    summary = {
        'output_h5ad': str(output.resolve()), 'output_sha256': sha256(output),
        'n_cells': int(combined.n_obs), 'n_features_union': int(combined.n_vars),
        'nnz': int(combined.X.nnz), 'total_umis': int(combined.X.sum()),
        'library_cell_counts': combined.obs['library_id'].value_counts().sort_index().astype(int).to_dict(),
        'annotation_status_counts': {str(k): int(v) for k, v in combined.var['symbol_mapping_status'].value_counts().items()},
        'mitochondrial_features_in_union': int(combined.var['is_mitochondrial'].sum()),
        'reference_gff_sha256': reference_summary['gff_sha256'], 'libraries': library_summaries,
    }
    write_json(summary_path, summary)
    return summary


def summarize_qc(adata, thresholds, scrublet_summary, input_path, reference_map, output,
                 min_genes, nmads, expected_doublet_rate, random_seed):
    by_library = {}
    for library_id, frame in adata.obs.groupby('library_id', observed=True):
        by_library[str(library_id)] = {
            'n_cells': int(len(frame)),
            'metric_medians': {metric: float(frame[metric].median()) for metric in QC_METRICS},
            'flag_counts': {column: int(frame[column].sum()) for column in QC_FLAG_COLUMNS},
            'flag_fractions': {column: float(frame[column].mean()) for column in QC_FLAG_COLUMNS},
            'thresholds': thresholds[str(library_id)], 'scrublet': scrublet_summary[str(library_id)],
        }
    return {
        'input_h5ad': str(Path(input_path).resolve()), 'reference_map': str(Path(reference_map).resolve()),
        'output_h5ad': str(Path(output).resolve()), 'cells_evaluated': int(adata.n_obs),
        'genes_evaluated': int(adata.n_vars), 'mitochondrial_features_used': int(adata.var['mt'].sum()),
        'annotation_status_counts': {str(k): int(v) for k, v in adata.var['symbol_mapping_status'].value_counts().items()},
        'filtering_applied': False,
        'robust_threshold_method': {'transform_for_genes_and_counts': 'log1p',
            'mad_definition': 'median absolute deviation without Gaussian scaling',
            'nmads': nmads, 'minimum_gene_floor': min_genes},
        'scrublet_expected_doublet_rate': expected_doublet_rate, 'random_seed': random_seed,
        'overall_flag_counts': {column: int(adata.obs[column].sum()) for column in QC_FLAG_COLUMNS},
        'overall_flag_fractions': {column: float(adata.obs[column].mean()) for column in QC_FLAG_COLUMNS},
        'by_library': by_library,
    }


def run_qc(input_path, reference_map, output, summary_path, quantiles_path, thresholds_path,
           *, min_genes=200, nmads=3.0,
           expected_doublet_rate=0.06, random_seed=20260811):
    """Calculate per-cell flags and per-library Scrublet without filtering."""
    import anndata as ad
    import scanpy as sc
    input_path, reference_map = Path(input_path), Path(reference_map)
    for path in (input_path, reference_map):
        if not path.is_file():
            raise FileNotFoundError(path)
    _prepare_outputs(output, summary_path, quantiles_path, thresholds_path)
    adata = ad.read_h5ad(input_path)
    if 'counts' not in adata.layers:
        raise RuntimeError("Input AnnData lacks layers['counts']")
    if not sparse.issparse(adata.layers['counts']):
        raise RuntimeError('Counts layer must be sparse')
    adata.X = adata.layers['counts']
    refresh_reference_annotations(adata, reference_map)
    adata.var['mt'] = adata.var['is_mitochondrial'].astype(bool)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    scrublet_summary = run_scrublet(adata, expected_doublet_rate, random_seed)
    thresholds = {library_id: robust_thresholds(group, min_genes, nmads)
                  for library_id, group in adata.obs.groupby('library_id', observed=True)}
    calculate_flags(adata, thresholds)
    write_table(make_quantile_table(adata), quantiles_path)
    threshold_table = pd.DataFrame.from_dict(thresholds, orient='index')
    threshold_table.index.name = 'library_id'
    write_table(threshold_table, thresholds_path, index=True)
    summary = summarize_qc(adata, thresholds, scrublet_summary, input_path, reference_map, output,
                           min_genes, nmads, expected_doublet_rate, random_seed)
    adata.uns['qc_screening'] = to_builtin(summary)
    _write_counts(adata, output)
    write_json(summary_path, summary)
    return summary


def summarize_filter_sensitivity(obs, masks):
    rows = []
    for strategy, keep in masks.items():
        for library_id, library_mask in obs.groupby('library_id', observed=True).groups.items():
            selected = keep.loc[library_mask]
            rows.append({'strategy': strategy, 'library_id': str(library_id),
                'input_cells': int(len(selected)), 'kept_cells': int(selected.sum()),
                'removed_cells': int((~selected).sum()), 'kept_fraction': float(selected.mean())})
    return pd.DataFrame.from_records(rows)


def run_filter(input_path, output, summary_path, sensitivity_path, *, min_cells_per_gene=3):
    """Apply conservative_mt10 and retain all four strategy summaries."""
    import anndata as ad
    import scanpy as sc
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    _prepare_outputs(output, summary_path, sensitivity_path)
    adata = ad.read_h5ad(input_path)
    required = {'qc_low_genes', 'qc_high_genes', 'qc_high_counts', 'qc_high_mt_5pct',
                'qc_high_mt_10pct', 'qc_keep_robust_no_doublet', 'predicted_doublet'}
    if missing := required - set(adata.obs.columns):
        raise RuntimeError(f'Missing QC columns: {sorted(missing)}')
    masks = strategy_masks(adata.obs)
    primary_keep = masks['conservative_mt10']
    adata.obs['qc_primary_keep'] = primary_keep.to_numpy(dtype=bool)
    write_table(summarize_filter_sensitivity(adata.obs, masks), sensitivity_path)
    removal_reason_counts = {
        'low_genes_robust': int(adata.obs['qc_low_genes'].sum()),
        'high_genes_robust': int(adata.obs['qc_high_genes'].sum()),
        'high_counts_robust': int(adata.obs['qc_high_counts'].sum()),
        'mitochondrial_percent_over_10': int(adata.obs['qc_high_mt_10pct'].sum()),
        'scrublet_automatic_call': int(adata.obs['predicted_doublet'].sum()),
    }
    input_cells = int(adata.n_obs)
    filtered = adata[primary_keep].copy()
    filtered.X = filtered.layers['counts']
    before_gene_filter = int(filtered.n_vars)
    sc.pp.filter_genes(filtered, min_cells=min_cells_per_gene)
    filtered.X = filtered.X.astype(np.int32, copy=False)
    filtered.layers['counts'] = filtered.X
    if not sparse.isspmatrix_csr(filtered.X):
        filtered.X = filtered.X.tocsr()
        filtered.layers['counts'] = filtered.X
    filtered.uns['qc_filtering'] = {
        'primary_strategy': 'conservative_mt10',
        'rules': ['remove per-library robust low n_genes', 'remove per-library robust high n_genes',
            'remove per-library robust high total_counts', 'remove pct_counts_mt > 10',
            'remove Scrublet automatic predicted_doublet'],
        'strict_robust_mt_strategy_used_for_primary': False, 'min_cells_per_gene': min_cells_per_gene,
    }
    summary = {
        'input_h5ad': str(input_path.resolve()), 'output_h5ad': str(Path(output).resolve()),
        'primary_strategy': 'conservative_mt10', 'input_cells': input_cells,
        'kept_cells': int(filtered.n_obs), 'removed_cells': int(input_cells - filtered.n_obs),
        'kept_fraction': float(filtered.n_obs / input_cells),
        'removal_reason_counts_nonexclusive': removal_reason_counts, 'input_genes': before_gene_filter,
        'kept_genes_min_3_cells': int(filtered.n_vars), 'removed_genes': int(before_gene_filter-filtered.n_vars),
        'kept_cells_by_library': {str(k): int(v) for k,v in filtered.obs['library_id'].value_counts().sort_index().items()},
        'kept_cells_by_company_sample': {str(k): int(v) for k,v in filtered.obs['company_sample'].value_counts().sort_index().items()},
        'strict_robust_strategy_kept_cells': int(masks['strict_robust'].sum()),
        'strict_robust_strategy_is_sensitivity_only': True,
    }
    filtered.uns['qc_filtering']['summary'] = to_builtin(summary)
    _write_counts(filtered, output)
    write_json(summary_path, summary)
    return summary


def run(ctx, task='reference'):
    """Resolve inputs once; outputs are named checkpoints, never substitutes."""
    paths = ctx.require(INPUTS_BY_TASK[task])
    options = ctx.config.get('counts_qc', {})
    if task == 'reference':
        return run_reference(paths['reference_gff'], ctx.output('reference_map.tsv'), ctx.output('reference_summary.json'))
    if task == 'import':
        return run_import(paths['matrix_root'], paths['sample_manifest'], paths['reference_map'],
            paths['reference_summary'], ctx.output('01_raw_counts.h5ad'), ctx.output('import_summary.json'))
    if task == 'qc':
        return run_qc(paths['raw_counts'], paths['reference_map'], ctx.output('02_qc_metrics.h5ad'),
            ctx.output('qc_summary.json'), ctx.output('qc_quantiles.tsv'), ctx.output('qc_thresholds.tsv'),
            min_genes=options.get('min_genes', 200), nmads=options.get('nmads', 3.0),
            expected_doublet_rate=options.get('expected_doublet_rate', 0.06),
            random_seed=options.get('random_seed', 20260811))
    if task == 'filter':
        return run_filter(paths['qc_counts'], ctx.output('03_filtered_counts.h5ad'),
            ctx.output('filter_summary.json'), ctx.output('qc_strategy_sensitivity.tsv'),
            min_cells_per_gene=options.get('min_cells_per_gene', 3))
    raise ValueError(f'Unknown counts/QC task: {task}')

