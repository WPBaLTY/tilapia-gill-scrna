# Tilapia gill analysis

Analysis code, reference inputs and recorded results for the Nile tilapia gill study submitted to *Marine Biotechnology*.

- `analysis.py`: 26 explicitly selected analysis stages and result export commands.
- `src/tilapia/`: quality control, atlas, pseudobulk, enrichment, state, regulatory and experimental-validation analyses.
- `inputs/`: 22 supplied input and reference files.
- `results/analysis_results.h5`: 158 tables and four metadata collections representing 295 scientific source files.
- `config.json`, `requirements.txt`, `manifest.json`: input contracts, dependency versions and SHA-256 file identities.

## Read recorded results

Use Python 3.12 with `h5py` installed. Result listing and export do not require the complete analysis environment.

```console
python analysis.py --results-list
python analysis.py --results-export de/apc_design_matrix_ref0 --format tsv --output ../apc_design_matrix.tsv
python analysis.py --results-export de/apc_design_matrix_ref0 --format csv --output ../apc_design_matrix.csv
```

Select an exact table ID from the listing. Exports contain the complete selected table, including source fields. Output must be a new file outside this package and its input trees. Import exported fields as text in spreadsheet software if exact decimal and missing-value strings are needed.

The master file groups results under `/tables/de`, `/tables/pathways`, `/tables/qc`, `/tables/regulatory`, `/tables/state` and `/tables/validation`. `/catalog_json` describes tables; `/source_index_json` maps original source identifiers to their rows and schemas. These identifiers are logical references inside HDF5, not missing files to download. Tables use ordered UTF-8 columns and compressed datasets; the wide count table uses a two-dimensional dataset. `__source_row` is zero-based and excludes the original header.

Retained source values, order, missing-value tokens and nonsignificant results are preserved. Distinct fits, biological experiments and sensitivity settings remain identifiable; repeated measurements are not additional biological replicates. The metabolism tables include a complete supporting sensitivity grid. The manuscript directly reports three ionocyte intersection modules; the additional module, membership and pavement grids are supporting results, not additional manuscript claims.

## Run analyses

Use Python 3.12.14 in a separate environment with the dependency versions pinned in `requirements.txt`.

```console
python analysis.py --list
python analysis.py --stage all --preflight
python analysis.py --stage qpcr:recompute --execute --output ../qpcr_run
```

The full workflow requires 11 external resources: provider count matrices, four H5AD checkpoints and six reference resources. Their exact paths, sizes, hashes, schemas and upstream producers are specified in `config.json`. Copy that file outside the package, set its external paths to the actual resources, then pass `--config ../local_config.json`. Bundled paths remain relative to the package; external relative paths resolve from the selected configuration file. No resource is downloaded or substituted automatically.

The workflow begins with filtered UMI count matrices. It does not process FASTQ files. The manuscript reports GEO **GSE348541**, samples **GSM10074946–GSM10074957**, private until 16 September 2027; reviewer access is supplied through submission. Exact matrix/checkpoint/reference access must be obtained from the authors. Remote availability has not been verified.

Each execution selects one concrete stage and writes to a new directory outside the package and inputs. Newly generated upstream products must be explicitly bound in a separate configuration before downstream use; no automatic end-to-end pipeline or automatic replacement of recorded results is provided. Preflight checks presence, hashes, declared schemas and installed-distribution metadata; it does not fit models.

## Processing details

The recorded inputs and code identify PISA `v0.12b-15-gb8ca969` in matrix headers, GO release `2026-06-15` in the ontology header, and Scrublet seed `20260811`. Embedding used 84,686 cells; the later rRNA flag excluded 43 cells from biological analyses, leaving 84,643. These version, seed and processing-order details need reconciliation with the manuscript Methods/S4 descriptions. Broad-class fits require at least 100 cells per pool; state fits require 30. All recorded broad-class fits meet the 100-cell criterion.

qPCR recomputation covers Ct statistics; primer and annotation evidence is provided in recorded assay metadata. Original primary and whole-gill per-gene fit flags were not retained, so missing adjusted P values are not retrospectively labelled as fit failures.

## GitHub storage

The HDF5 file exceeds GitHub's [100 MiB ordinary-file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). `.gitattributes` tracks it with [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). The supplied ZIP contains the complete data file. For a clone, install Git LFS and run `git lfs pull`; a pointer file alone cannot be read as results.

After installing Git LFS and creating an empty repository, run these commands in the package folder. Check that `git lfs ls-files` lists `results/analysis_results.h5` before committing.

```powershell
git init -b main
git lfs install
git add .gitattributes .gitignore README.md requirements.txt config.json manifest.json analysis.py src inputs results
git lfs ls-files
git commit -m "Add tilapia analysis and recorded results"
$repoUrl = Read-Host "Repository URL"
git remote add origin $repoUrl
git push -u origin main
```

No repository-wide license is specified. Third-party resource identifiers and source annotations are retained.
