"""Current TF evidence: transferred network, full-universe ULM/LOTO, target specificity, composition and identity sensitivity."""
from __future__ import annotations
from .runtime import read_table, source_member, iter_sources, source_provenance, source_size, source_exists, source_label, source_text, copy_source
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from datetime import datetime
from datetime import timezone
import sys
import platform
import itertools
import importlib.metadata as im
from scipy.stats import spearmanr
from .runtime import sha256 as sha
TMIN = 10

def prepared_y(values):
    y = np.asarray(values, dtype=np.float64)
    yc = y - y.mean(axis=1, keepdims=True)
    return (yc, (yc * yc).sum(axis=1))

def score(y, symbols, edges, min_targets=TMIN):
    """Same full-universe univariate regression t statistic as decoupler ULM.

    In x: target edges have the recorded signed weight; other genes have x=0.
    LOTO changes one edge to zero, not the gene-expression background.
    """
    sub = edges.loc[edges.target.isin(symbols)].copy()
    n = len(symbols)
    k = len(sub)
    yc, sy2 = prepared_y(y)
    index = pd.Index(symbols).get_indexer(sub.target)
    w = sub.weight.to_numpy(float)
    if k:
        cov = yc[:, index] @ w
        sx2 = (w * w).sum() - w.sum() ** 2 / n
        r = np.clip(cov / np.sqrt(sx2 * sy2), -1 + 1e-14, 1 - 1e-14) if sx2 > 0 else np.full(y.shape[0], np.nan)
        t = r * np.sqrt((n - 2) / (1 - r * r))
        signature = np.asarray(y)[:, index] @ w / np.abs(w).sum()
        contributions = yc[:, index] * w
    else:
        t = np.full(y.shape[0], np.nan)
        signature = t.copy()
        contributions = np.zeros((y.shape[0], 0))
    return dict(t=t if k >= min_targets else np.full(y.shape[0], np.nan), signature=signature, n=k, status='eligible' if k >= min_targets else 'insufficient_targets_tmin10', sub=sub, contributions=contributions, yc=yc, sy2=sy2, index=index, w=w)

def loto_scores(y, symbols, edges):
    full = score(y, symbols, edges)
    if full['n'] - 1 < TMIN:
        return (full, np.full((y.shape[0], full['n']), np.nan))
    w = full['w']
    n = len(symbols)
    cov = full['contributions'].sum(axis=1)
    remaining_cov = cov[:, None] - full['contributions']
    sx2 = (w * w).sum() - w * w - (w.sum() - w) ** 2 / n
    r = np.clip(remaining_cov / np.sqrt(full['sy2'][:, None] * sx2[None, :]), -1 + 1e-14, 1 - 1e-14)
    return (full, r * np.sqrt((n - 2) / (1 - r * r)))

class NetworkActivity:

    def __init__(self, ctx):
        self.ctx = ctx
        self.OUT = ctx.output('')
        self.HUMAN_NETWORK_FILE = ctx.input('collectri_human')
        self.NETWORK_FILE = ctx.output('collectri_ncbi_one_to_one_tilapia_targets.tsv.gz')
        self.ORTHOLOGY_FILE = ctx.input('strict_orthology')
        self.H5AD = ctx.input('atlas_h5ad')
        self.TMIN = 10
        self.POOL_ORDER = ['BR1', 'BR2', 'BR3', 'BR4', 'BR5', 'BR6']
        self.PRESPECIFIED_TFS = ['CEBPD', 'CEBPB', 'AP1', 'JUN', 'JUNB', 'FOS', 'ATF4', 'HIF1A', 'NFE2L2', 'XBP1', 'RELA', 'NFKB1', 'STAT3', 'TP53', 'RBPJ', 'KLF4', 'SOX4', 'MYB', 'ESRRG', 'FOXI1', 'FOXI2', 'HES1', 'PPARGC1A']

    def load_network(self) -> pd.DataFrame:
        human = read_table(self.HUMAN_NETWORK_FILE, sep='\t')
        human = human.copy()
        human['source'] = human['source'].astype(str).str.upper()
        human['target'] = human['target'].astype(str).str.upper()
        human['weight'] = pd.to_numeric(human['weight'], errors='coerce')
        human = human.dropna(subset=['source', 'target', 'weight'])
        human = human.groupby(['source', 'target'], as_index=False, observed=True)['weight'].mean()
        mapping = read_table(self.ORTHOLOGY_FILE, sep='\t')
        mapping = mapping.loc[mapping['strict_one_to_one'].astype(bool) & mapping['liana_compatible_feature'].astype(bool) & mapping['human_symbol'].notna() & mapping['tilapia_symbol'].notna()].copy()
        symbol_map = mapping.set_index(mapping['human_symbol'].astype(str).str.upper())['tilapia_symbol'].astype(str).str.upper()
        human['source_tilapia'] = human['source'].map(symbol_map)
        human['target_tilapia'] = human['target'].map(symbol_map)
        net = human.dropna(subset=['source_tilapia', 'target_tilapia']).copy()
        net = net.rename(columns={'source': 'source_human', 'target': 'target_human'})
        net['source'] = net['source_human']
        net['target'] = net['target_tilapia']
        net = net.groupby(['source', 'source_tilapia', 'target', 'target_human'], as_index=False, observed=True)['weight'].mean()
        net = net.loc[net['weight'].ne(0)].copy()
        net.to_csv(self.NETWORK_FILE, sep='\t', index=False, compression='gzip')
        return net

    def unique_gene_frame(self, frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
        work = frame[['gene', 'baseMean', *value_columns]].copy()
        work['symbol'] = work['gene'].astype(str).str.upper()
        work = work.sort_values(['symbol', 'baseMean'], ascending=[True, False], kind='mergesort')
        return work.drop_duplicates('symbol', keep='first').set_index('symbol')[value_columns]

    def run_contrast_activity(self, net: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        import decoupler as dc
        result_rows: list[pd.DataFrame] = []
        coverage_rows: list[dict[str, object]] = []
        for group, prefix in [('pavement cell', 'broad__pavement_cell'), ('ionocyte', 'broad__ionocyte')]:
            for contrast in ['24_vs_0', '96_vs_0', '96_vs_24']:
                de = read_table(source_member(self.ctx.input('de_results'), f'{prefix}__{contrast}.tsv.gz'), sep='\t')
                values = self.unique_gene_frame(de, ['stat', 'paired_log2cpm_diff_batch_1', 'paired_log2cpm_diff_batch_2'])
                values.columns = ['model', 'batch_1', 'batch_2']
                data = values.T
                acts, padj = dc.mt.ulm(data=data, net=net[['source', 'target', 'weight']], tmin=self.TMIN, empty=True)
                overlap = net.loc[net['target'].isin(data.columns)].groupby('source', observed=True)['target'].nunique()
                table = pd.DataFrame({'tf': acts.columns, 'activity_model': acts.loc['model'].to_numpy(), 'activity_batch_1': acts.loc['batch_1'].to_numpy(), 'activity_batch_2': acts.loc['batch_2'].to_numpy(), 'padj_model': padj.loc['model'].to_numpy()})
                table.insert(0, 'contrast', contrast)
                table.insert(0, 'cell_type', group)
                table['n_mapped_targets'] = table['tf'].map(overlap).fillna(0).astype(int)
                table['batch_same_direction'] = np.sign(table['activity_batch_1']).eq(np.sign(table['activity_batch_2'])) & np.sign(table['activity_batch_1']).ne(0)
                table['batch_direction_matches_model'] = table['batch_same_direction'] & np.sign(table['activity_model']).eq(np.sign(table['activity_batch_1']))
                table['direction_robust'] = table['padj_model'].lt(0.05) & table['batch_direction_matches_model']
                table['prespecified'] = table['tf'].isin(self.PRESPECIFIED_TFS)
                result_rows.append(table)
                coverage_rows.append({'cell_type': group, 'contrast': contrast, 'tested_matrix_symbols': int(data.shape[1]), 'network_targets_with_strict_ortholog_match': int(net['target'].isin(data.columns).sum()), 'unique_matched_target_symbols': int(net.loc[net['target'].isin(data.columns), 'target'].nunique()), 'eligible_regulators_tmin10': int(acts.shape[1])})
        all_results = pd.concat(result_rows, ignore_index=True)
        coverage = pd.DataFrame(coverage_rows)
        all_results.to_csv(self.OUT / 'contrast_level_collectri_ulm.tsv.gz', sep='\t', index=False, compression='gzip')
        coverage.to_csv(self.OUT / 'contrast_level_mapping_coverage.tsv', sep='\t', index=False)
        all_results.loc[all_results['prespecified']].to_csv(self.OUT / 'prespecified_tf_activity.tsv', sep='\t', index=False)
        all_results.loc[all_results['direction_robust']].sort_values(['cell_type', 'contrast', 'activity_model'], ascending=[True, True, False]).to_csv(self.OUT / 'all_direction_robust_tf_activity.tsv', sep='\t', index=False)
        return (all_results, coverage)

    def state_pool_activity(self, net: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        import decoupler as dc
        adata = ad.read_h5ad(self.H5AD)
        keep = ~adata.obs['exclude_from_biology_v5'].astype(bool).to_numpy()
        obs = adata.obs.loc[keep].copy()
        counts = adata.layers['counts'][keep]
        counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
        symbols = pd.Series(adata.var_names.astype(str).str.upper(), name='symbol')
        totals = np.asarray(counts.sum(axis=0)).ravel()
        gene_map = pd.DataFrame({'symbol': symbols, 'index': np.arange(adata.n_vars), 'total': totals})
        gene_map = gene_map.sort_values(['symbol', 'total'], ascending=[True, False], kind='mergesort').drop_duplicates('symbol')
        gene_map.sort_values('symbol', inplace=True)
        selected_idx = gene_map['index'].to_numpy(dtype=int)
        selected_symbols = gene_map['symbol'].to_numpy(dtype=str)
        sample_rows: list[np.ndarray] = []
        meta_rows: list[dict[str, object]] = []
        for cell_type in ['pavement cell', 'ionocyte']:
            states = sorted(obs.loc[obs['cell_type_v5'].astype(str).eq(cell_type), 'cell_state_v5'].astype(str).unique())
            for pool in self.POOL_ORDER:
                time_h = int(obs.loc[obs['biological_replicate'].astype(str).eq(pool), 'time_h'].iloc[0])
                for state in states:
                    mask = obs['biological_replicate'].astype(str).eq(pool).to_numpy() & obs['cell_state_v5'].astype(str).eq(state).to_numpy()
                    idx = np.flatnonzero(mask)
                    if len(idx) < 30:
                        continue
                    raw = np.asarray(counts[idx].sum(axis=0)).ravel()[selected_idx]
                    cpm = raw / max(raw.sum(), 1) * 1000000.0
                    sample_rows.append(np.log2(cpm + 0.5))
                    meta_rows.append({'sample': f'{cell_type}__{pool}__{state}', 'cell_type': cell_type, 'pool': pool, 'time_h': time_h, 'state': state, 'n_cells': int(len(idx))})
        expression = pd.DataFrame(sample_rows, index=[row['sample'] for row in meta_rows], columns=selected_symbols)
        meta = pd.DataFrame(meta_rows).set_index('sample')
        acts, _ = dc.mt.ulm(data=expression, net=net[['source', 'target', 'weight']], tmin=self.TMIN, empty=True)
        activity = acts.stack().rename('activity').reset_index().rename(columns={'level_0': 'sample', 'level_1': 'tf'})
        activity = activity.merge(meta.reset_index(), on='sample', how='left')
        activity['prespecified'] = activity['tf'].isin(self.PRESPECIFIED_TFS)
        activity.to_csv(self.OUT / 'state_pool_collectri_ulm.tsv.gz', sep='\t', index=False, compression='gzip')
        candidates = activity.loc[activity['prespecified']].copy()
        candidates['activity_z_within_cell_type_tf'] = candidates.groupby(['cell_type', 'tf'], observed=True)['activity'].transform(lambda values: (values - values.mean()) / values.std(ddof=0) if values.std(ddof=0) > 0 else 0)
        state_summary = candidates.groupby(['cell_type', 'state', 'tf'], observed=True).agg(median_activity=('activity', 'median'), median_activity_z=('activity_z_within_cell_type_tf', 'median'), n_pool_state_observations=('pool', 'nunique'), min_cells=('n_cells', 'min')).reset_index()
        state_summary.to_csv(self.OUT / 'prespecified_tf_state_localization.tsv', sep='\t', index=False)
        return (activity, state_summary)

    def analyze(self) -> None:
        import decoupler as dc
        self.OUT.mkdir(parents=True, exist_ok=True)
        net = self.load_network()
        all_results, coverage = self.run_contrast_activity(net)
        _, state_summary = self.state_pool_activity(net)
        matrix_symbols = {str(x).upper() for x in ad.read_h5ad(self.H5AD, backed='r').var_names}
        eligible = all_results.groupby('tf', observed=True)['n_mapped_targets'].max()
        manifest = {'method': 'decoupler ULM', 'decoupler_version': dc.__version__, 'network': 'human CollecTRI via OmniPath; complexes removed; regulator and targets transferred through NCBI one-to-one orthologs', 'network_interactions_after_deduplication': int(len(net)), 'network_regulators': int(net['source'].nunique()), 'network_targets': int(net['target'].nunique()), 'human_network_file_sha256': sha(self.HUMAN_NETWORK_FILE), 'orthology_file_sha256': sha(self.ORTHOLOGY_FILE), 'translated_network_content_sha256': hashlib.sha256(net.to_csv(sep='\t', index=False, lineterminator='\n').encode('utf-8')).hexdigest(), 'mapping': 'NCBI-computed one-to-one Nile tilapia--human ortholog for both TF and target; duplicate tilapia symbols resolved by highest baseMean/total count', 'matrix_symbols_matching_any_network_target': int(len(matrix_symbols & set(net['target']))), 'minimum_mapped_targets_per_regulator': self.TMIN, 'eligible_regulators_in_any_contrast': int((eligible >= self.TMIN).sum()), 'contrast_input': 'PyDESeq2 Wald statistic from ~ sequencing_batch + stage', 'direction_audit': 'ULM activity sign must agree in both batch-matched pool log2CPM-difference vectors and the model vector', 'state_input': 'log2(CPM + 0.5) from state-by-pool raw-count pseudobulk; minimum 30 cells', 'inference_scope': 'orthology-transferred TF-target activity hypothesis; not conserved regulatory edges, direct binding, perturbation, or causal evidence', 'prespecified_tfs': self.PRESPECIFIED_TFS, 'coverage_rows': coverage.to_dict(orient='records'), 'state_summary_rows': int(len(state_summary))}
        (self.OUT / 'tf_activity_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({key: manifest[key] for key in ['decoupler_version', 'network_interactions_after_deduplication', 'matrix_symbols_matching_any_network_target', 'eligible_regulators_in_any_contrast', 'state_summary_rows']}, indent=2))
        for group in ['pavement cell', 'ionocyte']:
            for contrast in ['24_vs_0', '96_vs_0', '96_vs_24']:
                view = all_results.loc[all_results['cell_type'].eq(group) & all_results['contrast'].eq(contrast) & all_results['prespecified']].sort_values('activity_model', key=lambda x: x.abs(), ascending=False)
                print(f'\n[{group} | {contrast}] prespecified')
                print(view[['tf', 'activity_model', 'padj_model', 'activity_batch_1', 'activity_batch_2', 'direction_robust', 'n_mapped_targets']].head(12).to_string(index=False))

class TargetSpecificity:

    def __init__(self, ctx):
        self.ctx = ctx
        self.OUT = ctx.output('')
        self.TMIN = TMIN
        self.COUNTS = ctx.input('state_pool_counts')
        self.META = ctx.input('state_pool_weights')
        self.TFS = ['RBPJ', 'SOX4', 'MYB', 'KLF4', 'RELA', 'NFKB1', 'PPARGC1A']
        self.CONTRASTS = [(24, 0), (96, 0), (96, 24)]
        self.VARIANTS = ['full', 'exclusive_selected7', 'remove_cycle', 'exclusive_selected7_remove_cycle']
        self.S_GENES = 'MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 MLF1IP HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8'.split()
        self.G2M_GENES = 'HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA'.split()
        self.CANONICAL = ['HES1', 'HES5', 'HEY1', 'HEY2', 'DTX1', 'NRARP']
        self.CANONICAL_SOURCE = {'HES1': '10.1093/nar/gkac601;PMC3734558', 'HEY1': '10.1093/nar/gkac601;PMC3734558', 'HES5': 'PMC3734558', 'HEY2': 'PMC3734558', 'DTX1': '10.1093/nar/gkac601', 'NRARP': '10.1093/nar/gkac601;PMC3734558'}
        self.MANIFEST = []
        self.VALIDATION = []

    def read(self, path):
        self.MANIFEST.append(source_provenance(path))
        return read_table(path, sep='\t')

    def save(self, df, name):
        df.to_csv(self.OUT / name, sep='\t', index=False, encoding='utf-8', float_format='%.12g')

    def check(self, label, value, passed):
        self.VALIDATION.append(dict(check=label, value=value, passed=bool(passed)))
        if not passed:
            raise AssertionError((label, value))

    def metadata_and_expression(self):
        meta = self.read(self.META)
        counts = self.read(self.COUNTS)
        self.check('count_rows_equal_metadata', len(counts), len(counts) == len(meta) == 60)
        self.check('all_feature_columns', counts.shape[1], counts.shape[1] == 37484)
        self.check('all_feature_uppercase_unique', counts.columns.str.upper().nunique(), counts.columns.str.upper().is_unique)
        self.check('count_row_sums_equal_metadata', float(np.max(np.abs(counts.sum(axis=1) - meta.total_umi))), np.allclose(counts.sum(axis=1), meta.total_umi))
        keep = meta.cell_type.eq('ionocyte').to_numpy()
        m = meta.loc[keep].reset_index(drop=True).copy()
        raw = counts.loc[keep].reset_index(drop=True)
        self.check('ionocyte_states_and_pools', len(m), len(m) == 18 and m.groupby('state').sample_id.nunique().eq(6).all())
        m['scope'] = 'state'
        m['row_id'] = m.sample_id + '__' + m.state
        broad_rows = []
        broad_raw = []
        for sample, inds in m.groupby('sample_id', sort=False).groups.items():
            mm = m.loc[inds]
            r = raw.loc[inds].sum(axis=0).to_numpy(float)
            broad_rows.append(dict(sample_id=sample, cell_type='ionocyte', state='all_ionocytes', time_h=int(mm.time_h.iloc[0]), sequencing_batch=mm.sequencing_batch.iloc[0], n_cells=int(mm.n_cells.sum()), total_umi=float(r.sum()), cell_fraction=1.0, umi_fraction=1.0, scope='broad', row_id=sample + '__all_ionocytes'))
            broad_raw.append(r)
        meta = pd.concat([m, pd.DataFrame(broad_rows)], ignore_index=True)
        raw = np.vstack([raw.values, np.asarray(broad_raw)])
        symbols = counts.columns.str.upper().to_numpy(str)
        cpm = raw / raw.sum(axis=1, keepdims=True) * 1000000.0
        expr = np.log2(cpm + 0.5)
        meta['mean_umi_per_cell'] = meta.total_umi / meta.n_cells
        self.save(meta, 'ionocyte_24_pseudobulk_metadata.tsv')
        pd.DataFrame(raw, columns=counts.columns).to_csv(self.OUT / 'ionocyte_24_all_feature_raw_counts.tsv.gz', sep='\t', index=False)
        np.savez_compressed(self.OUT / 'ionocyte_24_pseudobulk_arrays.npz', raw=raw, log2cpm=expr, symbols=symbols, rows=meta.row_id.to_numpy(str))
        return (meta, raw, cpm, expr, symbols, counts.columns.to_numpy(str))

    def analyze(self):
        self.OUT.mkdir(parents=True, exist_ok=True)
        plan = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), candidate_tfs=self.TFS[:-1], control='PPARGC1A', primary_comparison='96_vs_24', all_comparisons=['24_vs_0', '96_vs_0', '96_vs_24'], variants=self.VARIANTS, minimum_ulm_targets=self.TMIN, share_definition='Target belongs to >=2 among the frozen 7 TFs', cell_cycle_source='https://www.nature.com/articles/s41467-021-22640-z', cell_cycle_rule='Fixed published 43 S + 54 G2M genes, strict same frozen NCBI one-to-one mapping. Unmapped historical aliases not guessed.', canonical_notch_human=self.CANONICAL, canonical_caveat='mammalian external evidence, not directly validated tilapia targets', ulm_background='Original full gene universe remains unchanged when edges removed; additional no-cell-cycle-background result separately labelled', loto_rule='Every full-network mapped edge is removed once; preserve failure if <10 mapped targets remain', statistical_scope='Exploratory specificity diagnostics; 6 biological pools. No new P/FDR claims from subset/regulon tests. State-pools are not independent biological samples.', signed_signature='sum(weight*log2(CPM+.5))/sum(abs(weight)), explicitly NOT ULM', rna_scope='CPM is relative expression, not absolute transcript output; correlation with UMI/cell cannot establish causation')
        (self.OUT / 'FROZEN_ANALYSIS_PLAN.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf8')
        meta, raw, cpm, expr, symbols, features = self.metadata_and_expression()
        netall = self.read(self.ctx.input('tf_network'))
        human = self.read(self.ctx.input('collectri_human'))
        orth = self.read(self.ctx.input('strict_orthology'))
        orth = orth.loc[orth.strict_one_to_one.astype(bool) & orth.liana_compatible_feature.astype(bool) & orth.tilapia_symbol.notna() & orth.human_symbol.notna()].copy()
        orth['human_symbol'] = orth.human_symbol.str.upper()
        orth['tilapia_symbol'] = orth.tilapia_symbol.str.upper()
        smap = orth.set_index('human_symbol').tilapia_symbol.to_dict()
        cyc = pd.DataFrame(dict(human_symbol=self.S_GENES + self.G2M_GENES, phase=['S'] * len(self.S_GENES) + ['G2M'] * len(self.G2M_GENES)))
        self.check('fixed_cycle_gene_count', len(cyc), len(cyc) == 97)
        cyc['tilapia_symbol'] = cyc.human_symbol.map(smap)
        cyc['mapped'] = cyc.tilapia_symbol.isin(symbols)
        cyc['source'] = 'https://www.nature.com/articles/s41467-021-22640-z'
        self.save(cyc, 'fixed_cell_cycle_mapping.tsv')
        cycles = set(cyc.loc[cyc.mapped, 'tilapia_symbol'])
        canonical = pd.DataFrame(dict(human_symbol=self.CANONICAL))
        canonical['tilapia_symbol'] = canonical.human_symbol.map(smap)
        canonical['mapped'] = canonical.tilapia_symbol.isin(symbols)
        canonical['primary_literature'] = canonical.human_symbol.map(self.CANONICAL_SOURCE)
        canonical['evidence_scope'] = 'Mammalian direct binding/response evidence; tilapia target identity transferred by frozen one-to-one mapping'
        net = netall.loc[netall.source.isin(self.TFS)].copy()
        degree = net.groupby('target').source.nunique()
        net['n_selected7_regulators'] = net.target.map(degree)
        net['shared_selected7'] = net.n_selected7_regulators.gt(1)
        net['cell_cycle_target'] = net.target.isin(cycles)
        net['canonical_notch_target'] = net.target_human.isin(self.CANONICAL)
        self.save(net, 'all_frozen_candidate_target_edges.tsv')
        human_sub = human.loc[human.source.str.upper().isin(self.TFS)].copy()
        self.save(human_sub, 'all_original_human_candidate_edges.tsv.gz')
        overlap = []
        for a, b in itertools.combinations(self.TFS, 2):
            aa = set(net.loc[net.source == a, 'target'])
            bb = set(net.loc[net.source == b, 'target'])
            inter = aa & bb
            weights = net.set_index(['source', 'target']).weight
            overlap.append(dict(tf_a=a, tf_b=b, n_a=len(aa), n_b=len(bb), shared_n=len(inter), jaccard=len(inter) / len(aa | bb), concordant_weight_n=sum((np.sign(weights.loc[a, g]) == np.sign(weights.loc[b, g]) for g in inter)), shared_targets=';'.join(sorted(inter))))
        self.save(pd.DataFrame(overlap), 'candidate_network_pairwise_overlap.tsv')
        coverage = []
        edgevariants = {}
        for tf in self.TFS:
            base = net.loc[net.source == tf]
            for variant in self.VARIANTS:
                sub = base.copy()
                if 'exclusive' in variant:
                    sub = sub.loc[~sub.shared_selected7]
                if 'remove_cycle' in variant:
                    sub = sub.loc[~sub.cell_cycle_target]
                edgevariants[tf, variant] = sub
                coverage.append(dict(tf=tf, variant=variant, human_targets=human.loc[human.source.str.upper() == tf, 'target'].nunique(), original_mapped_targets=len(base), retained_mapped_targets=len(sub), state_eligible=len(sub) >= self.TMIN, removed_targets=';'.join(sorted(set(base.target) - set(sub.target)))))
        self.save(pd.DataFrame(coverage), 'target_coverage_by_sensitivity.tsv')
        print('COVERAGE\n', pd.DataFrame(coverage).to_string(index=False), flush=True)
        original_state = self.read(self.ctx.input('baseline_state_tf'))
        original_state = original_state.loc[original_state.cell_type.eq('ionocyte') & original_state.tf.isin(self.TFS)].copy()
        poolmap = {f'BR{i + 1}': p for i, p in enumerate(['CTL0h_1', 'CTL0h_2', 'ST24h_1', 'ST24h_2', 'ST96h_1', 'ST96h_2'])}
        original_state['sample_id'] = original_state.pool.map(poolmap)
        scorerows = []
        state_loto = []
        influence = []
        for tf in self.TFS:
            full, loto = loto_scores(expr, symbols, net.loc[net.source == tf])
            contrib = full['contributions']
            abs_sum = np.abs(contrib).sum(axis=1)
            for i, row in meta.iterrows():
                cs = np.abs(contrib[i])
                order = np.argsort(-cs)
                influence.append(dict(row_id=row.row_id, tf=tf, n_targets=full['n'], full_ulm=full['t'][i], top1_absolute_covariance_fraction=cs[order[0]] / abs_sum[i], top3_absolute_covariance_fraction=cs[order[:3]].sum() / abs_sum[i], effective_contributing_targets=abs_sum[i] ** 2 / (cs * cs).sum(), top_target=full['sub'].iloc[order[0]].target, loto_min=float(np.nanmin(loto[i])), loto_max=float(np.nanmax(loto[i])), loto_any_sign_flip=bool(np.any(np.sign(loto[i]) != np.sign(full['t'][i])))))
                for j, g in enumerate(full['sub'].target):
                    state_loto.append(dict(row_id=row.row_id, tf=tf, removed_target=g, full_ulm=full['t'][i], loto_ulm=loto[i, j], delta_ulm=loto[i, j] - full['t'][i], signed_covariance_contribution=contrib[i, j]))
            for variant in self.VARIANTS:
                result = score(expr, symbols, edgevariants[tf, variant])
                for i, row in meta.iterrows():
                    scorerows.append({**row.to_dict(), 'tf': tf, 'variant': variant, 'n_targets': result['n'], 'status': result['status'], 'ulm': result['t'][i], 'signed_mean_log2cpm_not_ulm': result['signature'][i]})
            keep = ~np.isin(symbols, list(cycles))
            result = score(expr[:, keep], symbols[keep], edgevariants[tf, 'remove_cycle'])
            for i, row in meta.iterrows():
                scorerows.append({**row.to_dict(), 'tf': tf, 'variant': 'remove_cycle_genes_from_background', 'n_targets': result['n'], 'status': result['status'], 'ulm': result['t'][i], 'signed_mean_log2cpm_not_ulm': result['signature'][i]})
        scores = pd.DataFrame(scorerows)
        self.save(scores, 'all_pool_state_sensitivity_scores.tsv')
        self.save(pd.DataFrame(state_loto), 'all_pool_state_leave_one_target_out.tsv.gz')
        self.save(pd.DataFrame(influence), 'pool_state_single_target_influence.tsv')
        rep = scores.loc[scores.scope.eq('state') & scores.variant.eq('full')].merge(original_state[['sample_id', 'state', 'tf', 'activity']], on=['sample_id', 'state', 'tf'], validate='1:1')
        rep['abs_error'] = abs(rep.ulm - rep.activity)
        self.save(rep, 'original_state_ULM_reproduction.tsv')
        self.check('state_ULM_reproduction_max_abs_error', float(rep.abs_error.max()), rep.abs_error.max() < 0.0001)
        scorechanges = []
        lotocontrasts = []
        for (state, tf, variant), sub in scores.groupby(['state', 'tf', 'variant'], sort=False):
            for a, b in self.CONTRASTS:
                out = dict(state=state, tf=tf, variant=variant, contrast=f'{a}_vs_{b}', n_targets=int(sub.n_targets.iloc[0]), status=sub.status.iloc[0])
                for col in ['ulm', 'signed_mean_log2cpm_not_ulm']:
                    deltas = []
                    for batch in ['batch_1', 'batch_2']:
                        av = sub.loc[sub.time_h.eq(a) & sub.sequencing_batch.eq(batch), col].iloc[0]
                        bv = sub.loc[sub.time_h.eq(b) & sub.sequencing_batch.eq(batch), col].iloc[0]
                        out[f'{col}_{batch}_difference'] = av - bv
                        deltas.append(av - bv)
                    out[f'{col}_mean_difference'] = float(np.mean(deltas))
                    out[f'{col}_both_batches_positive'] = bool(np.all(np.asarray(deltas) > 0))
                    out[f'{col}_both_batches_negative'] = bool(np.all(np.asarray(deltas) < 0))
                scorechanges.append(out)
        changes = pd.DataFrame(scorechanges)
        self.save(changes, 'all_state_and_broad_score_contrasts.tsv')
        loto = pd.DataFrame(state_loto).merge(meta, on='row_id', validate='m:1')
        for (state, tf, g), sub in loto.groupby(['state', 'tf', 'removed_target'], sort=False):
            for a, b in self.CONTRASTS:
                out = dict(state=state, tf=tf, removed_target=g, contrast=f'{a}_vs_{b}')
                for col in ['full_ulm', 'loto_ulm']:
                    deltas = []
                    for batch in ['batch_1', 'batch_2']:
                        av = sub.loc[sub.time_h.eq(a) & sub.sequencing_batch.eq(batch), col].iloc[0]
                        bv = sub.loc[sub.time_h.eq(b) & sub.sequencing_batch.eq(batch), col].iloc[0]
                        out[f'{col}_{batch}_difference'] = av - bv
                        deltas.append(av - bv)
                    out[f'{col}_mean_difference'] = float(np.mean(deltas))
                    out[f'{col}_both_batches_positive'] = bool(np.all(np.asarray(deltas) > 0))
                out['mean_direction_preserved'] = np.sign(out['full_ulm_mean_difference']) == np.sign(out['loto_ulm_mean_difference'])
                lotocontrasts.append(out)
        lotoc = pd.DataFrame(lotocontrasts)
        self.save(lotoc, 'all_state_and_broad_LOTO_contrasts.tsv.gz')
        ls = lotoc.groupby(['state', 'tf', 'contrast']).agg(n_loto=('removed_target', 'size'), full_mean_difference=('full_ulm_mean_difference', 'first'), loto_min_mean_difference=('loto_ulm_mean_difference', 'min'), loto_max_mean_difference=('loto_ulm_mean_difference', 'max'), fraction_mean_direction_preserved=('mean_direction_preserved', 'mean'), fraction_loto_both_batches_positive=('loto_ulm_both_batches_positive', 'mean')).reset_index()
        self.save(ls, 'state_and_broad_LOTO_stability_summary.tsv')
        old = self.read(self.ctx.input('baseline_tf_activity'))
        contrast_scores = []
        contrast_loto = []
        contrast_genes = []
        contrast_rep = []
        contrast_influence = []
        for a, b in self.CONTRASTS:
            contrast = f'{a}_vs_{b}'
            de = self.read(source_member(self.ctx.input('de_results'), f'broad__ionocyte__{contrast}.tsv.gz'))
            de['symbol'] = de.gene.str.upper()
            de = de.sort_values(['symbol', 'baseMean'], ascending=[True, False], kind='mergesort').drop_duplicates('symbol').set_index('symbol')
            cols = ['stat', 'paired_log2cpm_diff_batch_1', 'paired_log2cpm_diff_batch_2']
            data = de[cols].T.values
            genes = de.index.to_numpy(str)
            for tf in self.TFS:
                base = net.loc[net.source == tf]
                full, loo = loto_scores(data, genes, base)
                prior = old.loc[old.cell_type.eq('ionocyte') & old.contrast.eq(contrast) & old.tf.eq(tf)]
                if len(prior):
                    for j, oldcol in enumerate(['activity_model', 'activity_batch_1', 'activity_batch_2']):
                        contrast_rep.append(dict(tf=tf, contrast=contrast, vector=cols[j], old_score=float(prior[oldcol].iloc[0]), new_score=full['t'][j], abs_error=abs(full['t'][j] - prior[oldcol].iloc[0])))
                for variant in self.VARIANTS:
                    result = score(data, genes, edgevariants[tf, variant])
                    contrast_scores.append(dict(tf=tf, contrast=contrast, variant=variant, n_targets=result['n'], status=result['status'], ulm_model=result['t'][0], ulm_batch1=result['t'][1], ulm_batch2=result['t'][2], original_FDR=float(prior.padj_model.iloc[0]) if variant == 'full' and len(prior) else np.nan, all_three_positive=bool(np.all(result['t'] > 0)), all_three_negative=bool(np.all(result['t'] < 0))))
                for j, g in enumerate(full['sub'].target):
                    row = full['sub'].iloc[j]
                    dr = de.loc[g]
                    contrast_genes.append(dict(tf=tf, contrast=contrast, target=g, target_human=row.target_human, weight=row.weight, shared_selected7=row.shared_selected7, cell_cycle_target=row.cell_cycle_target, canonical_notch_target=row.canonical_notch_target, wald_stat=dr.stat, shrunken_log2FC=dr.log2FoldChange_shrunken, gene_FDR=dr.padj, batch1_log2CPM_difference=dr.paired_log2cpm_diff_batch_1, batch2_log2CPM_difference=dr.paired_log2cpm_diff_batch_2, signed_covariance_model=full['contributions'][0, j], both_batches_match_positive_tf=bool(dr.paired_log2cpm_diff_batch_1 * row.weight > 0 and dr.paired_log2cpm_diff_batch_2 * row.weight > 0)))
                    contrast_loto.append(dict(tf=tf, contrast=contrast, removed_target=g, full_ulm_model=full['t'][0], loto_ulm_model=loo[0, j], loto_ulm_batch1=loo[1, j], loto_ulm_batch2=loo[2, j], all_three_positive=bool(np.all(loo[:, j] > 0)), all_three_negative=bool(np.all(loo[:, j] < 0)), model_sign_preserved=bool(np.sign(loo[0, j]) == np.sign(full['t'][0]))))
                cs = np.abs(full['contributions'][0])
                order = np.argsort(-cs)
                contrast_influence.append(dict(tf=tf, contrast=contrast, n_targets=full['n'], top_target=full['sub'].iloc[order[0]].target, top1_absolute_covariance_fraction=cs[order[0]] / cs.sum(), top3_absolute_covariance_fraction=cs[order[:3]].sum() / cs.sum(), effective_contributing_targets=cs.sum() ** 2 / (cs * cs).sum(), loto_min_model=float(np.nanmin(loo[0])), loto_max_model=float(np.nanmax(loo[0])), fraction_loto_all_three_positive=float(np.mean(np.all(loo > 0, axis=0)))))
        self.save(pd.DataFrame(contrast_scores), 'contrast_Wald_ULM_sensitivities.tsv')
        self.save(pd.DataFrame(contrast_loto), 'contrast_Wald_ULM_leave_one_target_out.tsv.gz')
        self.save(pd.DataFrame(contrast_genes), 'all_target_frozen_DE_evidence.tsv')
        self.save(pd.DataFrame(contrast_influence), 'contrast_Wald_ULM_single_target_influence.tsv')
        cr = pd.DataFrame(contrast_rep)
        self.save(cr, 'original_contrast_ULM_reproduction.tsv')
        self.check('contrast_ULM_reproduction_max_abs_error', float(cr.abs_error.max()), cr.abs_error.max() < 0.0001)
        selected = sorted(set(net.target) | set(canonical.loc[canonical.mapped, 'tilapia_symbol']) | cycles)
        indices = pd.Index(symbols).get_indexer(selected)
        self.check('selected_targets_present', int((indices < 0).sum()), (indices >= 0).all())
        long = []
        for i, row in meta.iterrows():
            for g, j in zip(selected, indices):
                long.append({**row.to_dict(), 'target': g, 'matrix_feature': features[j], 'raw_umi': raw[i, j], 'CPM': cpm[i, j], 'log2_CPM_plus_0_5': expr[i, j], 'candidate_regulators': ';'.join(net.loc[net.target.eq(g), 'source']), 'cell_cycle_target': g in cycles, 'canonical_notch_target': g in set(canonical.tilapia_symbol)})
        expression = pd.DataFrame(long)
        self.save(expression, 'all_targets_24_pool_state_expression.tsv.gz')
        genechanges = []
        for (state, g), sub in expression.groupby(['state', 'target'], sort=False):
            for a, b in self.CONTRASTS:
                out = dict(state=state, target=g, contrast=f'{a}_vs_{b}')
                deltas = []
                for batch in ['batch_1', 'batch_2']:
                    av = sub.loc[sub.time_h.eq(a) & sub.sequencing_batch.eq(batch), 'log2_CPM_plus_0_5'].iloc[0]
                    bv = sub.loc[sub.time_h.eq(b) & sub.sequencing_batch.eq(batch), 'log2_CPM_plus_0_5'].iloc[0]
                    out[batch + '_difference'] = av - bv
                    deltas.append(av - bv)
                out['mean_difference'] = float(np.mean(deltas))
                out['both_batches_positive'] = bool(np.all(np.asarray(deltas) > 0))
                out['both_batches_negative'] = bool(np.all(np.asarray(deltas) < 0))
                genechanges.append(out)
        gchange = pd.DataFrame(genechanges)
        self.save(gchange, 'all_target_state_and_broad_expression_contrasts.tsv.gz')
        evidence = pd.DataFrame(contrast_genes)
        summ = []
        for (tf, contrast), sub in evidence.groupby(['tf', 'contrast'], sort=False):
            for label, ss in [('all', sub), ('exclusive', sub.loc[~sub.shared_selected7]), ('noncycle', sub.loc[~sub.cell_cycle_target]), ('exclusive_noncycle', sub.loc[~sub.shared_selected7 & ~sub.cell_cycle_target])]:
                support = ss.both_batches_match_positive_tf
                summ.append(dict(tf=tf, contrast=contrast, subset=label, n_targets=len(ss), both_batch_signed_positive_n=int(support.sum()), positive_signed_Wald_n=int((ss.wald_stat * ss.weight > 0).sum()), geneFDR05_and_both_batch_signed_positive_n=int((ss.gene_FDR.lt(0.05) & support & (ss.shrunken_log2FC * ss.weight > 0)).sum()), both_batch_supporting_targets=';'.join(ss.loc[support, 'target']), FDR05_both_batch_supporting_targets=';'.join(ss.loc[ss.gene_FDR.lt(0.05) & support & (ss.shrunken_log2FC * ss.weight > 0), 'target'])))
        self.save(pd.DataFrame(summ), 'distributed_target_support_summary.tsv')
        self.save(pd.DataFrame(self.MANIFEST).drop_duplicates('path'), 'input_SHA256_manifest.tsv')
        self.save(pd.DataFrame(self.VALIDATION), 'validation_checks.tsv')
        versions = {p: im.version(p) for p in ['numpy', 'pandas', 'scipy', 'decoupler', 'anndata']}
        versions.update(python=sys.version, platform=platform.platform(), TFs=self.TFS, analyses_complete=True, n_ionocyte_cells=int(meta.loc[meta.scope == 'state', 'n_cells'].sum()), n_biological_pools=6, n_state_pool_rows=18, no_new_inferential_tests=True, cell_cycle_genes_mapped=len(cycles))
        (self.OUT / 'run_manifest.json').write_text(json.dumps(versions, indent=2), encoding='utf8')
        print('LATE CONTRAST\n', pd.DataFrame(contrast_scores).query("contrast=='96_vs_24'").to_string(index=False), flush=True)

    def focused_followup(self):
        frozen = dict(reason='Assess PPARGC1A state contrast stability and KLF4 aggregate/state weighting sensitivity', tfs=['PPARGC1A', 'KLF4'], contrast='96_vs_24', variants=['full', 'exclusive_selected7_remove_cycle'], weighting_sensitivity=['observed_umi_weights', 'fixed_0h_mean_umi_weights', 'equal_state_weights'], rule='All genes combined as CPM before log2(CPM+.5) and ULM. Fixed weights are a descriptive synthetic expression profile, not TF-score decomposition.', inferred_unit='6 biological pools; 2 per stage; no new P/FDR')
        (self.OUT / 'FOCUSED_FOLLOWUP_PLAN.json').write_text(json.dumps(frozen, indent=2), encoding='utf8')
        a = np.load(self.OUT / 'ionocyte_24_pseudobulk_arrays.npz')
        expr = a['log2cpm']
        raw = a['raw']
        symbols = a['symbols']
        rows = a['rows']
        meta = read_table(self.OUT / 'ionocyte_24_pseudobulk_metadata.tsv', sep='\t')
        net = read_table(self.OUT / 'all_frozen_candidate_target_edges.tsv', sep='\t')
        assert list(meta.row_id) == list(rows)

        def edges(tf, variant):
            sub = net.loc[net.source.eq(tf)]
            if variant != 'full':
                sub = sub.loc[~sub.shared_selected7 & ~sub.cell_cycle_target]
            return sub
        detailed = []
        summary = []
        samplevals = []
        for tf in frozen['tfs']:
            for variant in frozen['variants']:
                res, loo = loto_scores(expr, symbols, edges(tf, variant))
                for state in meta.state.unique():
                    sel = meta.state.eq(state)
                    full_d = []
                    loo_d = []
                    for batch in ['batch_1', 'batch_2']:
                        i = meta.index[sel & meta.sequencing_batch.eq(batch) & meta.time_h.eq(96)][0]
                        j = meta.index[sel & meta.sequencing_batch.eq(batch) & meta.time_h.eq(24)][0]
                        full_d.append(res['t'][i] - res['t'][j])
                        loo_d.append(loo[i] - loo[j])
                        samplevals.append(dict(tf=tf, variant=variant, state=state, batch=batch, n_targets=res['n'], ulm_24h=res['t'][j], ulm_96h=res['t'][i], difference=res['t'][i] - res['t'][j], signed_mean_24h=res['signature'][j], signed_mean_96h=res['signature'][i], signed_mean_difference=res['signature'][i] - res['signature'][j]))
                    ld = np.array(loo_d)
                    fd = np.array(full_d)
                    both_preserved = np.all(np.sign(ld) == np.sign(fd[:, None]), axis=0)
                    for k, g in enumerate(res['sub'].target):
                        detailed.append(dict(tf=tf, variant=variant, state=state, removed_target=g, n_targets_after_omission=res['n'] - 1, full_batch1_difference=fd[0], full_batch2_difference=fd[1], loto_batch1_difference=ld[0, k], loto_batch2_difference=ld[1, k], both_batch_directions_preserved=both_preserved[k]))
                    summary.append(dict(tf=tf, variant=variant, state=state, n_targets=res['n'], batch1_difference=fd[0], batch2_difference=fd[1], n_loto=res['n'], n_loto_both_batch_directions_preserved=int(both_preserved.sum()), all_loto_both_batch_directions_preserved=bool(both_preserved.all()), loto_batch1_min=float(ld[0].min()), loto_batch1_max=float(ld[0].max()), loto_batch2_min=float(ld[1].min()), loto_batch2_max=float(ld[1].max()), destabilizing_targets=';'.join(res['sub'].target[~both_preserved])))
        self.save(pd.DataFrame(samplevals), 'focused_PPAR_KLF4_state_score_levels_and_differences.tsv')
        self.save(pd.DataFrame(summary), 'focused_PPAR_KLF4_state_LOTO_summary.tsv')
        self.save(pd.DataFrame(detailed), 'focused_PPAR_KLF4_state_LOTO_all.tsv.gz')
        sm = meta.loc[meta.scope.eq('state')].copy()
        reference = sm.loc[sm.time_h.eq(0)].groupby('state').umi_fraction.mean()
        weights = reference.rename('fixed_0h_mean_umi_weight').reset_index()
        weights['equal_state_weight'] = 1 / 3
        self.save(weights, 'focused_fixed_state_weights.tsv')
        profiles = []
        pm = []
        for sample, inds in sm.groupby('sample_id').groups.items():
            s = meta.loc[inds]
            statecpm = raw[inds] / raw[inds].sum(axis=1, keepdims=True) * 1000000.0
            for mode in frozen['weighting_sensitivity']:
                if mode == 'observed_umi_weights':
                    w = s.umi_fraction.to_numpy(float)
                elif mode == 'fixed_0h_mean_umi_weights':
                    w = reference.reindex(s.state).to_numpy(float)
                else:
                    w = np.repeat(1 / 3, 3)
                reconstructed = w @ statecpm
                if mode == 'observed_umi_weights':
                    idx = meta.index[meta.scope.eq('broad') & meta.sample_id.eq(sample)][0]
                    assert np.allclose(reconstructed, raw[idx] / raw[idx].sum() * 1000000.0, atol=1e-06)
                profiles.append(np.log2(reconstructed + 0.5))
                pm.append(dict(sample_id=sample, time_h=int(s.time_h.iloc[0]), sequencing_batch=s.sequencing_batch.iloc[0], weight_mode=mode))
        pm = pd.DataFrame(pm)
        profiles = np.array(profiles)
        scored = []
        deltas = []
        for tf in frozen['tfs']:
            for variant in frozen['variants']:
                res = score(profiles, symbols, edges(tf, variant))
                for i, row in pm.iterrows():
                    scored.append({**row.to_dict(), 'tf': tf, 'variant': variant, 'n_targets': res['n'], 'ulm_recomputed_from_mixed_gene_CPM': res['t'][i], 'signed_mean_log2CPM_not_ULM': res['signature'][i]})
        scored = pd.DataFrame(scored)
        self.save(scored, 'focused_gene_profile_weight_sensitivity_scores.tsv')
        for (tf, variant, mode), sub in scored.groupby(['tf', 'variant', 'weight_mode'], sort=False):
            out = dict(tf=tf, variant=variant, weight_mode=mode)
            for metric in ['ulm_recomputed_from_mixed_gene_CPM', 'signed_mean_log2CPM_not_ULM']:
                dd = []
                for batch in ['batch_1', 'batch_2']:
                    av = sub.loc[sub.time_h.eq(96) & sub.sequencing_batch.eq(batch), metric].iloc[0]
                    bv = sub.loc[sub.time_h.eq(24) & sub.sequencing_batch.eq(batch), metric].iloc[0]
                    out[metric + '_' + batch + '_difference'] = av - bv
                    dd.append(av - bv)
                out[metric + '_mean_difference'] = np.mean(dd)
            deltas.append(out)
        self.save(pd.DataFrame(deltas), 'focused_gene_profile_weight_sensitivity_contrasts.tsv')
        print(pd.DataFrame(summary).to_string(index=False))
        print(pd.DataFrame(deltas).to_string(index=False))

    def summarize_outputs(self):
        scores = read_table(self.OUT / 'contrast_Wald_ULM_sensitivities.tsv', sep='\t')
        key = scores.loc[scores.contrast.eq('96_vs_24') & scores.variant.isin(['full', 'exclusive_selected7_remove_cycle'])]
        self.save(key, 'KEY_late_contrast_ULM_summary.tsv')
        levels = read_table(self.OUT / 'focused_PPAR_KLF4_state_score_levels_and_differences.tsv', sep='\t')
        summary = read_table(self.OUT / 'focused_PPAR_KLF4_state_LOTO_summary.tsv', sep='\t')
        detailed = read_table(self.OUT / 'focused_PPAR_KLF4_state_LOTO_all.tsv.gz', sep='\t')
        weights = read_table(self.OUT / 'focused_fixed_state_weights.tsv', sep='\t')
        checks = [dict(check='focused_score_rows_2TF_2variants_4states_2batches', value=len(levels), passed=len(levels) == 32), dict(check='focused_LOTO_summary_rows', value=len(summary), passed=len(summary) == 16), dict(check='focused_LOTO_complete_rows', value=len(detailed), passed=len(detailed) == int(summary.n_loto.sum())), dict(check='focused_target_count_minimum_after_omission', value=int(detailed.n_targets_after_omission.min()), passed=detailed.n_targets_after_omission.ge(10).all()), dict(check='focused_all_LOTO_differences_finite', value=int(detailed[['loto_batch1_difference', 'loto_batch2_difference']].isna().sum().sum()), passed=not detailed[['loto_batch1_difference', 'loto_batch2_difference']].isna().any().any()), dict(check='fixed_0h_weights_sum', value=weights.fixed_0h_mean_umi_weight.sum(), passed=abs(weights.fixed_0h_mean_umi_weight.sum() - 1) < 1e-08)]
        self.save(pd.DataFrame(checks), 'focused_validation_checks.tsv')
        assert all((c['passed'] for c in checks))
        entries = []
        for path in sorted(self.OUT.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name == 'OUTPUT_SHA256.tsv':
                continue
            entries.append(dict(relative_path=path.relative_to(self.OUT).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
        self.save(pd.DataFrame(entries), 'OUTPUT_SHA256.tsv')
        for entry in entries:
            path = self.OUT / entry['relative_path']
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        print(json.dumps(dict(files_hashed=len(entries), bytes_hashed=sum((e['bytes'] for e in entries)), focused_checks_passed=len(checks), all_output_hashes_verified=True, manifest_sha256=sha(self.OUT / 'OUTPUT_SHA256.tsv')), indent=2))

class TargetIdentityCheck:

    def __init__(self, ctx):
        self.ctx = ctx
        self.BASE = ctx.output('')
        self.OUT = ctx.output('identity_crosscheck')
        self.OUT.mkdir(exist_ok=True)
        self.TFS = ['RBPJ', 'SOX4', 'MYB', 'KLF4', 'RELA', 'NFKB1', 'PPARGC1A']
        self.INPUTS = []

    def read(self, path, **kwargs):
        self.INPUTS.append(source_provenance(path))
        return read_table(path, sep='\t', **kwargs)

    def save(self, df, name):
        df.to_csv(self.OUT / name, sep='\t', index=False, float_format='%.12g')

    def analyze(self):
        audit = self.read(self.ctx.input('recovered_geneid_audit'), dtype=str).fillna('')
        loc = self.read(self.ctx.input('loc_conflicts'), dtype=str).fillna('')
        dis = self.read(self.ctx.input('geneid_disagreements'), dtype=str).fillna('')
        orth = self.read(self.ctx.input('strict_orthology'), dtype=str).fillna('')
        net = self.read(self.BASE / 'all_frozen_candidate_target_edges.tsv')
        orth = orth.loc[orth.strict_one_to_one.eq('True') & orth.liana_compatible_feature.eq('True')].copy()
        orth['tilapia_symbol'] = orth.tilapia_symbol.str.upper()
        orth['human_symbol'] = orth.human_symbol.str.upper()
        mapping = orth.groupby(['human_symbol', 'tilapia_symbol']).tilapia_gene_id.agg(lambda s: ';'.join(sorted(set(s))))
        a = audit.set_index('symbol_upper')
        assert a.index.is_unique
        locset = set(loc.matrix_feature.str.upper())
        disset = set(dis.gene.str.upper())

        def identity(human, gene):
            row = a.loc[gene] if gene in a.index else pd.Series(dtype=str)
            old = mapping.get((human, gene), '')
            exact = row.get('gene_id', '')
            collision = row.get('mapping_status', '') == 'excluded_multifeature_geneid_collision'
            disagreement = gene in disset or row.get('eggnog_geneid_status', '') == 'ncbi_eggnog_geneid_disagreement'
            mismatch = bool(old and exact and (old != exact))
            unresolved = not old or not exact or ';' in old
            flags = []
            if collision:
                flags.append('multiple_features_claim_same_GeneID')
            if disagreement:
                flags.append('native_eggNOG_GeneID_disagreement')
            if mismatch:
                flags.append('network_precise_GeneID_mismatch')
            if unresolved:
                flags.append('unresolved_precise_or_network_GeneID')
            bridge = row.get('eggnog_bridge_geneids', '')
            return dict(network_gene_id=old, precise_gene_id=exact, eggnog_bridge_geneids=bridge, network_eggnog_agreement=bool(old and bridge and (old == bridge)), known_conflict=collision or disagreement or mismatch, mapping_status=row.get('mapping_status', ''), eggnog_geneid_status=row.get('eggnog_geneid_status', ''), in_LOC_conflict_rows=gene in locset, multifeature_collision=collision, native_eggnog_disagreement=disagreement, network_precise_mismatch=mismatch, identity_unresolved=unresolved, identity_uncertain=bool(flags), identity_flags=';'.join(flags))
        edgecheck = pd.DataFrame([{**r.to_dict(), **identity(r.target_human, r.target)} for _, r in net.iterrows()])
        sourcecheck = pd.DataFrame([dict(tf=tf, source_tilapia=net.loc[net.source.eq(tf), 'source_tilapia'].iloc[0], **identity(tf, net.loc[net.source.eq(tf), 'source_tilapia'].iloc[0])) for tf in self.TFS])
        self.save(edgecheck, 'all_frozen_network_target_identity.tsv')
        self.save(sourcecheck, 'frozen_regulator_identity.tsv')
        coverage = []
        for tf in self.TFS:
            s = edgecheck.loc[edgecheck.source.eq(tf)]
            coverage.append(dict(tf=tf, full_targets=len(s), uncertain_targets=int(s.identity_uncertain.sum()), retained_identity_clear=int((~s.identity_uncertain).sum()), known_conflict_targets=int(s.known_conflict.sum()), unresolved_targets=int(s.identity_unresolved.sum()), known_conflict_names=';'.join(s.loc[s.known_conflict, 'target']), uncertain_target_names=';'.join(s.loc[s.identity_uncertain, 'target']), n_network_precise_mismatch=int(s.network_precise_mismatch.sum())))
        self.save(pd.DataFrame(coverage), 'identity_coverage_summary.tsv')
        arrays = self.BASE / 'ionocyte_24_pseudobulk_arrays.npz'
        self.INPUTS.append(dict(path=str(arrays), bytes=arrays.stat().st_size, sha256=sha(arrays)))
        x = np.load(arrays)
        expr = x['log2cpm']
        symbols = x['symbols']
        raw = x['raw']
        meta = self.read(self.BASE / 'ionocyte_24_pseudobulk_metadata.tsv')
        assert list(meta.row_id) == list(x['rows'])
        variants = ['full', 'full_exclude_known_conflicts', 'full_identity_clear', 'exclusive_selected7_remove_cycle', 'exclusive_selected7_remove_cycle_exclude_known_conflicts', 'exclusive_selected7_remove_cycle_identity_clear']

        def edges(tf, v):
            e = edgecheck.loc[edgecheck.source.eq(tf)]
            if v.startswith('exclusive'):
                e = e.loc[~e.shared_selected7 & ~e.cell_cycle_target]
            if v.endswith('exclude_known_conflicts'):
                e = e.loc[~e.known_conflict]
            if v.endswith('identity_clear'):
                e = e.loc[~e.identity_uncertain]
            return e
        levels = []
        contrasts = []
        looall = []
        loos = []
        for tf in self.TFS:
            for v in variants:
                s, loo = loto_scores(expr, symbols, edges(tf, v))
                for i, r in meta.iterrows():
                    levels.append({**r.to_dict(), 'tf': tf, 'variant': v, 'n_targets': s['n'], 'status': s['status'], 'ulm': s['t'][i]})
                for state in meta.state.unique():
                    d = []
                    ld = []
                    for batch in ['batch_1', 'batch_2']:
                        i = meta.index[meta.state.eq(state) & meta.time_h.eq(96) & meta.sequencing_batch.eq(batch)][0]
                        j = meta.index[meta.state.eq(state) & meta.time_h.eq(24) & meta.sequencing_batch.eq(batch)][0]
                        d.append(s['t'][i] - s['t'][j])
                        ld.append(loo[i] - loo[j])
                    contrasts.append(dict(tf=tf, variant=v, state=state, n_targets=s['n'], status=s['status'], batch1_difference=d[0], batch2_difference=d[1]))
                    if tf in ['RBPJ', 'PPARGC1A', 'KLF4'] and v.endswith('identity_clear'):
                        ld = np.array(ld)
                        same = np.all(np.sign(ld) == np.sign(np.array(d)[:, None]), axis=0)
                        finite = np.all(np.isfinite(ld), axis=0)
                        loos.append(dict(tf=tf, variant=v, state=state, n_targets=s['n'], batch1_difference=d[0], batch2_difference=d[1], n_loto_eligible=int(finite.sum()), n_loto_both_directions_preserved=int((same & finite).sum()), destabilizing_targets=';'.join(s['sub'].target[finite & ~same])))
                        for k, g in enumerate(s['sub'].target):
                            looall.append(dict(tf=tf, variant=v, state=state, removed_target=g, n_after=s['n'] - 1, batch1_difference=ld[0, k], batch2_difference=ld[1, k], both_signs_preserved=bool(same[k]), eligible=bool(finite[k])))
        self.save(pd.DataFrame(levels), 'identity_state_pool_ULM_scores.tsv')
        self.save(pd.DataFrame(contrasts), 'identity_state_late_contrasts.tsv')
        self.save(pd.DataFrame(loos), 'identity_key_state_LOTO_summary.tsv')
        self.save(pd.DataFrame(looall), 'identity_key_state_LOTO_all.tsv.gz')
        de = self.read(source_member(self.ctx.input('de_results'), 'broad__ionocyte__96_vs_24.tsv.gz'))
        de['symbol'] = de.gene.str.upper()
        de = de.sort_values(['symbol', 'baseMean'], ascending=[True, False], kind='mergesort').drop_duplicates('symbol').set_index('symbol')
        data = de[['stat', 'paired_log2cpm_diff_batch_1', 'paired_log2cpm_diff_batch_2']].T.values
        cc = []
        for tf in self.TFS:
            for v in variants:
                s = score(data, de.index.to_numpy(str), edges(tf, v))
                cc.append(dict(tf=tf, variant=v, n_targets=s['n'], status=s['status'], ulm_model=s['t'][0], ulm_batch1=s['t'][1], ulm_batch2=s['t'][2]))
        self.save(pd.DataFrame(cc), 'identity_late_contrast_Wald_ULM.tsv')
        sm = meta.loc[meta.scope.eq('state')]
        reference = sm.loc[sm.time_h.eq(0)].groupby('state').umi_fraction.mean()
        profiles = []
        pm = []
        for sample, inds in sm.groupby('sample_id').groups.items():
            m = meta.loc[inds]
            cpm = raw[inds] / raw[inds].sum(axis=1, keepdims=True) * 1000000.0
            for mode in ['observed_umi_weights', 'fixed_0h_mean_umi_weights', 'equal_state_weights']:
                w = m.umi_fraction.to_numpy(float) if mode == 'observed_umi_weights' else reference.reindex(m.state).to_numpy(float) if mode == 'fixed_0h_mean_umi_weights' else np.repeat(1 / 3, 3)
                profiles.append(np.log2(w @ cpm + 0.5))
                pm.append(dict(sample_id=sample, time_h=m.time_h.iloc[0], batch=m.sequencing_batch.iloc[0], mode=mode))
        pm = pd.DataFrame(pm)
        weighted = []
        for v in ['full_exclude_known_conflicts', 'exclusive_selected7_remove_cycle_exclude_known_conflicts', 'full_identity_clear', 'exclusive_selected7_remove_cycle_identity_clear']:
            s = score(np.array(profiles), symbols, edges('KLF4', v))
            for mode in pm['mode'].unique():
                d = []
                for batch in ['batch_1', 'batch_2']:
                    i = pm.index[pm['mode'].eq(mode) & pm.batch.eq(batch) & pm.time_h.eq(96)][0]
                    j = pm.index[pm['mode'].eq(mode) & pm.batch.eq(batch) & pm.time_h.eq(24)][0]
                    d.append(s['t'][i] - s['t'][j])
                weighted.append(dict(tf='KLF4', variant=v, weight_mode=mode, n_targets=s['n'], batch1_difference=d[0], batch2_difference=d[1]))
        self.save(pd.DataFrame(weighted), 'identity_KLF4_gene_profile_weight_sensitivity.tsv')
        self.save(pd.DataFrame(self.INPUTS), 'input_SHA256_manifest.tsv')
        prior = read_table(self.BASE / 'OUTPUT_SHA256.tsv', sep='\t')
        old_files_unchanged = all((sha(self.BASE / r.relative_path) == r.sha256 for _, r in prior.iterrows() if r.relative_path not in ['identity_crosscheck.py'] and (not r.relative_path.startswith('identity_crosscheck/'))))
        validation = dict(n_edges=len(edgecheck), n_unique_targets=int(edgecheck.target.nunique()), n_edges_precise_geneid_resolved=int(edgecheck.precise_gene_id.ne('').sum()), n_edges_network_precise_geneid_mismatch=int(edgecheck.network_precise_mismatch.sum()), n_edges_network_eggnog_geneid_agree=int(edgecheck.network_eggnog_agreement.sum()), n_features_known_conflict=int(edgecheck.loc[edgecheck.known_conflict, 'target'].nunique()), n_edges_known_conflict=int(edgecheck.known_conflict.sum()), n_unique_features_unresolved=int(edgecheck.loc[edgecheck.identity_unresolved, 'target'].nunique()), n_edges_native_eggnog_disagreement=int(edgecheck.native_eggnog_disagreement.sum()), original_deliverable_files_unchanged=old_files_unchanged, source_regulators_all_network_eggnog_agree=bool(sourcecheck.network_eggnog_agreement.all()), source_regulators_n_exact_symbol_unresolved=int(sourcecheck.identity_unresolved.sum()))
        assert old_files_unchanged
        (self.OUT / 'validation_summary.json').write_text(json.dumps(validation, indent=2), encoding='utf8')
        print('COVERAGE\n' + pd.DataFrame(coverage).to_string(index=False))
        print('SOURCE IDENTITY\n' + sourcecheck.to_string(index=False))
        print('WALD\n' + pd.DataFrame(cc).to_string(index=False))
        print('KEY LOTO\n' + pd.DataFrame(loos).to_string(index=False))
        print('KLF4 WEIGHTS\n' + pd.DataFrame(weighted).to_string(index=False))
INPUTS_BY_TASK = {'tf-activity': ['atlas_h5ad', 'collectri_human', 'strict_orthology', 'de_results'], 'tf-specificity': ['state_pool_counts', 'state_pool_weights', 'tf_network', 'collectri_human', 'strict_orthology', 'baseline_state_tf', 'baseline_tf_activity', 'de_results', 'recovered_geneid_audit', 'loc_conflicts', 'geneid_disagreements'], 'orthology': ['ncbi_gene_orthologs', 'human_gene_info', 'eggnog_bridge'], 'identity': ['atlas_h5ad', 'gene_annotation'], 'motif': ['genome_fasta', 'gene_annotation', 'tf_network', 'ionocyte_late_de', 'motif_klf4', 'motif_myb', 'motif_nfkb1', 'motif_rela', 'motif_sox4', 'motif_rbpj']}

def run(ctx, task='tf-specificity'):
    ctx.require(INPUTS_BY_TASK[task])
    if task in ['orthology', 'identity']:
        from . import gene_identity
        return gene_identity.run(ctx, task)
    if task == 'motif':
        from . import motif
        return motif.run(ctx, task)
    if task == 'tf-activity':
        NetworkActivity(ctx).analyze()
    else:
        a = TargetSpecificity(ctx)
        a.analyze()
        a.focused_followup()
        a.summarize_outputs()
        TargetIdentityCheck(ctx).analyze()
    return {'files': [str(p.relative_to(ctx.output(''))) for p in ctx.output('').rglob('*') if p.is_file()]}
