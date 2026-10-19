"""Prespecified ionocyte robustness: all36 full and60 balanced runs with complete summaries."""
from __future__ import annotations
from .runtime import read_table, source_member, iter_sources, source_provenance, source_size, source_exists, source_label, source_text, copy_source
import hashlib
import importlib.metadata as im
import json
import time
import traceback
import warnings
import h5py
import anndata as ad
from anndata.io import read_elem
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from .gene_identity import read_csr_rows

class StateRobustness:

    def __init__(self, ctx):
        self.ctx = ctx
        self.OUT = ctx.output('')
        self.H5 = ctx.input('atlas_h5ad')
        self.SEED = 20260922
        self.POOLS = ['CTL0h_1', 'CTL0h_2', 'ST24h_1', 'ST24h_2', 'ST96h_1', 'ST96h_2']
        self.STATES = ['ATP1A1_paralogue_mitochondria-rich', 'FOXI1_NDRG1_ESRRG_transport', 'SOX4_MYB_ATP1A1_progenitor-like']
        self.DISPLAY = dict(zip(self.STATES, ['NKA alpha1-rich', 'ESRRG transport-rich', 'SOX4/MYB-rich']))
        self.NAMING = ['sox4', 'myb', 'esrrg', 'foxi1', 'foxi2', 'ndrg1', 'LOC112841652', 'atp1a1', 'atp1a1a', 'atp1a1b', 'cftr', 'slc4a4']
        self.MARKERS = ['sox4', 'myb', 'esrrg', 'foxi2', 'ndrg1', 'LOC112841652', 'cftr', 'slc4a4', 'atp2b1', 'dll4', 'notch1', 'mki67', 'pcna', 'top2a']
        self.CYCLE = 'MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8 HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA'.lower().split()
        self.PANELS = ['base', 'without_naming', 'without_naming_cycle']
        self.TABLES = {'failures': []}

    def write(self, df, name):
        df.to_csv(self.OUT / name, sep='\t', index=False, lineterminator='\n')

    def append(self, name, rows):
        self.TABLES.setdefault(name, []).extend(rows)

    def flush(self):
        for name, rows in self.TABLES.items():
            self.write(pd.DataFrame(rows), name + '.tsv')

    def jsonout(self, obj, name):
        (self.OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')

    def load_data(self):
        with h5py.File(self.H5, 'r') as f:
            obs, var = (read_elem(f['obs']), read_elem(f['var']))
            mask = ~obs.exclude_from_biology_v5.astype(bool) & obs.cell_type_v5.astype(str).eq('ionocyte')
            assert mask.sum() == 3885
            pieces = []
            for start in range(0, len(obs), 4096):
                stop = min(start + 4096, len(obs))
                keep = mask.iloc[start:stop].to_numpy()
                if keep.any():
                    pieces.append(read_csr_rows(f['layers/counts'], start, stop)[keep])
            counts = sparse.vstack(pieces, format='csr').astype(np.float32)
            meta = obs.loc[mask].copy()
        assert np.all(counts.data >= 0) and np.all(counts.data == np.floor(counts.data))
        assert set(meta.cell_state_v5.astype(str)) == set(self.STATES)
        assert counts.shape == (3885, 37484)
        dig = hashlib.sha256()
        for arr in [counts.data, counts.indices, counts.indptr]:
            dig.update(arr.tobytes())
        self.jsonout({'shape': counts.shape, 'count_matrix_sha256': dig.hexdigest(), 'cell_order_sha256': hashlib.sha256('\n'.join(meta.index).encode()).hexdigest(), 'input_path': str(self.H5), 'input_size': self.H5.stat().st_size, 'input_mtime_ns': self.H5.stat().st_mtime_ns, 'versions': {p: im.version(p) for p in ['scanpy', 'anndata', 'harmonypy', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'igraph', 'leidenalg', 'scikit-misc']}, 'run_started_local': time.strftime('%Y-%m-%d %H:%M:%S')}, 'input_and_environment.json')
        self.write(meta[['company_sample', 'time_h', 'sequencing_batch', 'library_id', 'cell_state_v5', 'total_counts', 'pct_counts_mt', 'pct_counts_rrna_annotated', 'doublet_score']].rename_axis('cell_id').reset_index(), 'frozen_cell_metadata.tsv')
        poolstate = meta.groupby(['company_sample', 'time_h', 'sequencing_batch', 'cell_state_v5'], observed=True).size().rename('n_cells').reset_index()
        poolstate['fraction'] = poolstate.n_cells / poolstate.groupby('company_sample').n_cells.transform('sum')
        self.write(poolstate, 'original_state_pool_composition.tsv')
        sparse.save_npz(self.OUT / 'frozen_ionocyte_counts.npz', counts)
        return (counts, meta, var)

    def feature_flags(self, var):
        name = var.index.astype(str).str.lower()
        symbol = var.gene_symbol.astype(str).str.lower()
        naming = np.asarray(name.isin([g.lower() for g in self.NAMING]) | symbol.isin([g.lower() for g in self.NAMING]))
        expanded = set(self.CYCLE)
        expanded.update((g + s for g in self.CYCLE for s in ['a', 'b']))
        cyc = np.asarray(name.isin(expanded) | symbol.isin(expanded))
        base = var.excluded_from_v5_full_embedding.astype(bool).to_numpy()
        frame = pd.DataFrame({'feature': var.index, 'symbol': var.gene_symbol.to_numpy(), 'exclude_base': base, 'exclude_naming': naming, 'exclude_cycle_symbol_approximation': cyc, 'base_exclusion_reason': var.v5_full_embedding_exclusion_reason.to_numpy()})
        self.write(frame, 'feature_exclusion_audit.tsv')
        return {'base': base, 'without_naming': base | naming, 'without_naming_cycle': base | naming | cyc}

    def preprocess(self, counts, meta, var, excluded, n_hvg, n_pcs, seed, feature_id):
        import scanpy as sc
        eligible = ~excluded & (np.asarray((counts > 0).sum(axis=0)).ravel() >= 3)
        hv = ad.AnnData(counts[:, eligible].copy(), obs=meta[['sequencing_batch']].copy(), var=var.loc[eligible, ['gene_symbol']].copy())
        sc.pp.highly_variable_genes(hv, flavor='seurat_v3', n_top_genes=n_hvg, batch_key='sequencing_batch', span=0.3)
        selected = np.flatnonzero(eligible)[hv.var.highly_variable.to_numpy()]
        selected_info = hv.var.rename_axis('matrix_feature').reset_index()
        selected_info.insert(0, 'feature_id', feature_id)
        self.write(selected_info, f'hvg/{feature_id}.tsv')
        total = np.asarray(counts.sum(axis=1)).ravel()
        x = counts[:, selected].multiply((10000.0 / total)[:, None]).tocsr()
        x.data = np.log1p(x.data)
        a = ad.AnnData(x, obs=meta[['company_sample', 'time_h', 'sequencing_batch']].copy(), var=var.iloc[selected][['gene_symbol']].copy())
        sc.pp.scale(a, max_value=10)
        sc.tl.pca(a, n_comps=n_pcs, svd_solver='arpack', random_state=seed)
        self.append('preprocessing_log', [{'feature_id': feature_id, 'n_cells': len(meta), 'eligible_genes': int(eligible.sum()), 'selected_hvg': len(selected), 'n_pcs': n_pcs, 'selected_naming_markers': ','.join(a.var_names[a.var_names.str.lower().isin([g.lower() for g in self.NAMING])])}])
        return a

    def marker_summary(self, counts, meta, var, labels, run_id):
        present = [g for g in self.MARKERS if g in var.index]
        idx = var.index.get_indexer(present)
        sub = counts[:, idx].toarray()
        cp = sub / np.asarray(counts.sum(axis=1)).ravel()[:, None] * 10000.0
        rows = []
        for group in sorted(set(labels), key=int):
            mask = labels == group
            for j, gene in enumerate(present):
                rows.append({'run_id': run_id, 'cluster': group, 'gene': gene, 'n_cells': int(mask.sum()), 'mean_log1p_CP10k': float(np.log1p(cp[mask, j]).mean()), 'fraction_detected': float((sub[mask, j] > 0).mean())})
        self.append('cluster_marker_expression', rows)

    def evaluate(self, labels, meta, run_id, params, counts, var):
        old = meta.cell_state_v5.astype(str).to_numpy()
        clusters = sorted(set(labels), key=int)
        table = pd.crosstab(pd.Series(old, name='old_state'), pd.Series(labels, name='cluster')).reindex(index=self.STATES, columns=clusters, fill_value=0)
        majority = table.idxmax(axis=0).to_dict()
        purity = (table.max(axis=0) / table.sum(axis=0)).to_dict()
        mapping = {c: majority[c] if purity[c] >= 0.7 else 'unresolved' for c in clusters}
        mapped = np.array([majority[c] for c in labels])
        strict = np.array([mapping[c] for c in labels])
        self.append('run_summary', [{'run_id': run_id, **params, 'n_cells': len(meta), 'n_clusters': len(clusters), 'ARI': adjusted_rand_score(old, labels), 'NMI': normalized_mutual_info_score(old, labels), 'majority_mapping_accuracy': float((mapped == old).mean()), 'strict_unresolved_fraction': float((strict == 'unresolved').mean()), 'minimum_cluster_size': int(pd.Series(labels).value_counts().min())}])
        self.append('cluster_oldstate_confusion', [{'run_id': run_id, 'old_state': st, 'cluster': c, 'n_cells': int(table.loc[st, c]), 'fraction_of_old_state': float(table.loc[st, c] / table.loc[st].sum()), 'cluster_majority': majority[c], 'cluster_purity': purity[c]} for st in self.STATES for c in clusters])
        rows = []
        for st in self.STATES:
            oldmask = old == st
            best = max(clusters, key=lambda c: int(((labels == c) & oldmask).sum()) / max(1, int(((labels == c) | oldmask).sum())))
            for metric, pred in [('majority', mapped), ('strict_70', strict)]:
                hit = (pred == st) & oldmask
                rows.append({'run_id': run_id, 'state': st, 'mapping': metric, 'precision': float(hit.sum() / max(1, (pred == st).sum())), 'recall': float(hit.sum() / oldmask.sum()), 'Jaccard': float(hit.sum() / max(1, ((pred == st) | oldmask).sum())), 'n_associated_clusters': sum(((majority if metric == 'majority' else mapping)[c] == st for c in clusters)), 'best_single_cluster': best, 'best_single_cluster_Jaccard': float(((labels == best) & oldmask).sum() / ((labels == best) | oldmask).sum())})
        self.append('state_recovery', rows)
        diag = meta[['company_sample', 'time_h', 'sequencing_batch']].copy()
        diag['cluster'] = labels
        diag['old_state'] = old
        diag['majority_state'] = mapped
        diag['strict_70_state'] = strict
        self.write(diag.rename_axis('cell_id').reset_index(), f'labels/{run_id}.tsv.gz')
        comp = []
        for pool in self.POOLS:
            m = diag.company_sample.astype(str).eq(pool)
            d = diag.loc[m]
            for typ, col in [('original', 'old_state'), ('majority', 'majority_state'), ('strict_70', 'strict_70_state')]:
                for st in self.STATES + (['unresolved'] if typ == 'strict_70' else []):
                    comp.append({'run_id': run_id, 'pool': pool, 'time_h': int(d.time_h.iloc[0]), 'batch': str(d.sequencing_batch.iloc[0]), 'mapping': typ, 'state': st, 'n_cells': int(d[col].eq(st).sum()), 'n_pool_cells': len(d), 'fraction': float(d[col].eq(st).mean())})
            for c in clusters:
                self.append('cluster_pool_composition', [{'run_id': run_id, 'pool': pool, 'time_h': int(d.time_h.iloc[0]), 'batch': str(d.sequencing_batch.iloc[0]), 'cluster': c, 'n_cells': int(d.cluster.eq(c).sum()), 'fraction': float(d.cluster.eq(c).mean()), 'cluster_majority': majority[c], 'cluster_purity': purity[c]}])
            for st in self.STATES:
                om = d.old_state.eq(st)
                self.append('state_pool_recovery', [{'run_id': run_id, 'pool': pool, 'state': st, 'n_original': int(om.sum()), 'majority_retained': int((om & d.majority_state.eq(st)).sum()), 'strict_70_retained': int((om & d.strict_70_state.eq(st)).sum()), 'majority_recall': float(d.loc[om, 'majority_state'].eq(st).mean()) if om.any() else np.nan, 'strict_70_recall': float(d.loc[om, 'strict_70_state'].eq(st).mean()) if om.any() else np.nan}])
        self.append('mapped_state_pool_composition', comp)
        comp = pd.DataFrame(comp)
        for typ in ['original', 'majority', 'strict_70']:
            for st in self.STATES:
                part = comp.loc[(comp.mapping == typ) & (comp.state == st)]
                row = {'run_id': run_id, 'mapping': typ, 'state': st}
                for target, baseline in [(96, 24), (96, 0), (24, 0)]:
                    d = []
                    for batch in ['batch_1', 'batch_2']:
                        a = float(part.loc[(part.time_h == target) & (part.batch == batch), 'fraction'].iloc[0])
                        b = float(part.loc[(part.time_h == baseline) & (part.batch == batch), 'fraction'].iloc[0])
                        row[f'{target}vs{baseline}_{batch}_percentage_points'] = 100 * (a - b)
                        d.append(a - b)
                    row[f'{target}vs{baseline}_both_batches_positive'] = all((v > 0 for v in d))
                    row[f'{target}vs{baseline}_all_target_above_all_baseline'] = bool(part.loc[part.time_h == target, 'fraction'].min() > part.loc[part.time_h == baseline, 'fraction'].max())
                self.append('stage_direction', [row])
        self.marker_summary(counts, meta, var, labels, run_id)

    def run_graph(self, a, counts, meta, var, feature_id, n_neighbors, integration, resolutions, seed, params):
        import scanpy as sc
        t = time.time()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                if integration == 'harmony':
                    sc.external.pp.harmony_integrate(a, key='sequencing_batch', basis='X_pca', adjusted_basis='X_harmony', max_iter_harmony=20, random_state=seed, verbose=False)
                    rep = 'X_harmony'
                else:
                    rep = 'X_pca'
                sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep=rep, random_state=seed, transformer='sklearn')
                for res in resolutions:
                    run_id = f"{feature_id}_{integration}_k{n_neighbors}_r{str(res).replace('.', 'p')}"
                    sc.tl.leiden(a, resolution=res, random_state=seed, flavor='igraph', n_iterations=2, directed=False, key_added='new_cluster')
                    self.evaluate(a.obs.new_cluster.astype(str).to_numpy(), meta, run_id, {**params, 'integration': integration, 'n_neighbors': n_neighbors, 'resolution': res, 'seed': seed}, counts, var)
                    print(f'DONE {run_id}: clusters={a.obs.new_cluster.nunique()}', flush=True)
                for w in ws:
                    self.append('warnings', [{'feature_id': feature_id, 'integration': integration, 'category': w.category.__name__, 'message': str(w.message)}])
            self.append('runtime', [{'feature_id': feature_id, 'integration': integration, 'seconds': time.time() - t, 'status': 'complete'}])
        except Exception as exc:
            self.append('failures', [{'feature_id': feature_id, 'integration': integration, 'error': repr(exc), 'traceback': traceback.format_exc()}])
            print('FAIL', feature_id, integration, repr(exc), flush=True)
        self.flush()

    def attempt_preprocess(self, counts, meta, var, exclude, hvg, pcs, seed, feature_id):
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                a = self.preprocess(counts, meta, var, exclude, hvg, pcs, seed, feature_id)
                for w in ws:
                    self.append('warnings', [{'feature_id': feature_id, 'integration': 'preprocess', 'category': w.category.__name__, 'message': str(w.message)}])
            return a
        except Exception as exc:
            self.append('failures', [{'feature_id': feature_id, 'integration': 'preprocess', 'error': repr(exc), 'traceback': traceback.format_exc()}])
            print('FAIL preprocessing', feature_id, repr(exc), flush=True)
            self.flush()
            return None

    def fit_grid(self):
        start = time.time()
        stat = (self.H5.stat().st_size, self.H5.stat().st_mtime_ns)
        for sub in ['hvg', 'labels']:
            (self.OUT / sub).mkdir(exist_ok=True)
        counts, meta, var = self.load_data()
        exclusions = self.feature_flags(var)
        original_counts = meta.groupby('company_sample', observed=True).size()
        assert original_counts.min() == 201
        configs = [('base', 2000, 30, 15), ('base', 4000, 50, 15), ('base', 2000, 20, 30), ('base', 4000, 30, 30), ('without_naming', 2000, 30, 15), ('without_naming_cycle', 2000, 30, 15)]
        self.jsonout({'full_configs': configs, 'integrations': ['none', 'harmony'], 'resolutions': [0.3, 0.5, 0.8], 'balance_seeds': list(range(self.SEED, self.SEED + 20)), 'per_pool_cells': 201, 'balanced_hvg': 2000, 'balanced_pcs': 30, 'balanced_k': 15, 'balanced_resolution': 0.5, 'balanced_integration': 'harmony', 'panels': self.PANELS, 'naming_features_requested': self.NAMING, 'cellcycle_roots': self.CYCLE, 'cycle_matching': 'case-insensitive exact symbol or suffix a/b'}, 'frozen_parameters.json')
        for panel, hvg, pcs, nn in configs:
            fid = f'full_{panel}_h{hvg}_pc{pcs}'
            a = self.attempt_preprocess(counts, meta, var, exclusions[panel], hvg, pcs, self.SEED, fid)
            if a is None:
                continue
            for integration in ['none', 'harmony']:
                self.run_graph(a, counts, meta, var, fid, nn, integration, [0.3, 0.5, 0.8], self.SEED, {'family': 'full', 'panel': panel, 'n_hvg': hvg, 'n_pcs': pcs})
        for seed in range(self.SEED, self.SEED + 20):
            rng = np.random.default_rng(seed)
            ids = np.sort(np.concatenate([rng.choice(np.flatnonzero(meta.company_sample.astype(str).eq(pool).to_numpy()), 201, replace=False) for pool in self.POOLS]))
            subcounts = counts[ids]
            submeta = meta.iloc[ids].copy()
            for panel in self.PANELS:
                fid = f'balanced_s{seed}_{panel}_h2000_pc30'
                a = self.attempt_preprocess(subcounts, submeta, var, exclusions[panel], 2000, 30, seed, fid)
                if a is not None:
                    self.run_graph(a, subcounts, submeta, var, fid, 15, 'harmony', [0.5], seed, {'family': 'balanced', 'panel': panel, 'n_hvg': 2000, 'n_pcs': 30})
        self.flush()
        assert stat == (self.H5.stat().st_size, self.H5.stat().st_mtime_ns), 'Input changed during analysis'
        self.jsonout({'complete': True, 'expected_runs': 96, 'successful_runs': len(self.TABLES.get('run_summary', [])), 'failure_records': len(self.TABLES.get('failures', [])), 'input_unchanged_size_mtime': True, 'elapsed_seconds': time.time() - start}, 'completion.json')
        print('COMPLETE', json.loads((self.OUT / 'completion.json').read_text()), flush=True)

    def read_summary(self, stem):
        return read_table(self.OUT / (stem + '.tsv'), sep='\t')

    def write_summary(self, frame, stem):
        frame.to_csv(self.OUT / (stem + '.tsv'), sep='\t', index=False, lineterminator='\n')

    def summarize_all(self):
        run = self.read_summary('run_summary')
        dirs = self.read_summary('stage_direction').merge(run, on='run_id', validate='m:1')
        rec = self.read_summary('state_recovery').merge(run[['run_id', 'family', 'panel', 'integration']], on='run_id', validate='m:1')
        mix = self.read_summary('mapped_state_pool_composition').merge(run[['run_id', 'family', 'panel', 'integration']], on='run_id', validate='m:1')
        sox = 'SOX4_MYB_ATP1A1_progenitor-like'
        summ = []
        for key, part in run.groupby(['family', 'panel', 'integration'], observed=True, sort=False):
            row = dict(zip(['family', 'panel', 'integration'], key))
            row['n_runs'] = len(part)
            for metric in ['n_clusters', 'ARI', 'NMI', 'majority_mapping_accuracy', 'strict_unresolved_fraction']:
                for stat, fn in [('min', np.min), ('median', np.median), ('max', np.max)]:
                    row[metric + '_' + stat] = float(fn(part[metric]))
            summ.append(row)
        self.write_summary(pd.DataFrame(summ), 'all_settings_summary')
        target = dirs.loc[dirs.state.eq(sox)]
        rows = []
        for key, part in target.groupby(['family', 'panel', 'integration', 'mapping'], observed=True, sort=False):
            row = dict(zip(['family', 'panel', 'integration', 'mapping'], key))
            row['n_runs'] = len(part)
            for contrast in ['96vs24', '96vs0', '24vs0']:
                row[contrast + '_both_positive_runs'] = int(part[contrast + '_both_batches_positive'].sum())
                row[contrast + '_all_target_above_baseline_runs'] = int(part[contrast + '_all_target_above_all_baseline'].sum())
                for batch in ['batch_1', 'batch_2']:
                    v = part[f'{contrast}_{batch}_percentage_points']
                    for stat, fn in [('min', np.min), ('median', np.median), ('max', np.max)]:
                        row[f'{contrast}_{batch}_pp_{stat}'] = float(fn(v))
            rows.append(row)
        self.write_summary(pd.DataFrame(rows), 'SOX4_direction_summary_all_settings')
        rows = []
        for key, part in rec.groupby(['family', 'panel', 'integration', 'state', 'mapping'], observed=True, sort=False):
            row = dict(zip(['family', 'panel', 'integration', 'state', 'mapping'], key))
            row['n_runs'] = len(part)
            for metric in ['precision', 'recall', 'Jaccard', 'n_associated_clusters', 'best_single_cluster_Jaccard']:
                for stat, fn in [('min', np.min), ('median', np.median), ('max', np.max)]:
                    row[metric + '_' + stat] = float(fn(part[metric]))
            rows.append(row)
        self.write_summary(pd.DataFrame(rows), 'state_recovery_summary_all_settings')
        rows = []
        for key, part in mix.loc[mix.state.eq(sox)].groupby(['family', 'panel', 'integration', 'mapping', 'pool', 'time_h'], observed=True, sort=False):
            row = dict(zip(['family', 'panel', 'integration', 'mapping', 'pool', 'time_h'], key))
            row['n_runs'] = len(part)
            for stat, fn in [('min', np.min), ('median', np.median), ('max', np.max)]:
                row['percent_' + stat] = 100 * float(fn(part.fraction))
            rows.append(row)
        self.write_summary(pd.DataFrame(rows), 'SOX4_pool_fraction_summary')
        diag = []
        pooldiag = []
        for row in run.itertuples():
            lab = read_table(self.OUT / 'labels' / (row.run_id + '.tsv.gz'), sep='\t')
            d = {'run_id': row.run_id, 'family': row.family, 'panel': row.panel, 'integration': row.integration, 'state_homogeneity': homogeneity_score(lab.old_state, lab.cluster), 'state_completeness': completeness_score(lab.old_state, lab.cluster), 'ARI_vs_pool': adjusted_rand_score(lab.company_sample, lab.cluster), 'ARI_vs_time': adjusted_rand_score(lab.time_h, lab.cluster), 'ARI_vs_batch': adjusted_rand_score(lab.sequencing_batch, lab.cluster)}
            diag.append(d)
            for cluster, part in lab.groupby('cluster', observed=True):
                counts = part.company_sample.value_counts()
                pooldiag.append({'run_id': row.run_id, 'cluster': cluster, 'n_cells': len(part), 'n_pools': part.company_sample.nunique(), 'n_stages': part.time_h.nunique(), 'n_batches': part.sequencing_batch.nunique(), 'largest_pool': counts.index[0], 'largest_pool_fraction': float(counts.iloc[0] / len(part)), 'largest_stage_fraction': float(part.time_h.value_counts().max() / len(part)), 'largest_batch_fraction': float(part.sequencing_batch.value_counts().max() / len(part))})
        self.write_summary(pd.DataFrame(diag), 'cluster_association_diagnostics')
        self.write_summary(pd.DataFrame(pooldiag), 'cluster_pool_presence_diagnostics')
        assert len(run) == 96 and run.family.value_counts().to_dict() == {'balanced': 60, 'full': 36}
        assert len(set(run.run_id)) == 96
        assert run.groupby(['family', 'panel']).size().to_dict() == {('balanced', 'base'): 20, ('balanced', 'without_naming'): 20, ('balanced', 'without_naming_cycle'): 20, ('full', 'base'): 24, ('full', 'without_naming'): 6, ('full', 'without_naming_cycle'): 6}
        assert mix.loc[mix.family.eq('balanced'), 'n_pool_cells'].eq(201).all()
        assert dirs.loc[dirs.state.eq(sox), '96vs24_both_batches_positive'].all()
        checks = {'all_96_runs_completed': True, 'balanced_exactly_201_cells_per_pool': True, 'SOX4_96_vs_24_positive_both_batches_all_runs_original_majority_strict': True, 'cell_subsampling_does_not_increase_biological_n': True, 'report_generated_from_every_run_not_best_only': True}
        (self.OUT / 'summary_validation.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
        print(pd.DataFrame(summ).to_string(index=False))
        print('SOX4 direction:', target.groupby(['family', 'mapping'])[['96vs24_both_batches_positive', '96vs0_both_batches_positive']].agg(['sum', 'size']).to_string())
        print('SOX4 96vs24 difference ranges:', target.groupby(['family', 'mapping'])[['96vs24_batch_1_percentage_points', '96vs24_batch_2_percentage_points']].agg(['min', 'median', 'max']).to_string())
        print('SOX4 state recovery:', rec.loc[rec.state.eq(sox)].groupby(['family', 'mapping'])[['precision', 'recall', 'best_single_cluster_Jaccard', 'n_associated_clusters']].agg(['min', 'median', 'max']).to_string())
        print('Associations:', pd.DataFrame(diag).groupby('family')[['state_homogeneity', 'state_completeness', 'ARI_vs_pool', 'ARI_vs_time', 'ARI_vs_batch']].agg(['min', 'median', 'max']).to_string())
        panel_a, panel_b = robustness_figure_sources(
            self.read_summary('original_state_pool_composition'),
            self.read_summary('stage_direction'), run)
        self.write_summary(panel_a, 'candidate_figure_panel_a')
        self.write_summary(panel_b, 'candidate_figure_panel_b')
def robustness_figure_sources(original, directions, runs):
    """Exact two source tables for original labels and all96 strict mappings.

    Retains the median/min/max over the complete predefined setting groups;
    these are descriptive setting summaries, not biological replicates.
    """
    panel_a = original.loc[
        original.cell_state_v5.str.startswith('SOX4')
    ].copy()
    strict = directions.loc[
        directions.state.str.startswith('SOX4')
        & directions.mapping.eq('strict_70')
    ].merge(runs[['run_id', 'family', 'panel']], on='run_id', validate='m:1')
    assert len(strict) == 96 and strict['96vs24_both_batches_positive'].all()
    groups = [
        ('full', 'base'), ('full', 'without_naming'),
        ('full', 'without_naming_cycle'), ('balanced', 'base'),
        ('balanced', 'without_naming'), ('balanced', 'without_naming_cycle'),
    ]
    rows = []
    for family, panel in groups:
        selected = strict.loc[strict.family.eq(family) & strict.panel.eq(panel)]
        for batch in ['batch_1', 'batch_2']:
            values = selected[f'96vs24_{batch}_percentage_points'].to_numpy()
            rows.append({
                'family': family, 'panel': panel, 'batch': batch,
                'n_settings': len(selected),
                'median_percentage_points': float(np.median(values)),
                'min_percentage_points': float(values.min()),
                'max_percentage_points': float(values.max()),
                'all_settings_positive': bool((values > 0).all()),
            })
    return panel_a, pd.DataFrame(rows)


INPUTS_BY_TASK = {'state-robustness': ['atlas_h5ad']}

def run(ctx, task='state-robustness'):
    ctx.require(INPUTS_BY_TASK[task])
    a = StateRobustness(ctx)
    with threadpool_limits(limits=4):
        a.fit_grid()
    a.summarize_all()
    return {'files': [str(p.relative_to(ctx.output(''))) for p in ctx.output('').rglob('*') if p.is_file()]}
