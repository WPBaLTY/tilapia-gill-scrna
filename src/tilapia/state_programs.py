"""Pool-level state composition, program decomposition, respiratory classes and fixed-membership evidence."""
from __future__ import annotations
from .runtime import read_table, source_member, iter_sources, source_provenance, source_size, source_exists, source_label, source_text, copy_source
from .runtime import sha256 as sha
from pathlib import Path
import hashlib
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from anndata.io import read_elem
import json
import platform
import re
from pydeseq2.preprocessing import deseq2_norm
import pydeseq2
from pydeseq2.preprocessing import deseq2_norm_fit
from pydeseq2.preprocessing import deseq2_norm_transform
import gzip
import shutil
from .gene_identity import read_csr_rows

class StateComposition:

    def __init__(self, ctx):
        self.ctx = ctx
        self.OUT = ctx.output('')
        self.H5 = ctx.input('atlas_h5ad')
        self.POOLS = ['CTL0h_1', 'CTL0h_2', 'ST24h_1', 'ST24h_2', 'ST96h_1', 'ST96h_2']
        self.BR = dict(zip([f'BR{i}' for i in range(1, 7)], self.POOLS))
        self.STATES = ['ATP1A1_paralogue_mitochondria-rich', 'FOXI1_NDRG1_ESRRG_transport', 'SOX4_MYB_ATP1A1_progenitor-like']
        self.DISPLAY = dict(zip(self.STATES, ['NKA alpha1-rich', 'ESRRG transport-rich', 'SOX4/MYB-rich']))
        self.ION_GENES = ['cftr', 'LOC112841652', 'foxi2', 'slc4a4', 'esrrg', 'sox4', 'myb', 'atp2b1', 'notch1', 'dll4']
        self.PAV_GENES = ['epcam', 'cebpd', 'abca12', 'evpl', 'eppk1', 'ptgs2', 'lgals4', 'pkp3', 'perp', 'scel', 'muc15', 'hopx']
        self.TFS = ['PPARGC1A', 'RELA', 'RBPJ', 'SOX4', 'MYB', 'KLF4', 'NFKB1', 'TP53']
        self.CONTRASTS = [(24, 0), (96, 0), (96, 24)]
        self.inputs = []

    def track(self, path):
        record=source_provenance(path)
        record['input']=record.pop('path')
        self.inputs.append(record)
        return path

    def read_tsv(self, path):
        return read_table(self.track(path), sep='\t')

    def write(self, frame, name):
        frame.to_csv(self.OUT / name, sep='\t', index=False, lineterminator='\n')

    def compare_by_batch(self, frame, keys, value, transform=True):
        rows = []
        for group, part in frame.groupby(keys, observed=True, sort=False):
            if not isinstance(group, tuple):
                group = (group,)
            for target, baseline in self.CONTRASTS:
                row = dict(zip(keys, group))
                row['contrast'] = f'{target}_vs_{baseline}'
                diffs = []
                for batch in ['batch_1', 'batch_2']:
                    a = part.loc[(part.time_h == target) & (part.sequencing_batch == batch), value]
                    b = part.loc[(part.time_h == baseline) & (part.sequencing_batch == batch), value]
                    if len(a) != 1 or len(b) != 1:
                        raise ValueError((row, batch, len(a), len(b)))
                    av, bv = (float(a.iloc[0]), float(b.iloc[0]))
                    diff = np.log2(av + 0.5) - np.log2(bv + 0.5) if transform else av - bv
                    row[f'{batch}_difference'] = diff
                    diffs.append(diff)
                row['mean_difference'] = float(np.mean(diffs))
                row['both_batches_positive'] = bool(all((x > 0 for x in diffs)))
                row['both_batches_negative'] = bool(all((x < 0 for x in diffs)))
                row['batch_direction_agrees'] = bool(np.sign(diffs[0]) == np.sign(diffs[1]))
                row['difference_scale'] = 'log2(value + 0.5)' if transform else 'raw ULM activity'
                rows.append(row)
        return pd.DataFrame(rows)

    def analyze(self):
        self.OUT.mkdir(parents=True, exist_ok=True)
        self.track(self.H5)
        hstat = (self.H5.stat().st_size, self.H5.stat().st_mtime_ns)
        with h5py.File(self.H5, 'r') as f:
            obs = read_elem(f['obs'])
            var = read_elem(f['var'])
            primary = ~obs.exclude_from_biology_v5.astype(bool).to_numpy()
            assert primary.sum() == 84643
            relevant = primary & obs.cell_type_v5.astype(str).isin(['ionocyte', 'pavement cell']).to_numpy()
            idx = np.flatnonzero(relevant)
            meta = obs.iloc[idx].copy()
            genes = list(dict.fromkeys(self.ION_GENES + self.PAV_GENES))
            gidx = var.index.get_indexer(genes)
            assert (gidx >= 0).all(), np.array(genes)[gidx < 0]
            selected_raw = np.zeros((len(idx), len(genes)), dtype=np.float64)
            selected_x = np.zeros_like(selected_raw)
            totals = np.zeros(len(idx), dtype=np.float64)
            idx_to_sel = np.full(len(obs), -1, dtype=int)
            idx_to_sel[idx] = np.arange(len(idx))
            for start in range(0, len(obs), 4096):
                stop = min(start + 4096, len(obs))
                within = np.flatnonzero(relevant[start:stop])
                if not len(within):
                    continue
                dest = idx_to_sel[start + within]
                raw = read_csr_rows(f['layers/counts'], start, stop)[within]
                totals[dest] = np.asarray(raw.sum(axis=1)).ravel()
                selected_raw[dest] = raw[:, gidx].toarray()
                selected_x[dest] = read_csr_rows(f['X'], start, stop)[within][:, gidx].toarray()
                print(f'count extraction {stop}/{len(obs)}', flush=True)
        assert np.all(totals > 0)
        stored_totals = meta.total_counts.to_numpy()
        total_relative_error = float(np.max(np.abs(totals - stored_totals) / np.maximum(stored_totals, 1)))
        print('max relative difference current count sums versus stored obs total_counts', total_relative_error, flush=True)
        cp_cell = selected_raw / totals[:, None] * 1000000.0
        log_cp10k = np.log1p(cp_cell / 100)
        norm_error = float(np.max(np.abs(log_cp10k - selected_x)))
        print('max error current count-derived log1pCP10k versus saved X', norm_error, flush=True)
        assert norm_error < 0.0001
        selected_var = var.loc[genes, ['gene_symbol', 'gene_id', 'gff_gene_id', 'symbol_mapping_status']].reset_index(drop=True)
        selected_var.insert(0, 'matrix_feature', genes)
        selected_var['display_name'] = selected_var.matrix_feature.replace({'LOC112841652': 'NKA alpha1 (LOC112841652)', 'notch1': 'notch1b'})
        selected_var['additional_mapping_note'] = ''
        selected_var.loc[selected_var.matrix_feature.eq('notch1'), 'additional_mapping_note'] = 'Existing audited qPCR mapping: provider alias notch1 -> NCBI Gene 100699419 (notch1b); H5AD annotation itself remains unmatched.'
        self.write(selected_var, 'selected_gene_identifier_audit.tsv')
        m = meta.reset_index(drop=True)
        group_rows = []
        marker_rows = []
        state_rows = []
        broad_rows = []
        for (celltype, pool, state), inds in m.groupby(['cell_type_v5', 'company_sample', 'cell_state_v5'], observed=True).groups.items():
            inds = np.asarray(inds)
            mm = m.iloc[inds]
            base = {'cell_type': celltype, 'sample_id': pool, 'state': state, 'time_h': int(mm.time_h.iloc[0]), 'sequencing_batch': str(mm.sequencing_batch.iloc[0]), 'n_cells': len(inds), 'total_umi_all_features': float(totals[inds].sum())}
            group_rows.append(base)
            use = self.ION_GENES if celltype == 'ionocyte' else self.PAV_GENES
            for gene in use:
                j = genes.index(gene)
                meanlog = float(selected_x[inds, j].mean())
                det = float((selected_raw[inds, j] > 0).mean())
                row = {**base, 'gene': gene, 'raw_umi': int(selected_raw[inds, j].sum()), 'state_pseudobulk_cpm': float(selected_raw[inds, j].sum() / totals[inds].sum() * 1000000.0), 'mean_cell_cpm': float(cp_cell[inds, j].mean()), 'mean_log1p_cp10k': meanlog, 'fraction_detected': det}
                (state_rows if celltype == 'ionocyte' else marker_rows).append(row)
        group = pd.DataFrame(group_rows)
        iongroup = group.loc[group.cell_type == 'ionocyte'].copy()
        iongroup['cell_fraction'] = iongroup.n_cells / iongroup.groupby('sample_id').n_cells.transform('sum')
        iongroup['umi_fraction'] = iongroup.total_umi_all_features / iongroup.groupby('sample_id').total_umi_all_features.transform('sum')
        iongroup['mean_umi_per_cell'] = iongroup.total_umi_all_features / iongroup.n_cells
        assert len(iongroup) == 18 and iongroup.groupby('state').sample_id.nunique().eq(6).all()
        oldcounts = self.read_tsv(self.ctx.input('ionocyte_state_composition_by_pool'))
        oldcounts['sample_id'] = oldcounts.biological_replicate.map(self.BR)
        ck = iongroup.merge(oldcounts[['sample_id', 'cell_state_v5', 'n_cells']], left_on=['sample_id', 'state'], right_on=['sample_id', 'cell_state_v5'], suffixes=('', '_old'), validate='1:1')
        assert ck.n_cells.eq(ck.n_cells_old).all()
        self.write(iongroup, 'ionocyte_state_pool_weights.tsv')
        state = pd.DataFrame(state_rows)
        state = state.merge(iongroup[['sample_id', 'state', 'cell_fraction', 'umi_fraction', 'mean_umi_per_cell']], on=['sample_id', 'state'], validate='m:1')
        state['state_display'] = state.state.map(self.DISPLAY)
        self.write(state, 'ionocyte_state_pool_gene_expression.tsv')
        for pool, inds in m.loc[m.cell_type_v5.astype(str) == 'ionocyte'].groupby('company_sample', observed=True).groups.items():
            inds = np.asarray(inds)
            mm = m.iloc[inds]
            for gene in self.ION_GENES:
                j = genes.index(gene)
                broad_rows.append({'sample_id': pool, 'gene': gene, 'time_h': int(mm.time_h.iloc[0]), 'sequencing_batch': str(mm.sequencing_batch.iloc[0]), 'n_cells': len(inds), 'observed_pseudobulk_cpm': float(selected_raw[inds, j].sum() / totals[inds].sum() * 1000000.0), 'observed_mean_cell_cpm': float(cp_cell[inds, j].mean())})
        broad = pd.DataFrame(broad_rows)
        ref = iongroup.loc[iongroup.time_h == 0].groupby('state')[['cell_fraction', 'umi_fraction']].mean().rename(columns={'cell_fraction': 'baseline_mean_cell_weight', 'umi_fraction': 'baseline_mean_umi_weight'})
        ref['equal_state_weight'] = 1 / 3
        self.write(ref.reset_index(), 'fixed_state_reference_weights.tsv')
        mixrows = []
        for (pool, gene), part in state.groupby(['sample_id', 'gene'], observed=True):
            part = part.set_index('state').reindex(self.STATES)
            w = ref.reindex(self.STATES)
            vals = {'observed_pseudobulk_cpm': float((part.state_pseudobulk_cpm * part.umi_fraction).sum()), 'fixed_baseline_umi_weight_cpm': float((part.state_pseudobulk_cpm * w.baseline_mean_umi_weight).sum()), 'equal_state_weight_pseudobulk_cpm': float(part.state_pseudobulk_cpm.mean()), 'observed_mean_cell_cpm': float((part.mean_cell_cpm * part.cell_fraction).sum()), 'fixed_baseline_cell_weight_mean_cell_cpm': float((part.mean_cell_cpm * w.baseline_mean_cell_weight).sum()), 'equal_state_weight_mean_cell_cpm': float(part.mean_cell_cpm.mean())}
            base = {'sample_id': pool, 'gene': gene, 'time_h': int(part.time_h.iloc[0]), 'sequencing_batch': part.sequencing_batch.iloc[0]}
            check = broad.loc[(broad.sample_id == pool) & (broad.gene == gene)].iloc[0]
            for metric, value in vals.items():
                if metric.startswith('observed_'):
                    assert np.isclose(value, check[metric], rtol=1e-10, atol=1e-10)
                mixrows.append({**base, 'estimand': metric, 'expression': value})
        mixes = pd.DataFrame(mixrows)
        self.write(mixes, 'ionocyte_mixture_and_fixed_weight_expression.tsv')
        mixchange = self.compare_by_batch(mixes, ['gene', 'estimand'], 'expression')
        self.write(mixchange, 'ionocyte_fixed_weight_contrasts_descriptive.tsv')
        schange = self.compare_by_batch(state, ['gene', 'state'], 'state_pseudobulk_cpm')
        de_selected = []
        for state_name in self.STATES + ['ionocyte']:
            for a, b in self.CONTRASTS:
                stem = f'broad__ionocyte__{a}_vs_{b}' if state_name == 'ionocyte' else f'ionocyte_state__{state_name}__{a}_vs_{b}'
                de = self.read_tsv(source_member(self.ctx.input('de_results'), f'{stem}.tsv.gz'))
                picked = de.loc[de.gene.isin(self.ION_GENES)].copy()
                picked['state'] = state_name
                de_selected.append(picked)
        de_selected = pd.concat(de_selected, ignore_index=True)
        self.write(de_selected, 'existing_core_gene_DE_results.tsv')
        cols = ['gene', 'state', 'contrast', 'log2FoldChange_shrunken', 'padj', 'direction_robust', 'residual_df']
        schange = schange.merge(de_selected[cols], on=['gene', 'state', 'contrast'], how='left', validate='1:1')
        self.write(schange, 'ionocyte_state_gene_contrasts_with_existing_DE.tsv')
        pavmask = m.cell_type_v5.astype(str).eq('pavement cell').to_numpy()
        agg = []
        for st, inds in m.loc[pavmask].groupby('cell_state_v5', observed=True).groups.items():
            inds = np.asarray(inds)
            for gene in self.PAV_GENES:
                j = genes.index(gene)
                agg.append({'group': st, 'gene': gene, 'n_cells': len(inds), 'n_pools': m.iloc[inds].company_sample.nunique(), 'mean_log1p_cp10k': float(selected_x[inds, j].mean()), 'fraction_detected': float((selected_raw[inds, j] > 0).mean())})
        pav = pd.DataFrame(agg)
        pav['scaled_mean_within_gene'] = pav.groupby('gene').mean_log1p_cp10k.transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1))
        self.write(pav, 'pavement_state_marker_summary_expanded.tsv')
        self.write(pd.DataFrame(marker_rows), 'pavement_state_pool_marker_expression_expanded.tsv')
        old = self.read_tsv(self.ctx.input('pavement_state_marker_summary'))
        pcheck = pav.merge(old, on=['group', 'gene'], suffixes=('', '_old'), validate='1:1')
        expected_keys = {(state, gene) for state in pav['group'].unique() for gene in self.PAV_GENES}
        assert pav['group'].nunique() == 7
        assert set(zip(pav['group'], pav['gene'])) == expected_keys
        assert set(zip(old['group'], old['gene'])) == expected_keys
        assert len(pcheck) == len(expected_keys)
        assert np.allclose(pcheck.mean_log1p_cp10k, pcheck.mean_log1p_cp10k_old, atol=3e-05)
        assert np.allclose(pcheck.fraction_detected, pcheck.fraction_detected_old, atol=1e-12)
        checks = pd.DataFrame([{'check': 'primary_cells', 'value': int(primary.sum())}, {'check': 'ionocyte_cells', 'value': int(iongroup.n_cells.sum())}, {'check': 'ionocyte_state_pool_combinations', 'value': len(iongroup)}, {'check': 'min_state_pool_cells', 'value': int(iongroup.n_cells.min())}, {'check': 'max_error_existing_X_vs_log1p_cp10k', 'value': norm_error}, {'check': 'max_relative_difference_counts_row_sum_vs_stored_obs_total_counts', 'value': total_relative_error}, {'check': 'same_primary_count_as_frozen_audit', 'value': True}, {'check': 'observed_mixtures_reconstruct_direct_broad_estimands', 'value': True}, {'check': 'existing_shared_marker_means_and_detection_match', 'value': True}, {'check': 'H5AD_size_mtime_unchanged', 'value': hstat == (self.H5.stat().st_size, self.H5.stat().st_mtime_ns)}])
        self.write(checks, 'validation_checks.tsv')
        self.write(pd.DataFrame(self.inputs).drop_duplicates('input'), 'input_manifest.tsv')
        print(checks.to_string(index=False))
        print('FIXED WEIGHT CONTRASTS')
        print(mixchange.loc[mixchange.gene.isin(self.ION_GENES[:4]), ['gene', 'estimand', 'contrast', 'mean_difference', 'batch_direction_agrees']].to_string(index=False))

class MetabolicPrograms:

    def __init__(self, ctx):
        self.ctx = ctx
        self.OUT = ctx.output('')
        self.H5 = ctx.input('atlas_h5ad')
        self.K = ctx.input('matched_kegg_results')
        self.MODULES = {'KEGG:00190': 'Oxidative phosphorylation', 'KEGG:00020': 'TCA cycle', 'KEGG:00071': 'Fatty acid degradation', 'KEGG:00010': 'Glycolysis / Gluconeogenesis', 'KEGG:03010': 'Ribosome'}
        self.CONTRASTS = [(24, 0), (96, 0), (96, 24)]
        self.inputs = []

    def track(self, p):
        self.inputs.append(source_provenance(p))
        return p

    def tab(self, p):
        return read_table(self.track(p), sep='\t')

    def save(self, df, name):
        df.to_csv(self.OUT / name, sep='\t', index=False, lineterminator='\n')

    def get_sets(self):
        ua = self.tab(source_member(self.K, 'common_gene_universe_audit.tsv'))
        eligible = set(ua.loc[ua.common_mapping_eligible.astype(str).str.lower().eq('true'), 'gene'])
        nat = self.tab(source_member(self.K, 'native_kegg_union_catalog.tsv')).set_index('pathway')
        bridge = self.tab(self.ctx.input('eggnog_bridge'))
        bridge = bridge[bridge.in_emapper.astype(str).str.lower().eq('true')]
        sets = {}
        rows = []
        for pid, name in self.MODULES.items():
            n = set(str(nat.loc[pid, 'mapped_symbols']).split(';')) & eligible
            eid = pid.split(':')[1]
            e = set()
            for r in bridge.itertuples():
                if any((x in str(r.KEGG_Pathway_list).split(',') for x in ['ko' + eid, 'map' + eid])):
                    e.add(r.feature_symbol)
            e &= eligible
            for definition, gs in [('consensus_intersection', n & e), ('native', n), ('eggnog', e), ('union', n | e)]:
                sets[pid, definition] = sorted(gs)
                for g in sorted(gs):
                    rows.append(dict(pathway=pid, module=name, definition=definition, gene=g, in_native=g in n, in_eggnog=g in e))
        member = pd.DataFrame(rows)
        self.save(member, 'members.tsv')
        cov = member.groupby(['pathway', 'module', 'definition']).gene.nunique().reset_index(name='n_genes')
        cov['enough_for_summary'] = cov.n_genes >= 10
        self.save(cov, 'module_coverage.tsv')
        return sets

    def extract(self, sets):
        genes = sorted(set((g for gs in sets.values() for g in gs)))
        hstat = (self.H5.stat().st_size, self.H5.stat().st_mtime_ns)
        with h5py.File(self.track(self.H5), 'r') as f:
            obs = read_elem(f['obs'])
            var = read_elem(f['var'])
            relevant = ~obs.exclude_from_biology_v5.astype(bool) & obs.cell_type_v5.astype(str).isin(['ionocyte', 'pavement cell'])
            sc = 'cell_state_v5'
            assert sc in obs
            m = obs.loc[relevant, ['company_sample', 'cell_type_v5', sc, 'time_h', 'sequencing_batch']].copy()
            m.columns = ['sample_id', 'cell_type', 'state', 'time_h', 'sequencing_batch']
            for c in ['sample_id', 'cell_type', 'state', 'sequencing_batch']:
                m[c] = m[c].astype(str)
            m['time_h'] = m.time_h.astype(int)
            codes, labels = pd.factorize(pd.MultiIndex.from_frame(m[['sample_id', 'cell_type', 'state', 'time_h', 'sequencing_batch']]), sort=True)
            groups = pd.DataFrame(list(labels), columns=['sample_id', 'cell_type', 'state', 'time_h', 'sequencing_batch'])
            n = len(groups)
            rawsum = np.zeros((n, len(genes)))
            normsum = np.zeros_like(rawsum)
            detected = np.zeros_like(rawsum)
            allraw = np.zeros((n, len(var)), dtype=np.float64)
            umi = np.zeros(n)
            cells = np.bincount(codes, minlength=n)
            gi = var.index.get_indexer(genes)
            assert (gi >= 0).all()
            allcodes = np.full(len(obs), -1)
            allcodes[np.flatnonzero(relevant.to_numpy())] = codes
            for start in range(0, len(obs), 4096):
                stop = min(start + 4096, len(obs))
                keep = np.flatnonzero(allcodes[start:stop] >= 0)
                if len(keep) == 0:
                    continue
                raw = read_csr_rows(f['layers/counts'], start, stop)[keep]
                totals = np.asarray(raw.sum(axis=1)).ravel().astype(float)
                assert np.all(totals > 0)
                select = raw[:, gi].toarray().astype(float)
                cd = allcodes[start + keep]
                for code in np.unique(cd):
                    allraw[code] += np.asarray(raw[cd == code].sum(axis=0)).ravel()
                np.add.at(rawsum, cd, select)
                np.add.at(normsum, cd, select / totals[:, None] * 1000000.0)
                np.add.at(detected, cd, select > 0)
                np.add.at(umi, cd, totals)
            assert cells.sum() == 46346, (cells.sum(), groups)
            assert np.array_equal(allraw[:, gi], rawsum)
            assert (self.H5.stat().st_size, self.H5.stat().st_mtime_ns) == hstat
        groups['n_cells'] = cells
        groups['total_umi'] = umi
        groups['cell_fraction'] = groups.n_cells / groups.groupby(['cell_type', 'sample_id']).n_cells.transform('sum')
        groups['umi_fraction'] = groups.total_umi / groups.groupby(['cell_type', 'sample_id']).total_umi.transform('sum')
        self.save(groups, 'state_pool_weights.tsv')
        pd.DataFrame(allraw.astype(np.int64), columns=var.index).to_csv(self.OUT / 'counts.tsv.gz', sep='\t', index=False)
        frames = []
        for i, r in groups.iterrows():
            z = pd.DataFrame(dict(gene=genes, count=rawsum[i], CPM=rawsum[i] / umi[i] * 1000000.0, mean_cell_CPM=normsum[i] / cells[i], detection_fraction=detected[i] / cells[i]))
            for c, v in r.items():
                z[c] = v
            frames.append(z)
        expr = pd.concat(frames, ignore_index=True)
        self.save(expr, 'state_pool_gene_expression.tsv.gz')
        return (groups, expr)

    def summarize(self, groups, expr, sets):
        values = []
        linear = []
        refs = []
        for ct, gp in groups.groupby('cell_type', observed=True):
            ct_expr = expr[expr.cell_type == ct]
            for estimator, column, weight in [('umi_weighted', 'CPM', 'umi_fraction'), ('cell_weighted', 'mean_cell_CPM', 'cell_fraction')]:
                ref = gp[gp.time_h == 0].groupby('state')[weight].mean()
                assert np.isclose(ref.sum(), 1)
                for s, v in ref.items():
                    refs.append(dict(cell_type=ct, estimator=estimator, state=s, baseline_mean_weight=v))
                for (pid, definition), genes in sets.items():
                    if len(genes) < 10:
                        continue
                    part = ct_expr[ct_expr.gene.isin(genes)]
                    for pool, p in part.groupby('sample_id', observed=True):
                        meta = p.iloc[0]
                        mat = p.pivot(index='state', columns='gene', values=column).reindex(index=ref.index, columns=genes)
                        assert np.isfinite(mat.to_numpy()).all()
                        ws = gp[gp.sample_id == pool].set_index('state').loc[ref.index, weight]
                        summaries = {'observed': (mat.to_numpy() * ws.to_numpy()[:, None]).sum(axis=0), 'fixed_0h_composition': (mat.to_numpy() * ref.to_numpy()[:, None]).sum(axis=0)}
                        for st in mat.index:
                            summaries['state::' + st] = mat.loc[st].to_numpy()
                        for level, arr in summaries.items():
                            values.append(dict(cell_type=ct, sample_id=pool, time_h=int(meta.time_h), sequencing_batch=meta.sequencing_batch, pathway=pid, module=self.MODULES[pid], definition=definition, estimator=estimator, level=level, n_genes=len(genes), score=float(np.log2(arr + 0.5).mean()), sum_gene_CPM=float(arr.sum())))
                    for batch in ['batch_1', 'batch_2']:
                        b = gp[gp.sequencing_batch == batch]
                        for target, baseline in self.CONTRASTS:
                            ps = [b[b.time_h == t].sample_id.unique().item() for t in [baseline, target]]
                            x = [part[part.sample_id == p].groupby('state')[column].sum().reindex(ref.index).to_numpy() for p in ps]
                            w = [b[b.sample_id == p].set_index('state').loc[ref.index, weight].to_numpy() for p in ps]
                            inside = float(np.sum((w[1] + w[0]) / 2 * (x[1] - x[0])))
                            comp = float(np.sum((w[1] - w[0]) * (x[1] + x[0]) / 2))
                            total = float(np.dot(w[1], x[1]) - np.dot(w[0], x[0]))
                            assert np.isclose(inside + comp, total, atol=1e-07)
                            linear.append(dict(cell_type=ct, pathway=pid, module=self.MODULES[pid], definition=definition, estimator=estimator, sequencing_batch=batch, contrast=f'{target}_vs_{baseline}', total_CPM_change=total, within_state_component_CPM=inside, composition_component_CPM=comp))
        vals = pd.DataFrame(values)
        self.save(vals, 'module_pool_scores.tsv')
        self.save(pd.DataFrame(linear), 'symmetric_CPM_decomposition.tsv')
        self.save(pd.DataFrame(refs), 'fixed_reference_weights.tsv')
        contrasts = []
        keys = ['cell_type', 'pathway', 'module', 'definition', 'estimator', 'level', 'n_genes']
        for key, p in vals.groupby(keys, observed=True, sort=False):
            for target, baseline in self.CONTRASTS:
                r = dict(zip(keys, key))
                r['contrast'] = f'{target}_vs_{baseline}'
                for batch in ['batch_1', 'batch_2']:
                    q = p[p.sequencing_batch == batch]
                    r[batch] = float(q.loc[q.time_h == target, 'score'].item() - q.loc[q.time_h == baseline, 'score'].item())
                r['mean_difference'] = (r['batch_1'] + r['batch_2']) / 2
                r['direction_agrees'] = bool(np.sign(r['batch_1']) == np.sign(r['batch_2']))
                contrasts.append(r)
        con = pd.DataFrame(contrasts)
        self.save(con, 'module_contrasts.tsv')
        return (vals, con)

    def analyze(self):
        self.OUT.mkdir(parents=True, exist_ok=True)
        sets = self.get_sets()
        groups, expr = self.extract(sets)
        vals, con = self.summarize(groups, expr, sets)
        self.save(pd.DataFrame(self.inputs), 'input_manifest.tsv')
        (self.OUT / 'run_manifest.json').write_text(json.dumps(dict(status='complete', python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, n_cells=int(groups.n_cells.sum()), n_state_pools=len(groups), n_scores=len(vals), n_contrasts=len(con), script_sha256=sha(Path(__file__)), note='Exploratory; two independent tank pools per stage. No cell-level P-values; all definitions retained.'), ensure_ascii=False, indent=2), encoding='utf8')
        z = con[(con.definition == 'consensus_intersection') & (con.estimator == 'umi_weighted') & con.level.isin(['observed', 'fixed_0h_composition'])]
        print(z[['cell_type', 'module', 'level', 'contrast', 'batch_1', 'batch_2', 'mean_difference']].to_string(index=False), flush=True)

    def normalization_sensitivity(self):
        meta = read_table(self.OUT / 'state_pool_weights.tsv', sep='\t')
        rowmeta = meta.copy()
        rowmeta.insert(0, 'matrix_row_0based', np.arange(len(rowmeta)))
        rowmeta.insert(1, 'group_id', ['|'.join([str(r.cell_type), str(r.state), str(r.sample_id)]) for r in rowmeta.itertuples()])
        self.save(rowmeta, 'count_rows.tsv')
        counts = read_table(self.OUT / 'counts.tsv.gz', sep='\t')
        member = read_table(self.OUT / 'members.tsv', sep='\t')
        ann = read_table(self.ctx.input('gene_annotation'), sep='\t', dtype=str)
        bad = set(ann.loc[ann.biotype.eq('rRNA') | ann.is_mitochondrial.str.lower().eq('true'), 'gene_symbol'].dropna())
        good = ~counts.columns.isin(bad) & ~counts.columns.str.startswith('KEH09_')
        assert np.array_equal(counts.sum(axis=1).to_numpy(), meta.total_umi.to_numpy())
        sets = {key: p.gene.tolist() for key, p in member.groupby(['pathway', 'definition'], sort=False)}
        rows = []
        factors = []
        checks = []
        for ct, cm in meta.groupby('cell_type', sort=False):
            units = [('observed', cm.index.to_numpy())] + [('state::' + s, x.index.to_numpy()) for s, x in cm.groupby('state', sort=False)]
            for level, idx in units:
                md = cm.loc[idx].drop_duplicates('sample_id').set_index('sample_id')[['time_h', 'sequencing_batch']]
                raw = counts.loc[idx].groupby(meta.loc[idx, 'sample_id']).sum().reindex(md.index)
                eligible = good & (raw.sum(axis=0) >= 10) & ((raw > 0).sum(axis=0) >= 2)
                normalized, sf = deseq2_norm(raw.loc[:, eligible])
                pos = raw.loc[:, eligible].gt(0).all(axis=0)
                fit = raw.loc[:, eligible].loc[:, pos].to_numpy(float)
                manual = np.exp(np.median(np.log(fit) - np.log(fit).mean(axis=0), axis=1))
                assert np.allclose(manual, sf, rtol=1e-10)
                reference_library = float(np.median(raw.sum(axis=1).to_numpy() / sf))
                cpm = raw.div(raw.sum(axis=1), axis=0) * 1000000.0
                mor = raw.div(sf, axis=0) * 1000000.0 / reference_library
                for i, pool in enumerate(md.index):
                    factors.append(dict(cell_type=ct, level=level, sample_id=pool, size_factor=float(sf[i]), total_umi=int(raw.loc[pool].sum()), n_eligible_genes=int(eligible.sum()), n_all_six_positive=int(pos.sum()), CPM_to_MOR_multiplier=float(raw.loc[pool].sum() / sf[i] / reference_library)))
                for (pid, definition), genes in sets.items():
                    for norm, mat in [('CPM', cpm), ('median_ratio', mor)]:
                        score = np.log2(mat.loc[:, genes] + 0.5).mean(axis=1)
                        for target, baseline in self.CONTRASTS:
                            row = dict(cell_type=ct, level=level, pathway=pid, module=self.MODULES[pid], definition=definition, normalization=norm, contrast=f'{target}_vs_{baseline}', n_genes=len(genes))
                            for batch in ['batch_1', 'batch_2']:
                                a = md.index[(md.time_h == target) & (md.sequencing_batch == batch)].item()
                                b = md.index[(md.time_h == baseline) & (md.sequencing_batch == batch)].item()
                                row[batch] = float(score[a] - score[b])
                            row['mean_difference'] = (row['batch_1'] + row['batch_2']) / 2
                            row['direction_agrees'] = bool(np.sign(row['batch_1']) == np.sign(row['batch_2']))
                            rows.append(row)
        result = pd.DataFrame(rows)
        self.save(result, 'normalization_sensitivity.tsv')
        self.save(pd.DataFrame(factors), 'norm_factors.tsv')
        main = read_table(self.OUT / 'module_contrasts.tsv', sep='\t')
        check = result[result.normalization == 'CPM'].merge(main[main.estimator == 'umi_weighted'], on=['cell_type', 'level', 'pathway', 'module', 'definition', 'contrast', 'n_genes'], suffixes=('_new', '_old'), validate='one_to_one')
        err = float(np.max(np.abs(check.mean_difference_new - check.mean_difference_old)))
        assert err < 1e-12
        checks.append(dict(check='Independent broad and state CPM reconstruction', passed=True, max_error=err, n=len(check)))
        dec = read_table(self.OUT / 'symmetric_CPM_decomposition.tsv', sep='\t')
        err = float(np.max(np.abs(dec.total_CPM_change - dec.within_state_component_CPM - dec.composition_component_CPM)))
        checks.append(dict(check='Exact symmetric linear CPM decomposition', passed=err < 1e-07, max_error=err, n=len(dec)))
        expr = read_table(self.OUT / 'state_pool_gene_expression.tsv.gz', sep='\t')
        weightrows = []
        for ct, cm in meta.groupby('cell_type', sort=False):
            for estimator, col, wcol in [('umi_weighted', 'CPM', 'umi_fraction'), ('cell_weighted', 'mean_cell_CPM', 'cell_fraction')]:
                states = cm.state.unique()
                weights = {'equal_states': pd.Series(1 / len(states), index=states), 'mean_all_stages': cm.groupby('state')[wcol].mean(), 'mean_0h': cm[cm.time_h == 0].groupby('state')[wcol].mean()}
                for (pid, definition), genes in sets.items():
                    e = expr[(expr.cell_type == ct) & expr.gene.isin(genes)]
                    scores = {}
                    for ref, w in weights.items():
                        assert np.isclose(w.sum(), 1)
                        for pool, p in e.groupby('sample_id'):
                            mat = p.pivot(index='state', columns='gene', values=col).loc[w.index, genes]
                            scores[ref, pool] = float(np.log2((mat.to_numpy() * w.to_numpy()[:, None]).sum(axis=0) + 0.5).mean())
                        for target, baseline in self.CONTRASTS:
                            r = dict(cell_type=ct, estimator=estimator, reference=ref, pathway=pid, module=self.MODULES[pid], definition=definition, contrast=f'{target}_vs_{baseline}')
                            for batch in ['batch_1', 'batch_2']:
                                a = cm.loc[(cm.time_h == target) & (cm.sequencing_batch == batch), 'sample_id'].unique().item()
                                b = cm.loc[(cm.time_h == baseline) & (cm.sequencing_batch == batch), 'sample_id'].unique().item()
                                r[batch] = scores[ref, a] - scores[ref, b]
                            r['mean_difference'] = (r['batch_1'] + r['batch_2']) / 2
                            r['direction_agrees'] = bool(np.sign(r['batch_1']) == np.sign(r['batch_2']))
                            weightrows.append(r)
        self.save(pd.DataFrame(weightrows), 'reference_weight_sensitivity.tsv')
        self.save(pd.DataFrame(checks), 'validation_checks.tsv')
        (self.OUT / 'sensitivity_manifest.json').write_text(json.dumps({'pydeseq2': pydeseq2.__version__, 'normalization': 'median of ratios, each cell-type/state across six pools; original count>=10 and present>=2 and rRNA/mitochondrial/KEH09 exclusion for size-factor fitting; no DE model or P values refitted', 'interpretation': 'CPM estimates transcript allocation; median-ratio estimates expression relative to a majority-stable reference. Neither measures absolute RNA per cell or metabolic flux.', 'followup_reason': 'Check whether full-program directions depend on denominator and fixed state composition.'}, indent=2), encoding='utf8')
        z = result[(result.definition == 'consensus_intersection') & (result.cell_type == 'ionocyte') & (result.contrast == '96_vs_24') & result.pathway.eq('KEGG:00190')]
        print(z[['level', 'normalization', 'batch_1', 'batch_2', 'mean_difference']].to_string(index=False))

class RespiratoryComponents:

    def __init__(self, ctx):
        self.ctx = ctx
        self.OUT = ctx.output('')
        self.MET = ctx.input('metabolic_results')
        self.MAP = ctx.input('matched_kegg_results')
        self.ANN = ctx.input('gene_annotation')

    def table(self, path):
        return read_table(path, sep='\t', dtype=str)

    def save(self, df, name):
        df.to_csv(self.OUT / name, sep='\t', index=False, lineterminator='\n')

    def classify(self, description):
        d = str(description).lower()
        if re.search('v-type proton atpase|atpase.*h\\+.*(?:v0|v1)', d):
            return ('V_type_H_ATPase', 'description explicitly V-type or V0/V1 H+-ATPase', False, False)
        if 'nadh' in d and ('ubiquinone oxidoreductase' in d or 'dehydrogenase [ubiquinone]' in d):
            return ('respiratory_I_subunit', 'description explicitly NADH:ubiquinone oxidoreductase/dehydrogenase', True, True)
        if 'succinate dehydrogenase' in d:
            return ('respiratory_II_subunit', 'description explicitly succinate dehydrogenase', True, True)
        if 'ubiquinol-cytochrome c reductase' in d or 'cytochrome b-c1 complex' in d or d.strip() == 'cytochrome c-1':
            return ('respiratory_III_subunit', 'description explicitly ubiquinol-cytochrome c reductase/cytochrome b-c1/cytochrome c-1', True, True)
        if 'cytochrome c oxidase' in d:
            if 'assembly' in d or 'chaperone' in d:
                return ('respiratory_IV_assembly_associated', 'description explicitly COX assembly factor/chaperone, not direct catalytic subunit', False, True)
            if 'subunit' in d:
                return ('respiratory_IV_subunit', 'description explicitly cytochrome c oxidase subunit (including annotated -like)', True, True)
        if 'atp synthase' in d:
            return ('respiratory_V_ATP_synthase_subunit', 'description explicitly ATP synthase; not inferred from ATP gene-name prefix', True, True)
        if 'ndufa4 mitochondrial complex associated' in d:
            return ('mitochondrial_complex_unresolved', 'current GCF description does not resolve exact complex; not assigned from NDUF prefix', False, False)
        if 'phosphatase' in d:
            return ('other_phosphatase', 'phosphatase description does not establish respiratory-chain complex membership', False, False)
        return ('unresolved', 'description insufficient for prespecified mechanistic class', False, False)

    def analyze(self, final=False):
        members = self.table(source_member(self.MET, 'members.tsv'))
        members = members.loc[members.pathway.eq('KEGG:00190') & members.definition.eq('consensus_intersection')].copy()
        audit = self.table(source_member(self.MAP, 'all_feature_bidirectional_geneid_audit.tsv'))
        genome = self.table(self.ANN)
        features = members.merge(audit, on='gene', validate='1:1')
        ann = genome[['gene_id', 'gene_symbol', 'description', 'is_mitochondrial']].drop_duplicates()
        assert not ann.gene_id.duplicated().any()
        features = features.merge(ann, on='gene_id', how='left', validate='m:1', suffixes=('_audit', '_gcf'))
        assert features.description.notna().all()
        classes = features.description.map(self.classify)
        features['functional_class'] = [x[0] for x in classes]
        features['classification_evidence'] = [x[1] for x in classes]
        features['respiratory_subunit_core'] = [x[2] for x in classes]
        features['respiratory_plus_COX_assembly'] = [x[3] for x in classes]
        assert not features.is_mitochondrial.astype(str).str.lower().eq('true').any()
        features['classification_status'] = 'final_domain' if final else 'provisional_mapping_domain_do_not_cite'
        self.save(features, 'gene_classification_audit.tsv')
        self.save(features.groupby('functional_class', observed=True).size().rename('n_genes').reset_index(), 'functional_class_counts.tsv')
        manifest = {'status': 'final_domain' if final else 'provisional_mapping_domain_do_not_cite', 'n_genes': len(features), 'evidence': 'Exact input feature-to-GCF GeneID mapping audit joined to frozen assembly annotation description; no paralog identity inferred from name similarity.', 'files': {str(p): sha(p) for p in [source_member(self.MET, 'members.tsv'), source_member(self.MAP, 'all_feature_bidirectional_geneid_audit.tsv'), self.ANN]}, 'classes': features.functional_class.value_counts().to_dict()}
        (self.OUT / 'classification_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(features.groupby('functional_class', observed=True).size().to_string())
        if not final:
            return
        universe = self.table(source_member(self.MAP, 'common_gene_universe_audit.tsv'))
        eligible = set(universe.loc[universe.common_mapping_eligible.astype(str).str.lower().eq('true'), 'gene'])
        assert set(features.gene) <= eligible
        assert len(eligible) == 16575, 'Final audited domain changed: re-confirm before interpreting'
        manifest['final_common_mapping_eligible_count'] = len(eligible)
        for p in [source_member(self.MAP, 'common_gene_universe_audit.tsv'), source_member(self.MET, 'count_rows.tsv'), source_member(self.MET, 'counts.tsv.gz'), source_member(self.MET, 'norm_factors.tsv')]:
            manifest['files'][str(p)] = sha(p)
        (self.OUT / 'classification_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        meta = read_table(source_member(self.MET, 'count_rows.tsv'), sep='\t')
        counts = read_table(source_member(self.MET, 'counts.tsv.gz'), sep='\t')
        factors = read_table(source_member(self.MET, 'norm_factors.tsv'), sep='\t')
        sets = {'respiratory_subunit_core': features.loc[features.respiratory_subunit_core, 'gene'].tolist(), 'respiratory_plus_COX_assembly': features.loc[features.respiratory_plus_COX_assembly, 'gene'].tolist(), 'V_type_H_ATPase': features.loc[features.functional_class.eq('V_type_H_ATPase'), 'gene'].tolist()}
        rows = []
        for state, part in meta.loc[meta.cell_type.eq('ionocyte')].groupby('state', sort=False):
            sf = factors.loc[factors.cell_type.eq('ionocyte') & factors.level.eq('state::' + state)].set_index('sample_id')
            for m in part.itertuples():
                raw = counts.iloc[int(m.matrix_row_0based)]
                assert np.isclose(raw.sum(), m.total_umi)
                cpm = raw / float(m.total_umi) * 1000000.0
                mor = cpm * sf.loc[m.sample_id, 'CPM_to_MOR_multiplier']
                for group, genes in sets.items():
                    for norm, values in [('CPM', cpm), ('median_ratio', mor)]:
                        rows.append({'state': state, 'sample_id': m.sample_id, 'time_h': int(m.time_h), 'sequencing_batch': m.sequencing_batch, 'group': group, 'n_genes': len(genes), 'normalization': norm, 'score': float(np.log2(values.loc[genes] + 0.5).mean())})
        scores = pd.DataFrame(rows)
        self.save(scores, 'state_pool_component_scores.tsv')
        con = []
        for key, part in scores.groupby(['state', 'group', 'n_genes', 'normalization'], sort=False):
            row = dict(zip(['state', 'group', 'n_genes', 'normalization'], key))
            row['contrast'] = '96_vs_24'
            for batch in ['batch_1', 'batch_2']:
                p = part.loc[part.sequencing_batch.eq(batch)]
                row[batch] = float(p.loc[p.time_h.eq(96), 'score'].item() - p.loc[p.time_h.eq(24), 'score'].item())
            row['mean_difference'] = (row['batch_1'] + row['batch_2']) / 2
            row['both_negative'] = row['batch_1'] < 0 and row['batch_2'] < 0
            row['both_positive'] = row['batch_1'] > 0 and row['batch_2'] > 0
            con.append(row)
        c = pd.DataFrame(con)
        self.save(c, 'component_late_contrasts.tsv')
        assert len(scores) == 18 * 3 * 2
        assert scores.groupby(['state', 'group', 'normalization']).sample_id.nunique().eq(6).all()
        print(c.to_string(index=False))

def read_h5_column(group, key):
    x = group[key]
    if isinstance(x, h5py.Group):
        cats = x['categories'][:]
        cats = np.array([v.decode() if isinstance(v, bytes) else v for v in cats], dtype=object)
        codes = x['codes'][:]
        return np.array([cats[k] if k >= 0 else None for k in codes])
    arr = x[:]
    return np.array([v.decode() if isinstance(v, bytes) else v for v in arr]) if arr.dtype.kind in ['S', 'O'] else arr

def cross_cell_programs(ctx):
    bridge_path = ctx.input('eggnog_bridge')
    bridge = read_table(bridge_path, sep='\t', low_memory=False)
    bridge_sha = sha(bridge_path)
    n = str(bridge_path)
    bridge.loc[bridge.feature_symbol.eq('LOC102081935')].to_csv(ctx.output('LOC102081935_frozen_eggNOG_annotation.tsv'), sep='\t', index=False)
    ann = read_table(ctx.input('gene_annotation'), sep='\t', dtype=str)
    ann.loc[ann.gene_symbol.eq('LOC102081935')].to_csv(ctx.output('LOC102081935_frozen_RefSeq_annotation.tsv'), sep='\t', index=False)
    member = read_table(ctx.input('fixed_module_membership'), sep='\t')
    ox = member.loc[member.module.eq('OXPHOS_consensus_103'), 'gene'].tolist()
    assert len(ox) == 103 and len(set(ox)) == 103
    modules = {'OXPHOS_consensus_103': sorted(ox)}
    for label, term in [('Antigen_processing_ko04612', 'ko04612'), ('NFkB_signaling_ko04064', 'ko04064')]:
        modules[label] = sorted(set(bridge.loc[bridge.in_emapper.fillna(False).astype(bool) & bridge.KEGG_Pathway_list.fillna('').map(lambda v: term in v.split(',')), 'feature_symbol'].astype(str)))
    assert all((set(vals) == set(member.loc[member.module.eq(name), 'gene']) for name, vals in modules.items())), 'Fixed module membership differs from its frozen identity'
    pd.DataFrame([{'module': k, 'gene': g, 'source': 'frozen_consensus_intersection' if k.startswith('OX') else 'frozen_eggNOG_membership'} for k, v in modules.items() for g in v]).to_csv(ctx.output('') / 'fixed_module_membership.tsv', sep='\t', index=False)
    ar = np.load(ctx.input('atlas_group_counts'))
    counts = ar['counts']
    genes = ar['genes']
    det = ar['detected_cells']
    gm = read_table(ctx.input('atlas_group_metadata'), sep='\t')
    bad = set(ann.loc[ann.biotype.eq('rRNA') | ann.is_mitochondrial.str.lower().eq('true'), 'gene_symbol'].dropna())
    nuclear = np.array([g not in bad and (not g.startswith('KEH09_')) for g in genes])
    gi = {g: i for i, g in enumerate(genes)}
    assert all((g in gi for vals in modules.values() for g in vals))
    scores = []
    gene_rows = []
    ratio_den = np.zeros(len(gm))
    sf_rows = []
    for ct, idxs in gm.groupby('cell_type').groups.items():
        idxs = np.array(list(idxs))
        mat = counts[idxs]
        elig = (mat.sum(axis=0) >= 10) & ((mat > 0).sum(axis=0) >= 2) & nuclear
        logmeans, finite = deseq2_norm_fit(mat[:, elig])
        _, sf = deseq2_norm_transform(mat[:, elig], logmeans, finite)
        ratio_den[idxs] = sf * np.median(mat.sum(axis=1) / sf)
        for j, i in enumerate(idxs):
            sf_rows.append({'cell_type': ct, 'pool': gm.iloc[i].pool, 'size_factor': sf[j], 'eligible_genes': int(elig.sum()), 'ratio_reference_median_total_over_SF': float(np.median(mat.sum(axis=1) / sf))})
        if ct == 'dendritic/APC-like':
            frozen = read_table(ctx.input('apc_fit_metadata'), sep='\t')
            assert np.allclose(sf, frozen.size_factors.to_numpy(), rtol=1e-12)
    pd.DataFrame(sf_rows).to_csv(ctx.output('') / 'median_ratio_size_factors_by_celltype.tsv', sep='\t', index=False)
    for module, vals in modules.items():
        idx = np.array([gi[g] for g in vals])
        x = counts[:, idx]
        for norm, den in [('all_features_CPM', counts.sum(axis=1)), ('exclude_mito_rRNA_KEH_CPM', counts[:, nuclear].sum(axis=1)), ('median_ratio_per_million', ratio_den)]:
            cpm = x / den[:, None] * 1000000.0
            score = np.log2(cpm + 0.5).mean(axis=1)
            for j, row in gm.iterrows():
                scores.append({**row.to_dict(), 'module': module, 'normalization': norm, 'n_members_fixed': len(vals), 'n_members_detected_pool': int((x[j] > 0).sum()), 'score_mean_log2_CPM_plus_0p5': score[j]})
        for j, row in gm.iterrows():
            for k, g in enumerate(vals):
                gene_rows.append({'cell_type': row.cell_type, 'pool': row.pool, 'stage': row.stage, 'module': module, 'gene': g, 'raw_UMI': int(x[j, k]), 'CPM_all_features': x[j, k] / row.total_UMI * 1000000.0, 'detected_cells': int(det[j, idx[k]])})
    scores = pd.DataFrame(scores)
    scores.to_csv(ctx.output('') / 'fixed_modules_14_celltypes_6_pools.tsv', sep='\t', index=False)
    pd.DataFrame(gene_rows).to_csv(ctx.output('') / 'fixed_module_gene_sample_evidence.tsv.gz', sep='\t', index=False, compression='gzip')
    poolmap = {0: ['BR1', 'BR2'], 24: ['BR4', 'BR3'], 96: ['BR5', 'BR6']}
    changes = []
    for (ct, mo, no), sub in scores.groupby(['cell_type', 'module', 'normalization']):
        s = sub.set_index('pool').score_mean_log2_CPM_plus_0p5
        for t, b in [(24, 0), (96, 0), (96, 24)]:
            d = [s[poolmap[t][i]] - s[poolmap[b][i]] for i in [0, 1]]
            changes.append({'cell_type': ct, 'module': mo, 'normalization': no, 'contrast': f'{t}_vs_{b}', 'mean_delta_score': np.mean(d), 'batch1_delta': d[0], 'batch2_delta': d[1], 'both_batches_same_direction': bool(np.sign(d[0]) == np.sign(d[1]) and d[0] != 0), 'min_cells': int(sub.n_cells.min()), 'all6_at_least100': bool(sub.n_cells.min() >= 100)})
    changes = pd.DataFrame(changes)
    changes.to_csv(ctx.output('') / 'fixed_module_stage_differences_all.tsv', sep='\t', index=False)
    comp = changes.pivot(index=['cell_type', 'module', 'contrast'], columns='normalization', values='mean_delta_score').reset_index()
    comp['normalizations_same_direction'] = (np.sign(comp.all_features_CPM) == np.sign(comp.exclude_mito_rRNA_KEH_CPM)) & (np.sign(comp.all_features_CPM) == np.sign(comp.median_ratio_per_million))
    comp.to_csv(ctx.output('') / 'normalization_direction_sensitivity.tsv', sep='\t', index=False)
    with h5py.File(ctx.input('atlas_h5ad'), 'r') as f:
        obs = f['obs']
        ct = read_h5_column(obs, 'cell_type_v5')
        state = read_h5_column(obs, 'cell_state_v5')
        pools = read_h5_column(obs, 'biological_replicate')
        keep = ~read_h5_column(obs, 'exclude_from_biology_v5').astype(bool) & (ct == 'dendritic/APC-like')
        assert keep.sum() == 4264
        states = sorted(set(state[keep]))
        keys = [(s, p) for s in states for p in ['BR1', 'BR2', 'BR3', 'BR4', 'BR5', 'BR6']]
        lookup = {k: i for i, k in enumerate(keys)}
        labs = np.array([lookup.get((s, p), -1) for s, p in zip(state, pools)])
        nc = np.bincount(labs[keep], minlength=len(keys))
        sc = np.zeros((len(keys), len(genes)), dtype=np.int64)
        x = f['layers/counts']
        ptr = x['indptr'][:]
        for lo in range(0, len(pools), 1000):
            hi = min(lo + 1000, len(pools))
            pos = np.flatnonzero(keep[lo:hi])
            if not len(pos):
                continue
            a, b = (ptr[lo], ptr[hi])
            block = sparse.csr_matrix((x['data'][a:b], x['indices'][a:b], ptr[lo:hi + 1] - a), shape=(hi - lo, len(genes)))
            agg = sparse.csr_matrix((np.ones(len(pos), dtype=np.int64), (labs[lo:hi][pos], pos)), shape=(len(keys), hi - lo))
            sc += (agg @ block).toarray()
    apcidx = np.flatnonzero(gm.cell_type.eq('dendritic/APC-like'))
    assert np.array_equal(sc.reshape(2, 6, len(genes)).sum(axis=0), counts[apcidx])
    np.savez_compressed(ctx.output('') / 'APC_two_state_raw_counts.npz', counts=sc, genes=genes, states=np.array([s for s, p in keys]), pools=np.array([p for s, p in keys]))
    loc = gi['LOC102081935']
    state_rows = []
    for j, (s, p) in enumerate(keys):
        original = gm.iloc[apcidx[[a for a in range(6) if gm.iloc[apcidx[a]].pool == p][0]]]
        state_rows.append({'state': s, 'pool': p, 'stage': original.stage, 'sequencing_batch': original.sequencing_batch, 'n_cells': nc[j], 'cell_fraction_within_APC': nc[j] / original.n_cells, 'total_UMI': sc[j].sum(), 'UMI_fraction_within_APC': sc[j].sum() / original.total_UMI, 'LOC102081935_UMI': sc[j, loc], 'LOC102081935_CPM': sc[j, loc] / sc[j].sum() * 1000000.0})
    pd.DataFrame(state_rows).to_csv(ctx.output('') / 'APC_state_coverage_and_candidate_expression.tsv', sep='\t', index=False)
    manifest = {'selection': 'exploratory extension after stage1; no prespecification claim', 'modules': {k: len(v) for k, v in modules.items()}, 'membership_source_file': str(ctx.input('fixed_module_membership')), 'bridge_entry': n, 'bridge_sha256': bridge_sha, 'score': 'mean over fixed members of log2(expression+0.5) per independent tank pool', 'normalizations': ['all37484featuredenominator', 'exclude_mito_rRNA_KEH_denominator', 'median_ratio_per_million: count / SF * 1e6 / median(total_UMI / SF) within each cell type, same scaling as existing state sensitivity; factors use frozen gene filter and PyDESeq2 median-ratio fit only'], 'APC_size_factors_equal_original_model': True, 'comparisons': 'within cell type; batch-indexed stage differences; two independent tank pools/stage, not longitudinal fish', 'no_new_hypothesis_tests': True, 'APC_state_labels': states, 'APC_state_counts_sum_exactly_to_broad': True}
    (ctx.output('') / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(manifest, ensure_ascii=False))
    print(changes[changes.cell_type.isin(['T cell', 'macrophage', 'ionocyte', 'dendritic/APC-like']) & (changes.normalization == 'all_features_CPM')].to_string(index=False))
    print(pd.DataFrame(state_rows).to_string(index=False))

def validate_oxphos_identity(fixed_membership, route_membership, genes,
                            tested_backgrounds, contract, gmt_sha256):
    """Fail closed against the independently frozen complete membership contract.

    Gene identities are case-sensitive. Repeated contrast rows are expected, but
    duplicate genes within one module/contrast are invalid. No expected member
    set or background is inferred from the data being checked.
    """
    prefix = 'OXPHOS identity contract: '

    def require(condition, message):
        if not condition:
            raise ValueError(prefix + message)

    def unique(values, label):
        values = list(values)
        require(all(isinstance(g, str) and g and g == g.strip() for g in values),
                label + ' contains missing, blank, or non-exact gene identifiers')
        require(len(values) == len(set(values)), label + ' contains duplicate gene identifiers')
        return set(values)

    def digest(values):
        return hashlib.sha256(('\n'.join(sorted(values)) + '\n').encode('utf8')).hexdigest()

    require(isinstance(contract, dict), 'missing frozen config contracts.oxphos_identity')
    scope = {'schema_version': 1, 'module': 'OXPHOS_consensus_103',
             'cell_type': 'ionocyte', 'route': 'gprofiler', 'term': 'KEGG:00190',
             'contrasts': ['24_vs_0', '96_vs_0', '96_vs_24']}
    require(all(contract.get(k) == v for k, v in scope.items()),
            'configuration has the wrong fixed module, route, term, or contrasts')
    required = {'fixed_members', 'route_members', 'only_fixed', 'only_route',
                'background', 'contrast_resources'}
    require(required <= contract.keys(), 'incomplete frozen configuration')
    expected_fixed = unique(contract['fixed_members'], 'frozen fixed member contract')
    expected_route = unique(contract['route_members'], 'frozen route member contract')
    require(len(expected_fixed) == len(expected_route) == 103,
            'the frozen complete member lists must each contain 103 genes')
    only_fixed = unique(contract['only_fixed'], 'frozen fixed-only contract')
    only_route = unique(contract['only_route'], 'frozen route-only contract')
    require(only_fixed == {'LOC100690057', 'LOC100690645'} and
            only_route == {'LOC100692041', 'LOC100696271'},
            'configuration does not preserve the predefined two-out/two-in identities')
    require(expected_fixed - expected_route == only_fixed and
            expected_route - expected_fixed == only_route,
            'complete frozen lists disagree with the predefined symmetric difference')
    require({'module', 'gene'} <= set(fixed_membership.columns),
            'fixed membership is missing module/gene columns')
    actual_fixed = unique(fixed_membership.loc[
        fixed_membership.module.eq(scope['module']), 'gene'], 'fixed module')
    require(actual_fixed == expected_fixed,
            'fixed complete membership differs; missing=' + str(sorted(expected_fixed - actual_fixed)) +
            ', extra=' + str(sorted(actual_fixed - expected_fixed)))
    background = unique(genes, 'complete atlas feature background')
    bg_contract = contract['background']
    require(len(background) == bg_contract.get('n_genes') and
            digest(background) == bg_contract.get('members_sha256'),
            'complete atlas feature background differs from the frozen identity')
    require(expected_fixed | expected_route <= background,
            'a fixed or route member is absent from the atlas feature background')
    columns = {'cell_type', 'route', 'term', 'contrast', 'gene'}
    require(columns <= set(route_membership.columns), 'route membership is missing identity columns')
    route_rows = route_membership.loc[
        route_membership.cell_type.eq(scope['cell_type']) &
        route_membership.route.eq(scope['route']) & route_membership.term.eq(scope['term'])]
    contrasts = scope['contrasts']
    require(set(route_rows.contrast) == set(contrasts),
            'the required route must occur in exactly the three frozen contrasts')
    require(set(tested_backgrounds) == set(contrasts) and
            set(gmt_sha256) == set(contrasts) and
            set(contract['contrast_resources']) == set(contrasts),
            'missing or extra route background/resource contrasts')
    for contrast in contrasts:
        actual = unique(route_rows.loc[route_rows.contrast.eq(contrast), 'gene'],
                        'route ' + contrast)
        require(actual == expected_route,
                'route complete membership differs for ' + contrast +
                '; missing=' + str(sorted(expected_route - actual)) +
                ', extra=' + str(sorted(actual - expected_route)))
        require(actual_fixed - actual == only_fixed and actual - actual_fixed == only_route,
                'observed symmetric difference differs for ' + contrast)
        tested = unique(tested_backgrounds[contrast], 'tested DE background ' + contrast)
        resource = contract['contrast_resources'][contrast]
        require(len(tested) == resource.get('background_count') and
                digest(tested) == resource.get('background_members_sha256'),
                'complete tested DE background differs for ' + contrast)
        require(tested <= background and actual <= tested,
                'route/background genes are outside their parent background for ' + contrast)
        require(gmt_sha256[contrast] == resource.get('gmt_sha256'),
                'frozen gprofiler GMT resource identity differs for ' + contrast)
    return {'v5_103_intersection_unchanged': actual_fixed == expected_fixed,
            'v6_route_specific_103_is_not_same_set': expected_route != actual_fixed,
            'only_route_specific': sorted(expected_route - actual_fixed),
            'only_v5_original': sorted(actual_fixed - expected_route),
            'validation': {'complete_member_lists': True, 'unique_members': True,
                           'atlas_background': True, 'three_tested_DE_backgrounds': True,
                           'three_frozen_gprofiler_resources': True,
                           'predefined_symmetric_difference': True},
            'note': 'Both contain 103 genes but differ by two fixed-only and two route-only genes. '
                    'Complete frozen identities, uniqueness, route resources and backgrounds were verified. '
                    'Main metabolic 103 scores retain the fixed members; route-specific means are separately labeled checks.'}


def check_route_identity_inputs(ctx):
    """Validate all identity inputs before route_evidence writes any output."""
    contract = ctx.config.get('contracts', {}).get('oxphos_identity')
    if not isinstance(contract, dict):
        raise ValueError('OXPHOS identity contract: missing frozen config contracts.oxphos_identity')
    fixed = read_table(ctx.input('fixed_module_membership'), sep='\t')
    with np.load(ctx.input('atlas_group_counts'), allow_pickle=True) as arr:
        genes = arr['genes'].tolist()
    members, tested, hashes = [], {}, {}
    for contrast in ['24_vs_0', '96_vs_0', '96_vs_24']:
        dataset = 'broad__ionocyte__' + contrast
        source = source_member(ctx.input('matched_kegg_results'), Path('gene_sets') / (dataset + '__gprofiler.gmt'))
        rows = [line.split('\t') for line in source_text(source).splitlines()]
        selected = [row for row in rows if row[0] == 'KEGG:00190']
        if len(selected) != 1 or len(selected[0]) < 3:
            raise ValueError('OXPHOS identity contract: expected exactly one KEGG:00190 GMT row for ' + contrast)
        members.extend({'cell_type': 'ionocyte', 'route': 'gprofiler', 'term': 'KEGG:00190',
                        'contrast': contrast, 'gene': gene} for gene in selected[0][2:])
        hashes[contrast] = sha(source)
        tested[contrast] = read_table(source_member(ctx.input('de_results'), dataset + '.tsv.gz'),
                                       sep='\t', usecols=['gene']).gene.tolist()
    return validate_oxphos_identity(fixed, pd.DataFrame(members), genes, tested, contract, hashes)


def route_evidence(ctx):
    identity_note = check_route_identity_inputs(ctx)

    def asset(relative):
        prefixes = {'Current_Analyses/kegg_mapping_recovery/': 'matched_kegg_results', 'Data/02_DE/': 'de_results'}
        for prefix, key in prefixes.items():
            if relative.startswith(prefix):
                path = source_member(ctx.input(key),relative[len(prefix):])
                if not source_exists(path):
                    raise FileNotFoundError(str(path))
                return path
        raise ValueError('Unknown evidence asset: ' + relative)

    def read_frozen_bytes(relative):
        path = asset(relative)
        return source_text(path).encode('utf-8')

    def read_frozen_frame(relative):
        return read_table(asset(relative), sep='\t')
    scope = {'check1': 'PVC tight junction KEGG:04530: all three contrasts, full members, all leading edges, both batches', 'check2': 'previously reported ionocyte oxidative/TCA/fatty-acid programs and transport sentinels; eight existing T/macrophage programs; APC candidate/state and existing common scores', 'check3': 'PVC Notch and fatty-acid degradation: full three comparisons, leading genes, effect and batch directions', 'selected_pathways': {'pavement cell': ['KEGG:04530', 'KEGG:04330', 'KEGG:00071'], 'ionocyte': ['KEGG:00190', 'KEGG:00020', 'KEGG:00071']}, 'rule': 'No new significance tests; no changing FDR thresholds; no searching additional pathways. Fixed-member CPM summaries only describe pooled expression; separately shrunken contrasts never subtracted to build trajectories.'}
    ctx.output('prespecified_scope.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2), encoding='utf8')
    arr = np.load(ctx.input('atlas_group_counts'), allow_pickle=True)
    genes = arr['genes'].astype(str)
    idx = {g: i for i, g in enumerate(genes)}
    meta = read_table(ctx.input('atlas_group_metadata'), sep='\t')
    counts = arr['counts']
    ann = read_table(ctx.input('gene_annotation'), sep='\t').drop_duplicates('gene_symbol').set_index('gene_symbol')
    selected = []
    scores = []
    detailed = []
    edges = []
    summaries = []
    members = []
    gene_pools = []
    comp = read_frozen_frame('Current_Analyses/kegg_mapping_recovery/matched_route_comparison.tsv.gz')
    for cell, terms in scope['selected_pathways'].items():
        for contrast in ['24_vs_0', '96_vs_0', '96_vs_24']:
            dataset = 'broad__' + cell.replace(' ', '_') + '__' + contrast
            co = comp[(comp.dataset == dataset) & comp.Term.isin(terms)]
            assert len(co) == len(terms)
            selected.append(co)
            de = read_frozen_frame('Data/02_DE/' + dataset + '.tsv.gz').set_index('gene')
            for route in ['gprofiler', 'eggnog']:
                gmt = {line.split('\t')[0]: line.split('\t')[2:] for line in read_frozen_bytes('Current_Analyses/kegg_mapping_recovery/gene_sets/' + dataset + '__' + route + '.gmt').decode().splitlines()}
                le = read_frozen_frame('Current_Analyses/kegg_mapping_recovery/leading_edges/' + dataset + '__' + route + '.tsv.gz')
                for term in terms:
                    gs = gmt[term]
                    dd = de.reindex(gs).copy()
                    dd['gene'] = gs
                    dd['term'] = term
                    dd['route'] = route
                    dd['cell_type'] = cell
                    dd['annotation'] = [ann.loc[g, 'description'] if g in ann.index else '' for g in gs]
                    detailed.append(dd.reset_index(drop=True))
                    ll = le[le.pathway == term].copy()
                    nes = float(co.loc[co.Term == term, 'NES_' + route].iloc[0])
                    q = float(co.loc[co.Term == term, 'FDR q-val_' + route].iloc[0])
                    sgn = np.sign(nes)
                    good = (np.sign(ll.paired_log2cpm_diff_batch_1) == sgn) & (np.sign(ll.paired_log2cpm_diff_batch_2) == sgn)
                    ll['NES'] = nes
                    ll['q'] = q
                    ll['both_batch_match_NES'] = good
                    ll['annotation'] = [ann.loc[g, 'description'] if g in ann.index else '' for g in ll.gene]
                    edges.append(ll)
                    summaries.append({'cell_type': cell, 'contrast': contrast, 'route': route, 'term': term, 'NES': nes, 'q': q, 'n_members': len(gs), 'n_leading': len(ll), 'leading_both_batch_match_NES': int(good.sum()), 'leading_fraction_both_match': float(good.mean()), 'leading_median_shrunk_LFC': float(ll.log2FoldChange_shrunken.median()), 'leading_median_batch1': float(ll.paired_log2cpm_diff_batch_1.median()), 'leading_median_batch2': float(ll.paired_log2cpm_diff_batch_2.median())})
                    members.extend(({'cell_type': cell, 'contrast': contrast, 'route': route, 'term': term, 'gene': g} for g in gs))
                    if contrast == '24_vs_0':
                        for k, m in meta[meta.cell_type == cell].iterrows():
                            cpm = counts[k] / m.total_UMI * 1000000.0
                            vals = np.log2(cpm[[idx[g] for g in gs]] + 0.5)
                            scores.append({'cell_type': cell, 'route': route, 'term': term, **m[['pool', 'stage', 'sequencing_batch']].to_dict(), 'n_members': len(gs), 'score': float(vals.mean())})
            if contrast == '24_vs_0':
                poolgenes = set((g for row in members if row['cell_type'] == cell for g in [row['gene']]))
                poolgenes.update(['cebpd', 'scel', 'hopx', 'muc15', 'abca12', 'evpl'] if cell == 'pavement cell' else ['cftr', 'LOC112841652', 'LOC100699477', 'foxi2', 'slc4a4', 'trpv6', 'atp1b1b', 'foxi3b'])
                for k, m in meta[meta.cell_type == cell].iterrows():
                    for g in sorted(poolgenes):
                        if g in idx:
                            gene_pools.append({'cell_type': cell, 'gene': g, **m[['pool', 'stage', 'sequencing_batch']].to_dict(), 'count': int(counts[k, idx[g]]), 'CPM_all_features': float(counts[k, idx[g]] / m.total_UMI * 1000000.0), 'annotation': ann.loc[g, 'description'] if g in ann.index else ''})
    for cell, sg in [('pavement cell', ['cebpd', 'scel', 'hopx', 'muc15', 'abca12', 'evpl']), ('ionocyte', ['cftr', 'LOC112841652', 'LOC100699477', 'foxi2', 'slc4a4', 'trpv6', 'atp1b1b', 'foxi3b'])]:
        dd = []
        for contrast in ['24_vs_0', '96_vs_0', '96_vs_24']:
            a = read_frozen_frame('Data/02_DE/broad__' + cell.replace(' ', '_') + '__' + contrast + '.tsv.gz')
            dd.append(a[a.gene.isin(sg)])
        pd.concat(dd).to_csv(ctx.output('') / (cell.replace(' ', '_') + '_prespecified_sentinels.tsv'), sep='\t', index=False)
    member = pd.DataFrame(members)
    for key, g in member.groupby(['cell_type', 'route', 'term']):
        sets = [set(q.gene) for _, q in g.groupby('contrast')]
        assert len(sets) == 3 and sets[0] == sets[1] == sets[2], key
    pool = pd.DataFrame(scores)
    delta = []
    for key, g in pool.groupby(['cell_type', 'route', 'term']):
        row = dict(zip(['cell_type', 'route', 'term'], key))
        for num, den in [(24, 0), (96, 0), (96, 24)]:
            rr = {**row, 'contrast': f'{num}_vs_{den}'}
            for b in ['batch_1', 'batch_2']:
                a = g[g.sequencing_batch == b].set_index('stage')
                rr[b] = float(a.loc[num, 'score'] - a.loc[den, 'score'])
            rr['mean_delta'] = (rr['batch_1'] + rr['batch_2']) / 2
            rr['same_batch_direction'] = np.sign(rr['batch_1']) == np.sign(rr['batch_2'])
            delta.append(rr)
    outs = {'three_comparison_KEGG_frozen.tsv': pd.concat(selected), 'all_members_with_frozen_DE.tsv': pd.concat(detailed), 'leading_edges_with_two_batch_check.tsv': pd.concat(edges), 'leading_edge_summary.tsv': pd.DataFrame(summaries), 'fixed_membership_verified.tsv': member, 'fixed_members_pool_scores.tsv': pool, 'fixed_members_stage_changes.tsv': pd.DataFrame(delta), 'selected_gene_pool_CPM.tsv': pd.DataFrame(gene_pools)}
    im = read_table(ctx.input('immune_leading_edge_audit'), sep='\t')
    selected_im = []
    for cell, terms in [('T cell', ['ko04660', 'ko04064', 'ko04612', 'ko00190']), ('macrophage', ['GO:0002526', 'ko04612', 'ko04145', 'ko00190'])]:
        selected_im.append(im[(im.group == cell) & im.Term.isin(terms)])
    outs['existing_immune_programs_all_three.tsv'] = pd.concat(selected_im)
    common = read_table(ctx.input('fixed_module_stage_contrasts'), sep='\t')
    outs['existing_common_scores_relevant_cells.tsv'] = common[common.cell_type.isin(['ionocyte', 'T cell', 'macrophage', 'dendritic/APC-like'])]
    apc = []
    for c in ['24_vs_0', '96_vs_0', '96_vs_24']:
        d = read_table(source_member(ctx.input('apc_de_results'), 'broad__dendritic_APC-like__' + c + '.tsv.gz'), sep='\t')
        apc.append(d[d.gene == 'LOC102081935'])
    outs['APC_candidate_separately_shrunken_contrasts.tsv'] = pd.concat(apc)
    for n, d in outs.items():
        d.to_csv(ctx.output('') / n, sep='\t', index=False)
    copy_source(ctx.input('apc_state_coverage_and_candidate_expression'), ctx.output('APC_state_coverage_and_candidate_expression.tsv'))
    sf = read_table(ctx.input('cross_cell_size_factors'), sep='\t').set_index(['cell_type', 'pool'])
    ann2 = read_table(ctx.input('gene_annotation'), sep='\t', dtype=str)
    bad = set(ann2.loc[ann2.biotype.eq('rRNA') | ann2.is_mitochondrial.str.lower().eq('true'), 'gene_symbol'].dropna())
    good = np.array([g not in bad and (not g.startswith('KEH09_')) for g in genes])
    ss = []
    for key, rows in member[member.contrast == '24_vs_0'].groupby(['cell_type', 'route', 'term']):
        for k, m in meta[meta.cell_type == key[0]].iterrows():
            f = sf.loc[key[0], m.pool]
            denoms = {'all_features_CPM': m.total_UMI, 'exclude_mito_rRNA_KEH_CPM': counts[k, good].sum(), 'existing_v5_median_ratio': f.size_factor * f.ratio_reference_median_total_over_SF}
            for norm, den in denoms.items():
                x = counts[k, [idx[g] for g in rows.gene]] / den * 1000000.0
                ss.append(dict(zip(['cell_type', 'route', 'term'], key)) | {'normalization': norm, 'pool': m.pool, 'stage': m.stage, 'batch': m.sequencing_batch, 'score': float(np.log2(x + 0.5).mean())})
    ss = pd.DataFrame(ss)
    ss.to_csv(ctx.output('route_members_pool_scores_three_scales.tsv'), sep='\t', index=False)
    normrows = []
    for key, g in ss.groupby(['cell_type', 'route', 'term', 'normalization']):
        for a, b in [(24, 0), (96, 0), (96, 24)]:
            row = dict(zip(['cell_type', 'route', 'term', 'normalization'], key)) | {'contrast': f'{a}_vs_{b}'}
            for batch in ['batch_1', 'batch_2']:
                t = g[g.batch == batch].set_index('stage').score
                row[batch] = float(t[a] - t[b])
            row['mean_delta'] = (row['batch_1'] + row['batch_2']) / 2
            normrows.append(row)
    pd.DataFrame(normrows).to_csv(ctx.output('route_members_differences_three_scales.tsv'), sep='\t', index=False)
    ctx.output('module_identity_note.json').write_text(json.dumps(identity_note, indent=2), encoding='utf8')
    print(pd.DataFrame(summaries).round(4).to_string(index=False))
    print('\nFULL-MEMBER SCORE DELTAS\n', pd.DataFrame(delta).round(4).to_string(index=False))
INPUTS_BY_TASK = {'state-composition': ['atlas_h5ad', 'ionocyte_state_composition_by_pool', 'de_results', 'pavement_state_marker_summary'], 'metabolism': ['atlas_h5ad', 'matched_kegg_results', 'eggnog_bridge', 'gene_annotation'], 'respiratory': ['metabolic_results', 'matched_kegg_results', 'gene_annotation'], 'cross-cell': ['atlas_h5ad', 'atlas_group_counts', 'atlas_group_metadata', 'apc_fit_metadata', 'eggnog_bridge', 'gene_annotation', 'fixed_module_membership'], 'route-evidence': ['atlas_group_counts', 'atlas_group_metadata', 'gene_annotation', 'matched_kegg_results', 'de_results', 'immune_leading_edge_audit', 'fixed_module_stage_contrasts', 'apc_de_results', 'apc_state_coverage_and_candidate_expression', 'cross_cell_size_factors', 'fixed_module_membership'], 'state-robustness': ['atlas_h5ad']}

def run(ctx, task='metabolism'):
    ctx.require(INPUTS_BY_TASK[task])
    if task == 'state-composition':
        StateComposition(ctx).analyze()
    elif task == 'metabolism':
        a = MetabolicPrograms(ctx)
        a.analyze()
        a.normalization_sensitivity()
    elif task == 'respiratory':
        RespiratoryComponents(ctx).analyze(final=True)
    elif task == 'cross-cell':
        cross_cell_programs(ctx)
    elif task == 'route-evidence':
        route_evidence(ctx)
    elif task == 'state-robustness':
        from . import state_robustness
        return state_robustness.run(ctx, task)
    return {'files': [str(p.relative_to(ctx.output(''))) for p in ctx.output('').rglob('*') if p.is_file()]}
