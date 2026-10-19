"""Exact identifier, orthology, LOC collision and state-marker evidence for the current analysis."""
from __future__ import annotations
from .runtime import read_table
import gzip
import json
import pandas as pd
import h5py
from anndata.io import read_elem
import re
from scipy import sparse


def read_csr_rows(group, start, stop):
    """Read contiguous rows from an AnnData CSR group without materializing the atlas."""
    ptr = group['indptr'][start:stop + 1]
    a, b = (int(ptr[0]), int(ptr[-1]))
    return sparse.csr_matrix((group['data'][a:b], group['indices'][a:b], ptr - ptr[0]), shape=(stop - start, int(group.attrs['shape'][1])))


class GeneIdentity:

    def __init__(self, ctx):
        self.ctx = ctx
        self.OUT = ctx.output('')
        self.H5 = ctx.input('atlas_h5ad')

    def write(self, x, name):
        x.to_csv(self.OUT / name, sep='\t', index=False, lineterminator='\n')


    def clean_id(self, v):
        return str(v).replace('.0', '') if pd.notna(v) else ''




    def audit_current_mapping(self):
        before = (self.H5.stat().st_size, self.H5.stat().st_mtime_ns)
        ann = read_table(self.ctx.input('gene_annotation'), sep='\t', dtype=str).fillna('')
        with h5py.File(self.H5, 'r') as f:
            var = read_elem(f['var'])
            obs = read_elem(f['obs'])
            unique_ann = ann.loc[~ann.gene_id.duplicated(False)]
            by_id = unique_ann.set_index('gene_id')
            by_symbol = ann.loc[~ann.gene_symbol.duplicated(False)].set_index('gene_symbol')
            claims = {}
            for feature, row in var.iterrows():
                ids = set()
                old = self.clean_id(row.gene_id)
                if old:
                    ids.add(old)
                if feature in by_symbol.index:
                    ids.add(by_symbol.loc[feature, 'gene_id'])
                if re.fullmatch('LOC\\d+', feature) and feature[3:] in by_id.index:
                    ids.add(feature[3:])
                for geneid in ids:
                    claims.setdefault(geneid, []).append(feature)
            loc = []
            for feature, row in var.iterrows():
                if not re.fullmatch('LOC\\d+', feature):
                    continue
                geneid = feature[3:]
                old = self.clean_id(row.gene_id)
                hits = ann.loc[ann.gene_id.eq(geneid)]
                others = sorted(set(claims.get(geneid, [])) - {feature})
                r = {'matrix_feature': feature, 'literal_gene_id': geneid, 'original_h5ad_gene_id': old, 'original_mapping_status': row.symbol_mapping_status, 'assembly_records': len(hits), 'current_assembly_symbol': ';'.join(hits.gene_symbol), 'other_feature_columns_claiming_same_gene_id': ';'.join(others)}
                if hits.empty:
                    status = 'not_in_exact_assembly'
                elif len(hits) != 1:
                    status = 'nonunique_assembly_gene_id'
                elif old and old != geneid:
                    status = 'conflicting_original_h5ad_gene_id'
                elif others:
                    status = 'collision_other_independent_feature_column'
                elif old == geneid:
                    status = 'already_exact_gene_id'
                else:
                    status = 'safe_candidate_numeric_gene_id_recovery'
                r['status'] = status
                r['safe_bridge_candidate'] = status in ('already_exact_gene_id', 'safe_candidate_numeric_gene_id_recovery')
                r['symbol_changes_from_literal_LOC'] = len(hits) == 1 and hits.iloc[0].gene_symbol != feature
                loc.append(r)
            loc = pd.DataFrame(loc)
            self.write(loc, 'all_LOC_feature_geneid_bridge_audit.tsv')
            self.write(loc.loc[loc.status.eq('collision_other_independent_feature_column') | loc.status.eq('conflicting_original_h5ad_gene_id')], 'all_LOC_geneid_conflicts.tsv')
            self.write(loc.groupby(['status', 'symbol_changes_from_literal_LOC']).size().reset_index(name='n_features'), 'all_LOC_bridge_summary.tsv')
        assert before == (self.H5.stat().st_size, self.H5.stat().st_mtime_ns)



class StrictOrthology:

    def __init__(self, ctx):
        self.ctx = ctx
        self.ORTHOLOGS = ctx.input('ncbi_gene_orthologs')
        self.HUMAN_INFO = ctx.input('human_gene_info')
        self.BRIDGE = ctx.input('eggnog_bridge')
        self.TILAPIA_TAXID = 8128
        self.HUMAN_TAXID = 9606

    def read_tilapia_human_pairs(self) -> pd.DataFrame:
        pairs: set[tuple[int, int]] = set()
        with gzip.open(self.ORTHOLOGS, 'rt', encoding='utf-8') as handle:
            header = next(handle).rstrip('\n').split('\t')
            if header != ['#tax_id', 'GeneID', 'relationship', 'Other_tax_id', 'Other_GeneID']:
                raise RuntimeError(f'Unexpected NCBI ortholog header: {header}')
            for line in handle:
                tax_id, gene_id, relationship, other_tax_id, other_gene_id = line.rstrip('\n').split('\t')
                if relationship != 'Ortholog':
                    continue
                if tax_id == str(self.TILAPIA_TAXID) and other_tax_id == str(self.HUMAN_TAXID):
                    pairs.add((int(gene_id), int(other_gene_id)))
                elif tax_id == str(self.HUMAN_TAXID) and other_tax_id == str(self.TILAPIA_TAXID):
                    pairs.add((int(other_gene_id), int(gene_id)))
        frame = pd.DataFrame(sorted(pairs), columns=['tilapia_gene_id', 'human_gene_id'])
        frame['n_human_per_tilapia'] = frame.groupby('tilapia_gene_id')['human_gene_id'].transform('nunique')
        frame['n_tilapia_per_human'] = frame.groupby('human_gene_id')['tilapia_gene_id'].transform('nunique')
        frame['strict_one_to_one'] = frame['n_human_per_tilapia'].eq(1) & frame['n_tilapia_per_human'].eq(1)
        return frame

    def human_gene_symbols(self) -> pd.DataFrame:
        frame = read_table(self.HUMAN_INFO, sep='\t', compression='gzip', usecols=['#tax_id', 'GeneID', 'Symbol'], dtype={'#tax_id': int, 'GeneID': int, 'Symbol': str})
        frame = frame.loc[frame['#tax_id'].eq(self.HUMAN_TAXID), ['GeneID', 'Symbol']].copy()
        frame.columns = ['human_gene_id', 'human_symbol']
        frame['human_symbol'] = frame['human_symbol'].str.upper()
        return frame.drop_duplicates('human_gene_id')

    def tilapia_matrix_symbols(self) -> pd.DataFrame:
        frame = read_table(self.BRIDGE, sep='\t', usecols=['feature_symbol', 'gene_id'])
        frame = frame.dropna(subset=['gene_id']).copy()
        frame['tilapia_gene_id'] = frame['gene_id'].astype(int)
        frame['tilapia_symbol'] = frame['feature_symbol'].astype(str).str.upper()
        if frame['tilapia_gene_id'].duplicated().any():
            duplicated = frame.loc[frame['tilapia_gene_id'].duplicated(False), 'tilapia_gene_id'].unique()
            raise RuntimeError(f'GeneID-to-feature mapping is not unique for {len(duplicated)} genes')
        return frame[['tilapia_gene_id', 'tilapia_symbol']]

    def translated_mapping(self) -> tuple[pd.DataFrame, dict[str, int]]:
        pairs = self.read_tilapia_human_pairs()
        strict = pairs.loc[pairs['strict_one_to_one']].copy()
        mapping = strict.merge(self.human_gene_symbols(), on='human_gene_id', how='left')
        mapping = mapping.merge(self.tilapia_matrix_symbols(), on='tilapia_gene_id', how='left')
        mapping['measured_feature'] = mapping['tilapia_symbol'].notna()
        mapping['liana_compatible_feature'] = mapping['measured_feature'] & ~mapping['tilapia_symbol'].fillna('').str.contains('_', regex=False)
        counts = {'ncbi_tilapia_human_pairs': int(len(pairs)), 'strict_one_to_one_pairs': int(len(strict)), 'strict_pairs_with_human_symbol': int(mapping['human_symbol'].notna().sum()), 'strict_pairs_with_measured_tilapia_feature': int(mapping['measured_feature'].sum()), 'strict_pairs_with_liana_compatible_feature': int(mapping['liana_compatible_feature'].sum()), 'excluded_underscore_ambiguous_features': int((mapping['measured_feature'] & ~mapping['liana_compatible_feature']).sum())}
        return (mapping, counts)
INPUTS_BY_TASK = {'orthology': ['ncbi_gene_orthologs', 'human_gene_info', 'eggnog_bridge'], 'identity': ['atlas_h5ad', 'gene_annotation']}

def run(ctx, task='identity'):
    ctx.require(INPUTS_BY_TASK[task])
    if task == 'orthology':
        mapping, counts = StrictOrthology(ctx).translated_mapping()
        path = ctx.output('ncbi_tilapia_human_one_to_one_orthologs.tsv.gz')
        mapping.to_csv(path, sep='\t', index=False, compression='gzip')
        ctx.output('orthology_mapping_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf8')
    else:
        # Audit exact-assembly GeneIDs and collisions without changing features.
        analysis = GeneIdentity(ctx)
        analysis.audit_current_mapping()
    return {'files': [p.name for p in ctx.output('').iterdir() if p.is_file()]}
