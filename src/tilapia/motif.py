"""Exact frozen-assembly promoter scans and six-TF adjusted motif enrichment."""
from __future__ import annotations
from .runtime import read_table, source_member, iter_sources, source_provenance, source_size, source_exists, source_label, source_text, copy_source
import gzip
import json
from pathlib import Path
from Bio import motifs
from Bio.Seq import Seq
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from .runtime import sha256 as sha

class PromoterMotifs:

    def __init__(self, ctx):
        self.ctx = ctx
        self.OUT = ctx.output('')
        self.GENOME = ctx.input('genome_fasta')
        self.ANNOTATION = ctx.input('gene_annotation')
        self.NETWORK = ctx.input('tf_network')
        self.DE = ctx.input('ionocyte_late_de')
        self.MOTIF_FILES = {tf: ctx.input('motif_' + tf.lower()) for tf in ['KLF4', 'MYB', 'NFKB1', 'RELA', 'SOX4', 'RBPJ']}
        self.PROMOTER_UPSTREAM = 1500
        self.PROMOTER_DOWNSTREAM = 250
        self.MOTIF_FPR = 0.0001
        self.COMPLEMENT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')

    def prepare_gene_table(self) -> pd.DataFrame:
        annotation = read_table(self.ANNOTATION, sep='\t', low_memory=False)
        de = read_table(self.DE, sep='\t')
        de['symbol_upper'] = de['gene'].astype(str).str.upper()
        de = de.sort_values(['symbol_upper', 'baseMean'], ascending=[True, False]).drop_duplicates('symbol_upper')
        annotation['symbol_upper'] = annotation['gene_symbol'].astype(str).str.upper()
        annotation = annotation.loc[annotation['chromosome'].notna() & annotation['start'].notna() & annotation['end'].notna() & annotation['strand'].isin(['+', '-'])].copy()
        annotation['mapping_priority'] = annotation['symbol_mapping_status'].map({'unique': 0}).fillna(1)
        annotation = annotation.sort_values(['symbol_upper', 'mapping_priority', 'start'], ascending=[True, True, True]).drop_duplicates('symbol_upper')
        table = de.merge(annotation[['symbol_upper', 'gene_symbol', 'gene_id', 'chromosome', 'start', 'end', 'strand']], on='symbol_upper', how='inner')
        table = table.loc[table['baseMean'].ge(5)].copy()
        table['start'] = table['start'].astype(int)
        table['end'] = table['end'].astype(int)
        return table

    def fasta_records(self, path: Path):
        name = None
        sequence_parts: list[str] = []
        with gzip.open(path, 'rt', encoding='ascii') as handle:
            for line in handle:
                if line.startswith('>'):
                    if name is not None:
                        yield (name, ''.join(sequence_parts))
                    name = line[1:].split()[0]
                    sequence_parts = []
                else:
                    sequence_parts.append(line.strip().upper())
            if name is not None:
                yield (name, ''.join(sequence_parts))

    def extract_promoters(self, table: pd.DataFrame) -> pd.DataFrame:
        intervals = {chromosome: frame.copy() for chromosome, frame in table.groupby('chromosome', observed=True)}
        sequences: dict[str, str] = {}
        truncated: dict[str, bool] = {}
        for chromosome, sequence in self.fasta_records(self.GENOME):
            frame = intervals.get(chromosome)
            if frame is None:
                continue
            chromosome_length = len(sequence)
            for row in frame.itertuples(index=False):
                if row.strand == '+':
                    left = max(0, int(row.start) - 1 - self.PROMOTER_UPSTREAM)
                    right = min(chromosome_length, int(row.start) - 1 + self.PROMOTER_DOWNSTREAM)
                    promoter = sequence[left:right]
                else:
                    left = max(0, int(row.end) - self.PROMOTER_DOWNSTREAM)
                    right = min(chromosome_length, int(row.end) + self.PROMOTER_UPSTREAM)
                    promoter = sequence[left:right].translate(self.COMPLEMENT)[::-1]
                sequences[row.symbol_upper] = promoter
                truncated[row.symbol_upper] = len(promoter) < self.PROMOTER_UPSTREAM + self.PROMOTER_DOWNSTREAM
        result = table.loc[table['symbol_upper'].isin(sequences)].copy()
        result['promoter_sequence'] = result['symbol_upper'].map(sequences)
        result['promoter_length'] = result['promoter_sequence'].str.len()
        result['promoter_truncated'] = result['symbol_upper'].map(truncated)
        result['gc_fraction'] = result['promoter_sequence'].map(lambda sequence: (sequence.count('G') + sequence.count('C')) / max(sum((sequence.count(base) for base in 'ACGT')), 1))
        result['ambiguous_fraction'] = result['promoter_sequence'].str.count('N') / result['promoter_length']
        return result

    def load_pssm(self, path: Path):
        record = json.loads(path.read_text(encoding='utf-8'))
        motif = motifs.Motif(counts=record['pfm'])
        motif.pseudocounts = 0.8
        motif.background = {base: 0.25 for base in 'ACGT'}
        pssm = motif.pssm
        threshold = float(pssm.distribution(background=motif.background, precision=10000).threshold_fpr(self.MOTIF_FPR))
        return (record, pssm, pssm.reverse_complement(), threshold)

    def scan_promoters(self, promoters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        scans = promoters[['symbol_upper', 'gene_symbol', 'gene_id', 'baseMean', 'stat', 'log2FoldChange_shrunken', 'direction_robust', 'chromosome', 'start', 'end', 'strand', 'promoter_length', 'promoter_truncated', 'gc_fraction', 'ambiguous_fraction']].copy()
        motif_meta = []
        for tf, path in self.MOTIF_FILES.items():
            record, pssm, reverse_pssm, threshold = self.load_pssm(path)
            hit_values = []
            best_values = []
            count_values = []
            for sequence in promoters['promoter_sequence']:
                forward = np.asarray(pssm.calculate(Seq(sequence)), dtype=float)
                reverse = np.asarray(reverse_pssm.calculate(Seq(sequence)), dtype=float)
                combined = np.concatenate([forward[np.isfinite(forward)], reverse[np.isfinite(reverse)]])
                if combined.size:
                    best_values.append(float(combined.max()))
                    count_values.append(int((combined >= threshold).sum()))
                    hit_values.append(bool((combined >= threshold).any()))
                else:
                    best_values.append(np.nan)
                    count_values.append(0)
                    hit_values.append(False)
            scans[f'{tf}_motif_hit'] = hit_values
            scans[f'{tf}_motif_count'] = count_values
            scans[f'{tf}_best_score'] = best_values
            motif_meta.append({'tf': tf, 'matrix_id': record['matrix_id'], 'motif_name': record['name'], 'tax_group': record['tax_group'], 'motif_type': record['type'], 'motif_length': int(pssm.length), 'scan_threshold': threshold, 'threshold_definition': f'JASPAR PSSM threshold_fpr({self.MOTIF_FPR}) under uniform background', 'motif_file': str(path), 'motif_file_sha256': sha(path)})
        return (scans, pd.DataFrame(motif_meta))

    def fit_enrichment(self, scans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        network = read_table(self.NETWORK, sep='\t')
        network['target_upper'] = network['target'].astype(str).str.upper()
        target_rows = []
        results = []
        for tf in self.MOTIF_FILES:
            targets = set(network.loc[network['source'].eq(tf), 'target_upper'])
            frame = scans.copy()
            frame['translated_collectri_target'] = frame['symbol_upper'].isin(targets)
            frame['tf'] = tf
            frame['motif_hit'] = frame[f'{tf}_motif_hit'].astype(bool)
            frame['best_score'] = frame[f'{tf}_best_score']
            frame['log1p_baseMean'] = np.log1p(frame['baseMean'])
            target_rows.append(frame[['tf', 'symbol_upper', 'gene_symbol', 'gene_id', 'translated_collectri_target', 'motif_hit', 'best_score', 'baseMean', 'log1p_baseMean', 'gc_fraction', 'promoter_length', 'promoter_truncated', 'ambiguous_fraction', 'stat', 'log2FoldChange_shrunken', 'direction_robust']])
            contingency = pd.crosstab(frame['translated_collectri_target'], frame['motif_hit'])
            for value in [False, True]:
                if value not in contingency.index:
                    contingency.loc[value] = 0
                if value not in contingency.columns:
                    contingency[value] = 0
            contingency = contingency.sort_index().sort_index(axis=1)
            raw_or, raw_p = fisher_exact([[contingency.loc[True, True], contingency.loc[True, False]], [contingency.loc[False, True], contingency.loc[False, False]]], alternative='greater')
            model_frame = frame.loc[frame['ambiguous_fraction'].le(0.1)].copy()
            x = pd.DataFrame({'const': 1.0, 'target': model_frame['translated_collectri_target'].astype(float), 'gc_z': (model_frame['gc_fraction'] - model_frame['gc_fraction'].mean()) / model_frame['gc_fraction'].std(ddof=0), 'expression_z': (model_frame['log1p_baseMean'] - model_frame['log1p_baseMean'].mean()) / model_frame['log1p_baseMean'].std(ddof=0)}, index=model_frame.index)
            fit = sm.GLM(model_frame['motif_hit'].astype(int), x, family=sm.families.Binomial()).fit(cov_type='HC3')
            coefficient = float(fit.params['target'])
            lower, upper = fit.conf_int().loc['target'].astype(float)
            results.append({'tf': tf, 'n_translated_collectri_targets': len(targets), 'n_targets_with_testable_promoter': int(frame['translated_collectri_target'].sum()), 'n_background_promoters': int((~frame['translated_collectri_target']).sum()), 'target_motif_hits': int(contingency.loc[True, True]), 'background_motif_hits': int(contingency.loc[False, True]), 'target_hit_fraction': float(contingency.loc[True, True] / max(contingency.loc[True].sum(), 1)), 'background_hit_fraction': float(contingency.loc[False, True] / max(contingency.loc[False].sum(), 1)), 'raw_fisher_odds_ratio': raw_or, 'raw_fisher_one_sided_p': raw_p, 'adjusted_log_odds_target': coefficient, 'adjusted_odds_ratio_target': float(np.exp(coefficient)), 'adjusted_odds_ratio_ci95_low': float(np.exp(lower)), 'adjusted_odds_ratio_ci95_high': float(np.exp(upper)), 'adjusted_target_p': float(fit.pvalues['target']), 'covariates': 'promoter GC fraction and log1p ionocyte pseudobulk baseMean'})
        result = pd.DataFrame(results)
        result['adjusted_target_fdr_bh'] = multipletests(result['adjusted_target_p'], method='fdr_bh')[1]
        result['sequence_support'] = result['adjusted_target_fdr_bh'].lt(0.05) & result['adjusted_odds_ratio_target'].gt(1)
        return (pd.concat(target_rows, ignore_index=True), result)

    def analyze(self) -> None:
        self.OUT.mkdir(parents=True, exist_ok=True)
        gene_table = self.prepare_gene_table()
        promoters = self.extract_promoters(gene_table)
        scans, motif_catalog = self.scan_promoters(promoters)
        target_detail, enrichment = self.fit_enrichment(scans)
        promoter_export = scans.drop(columns=[column for column in scans.columns if column.endswith('_motif_count')])
        promoter_export.to_csv(self.OUT / 'tilapia_promoter_motif_scan.tsv.gz', sep='\t', index=False, compression='gzip')
        target_detail.to_csv(self.OUT / 'six_tf_target_promoter_detail.tsv.gz', sep='\t', index=False, compression='gzip')
        motif_catalog.to_csv(self.OUT / 'jaspar_motif_catalog.tsv', sep='\t', index=False)
        enrichment.to_csv(self.OUT / 'six_tf_promoter_motif_enrichment.tsv', sep='\t', index=False)
        manifest = {'reference_assembly': 'GCF_001858045.2 O_niloticus_UMD_NMBU', 'reference_genome': str(self.GENOME), 'reference_genome_sha256': sha(self.GENOME), 'gene_annotation': str(self.ANNOTATION), 'gene_annotation_sha256': sha(self.ANNOTATION), 'promoter_window': f'-{self.PROMOTER_UPSTREAM}/+{self.PROMOTER_DOWNSTREAM} bp relative to strand-aware annotated TSS', 'promoters_tested': len(promoters), 'promoters_with_ambiguous_fraction_gt_0_10_excluded_from_regression': int(promoters['ambiguous_fraction'].gt(0.1).sum()), 'motifs': motif_catalog.to_dict(orient='records'), 'target_definition': 'strict NCBI one-to-one translated human CollecTRI targets for each TF; no selection by tilapia DE direction or motif score', 'background': 'all ionocyte genes with baseMean >= 5, valid exact-reference promoter, and <=10% ambiguous promoter sequence', 'model': 'binomial GLM motif_hit ~ target + promoter_GC_z + log1p_baseMean_z; HC3 robust covariance; BH correction across six prespecified TFs', 'inference_scope': 'promoter motif enrichment indicates sequence compatibility only; it is not TF binding, perturbation, chromatin-accessibility, or causal evidence', 'network_file': str(self.NETWORK), 'network_sha256': sha(self.NETWORK), 'late_contrast': 'ionocyte 96_vs_24; motif target definition itself is contrast-independent'}
        (self.OUT / 'motif_support_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({'promoters_tested': len(promoters), 'enrichment': enrichment.to_dict(orient='records')}, indent=2))
INPUTS_BY_TASK = {'motif': ['genome_fasta', 'gene_annotation', 'tf_network', 'ionocyte_late_de', 'motif_klf4', 'motif_myb', 'motif_nfkb1', 'motif_rela', 'motif_sox4', 'motif_rbpj']}

def run(ctx, task='motif'):
    ctx.require(INPUTS_BY_TASK[task])
    PromoterMotifs(ctx).analyze()
    return {'files': [p.name for p in ctx.output('').iterdir() if p.is_file()]}
