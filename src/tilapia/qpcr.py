"""Recorded-well qPCR normalization, reference stability, and descriptive comparison.

All recorded wells are retained; amplification is assumed to double per cycle.
"""
from .runtime import read_table, write_table, write_json


TARGETS = ('cftr','LOC112841652','LOC100699477','LOC102081935','LOC106098517',
           'LOC106098252','LOC102079674','LOC100691161','notch1b','dll4')
REFERENCES = ('rpl8','eef1db')
CANDIDATES = ('gnb1a','actb2','rpl8','eef1db','hdlbpa','atp13a3','rps18','rpl13a')
SAMPLES = ('0h_1','0h_2','24h_1','24h_2','96h_1','96h_2')
STAGES = ('0h','24h','96h')


def calculate(wells):
    """Return all 60 individual fold values and 30 arithmetic mean/sample SD bars."""
    import numpy as np
    import pandas as pd
    required = {'Gene','Sample','Cq_numeric','well_type'}
    if not required.issubset(wells.columns):
        raise ValueError(f'Missing recorded-well columns: {sorted(required-set(wells.columns))}')
    if {'Plate','Well'}.issubset(wells.columns) and wells.duplicated(['Plate','Well']).any():
        raise ValueError('Duplicate Plate/Well identifiers; no well can be silently removed.')
    if set(wells.well_type) != {'sample','record_label_NTC'}:
        raise ValueError('Frozen well_type labels must be sample or record_label_NTC; unknown controls cannot be classified as NTC.')
    sample_wells = wells.loc[wells.well_type.eq('sample')].copy()
    if len(wells)!=252 or len(sample_wells)!=216:
        raise ValueError('Frozen qPCR requires all 252 records: 216 sample wells and 36 NTCs.')
    if set(sample_wells.Sample)!=set(SAMPLES) or set(sample_wells.Gene)!=set(TARGETS+CANDIDATES):
        raise ValueError('Expected all 10 targets, 8 candidate references, and six recorded sample labels.')
    sample_wells['Cq_numeric']=pd.to_numeric(sample_wells.Cq_numeric,errors='raise')
    if not np.isfinite(sample_wells.Cq_numeric).all():
        raise ValueError('Sample Ct values must be finite; missing wells are not imputed.')
    means=sample_wells.groupby(['Gene','Sample'],sort=True).Cq_numeric.agg(['mean','count'])
    if len(means)!=108 or not means['count'].eq(2).all():
        raise ValueError('Exactly two technical wells are required for every gene/sample.')
    records=[]
    for gene in TARGETS:
        vals=[]
        for sample in SAMPLES:
            mean_ct=float(means.loc[(gene,sample),'mean'])
            ref_ct=float(np.mean([means.loc[(ref,sample),'mean'] for ref in REFERENCES]))
            vals.append(dict(gene=gene,sample=sample,stage=sample.split('_')[0],sample_Ct=mean_ct,
                             reference_Ct=ref_ct,delta_Ct=mean_ct-ref_ct,technical_wells=2))
        calibrator=np.mean([d['delta_Ct'] for d in vals if d['stage']=='0h'])
        for d in vals:
            d['calibrator_0h_mean_delta_Ct']=float(calibrator)
            d['delta_delta_Ct']=d['delta_Ct']-calibrator
            d['negative_delta_delta_Ct']=-d['delta_delta_Ct']
            d['relative_expression']=2.0**(-d['delta_delta_Ct'])
            records.append(d)
    data=pd.DataFrame(records)
    summary=data.groupby(['gene','stage'],sort=False).relative_expression.agg(arithmetic_mean='mean',sample_SD='std',n='size').reset_index()
    summary['SD_lower']=summary.arithmetic_mean-summary.sample_SD
    summary['SD_upper']=summary.arithmetic_mean+summary.sample_SD
    summary['geometric_mean']=summary.apply(lambda r:float(np.exp(np.log(data.loc[data.gene.eq(r.gene)&data.stage.eq(r.stage),'relative_expression']).mean())),axis=1)
    summary['mean_negative_delta_delta_Ct']=summary.apply(lambda r:float(data.loc[data.gene.eq(r.gene)&data.stage.eq(r.stage),'negative_delta_delta_Ct'].mean()),axis=1)
    ntc=wells.loc[wells.well_type.ne('sample')].copy()
    if not ntc.Sample.eq('NTC').all() or not ntc.groupby('Gene').size().reindex(TARGETS+CANDIDATES).eq(2).all():
        raise ValueError('Exactly two recorded NTC wells are required for each of the 18 assayed genes.')
    ntc['Cq_numeric']=pd.to_numeric(ntc.Cq_numeric,errors='coerce')
    audit={'all_recorded_wells':len(wells),'sample_wells':len(sample_wells),'NTC_wells':len(ntc),
           'finite_NTC':int(np.isfinite(ntc.Cq_numeric).sum()),'NTC_Ct_min':float(ntc.Cq_numeric.min()),
           'NTC_Ct_max':float(ntc.Cq_numeric.max()),'technical_mean_rows':len(means),'target_sample_rows':len(data),'bars':len(summary),
           'no_well_exclusion':True,'noRT_status':'not_performed','efficiency':'equal_doubling_assumption_not_measured',
           'new_tests':0,'model_refits':0,'baseline_arithmetic_means_forced_to_one':False}
    return {'samples':data,'bars':summary,'technical_means':means,'NTC_records':ntc,'audit':audit}











SOURCE_STATUS = 'available_Ct_record_direct_instrument_export_provenance_unestablished'
ASSUMPTION = 'equal_amplification_assumed_doubling_not_measured'
CORE = ('cftr', 'LOC112841652', 'notch1b', 'dll4')
INPUTS_BY_TASK = {'recompute': ['recorded_wells', 'scrna_frozen_comparator']}


def recompute_records(wells, comparator):
    """All 252 wells, detailed reference/NTC tables and all 20 comparisons.

    Ct calculations use the same recorded data and equal-doubling assumption as
    calculate(). The detailed tables retain the original current-SI schemas.
    """
    import numpy as np
    import pandas as pd
    validated = calculate(wells)
    raw = wells.copy()
    sample = raw.loc[raw.Sample.ne('NTC')].copy()
    targets = sorted(comparator.gene.unique())
    assert set(targets) == set(TARGETS)
    means = sample.groupby(['Gene','Sample'], sort=True).agg(
        mean_Cq=('Cq_numeric','mean'), technical_Cq_min=('Cq_numeric','min'),
        technical_Cq_max=('Cq_numeric','max'), technical_Cq_SD=('Cq_numeric','std'),
        n_technical_wells=('Cq_numeric','size')).reset_index().rename(columns={'Gene':'gene','Sample':'sample'})
    means['technical_Cq_range'] = means.technical_Cq_max - means.technical_Cq_min
    means['stage'] = means['sample'].str.split('_').str[0]
    means['gene_role'] = means.gene.map(lambda g:'chosen_reference' if g in REFERENCES else 'candidate_reference' if g in CANDIDATES else 'target')
    means['source_status'] = SOURCE_STATUS
    means['noRT_status'] = 'not_performed'
    cq = means.pivot(index='sample',columns='gene',values='mean_Cq').loc[list(SAMPLES)]
    def ct_M(genes):
        return pd.Series({g:np.mean([(cq[g]-cq[k]).std(ddof=1) for k in genes if k!=g]) for g in genes})
    mvals = ct_M(CANDIDATES)
    refrows = []
    for gene in CANDIDATES:
        refrows.append(dict(gene=gene,chosen_reference=gene in REFERENCES,n_samples=6,
            Cq_mean=cq[gene].mean(),Cq_SD=cq[gene].std(ddof=1),Cq_min=cq[gene].min(),Cq_max=cq[gene].max(),
            pairwise_Cq_difference_M_eight_candidates=mvals[gene],
            mean_Cq_24h_minus_0h=cq.loc[['24h_1','24h_2'],gene].mean()-cq.loc[['0h_1','0h_2'],gene].mean(),
            mean_Cq_96h_minus_0h=cq.loc[['96h_1','96h_2'],gene].mean()-cq.loc[['0h_1','0h_2'],gene].mean(),
            interpretation='relative_Cq_stability_only_not_absolute_stability',efficiency_assumption=ASSUMPTION))
    refs = pd.DataFrame(refrows).sort_values('pairwise_Cq_difference_M_eight_candidates')
    remaining=list(CANDIDATES); steps=[]
    while len(remaining)>2:
        m=ct_M(remaining); worst=m.idxmax()
        steps.append(dict(step=len(steps)+1,n_remaining=len(remaining),removed=worst,removed_M=m[worst],remaining_after=';'.join(g for g in remaining if g!=worst)))
        remaining.remove(worst)
    steps.append(dict(step=len(steps)+1,n_remaining=2,removed='',removed_M=np.nan,remaining_after=';'.join(remaining)))
    assert set(remaining)==set(REFERENCES)
    nf=cq[list(REFERENCES)].mean(axis=1); poolrows=[]
    for gene in sorted(cq.columns):
        dcq=cq[gene]-nf; baseline=dcq.loc[['0h_1','0h_2']].mean()
        for sm in SAMPLES:
            lfc=-(dcq[sm]-baseline)
            poolrows.append(dict(gene=gene,sample=sm,stage=sm.split('_')[0],mean_Cq=cq.loc[sm,gene],reference_mean_Cq=nf[sm],delta_Cq=dcq[sm],baseline_mean_delta_Cq=baseline,negative_delta_delta_Cq=lfc,qpcr_log2FC_equal_eff_assumption=lfc,relative_fold_equal_eff_assumption=2.0**lfc,gene_role='target' if gene in targets else 'chosen_reference' if gene in REFERENCES else 'candidate_reference',efficiency_assumption=ASSUMPTION,noRT_status='not_performed',source_status=SOURCE_STATUS))
    pools=pd.DataFrame(poolrows); contrasts=[]
    for gene in targets:
        for stage in ['24h','96h']:
            sub=pools[pools.gene.eq(gene)&pools.stage.eq(stage)]
            ss=comparator[comparator.gene.eq(gene)&comparator.stage.eq(stage)].iloc[0]
            lfc=sub.qpcr_log2FC_equal_eff_assumption.mean()
            contrasts.append(dict(gene=gene,stage=stage,n_sample_labels=2,n_technical_wells_per_sample=2,qpcr_log2FC_equal_eff_assumption=lfc,relative_fold_equal_eff_assumption=2**lfc,qpcr_sample_1_log2FC=float(sub.iloc[0].qpcr_log2FC_equal_eff_assumption),qpcr_sample_2_log2FC=float(sub.iloc[1].qpcr_log2FC_equal_eff_assumption),qpcr_sample_SD_log2FC=sub.qpcr_log2FC_equal_eff_assumption.std(ddof=1),scrna_log2FC_shrunken=ss.scrna_log2FC_shrunken,matrix_gene=ss.matrix_gene,same_direction=bool(np.sign(lfc)==np.sign(ss.scrna_log2FC_shrunken)),core_figure=gene in CORE,efficiency_assumption=ASSUMPTION,evidence_level='descriptive_same_batch_tissue_comparison_not_independent_validation',noRT_status='not_performed'))
    contrasts=pd.DataFrame(contrasts); sensitivity=[]
    for refset in [list(REFERENCES),['rpl8'],['eef1db']]:
        nn=cq[refset].mean(axis=1)
        for gene in targets:
            d=cq[gene]-nn
            for stage in ['24h','96h']:
                val=-(d.loc[[stage+'_1',stage+'_2']].mean()-d.loc[['0h_1','0h_2']].mean())
                prim=contrasts[contrasts.gene.eq(gene)&contrasts.stage.eq(stage)].iloc[0]
                sensitivity.append(dict(gene=gene,stage=stage,references='+'.join(refset),qpcr_log2FC_equal_eff_assumption=val,difference_from_two_reference_log2FC=val-prim.qpcr_log2FC_equal_eff_assumption,same_direction_as_two_reference=bool(np.sign(val)==np.sign(prim.qpcr_log2FC_equal_eff_assumption)),same_direction_as_scrna=bool(np.sign(val)==np.sign(prim.scrna_log2FC_shrunken)),efficiency_assumption=ASSUMPTION))
    ntcrows=[]
    for gene in sorted(cq.columns):
        n=raw[raw.Gene.eq(gene)&raw.Sample.eq('NTC')]; s=sample[sample.Gene.eq(gene)]
        earliest=n.Cq_numeric.min(); weakest=s.Cq_numeric.max()
        ntcrows.append(dict(gene=gene,gene_role='target' if gene in targets else 'chosen_reference' if gene in REFERENCES else 'candidate_reference',n_sample_wells=len(s),n_NTC_labelled_wells=len(n),n_NTC_numeric_Cq=int(n.Cq_numeric.notna().sum()),n_NTC_ND=int(n.Cq_numeric.isna().sum()),earliest_NTC_Cq=earliest,latest_NTC_Cq=n.Cq_numeric.max(),weakest_sample_well_Cq=weakest,earliest_NTC_minus_weakest_sample_Cq=earliest-weakest,max_sample_technical_Cq_range=means[means.gene.eq(gene)].technical_Cq_range.max(),noRT_status='not_performed',amplification_efficiency_status='not_experimentally_verified',specificity_status='not_established_from_available_Ct_records',source_status=SOURCE_STATUS))
    tables={
        'qpcr_all_252_recorded_wells.tsv':raw,
        'qpcr_all_108_gene_sample_Cq_means.tsv':means,
        'qpcr_eight_reference_candidates_Cq_stability.tsv':refs,
        'qpcr_Cq_stability_stepwise.tsv':pd.DataFrame(steps),
        'qpcr_all_108_sample_normalized_Cq.tsv':pools,
        'qpcr_60_target_sample_values.tsv':pools[pools.gene.isin(targets)],
        'qpcr_20_target_stage_comparisons.tsv':contrasts,
        'qpcr_discordant_comparisons.tsv':contrasts[~contrasts.same_direction],
        'qpcr_single_reference_sensitivity.tsv':pd.DataFrame(sensitivity),
        'qpcr_18_assay_Ct_and_NTC_status.tsv':pd.DataFrame(ntcrows),
        'qpcr_core_comparisons.tsv':contrasts[contrasts.core_figure],
        'qpcr_reference_sample_Cq.tsv':means[means.gene.isin(REFERENCES)],
        'qpcr_target_sample_values.tsv':validated['samples'],
        'qpcr_target_bar_summary.tsv':validated['bars'],
    }
    tables['qPCR_108_sample_Ct_recomputed.tsv'] = means[
        ['gene','sample','mean_Cq','technical_Cq_min','technical_Cq_max','n_technical_wells','technical_Cq_range']].rename(columns={
        'gene':'Gene','sample':'Sample','mean_Cq':'mean','technical_Cq_min':'min','technical_Cq_max':'max',
        'n_technical_wells':'count','technical_Cq_range':'technical_range'})
    tables['qPCR_20_contrasts_recomputed.tsv'] = contrasts[
        ['gene','stage','qpcr_sample_1_log2FC','qpcr_sample_2_log2FC','qpcr_log2FC_equal_eff_assumption']].rename(columns={
        'qpcr_sample_1_log2FC':'sample1','qpcr_sample_2_log2FC':'sample2',
        'qpcr_log2FC_equal_eff_assumption':'log2FC_assuming_equal_doubling_efficiency'})
    return tables, validated['audit']
















def run(ctx, *, task='recompute'):
    ctx.require(INPUTS_BY_TASK[task])
    if task=='recompute':
        wells=read_table(ctx.input('recorded_wells'))
        tables,audit=recompute_records(wells,read_table(ctx.input('scrna_frozen_comparator')))
        # Keep the current numerical/QC outputs; omit old display subsets,
        # duplicate schema aliases and single-reference sensitivity exports.
        omitted = {'qpcr_core_comparisons.tsv', 'qpcr_reference_sample_Cq.tsv',
                   'qpcr_single_reference_sensitivity.tsv',
                   'qPCR_108_sample_Ct_recomputed.tsv', 'qPCR_20_contrasts_recomputed.tsv'}
        tables = {name: table for name, table in tables.items() if name not in omitted}
        for name,table in tables.items():
            write_table(table,ctx.output(name),float_format='%.12g',na_rep='')
        write_json(ctx.output('qpcr_audit.json'),audit)
        return {'tables':list(tables),'audit':'qpcr_audit.json'}
    raise ValueError(task)
