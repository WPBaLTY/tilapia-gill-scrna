#!/usr/bin/env python3
"""Explicit analysis stages. Listing uses the standard library; result preflight needs h5py."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import ast
import csv
import gzip
import importlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from tilapia.runtime import Context, SourceRef, declared_inputs, write_json, to_builtin

DISTRIBUTIONS = {
 'numpy':'numpy','pandas':'pandas','scipy':'scipy','matplotlib':'matplotlib',
 'PIL':'pillow','statsmodels':'statsmodels','anndata':'anndata',
 'scanpy':'scanpy','h5py':'h5py','pydeseq2':'pydeseq2','gseapy':'gseapy',
 'harmonypy':'harmonypy','leidenalg':'leidenalg','igraph':'igraph',
 'decoupler':'decoupler','sklearn':'scikit-learn','skmisc':'scikit-misc',
 'skimage':'scikit-image','seaborn':'seaborn','networkx':'networkx',
 'Bio':'biopython','threadpoolctl':'threadpoolctl',
}

def dependency_requirements(module_name):
    """Conservative module dependency closure; never import scientific modules."""
    found, visited = set(), set()
    def visit(name):
        if name in visited:
            return
        visited.add(name)
        path = ROOT/'src'/Path(*name.split('.')).with_suffix('.py')
        if not path.is_file():
            return
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split('.')[0]
                    if top in DISTRIBUTIONS:
                        found.add(DISTRIBUTIONS[top])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.module:
                        visit('tilapia.'+node.module.split('.')[0])
                    else:
                        for alias in node.names:
                            visit('tilapia.'+alias.name)
                elif node.module and node.module.split('.')[0] in DISTRIBUTIONS:
                    found.add(DISTRIBUTIONS[node.module.split('.')[0]])
    visit(module_name)
    # Libraries dynamically required by the retained Scanpy atlas/reclustering calls.
    if module_name in ('tilapia.atlas','tilapia.state_programs'):
        found.update(['scanpy','harmonypy','leidenalg','igraph','scikit-misc'])
    return sorted(found)

def dependency_status(module_name):
    pins = {}
    for name, version in json.loads((ROOT/'config.json').read_text(encoding='utf-8'))['historical_analysis_pins'].items():
        pins[name.lower()] = version
    rows=[]
    for name in dependency_requirements(module_name):
        try:
            version=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version=None
        pin=pins.get(name.lower())
        rows.append({'distribution':name,'installed_version':version,
                     'historical_recorded_pin':pin,'available':version is not None,
                     'pin_matches':version==pin if version and pin else None})
    return rows

def validate_header(path, entry):
    schema=entry.get('schema')
    if not isinstance(schema,dict) or not isinstance(schema.get('columns'),list):
        return 'not_checked_no_machine_readable_header_contract'
    fmt=schema.get('format','').upper()
    if fmt not in ('TSV','CSV'):
        return 'not_checked_non_tabular'
    if isinstance(path,SourceRef):
        header=path.record['original_columns']
    else:
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8-sig',newline='') as stream:
            header=next(csv.reader(stream,delimiter='\t' if fmt=='TSV' else ','))
    missing=sorted(set(schema['columns'])-set(header))
    if missing:
        raise ValueError(f'Missing required header fields: {missing}')
    if schema.get('column_count') is not None and len(header) != schema['column_count']:
        raise ValueError('Header column count differs from the recorded schema.')
    if schema.get('header_sha256'):
        import hashlib
        digest = hashlib.sha256(('\t'.join(header)+'\n').encode('utf-8')).hexdigest()
        if digest != schema['header_sha256']:
            raise ValueError('Full header identity/order differs from the recorded schema.')
    return 'required_columns_present'

def inspect_stage(stage, spec, args, execution=False):
    profile=args.profile or spec.get('default_profile')
    ctx=Context(args.config,args.inputs,args.output if execution else None,profile)
    if profile and ctx.config.get('profiles',{}).get(profile,{}).get('task') != spec['task']:
        raise ValueError(f'Profile {profile!r} does not belong to stage {stage!r}.')
    ctx.check_source_integrity()
    required=list(declared_inputs(spec['module'],spec['task']))
    if required != spec['required_inputs']:
        raise ValueError(f'Stage registry/input contract mismatch: {stage}')
    resources=[]
    for key in required:
        entry=ctx.config.get('inputs',{}).get(key,{})
        row={'key':key,'external':entry.get('external',False),'kind':entry.get('kind','file')}
        try:
            path=ctx.input(key)
            row.update({'status':'available','path':str(path),'header_check':validate_header(path,entry)})
            if isinstance(path,SourceRef):
                row['container']=path.record['container']
                row['source_identity']='Original logical source; token-validated slice of merged container'
        except (ValueError,KeyError,FileNotFoundError,OSError,StopIteration) as exc:
            row.update({'status':'blocked','reason':str(exc)})
        resources.append(row)
    dependencies=dependency_status(spec['module'])
    blocked_inputs=any(row['status']!='available' for row in resources)
    blocked_packages=any(not row['available'] for row in dependencies)
    return ctx, {'stage':stage,'profile':profile,
        'status':'blocked' if blocked_inputs or blocked_packages else 'preflight_passed',
        'analysis_executed':False,'resources':resources,'dependencies':dependencies,
        'scope':'Presence, declared table headers, frozen hashes and installed distribution metadata only; no H5AD content validation, model fit, or fresh installation.'}

def write_result_table(group, stream, output_format):
    """Stream a complete stored logical table as UTF-8 tokens, without inference."""
    columns=group['columns'].asstr()[:].tolist()
    rows,cols=int(group.attrs['rows']),int(group.attrs['cols'])
    if len(columns)!=cols:
        raise ValueError('Stored result column count differs.')
    storage=group.attrs['storage']
    if isinstance(storage,bytes): storage=storage.decode('utf-8')
    if storage=='columns':
        datasets=[group['data'][f'c{i:06d}'] for i in range(cols)]
        if any(dataset.shape!=(rows,) for dataset in datasets):
            raise ValueError('Stored result column shape differs.')
        row_bytes=sum(dataset.dtype.itemsize+64 for dataset in datasets)
    elif storage=='matrix':
        dataset=group['data/values']
        if dataset.shape!=(rows,cols):
            raise ValueError('Stored result matrix shape differs.')
        row_bytes=cols*(dataset.dtype.itemsize+64)
    else:
        raise ValueError('Unsupported result storage: '+str(storage))
    chunk_rows=max(1,min(4096,(16*1024*1024)//max(1,row_bytes)))
    writer=csv.writer(stream,delimiter='\t' if output_format=='tsv' else ',',lineterminator='\n')
    writer.writerow(columns)
    for start in range(0,rows,chunk_rows):
        stop=min(rows,start+chunk_rows)
        if storage=='columns':
            blocks=[part[start:stop] for part in datasets]
            values=zip(*blocks)
        else:
            values=dataset[start:stop,:]
        writer.writerows([value.decode('utf-8') for value in row] for row in values)

def inspect_results(args):
    """List/export recorded results only; no scientific task is imported."""
    import h5py
    ctx=Context()
    ctx.check_source_integrity()
    path=ctx.data_root/ctx.source_index_relative
    with h5py.File(path,'r') as handle:
        catalog=json.loads(handle['/catalog_json'].asstr()[()])
        tables={}
        for record in catalog['tables']:
            group=record['hdf_group']
            if not group.startswith('/tables/') or group=='/tables/':
                raise ValueError('Invalid result catalog table group.')
            table_id=group[len('/tables/'):]
            if table_id in tables:
                raise ValueError('Duplicate result catalog table ID.')
            tables[table_id]=record
        if args.results_list:
            print(json.dumps({'analysis_executed':False,'table_count':len(tables),
                'tables':[{'table_id':key,'rows':record['rows'],'columns':record['cols']}
                          for key,record in tables.items()]},ensure_ascii=False,indent=2))
            return 0
        if args.results_export not in tables:
            raise ValueError('Unknown result table ID; use --results-list: '+args.results_export)
        destination=args.output.resolve()
        if destination.exists():
            raise FileExistsError('Result output already exists; no file was overwritten: '+str(destination))
        if not destination.parent.is_dir():
            raise FileNotFoundError('Result output parent directory does not exist: '+str(destination.parent))
        # Apply the same package, submission and configured-input protection as execution.
        ctx.output_dir=destination
        ctx._validate_output_root()
        record=tables[args.results_export]
        with destination.open('x',encoding='utf-8',newline='') as stream:
            write_result_table(handle[record['hdf_group']],stream,args.format or 'tsv')
        print(json.dumps({'analysis_executed':False,'table_id':args.results_export,
            'rows':record['rows'],'columns':record['cols'],'format':args.format or 'tsv',
            'output':str(destination)},ensure_ascii=False,indent=2))
        return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    result_modes=parser.add_mutually_exclusive_group()
    result_modes.add_argument('--results-list',action='store_true',help='List the recorded logical result table IDs.')
    result_modes.add_argument('--results-export',metavar='TABLE_ID',help='Export one complete recorded logical table, including source columns.')
    parser.add_argument('--format',choices=('tsv','csv'),help='Result-export format; default: tsv.')
    parser.add_argument('--list',action='store_true',help='List the retained analysis stages.')
    parser.add_argument('--stage',help='Exact module:task identifier from --list, or all for preflight only.')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight',action='store_true',help='Check inputs/dependency metadata without running analysis.')
    modes.add_argument('--execute',action='store_true',help='Run one scientific stage after preflight. Not used for restoration validation.')
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--inputs',type=Path,help='Override package root for bundled inputs/ and results/ paths.')
    parser.add_argument('--profile',help='Optional exact recovered-kegg profile; stage default is preferred.')
    parser.add_argument('--output',type=Path,help='Execution directory or result-export file outside code and input trees.')
    supplied=sys.argv[1:] if argv is None else argv
    args=parser.parse_args(argv)
    if args.results_list or args.results_export is not None:
        forbidden={'--list','--stage','--preflight','--execute','--config','--inputs','--profile'}
        if any(arg.split('=',1)[0] in forbidden for arg in supplied):
            parser.error('Result commands cannot be combined with analysis or input-override options.')
        if args.results_list and (args.output is not None or args.format is not None):
            parser.error('--results-list does not accept --output or --format.')
        if args.results_export is not None and args.output is None:
            parser.error('--results-export requires --output FILE.')
        try:
            return inspect_results(args)
        except (ValueError,KeyError,FileNotFoundError,OSError,ImportError) as exc:
            print(json.dumps({'status':'blocked','analysis_executed':False,'error':str(exc)},ensure_ascii=False),file=sys.stderr)
            return 2
    if args.format is not None:
        parser.error('--format requires --results-export.')
    registry=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))['stages']
    if args.list:
        if args.stage or args.execute or args.preflight or args.output:
            parser.error('--list is independent of stage, execution, preflight and output.')
        print(json.dumps({'stage_count':len(registry),'stages':registry,'analysis_executed':False},indent=2))
        return 0
    if not args.stage:
        if args.preflight: args.stage='all'
        else: parser.error('Choose --list or --stage MODULE:TASK with --preflight/--execute.')
    if args.stage!='all' and args.stage not in registry:
        parser.error('Unknown analysis stage; use --list.')
    if args.execute and (args.stage=='all' or args.output is None):
        parser.error('Execution requires one concrete --stage and --output; there is no automatic full-pipeline run.')
    if not args.execute and (not args.preflight or args.output is not None):
        parser.error('Use --preflight without --output, or --execute with --output.')
    stages=registry if args.stage=='all' else {args.stage:registry[args.stage]}
    try:
        reports=[]
        for stage,spec in stages.items():
            ctx,report=inspect_stage(stage,spec,args,args.execute)
            reports.append(report)
        result={'analysis_executed':False,'stage_count':len(reports),'reports':reports}
        if not args.execute:
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 2 if any(r['status']=='blocked' for r in reports) else 0
        if report['status']=='blocked':
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 2
        # No scientific module is imported before all resource guards pass.
        ctx.prepare_environment()
        manifest={'stage':stage,'inputs':report['resources'],'dependencies':report['dependencies'],
                  'analysis_executed':True,'status':'started','profile':report['profile']}
        write_json(ctx.output('execution_manifest.json'),manifest)
        try:
            module=importlib.import_module(spec['module'])
            products=module.run(ctx,task=spec['task'])
            manifest.update({'status':'completed','products':to_builtin(products)})
        except Exception as exc:
            manifest.update({'status':'failed','error':str(exc),'error_type':type(exc).__name__})
            write_json(ctx.output('execution_manifest.json'),manifest)
            print(json.dumps(to_builtin(manifest),ensure_ascii=False,indent=2),file=sys.stderr)
            return 1
        write_json(ctx.output('execution_manifest.json'),manifest)
        print(json.dumps(to_builtin(manifest),ensure_ascii=False,indent=2))
        return 0
    except (ValueError,KeyError,FileNotFoundError,OSError,ImportError) as exc:
        print(json.dumps({'status':'blocked','analysis_executed':False,'error':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
