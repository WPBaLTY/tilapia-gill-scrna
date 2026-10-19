"""Shared configuration, immutable inputs and isolated output handling.

Scientific functions use explicit paths/arrays; this module performs no analysis.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


class SourceRef:
    """A logical recorded source, deliberately not a filesystem Path."""
    def __init__(self, store, logical_path):
        self.store, self.logical_path = store, logical_path
        self.record = store.index['sources'][logical_path]
    @property
    def name(self): return Path(self.logical_path).name
    @property
    def suffix(self): return Path(self.logical_path).suffix
    def __str__(self): return self.logical_path
    def __lt__(self, other): return str(self) < str(other)


class SourceDirectory:
    """A named collection in source_index, with no virtual filesystem methods."""
    def __init__(self, store, prefix):
        self.store, self.prefix = store, prefix.rstrip('/')
    def __str__(self): return self.prefix


class SourceStore:
    """Read the result master file without extracting legacy files to disk."""
    def __init__(self, root, index_path, manifest_path=None):
        self.manifest_path=Path(manifest_path) if manifest_path is not None else Path(root)/"manifest.json"
        self.root = Path(root).resolve()
        self.index_path = Path(index_path)
        # h5py is needed for storage access, not for --list or module discovery.
        try:
            import h5py
        except ImportError as exc:
            raise ImportError('Reading recorded results requires h5py; install the analysis environment.') from exc
        with h5py.File(self.index_path,'r') as handle:
            self.index = json.loads(handle['/source_index_json'].asstr()[()])
        if self.index.get('schema_version') != 2:
            raise ValueError('Unsupported recorded-source index schema')
        self.provenance_columns = self.index['provenance_columns']
        expected = ['__source_id','__source_row','__source_cell_type',
                    '__source_state','__source_contrast','__source_route']
        if self.provenance_columns not in (expected,expected+['__source_role']):
            raise ValueError('Recorded-source provenance column contract differs')
        self._verified = set()
        self._row_cache = {}
        self._verified_containers = set()

    def ref(self, logical_path):
        logical_path = str(logical_path).replace('\\', '/')
        if logical_path not in self.index['sources']:
            raise FileNotFoundError('No recorded source: '+logical_path)
        return SourceRef(self, logical_path)

    def container(self, source):
        p = (self.root/source.record['container']).resolve()
        if self.root not in p.parents or p != self.index_path.resolve() or not p.is_file():
            raise FileNotFoundError('Invalid merged container: '+str(p))
        if str(p) not in self._verified_containers:
            record=source.record
            if record.get('container_bytes') is not None and p.stat().st_size!=record['container_bytes']:
                raise ValueError('Merged container size differs: '+str(p))
            if record.get('container_sha256') and sha256(p)!=record['container_sha256']:
                raise ValueError('Merged container SHA256 differs: '+str(p))
            manifest_path=self.manifest_path
            if manifest_path.is_file():
                manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                records={x['path']:x for x in manifest['files']}
                relative=p.relative_to(self.root).as_posix()
                expected=records.get(relative)
                if expected is None or p.stat().st_size!=expected['bytes'] or sha256(p)!=expected['sha256']:
                    raise ValueError('Merged container integrity differs: '+relative)
            self._verified_containers.add(str(p))
        return p

    def rows(self, source):
        """Preserve original text tokens; infer types only after source selection."""
        import h5py
        key=source.logical_path
        if key in self._row_cache:
            return self._row_cache[key]
        record=source.record
        if record['kind'] not in ('table','gmt'):
            raise ValueError('Not a tabular recorded source: '+key)
        path=self.container(source)
        selected=[]
        with h5py.File(path,'r') as handle:
            group=handle[record['hdf_group']]
            header=group['columns'].asstr()[:].tolist()
            total_rows,total_columns=int(group.attrs['rows']),int(group.attrs['cols'])
            start,stop=record['hdf_row_start'],record['hdf_row_stop']
            if type(start) is not int or type(stop) is not int or not 0<=start<=stop<=total_rows:
                raise ValueError('Invalid HDF source row range: '+key)
            if len(header)!=total_columns:
                raise ValueError('HDF column count differs: '+key)
            nmeta=len(self.provenance_columns)
            if header[:nmeta]!=self.provenance_columns:
                raise ValueError('Merged provenance header differs: '+str(path))
            payload=header[nmeta:]
            if record['kind']=='table' and payload!=record['original_columns']:
                raise ValueError('Merged/original column order differs: '+key)
            storage=group.attrs['storage']
            if isinstance(storage,bytes): storage=storage.decode('utf-8')
            if storage=='columns':
                columns=[]
                for index in range(total_columns):
                    dataset=group['data'][f'c{index:06d}']
                    if dataset.shape!=(total_rows,):
                        raise ValueError('HDF column shape differs: '+key)
                    columns.append([value.decode('utf-8') for value in dataset[start:stop]])
                source_rows=zip(*columns)
            elif storage=='matrix':
                dataset=group['data/values']
                if dataset.shape!=(total_rows,total_columns):
                    raise ValueError('HDF matrix shape differs: '+key)
                values=dataset[start:stop,:]
                source_rows=([value.decode('utf-8') for value in row] for row in values)
            else:
                raise ValueError('Unsupported HDF table storage: '+str(storage))
            for row in source_rows:
                if row[0]!=record.get('source_id',key):
                    raise ValueError('HDF row range includes a different source: '+key)
                selected.append((int(row[1]),list(row[nmeta:])))
        if record['kind']=='table':
            selected.sort(key=lambda x:x[0])
            if [i for i,_ in selected]!=list(range(record['source_rows'])):
                raise ValueError('Source row count/order is incomplete or duplicated: '+key)
            rows=[row for _,row in selected]
        else:
            # GMT is a list of ordered terms, each with an ordered member list.
            required=['term_id','description','gene','member_index']
            if not set(required).issubset(payload):
                raise ValueError('GMT membership columns are incomplete')
            ix={name:payload.index(name) for name in required}
            terms={}
            for number,row in selected:
                term,description=row[ix['term_id']],row[ix['description']]
                entry=terms.setdefault(number,(term,description,[]))
                if entry[:2]!=(term,description):
                    raise ValueError('GMT term identity differs within one source row')
                entry[2].append((int(row[ix['member_index']]),row[ix['gene']]))
            if sorted(terms)!=list(range(record['source_rows'])):
                raise ValueError('GMT source term order/count differs')
            rows=[]
            for number in sorted(terms):
                term,description,members=terms[number];members.sort()
                if [i for i,_ in members]!=list(range(len(members))):
                    raise ValueError('GMT member order is incomplete or duplicated')
                rows.append([term,description]+[g for _,g in members])
        digest=hashlib.sha256()
        for row in rows:
            digest.update((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))
        if digest.hexdigest()!=record['token_sha256']:
            raise ValueError('Original source tokens differ: '+key)
        if len(self._row_cache)>=2:
            self._row_cache.pop(next(iter(self._row_cache)))
        self._row_cache[key]=rows
        self._verified.add(key)
        return rows

    def verify(self, source):
        if source.logical_path in self._verified:
            return
        if source.record['kind'] in ('table','gmt'):
            self.rows(source)
        elif source.record['kind']=='json':
            read_source_json(source)
            self._verified.add(source.logical_path)
        else:
            raise ValueError('Unsupported recorded source kind')


def source_member(directory, relative):
    """Resolve one original member from an indexed collection or a real directory."""
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Source member must remain inside its collection')
    if isinstance(directory,SourceDirectory):
        return directory.store.ref(directory.prefix+'/'+relative.as_posix())
    return Path(directory)/relative


def iter_sources(directory, pattern='*'):
    import fnmatch
    if isinstance(directory,SourceDirectory):
        prefix=directory.prefix+'/'
        return [directory.store.ref(key) for key in sorted(directory.store.index['sources'])
                if key.startswith(prefix) and fnmatch.fnmatchcase(key[len(prefix):],pattern)
                and '/' not in key[len(prefix):]]
    return sorted(Path(directory).glob(pattern))


def source_size(source):
    return source.record['original_bytes'] if isinstance(source,SourceRef) else Path(source).stat().st_size


def source_exists(source):
    if isinstance(source,SourceRef):
        source.store.verify(source)
        return True
    return Path(source).is_file()


def source_provenance(source):
    if isinstance(source,SourceRef):
        source.store.verify(source)
        return {'path':source.logical_path,'bytes':source.record['original_bytes'],
                'sha256':source.record['original_sha256'],
                'container':source.record['container'],'token_sha256':source.record.get('token_sha256'),
                'identity_scope':'original source identity; selected merged tokens validated'}
    path=Path(source);st=path.stat()
    return {'path':str(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,
            'sha256':sha256(path) if st.st_size<100000000 else 'not_computed_large_file'}


def source_label(source):
    return str(source) if isinstance(source,SourceRef) else str(Path(source).resolve())


def source_text(source):
    if isinstance(source,SourceRef):
        import csv
        import io
        record=source.record
        rows=source.store.rows(source)
        if record['kind']=='gmt':
            return ''.join('\t'.join(row)+'\n' for row in rows)
        buffer=io.StringIO(newline='')
        delimiter=record.get('delimiter','\t')
        if delimiter==r'\t': delimiter='\t'
        writer=csv.writer(buffer,delimiter=delimiter,lineterminator='\n')
        writer.writerow(record['original_columns']);writer.writerows(rows)
        return buffer.getvalue()
    import gzip
    path=Path(source)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as stream:
        return stream.read()


def copy_source(source, destination):
    if isinstance(source,SourceRef):
        destination=Path(destination)
        if destination.suffix=='.gz':
            import gzip
            with gzip.open(destination,'wt',encoding='utf-8',newline='') as stream:
                stream.write(source_text(source))
        else:
            destination.write_text(source_text(source),encoding='utf-8',newline='')
    else:
        import shutil
        shutil.copy2(source,destination)
    return destination


def read_source_json(source):
    if isinstance(source,SourceRef):
        import h5py
        record=source.record
        with h5py.File(source.store.container(source),'r') as handle:
            value=json.loads(handle[record['hdf_group']+'/raw_json'].asstr()[()])
        return value[record['object_key']] if record.get('object_key') is not None else value
    return json.loads(source_text(source))


def sha256(path):
    if isinstance(path,SourceRef):
        path.store.verify(path)
        return path.record['original_sha256']
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def to_builtin(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(value), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def read_table(path, **kwargs):
    import pandas as pd
    if isinstance(path,SourceRef):
        import io
        kwargs.pop('compression',None)
        return pd.read_csv(io.StringIO(source_text(path)),sep=kwargs.pop('sep','\t'),**kwargs)
    return pd.read_csv(path, sep=kwargs.pop('sep', '\t'), **kwargs)


def write_table(table, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep=kwargs.pop('sep', '\t'), index=kwargs.pop('index', False), **kwargs)
    return path


def as_csr(matrix):
    from scipy import sparse
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


class Context:
    """Resolve named inputs from one configuration and protect prior results."""
    def __init__(self, config_path=None, data_root=None, output_dir=None, profile=None):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config_path = Path(config_path or self.repo_root / 'config.json').resolve()
        self.config = json.loads(self.config_path.read_text(encoding='utf-8'))
        if profile is not None:
            profiles = self.config.get('profiles', {})
            if profile not in profiles:
                raise KeyError(f'Unknown input profile {profile!r}.')
            self.config['inputs'].update(profiles[profile].get('inputs', {}))
        self.profile = profile
        self.data_root = Path(data_root or self.repo_root).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self._source_checked = False
        self._checked_inputs = set()
        release_config=json.loads((self.repo_root/'config.json').read_text(encoding='utf-8'))
        self.source_index_relative=release_config.get('source_index','results/analysis_results.h5')
        if self.config.get('source_index',self.source_index_relative)!=self.source_index_relative:
            raise ValueError('A local config cannot replace the released source-index identity')
        index_path=self.data_root/self.source_index_relative
        self.source_store=SourceStore(self.data_root,index_path,self.repo_root/'manifest.json') if index_path.is_file() else None
        if self.output_dir is not None:
            self._validate_output_root()

    def _validate_output_root(self):
        out = self.output_dir
        protected_roots = [self.repo_root, self.data_root]
        # Protect the whole submission when installed in its code directory;
        # standalone repositories also protect their own input and result trees.
        for ancestor in self.repo_root.parents:
            if ancestor.name in ('03_Code', '03_GitHub', 'GitHub_Analysis') or ((ancestor / 'render.py').is_file() and (ancestor / 'renderers').is_dir()):
                protected_roots.append(ancestor)
            if (ancestor / '01_Submission').is_dir() and any((ancestor / name).is_dir() for name in ('03_Code', '03_GitHub', 'GitHub_Analysis')):
                protected_roots.append(ancestor)
        for raw in self.config.get('protected_paths', []):
            path = Path(raw)
            protected_roots.append((path if path.is_absolute() else self.config_path.parent / path).resolve())
        for protected in protected_roots:
            if out == protected or protected in out.parents or out in protected.parents:
                raise ValueError('Output must be outside the code/input trees and cannot contain either tree.')
        for entry in self.config.get('inputs', {}).values():
            if not isinstance(entry, dict) or not entry.get('path'):
                continue
            path = Path(entry['path'])
            if not path.is_absolute():
                path = (self.config_path.parent if entry.get('external') else self.data_root) / path
            self.protect_input_path(path, entry.get('kind', 'file'))
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError('Output must be a new or empty directory; existing results are never overwritten.')

    def protect_input_path(self, path, kind='file'):
        """An execution output may neither contain nor nest inside an input tree."""
        if self.output_dir is None:
            return
        path = Path(path).resolve()
        out = self.output_dir
        if out == path or out in path.parents or (kind == 'directory' and path in out.parents):
            raise ValueError(f'Output overlaps a configured input: {path}')

    def check_source_integrity(self):
        if self._source_checked:
            return
        failures = []
        manifest = json.loads((self.repo_root / 'manifest.json').read_text(encoding='utf-8'))
        for record in manifest['files']:
            relative, expected = record['path'], record['sha256']
            if not (relative.endswith('.py') or relative in ('config.json', 'requirements.txt',self.source_index_relative)):
                continue
            base=self.data_root if relative==self.source_index_relative else self.repo_root
            path = (base / relative).resolve()
            if base not in path.parents or not path.is_file() or sha256(path) != expected:
                failures.append(relative)
        if failures:
            raise ValueError('Missing or changed release code: ' + ', '.join(failures))
        self._source_checked = True

    def input(self, key):
        entries = self.config.get('inputs', {})
        if key not in entries:
            raise KeyError(f'Input {key!r} is not configured.')
        entry = entries[key]
        if isinstance(entry, str):
            entry = {'path': entry}
        raw = entry.get('path')
        if not raw:
            raise FileNotFoundError(f'{key}: supply {entry.get("description", "the actual external input")} in config.json; no substitute is inferred.')
        if self.source_store is not None and not entry.get('external',False):
            logical=str(raw).replace('\\','/')
            indexed=self.source_store.index['sources']
            if logical in indexed:
                source=self.source_store.ref(logical)
                if entry.get('expected_bytes') is not None and source_size(source)!=entry['expected_bytes']:
                    raise ValueError(key+': original source size contract differs')
                if entry.get('sha256') and source.record['original_sha256']!=entry['sha256']:
                    raise ValueError(key+': original source identity differs')
                self.source_store.verify(source)
                self._checked_inputs.add(key)
                return source
            if entry.get('kind')=='directory' and any(k.startswith(logical.rstrip('/')+'/') for k in indexed):
                directory=SourceDirectory(self.source_store,logical)
                for record in entry.get('required_files',[]):
                    source=source_member(directory,record['path'])
                    if record.get('bytes') is not None and source_size(source)!=record['bytes']:
                        raise ValueError(key+': original member size differs: '+record['path'])
                    if record.get('sha256') and source.record['original_sha256']!=record['sha256']:
                        raise ValueError(key+': original member hash differs: '+record['path'])
                    self.source_store.verify(source)
                self._checked_inputs.add(key)
                return directory
        path = Path(raw)
        if not path.is_absolute():
            base = self.config_path.parent if entry.get('external', False) else self.data_root
            path = base / path
        path = path.resolve()
        self.protect_input_path(path, entry.get('kind', 'file'))
        if entry.get('kind') == 'directory':
            exists = path.is_dir()
        else:
            exists = path.is_file()
        if not exists:
            raise FileNotFoundError(f'{key}: required {entry.get("kind", "file")} not found: {path}')
        if key not in self._checked_inputs:
            if entry.get('expected_bytes') is not None and path.is_file() and path.stat().st_size != entry['expected_bytes']:
                raise ValueError(f'{key}: file size differs from the recorded resource.')
            if entry.get('sha256') and sha256(path) != entry['sha256']:
                raise ValueError(f'{key}: input SHA256 differs from the frozen resource; no calculation was run.')
            for record in entry.get('required_files', []):
                member = (path / record['path']).resolve()
                if path not in member.parents or not member.is_file():
                    raise FileNotFoundError(f'{key}: missing/invalid required member {record["path"]}')
                if record.get('bytes') is not None and member.stat().st_size != record['bytes']:
                    raise ValueError(f'{key}: required member size differs: {record["path"]}')
                if record.get('sha256') and sha256(member) != record['sha256']:
                    raise ValueError(f'{key}: required member hash differs: {record["path"]}')
        self._checked_inputs.add(key)
        return path

    def table(self, key, member=None, **kwargs):
        source=self.input(key)
        return read_table(source_member(source,member) if member is not None else source,**kwargs)

    def sources(self, key, pattern='*'):
        return iter_sources(self.input(key),pattern)

    def require(self, keys):
        failures = []
        resolved = {}
        for key in keys:
            try:
                resolved[key] = self.input(key)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                failures.append(str(exc))
        if failures:
            raise FileNotFoundError('Required inputs are unavailable:\n' + '\n'.join(failures))
        return resolved

    def output(self, relative=''):
        if self.output_dir is None:
            raise ValueError('--output is required for execution.')
        relative = Path(relative)
        if relative.is_absolute():
            raise ValueError('Output names must be relative to the isolated output directory.')
        path = (self.output_dir / relative).resolve()
        if path != self.output_dir and self.output_dir not in path.parents:
            raise ValueError('Output path escaped the isolated output directory.')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def prepare_environment(self):
        sys.dont_write_bytecode = True
        cache = self.output('_cache')
        cache.mkdir(exist_ok=True)
        for name, value in {
            'PYTHONDONTWRITEBYTECODE':'1', 'MPLBACKEND':'Agg',
            'MPLCONFIGDIR':str(cache / 'matplotlib'), 'NUMBA_CACHE_DIR':str(cache / 'numba'),
            'XDG_CACHE_HOME':str(cache), 'TMP':str(cache), 'TEMP':str(cache),
        }.items():
            os.environ[name] = value


def declared_inputs(module_name, task):
    """Read a literal task-input contract without importing plotting/model libraries."""
    repo = Path(__file__).resolve().parents[2]
    module_path = repo / 'src' / Path(*module_name.split('.')).with_suffix('.py')
    for node in ast.parse(module_path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'INPUTS_BY_TASK' for target in node.targets):
            return ast.literal_eval(node.value)[task]
    raise ValueError(f'{module_name} must declare literal INPUTS_BY_TASK for preflight.')


def cli(module_name, tasks, default_task=None):
    """Common CLI; analysis modules declare INPUTS_BY_TASK and run(ctx, task=...)."""
    description = 'experimental validation' if isinstance(module_name, dict) else module_name.rsplit('.', 1)[-1]
    parser = argparse.ArgumentParser(description=f'Run {description} using explicit configured inputs.')
    parser.add_argument('task', nargs='?', choices=tasks, default=None)
    parser.add_argument('--task', dest='task_option', choices=tasks)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--profile', help='Named resource profile from config.json; never inferred from output tables.')
    parser.add_argument('--output', '--work-dir', dest='output', type=Path)
    parser.add_argument('--preflight', action='store_true', help='Check required inputs only; do not execute an analysis.')
    args = parser.parse_args()
    task = args.task_option or args.task or default_task
    if task is None or (args.task_option and args.task and args.task_option != args.task):
        parser.error('Choose one unambiguous task.')
    if not args.preflight and args.output is None:
        parser.error('--output is required for execution.')
    if isinstance(module_name, dict):
        module_name = module_name[task]
    try:
        ctx = Context(args.config, args.inputs, args.output, args.profile)
        if args.profile and ctx.config['profiles'][args.profile].get('task', task) != task:
            raise ValueError(f'Input profile {args.profile!r} is not intended for task {task!r}.')
        ctx.check_source_integrity()
        required = declared_inputs(module_name, task)
        ctx.require(required)
        if args.preflight:
            print(json.dumps({'task':task,'status':'inputs_available','analysis_executed':False,'inputs':required},indent=2))
            return 0
        ctx.prepare_environment()
        module = importlib.import_module(module_name)
        result = module.run(ctx, task=task)
        print(json.dumps(to_builtin({'task':task,'status':'completed','output':ctx.output_dir,'products':result}),indent=2))
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, NotADirectoryError) as exc:
        parser.error(str(exc))


def parse_public_sources(repo_root):
    paths = sorted(Path(repo_root).rglob('*.py'))
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    return len(paths)
