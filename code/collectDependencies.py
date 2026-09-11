#!/usr/bin/env python3
import os
import sys
import ast
from collections import defaultdict

# use importlib.metadata (py3.8+) or fall back to importlib_metadata
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

# built-in modules in the current interpreter
BUILTIN_MODULES = set(sys.builtin_module_names)


def find_imports_in_file(path):
    """Parse a .py file and return a set of top-level module names imported."""
    modules = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
    except (FileNotFoundError, PermissionError, OSError):
        return modules

    try:
        tree = ast.parse(source, filename=path)
    except Exception:
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split('.')[0])
    return modules


def collect_dependencies(root_dir="."):
    pkg_to_files = defaultdict(set)

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.py'):
                fullpath = os.path.join(dirpath, fname)
                mods = find_imports_in_file(fullpath)
                for m in mods:
                    pkg_to_files[m].add(fullpath)

    deps = {}
    for pkg in sorted(pkg_to_files):
        if pkg in BUILTIN_MODULES:
            ver = 'built-in'
        else:
            try:
                ver = version(pkg)
            except PackageNotFoundError:
                ver = 'UNKNOWN'
        deps[pkg] = (ver, sorted(pkg_to_files[pkg]))
    return deps


def write_outputs(deps):
    # Write dependencies.txt
    with open('dependencies.txt', 'w', encoding='utf-8') as out:
        out.write(f"Python interpreter: {sys.executable}\n")
        out.write(f"Python version: {sys.version.replace(os.linesep, ' ')}\n\n")
        out.write('Detected packages and modules:\n\n')
        for pkg, (ver, files) in deps.items():
            out.write(f"- {pkg}: {ver}\n")
            out.write('  Referenced in:\n')
            for fp in files:
                out.write(f"    * {fp}\n")
            out.write('\n')

    # Write requirements.txt
    with open('requirements.txt', 'w', encoding='utf-8') as req:
        for pkg, (ver, _) in deps.items():
            if ver not in ('UNKNOWN', 'built-in'):
                req.write(f"{pkg}=={ver}\n")
            elif ver == 'built-in':
                req.write(f"# {pkg} is built into Python and doesn't require installation\n")
            else:
                req.write(f"# {pkg} version not found; supply manually if needed\n")


def main():
    print('Scanning for Python imports...')
    deps = collect_dependencies()
    write_outputs(deps)
    print('→ Wrote dependencies.txt and requirements.txt')


if __name__ == '__main__':
    main()

