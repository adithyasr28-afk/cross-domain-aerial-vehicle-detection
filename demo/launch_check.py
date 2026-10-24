"""Offline launch preflight shared by the Windows and POSIX launchers."""
from pathlib import Path
import importlib.util
import os
import runpy
import sys


def main():
    os.chdir(Path(__file__).resolve().parent)
    packages = {'cv2': 'opencv-python', 'gradio': 'gradio', 'matplotlib': 'matplotlib',
                'numpy': 'numpy', 'torch': 'torch', 'torchvision': 'torchvision',
                'ultralytics': 'ultralytics', 'PIL': 'pillow'}
    missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        print('ERROR: Missing package(s): ' + ', '.join(missing), flush=True)
        print('Setup once while online: python -m pip install -r requirements.txt\nSee README.md; this launcher never installs or downloads anything.', flush=True)
        return 1
    print('Starting local models; please wait for Models ready. The browser opens after the server responds.', flush=True)
    args = [arg for arg in sys.argv[1:] if arg != '--no-browser']
    if '--no-browser' not in sys.argv and '--open-browser' not in args:
        args.append('--open-browser')
    sys.argv = ['app.py', *args]
    try:
        runpy.run_path('app.py', run_name='__main__')
    except ModuleNotFoundError as exc:
        print(f'ERROR: A dependency is missing: {exc.name}. Install requirements.txt; see README.md.', flush=True)
        return 1
    except KeyboardInterrupt:
        print('Demo stopped.')
    except Exception as exc:
        print(f'ERROR: Demo could not start: {exc}. See README.md and FINAL_STATUS.md.', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
