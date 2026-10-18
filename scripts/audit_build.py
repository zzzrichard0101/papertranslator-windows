"""Reject build-machine profile paths before publishing the Windows archive."""
from pathlib import Path
from types import CodeType
import os
from PyInstaller.archive.readers import CArchiveReader


def audit():
    project=Path(__file__).resolve().parents[1];folder=project/'dist/PaperTranslator'
    reader=CArchiveReader(str(folder/'PaperTranslator.exe'))
    pyz=reader.open_embedded_archive('PYZ.pyz');issues=[];count=0
    def inspect_code(code):
        nonlocal count
        count+=1
        if ':\\Users\\' in code.co_filename or ':/Users/' in code.co_filename:
            issues.append('personal profile in project code filename')
        for value in code.co_consts:
            if isinstance(value,CodeType):inspect_code(value)
    for name in pyz.toc:
        if name.startswith('papertranslator'):inspect_code(pyz.extract(name))
    home=os.environ.get('USERPROFILE')
    if not home:raise RuntimeError('Run this audit on the Windows build machine.')
    needles=[value.encode(encoding) for value in [home,home.replace('\\','/')] for encoding in ['utf-8','utf-16-le']]
    for path in folder.rglob('*'):
        if path.is_file():
            data=path.read_bytes()
            if any(needle in data for needle in needles):issues.append('profile path in '+path.relative_to(folder).as_posix())
    if issues:raise RuntimeError('; '.join(issues))
    print(f'Compiled build audit: {count} project code objects; no build-machine profile paths found.')


if __name__=='__main__':audit()
