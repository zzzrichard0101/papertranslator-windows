"""Build an app archive from a clean tagged source tree; never copy local jobs."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile
from papertranslator import __version__

PROJECT=Path(__file__).resolve().parents[1]


def main():
    from audit_build import audit
    subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=PROJECT,check=True,capture_output=True)
    subprocess.run(['git','diff','--exit-code','HEAD','--'],cwd=PROJECT,check=True)
    subprocess.run(['git','diff','--cached','--exit-code'],cwd=PROJECT,check=True)
    subprocess.run(['git','ls-files','--error-unmatch','LICENSE','README.md','NOTICE.txt'],cwd=PROJECT,check=True,capture_output=True)
    subprocess.run([__import__('sys').executable,str(PROJECT/'scripts/audit_public.py')],cwd=PROJECT,check=True)
    audit()
    release=PROJECT/'release';release.mkdir(exist_ok=True)
    folder=release/f'PaperTranslator-Windows-v{__version__}'
    if folder.exists():raise RuntimeError('Release folder already exists; choose a fresh version or inspect it.')
    shutil.copytree(PROJECT/'dist/PaperTranslator',folder)
    for name in ['LICENSE','README.md','NOTICE.txt','DEVELOPMENT.txt','CHANGELOG.md','SECURITY.md']:
        shutil.copyfile(PROJECT/name,folder/name)
    shutil.copytree(PROJECT/'docs',folder/'docs')
    shutil.copytree(PROJECT/'exports/dependency-notices',folder/'dependency-notices')
    # Source is the exact Git commit, excluding untracked/ignored machine files.
    source=release/f'PaperTranslator-Windows-v{__version__}-source.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=PaperTranslator-Windows/','-o',str(source),'HEAD'],cwd=PROJECT,check=True)
    archive=release/(folder.name+'-win-x64.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as out:
        for path in sorted(folder.rglob('*')):
            if path.is_file():out.write(path,path.relative_to(release))
    hashes=[]
    for path in [archive,source]:
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        hashes.append(f'{digest}  {path.name}')
    (release/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n',encoding='utf-8')
    print(json.dumps(dict(version=__version__,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),archives=[p.name for p in [archive,source]])))


if __name__=='__main__':main()
