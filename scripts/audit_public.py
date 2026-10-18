"""Check exact Git index blobs. Report locations/categories, never secret values."""
from pathlib import Path
import re
import subprocess
import sys

TOP={'README.md','NOTICE.txt','LICENSE','SECURITY.md','CONTRIBUTING.md','CHANGELOG.md',
     'AGENTS.md','DEVELOPMENT.txt','pyproject.toml','uv.lock','requirements-windows.lock','.gitignore','.gitattributes'}
DIRECTORIES={'src','tests','scripts','docs','.github'}
DATA={'benchmarks/model-evaluation.json','benchmarks/fresh-evaluation.json'}
PATTERNS={
    'provider credential':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})'),
    'private key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential URL':re.compile(r'https?://[^/\s:@]+:[^@\s/]+@'),
    'personal machine path':re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+[^\s"\x27<>]+|/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'literal credential':re.compile(r'''(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*[=:]\s*["']([^"'\r\n]{16,})["']'''),
}


def findings(path,data):
    result=[];parts=Path(path).parts
    if not (path in TOP or path in DATA or (parts and parts[0] in DIRECTORIES)):
        result.append((path,0,'file outside publication allowlist'))
    if '__pycache__' in parts or Path(path).suffix.lower() in {'.pyc','.pdf','.gguf','.exe','.dll','.pyd','.log','.zip','.png','.jpg'}:
        result.append((path,0,'generated/private/binary file'))
    if len(data)>1024*1024:result.append((path,0,'file larger than 1 MiB'))
    try:text=data.decode('utf-8-sig')
    except UnicodeDecodeError:return result+[(path,0,'non UTF-8/binary data')]
    for category,pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            result.append((path,text.count('\n',0,match.start())+1,category))
    return result


def main():
    project=Path(__file__).resolve().parents[1]
    paths=subprocess.check_output(['git','ls-files','--cached','-z'],cwd=project).decode().split('\0')
    paths=[p for p in paths if p]
    if not paths:raise SystemExit('Stage the publication allowlist before auditing.')
    issues=[]
    for path in paths:
        data=subprocess.check_output(['git','show',':'+path],cwd=project)
        issues.extend(findings(path,data))
    for path,line,category in issues:print(f'{path}:{line}: {category}')
    print(f'Public index audit: {len(paths)} files, {len(issues)} findings. No file contents printed.')
    return int(bool(issues))


if __name__=='__main__':sys.exit(main())
