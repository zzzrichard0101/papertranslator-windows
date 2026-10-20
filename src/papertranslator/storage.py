from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata
import uuid


def root():
    p = Path(os.environ.get('PAPERTRANSLATOR_HOME', Path(os.environ['LOCALAPPDATA']) / 'PaperTranslator'))
    p.mkdir(parents=True, exist_ok=True)
    return p


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic_json(path, value, *, compact=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, ensure_ascii=False, indent=None if compact else 2,
                      separators=(',', ':') if compact else None)
            f.flush(); os.fsync(f.fileno())
        # Windows readers/antivirus can briefly hold destination without DELETE sharing.
        for attempt in range(12):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt==11: raise
                time.sleep(min(.01*2**attempt,.2))
    finally:
        temp.unlink(missing_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pages(spec, count):
    if not spec.strip():
        return list(range(count))
    result = set()
    for token in spec.split(','):
        token = token.strip()
        match = re.fullmatch(r'(\d+)\s*(?:-\s*(\d*)\s*)?', token)
        if not match:
            raise ValueError('쪽 범위는 1-12,15,20- 형식으로 입력하세요.')
        first = int(match[1]); last = int(match[2] or count) if '-' in token else first
        if not 1 <= first <= last <= count:
            raise ValueError(f'쪽 범위가 잘못되었습니다. 전체 {count}쪽입니다.')
        result.update(range(first-1, last))
    return sorted(result)


def safe_title(value):
    value = unicodedata.normalize('NFC', value)
    value = re.sub(r'[\x00-\x1f<>:"/\\|?*\[\]#^]', '_', value).strip(' .')[:72].rstrip(' .')
    if not value:
        value = '논문'
    if re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', value, re.I):
        value = '_' + value
    return value


@contextmanager
def worker_lock():
    """OS byte lock is released even after abnormal process termination."""
    import msvcrt
    path = root() / 'worker.lock'
    with path.open('a+b') as f:
        f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('다른 번역 작업이 실행 중입니다. 완료 또는 취소 후 다시 시작하세요.') from exc
        try:
            yield
        finally:
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
