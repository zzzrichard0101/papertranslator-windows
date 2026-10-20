"""Durable per-record commits; JSON snapshots remain readable by older tools.

The database is authoritative once an epoch exists, including an empty epoch.
Never silently fall back to an older snapshot after a database error.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading

from .storage import atomic_json, read_json


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


class Checkpoint:
    def __init__(self, folder, stamp):
        self.folder = Path(folder)
        self.stamp = stamp
        self.lock = threading.RLock()
        self.key_locks = {}
        self.db = sqlite3.connect(self.folder/'checkpoint.sqlite3', timeout=10, check_same_thread=False)
        try:
            self._initialize(stamp)
        except BaseException:
            self.db.close();raise

    def _initialize(self,stamp):
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS epochs (stamp TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                stamp TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY (stamp, id));
            CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, data TEXT NOT NULL);
        ''')
        with self.db:
            if not self.db.execute('SELECT 1 FROM epochs WHERE stamp=?', (stamp,)).fetchone():
                old = read_json(self.folder/'checkpoint.json', {})
                records = old.get('records', {}) if old.get('fingerprint') == stamp else {}
                self.db.executemany('INSERT INTO records VALUES (?, ?, ?)',
                    ((stamp, key, encode(record)) for key, record in records.items()))
                self.db.execute('INSERT INTO epochs VALUES (?)', (stamp,))
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('current', stamp))

    def load(self):
        with self.lock:
            return {key: json.loads(data) for key, data in self.db.execute(
                'SELECT id, data FROM records WHERE stamp=? ORDER BY rowid', (self.stamp,))}

    def put(self, record):
        data = encode(record)
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO records VALUES (?, ?, ?)',
                            (self.stamp, record['id'], data))

    def remove(self, ids):
        with self.lock, self.db:
            self.db.executemany('DELETE FROM records WHERE stamp=? AND id=?',
                                ((self.stamp, key) for key in ids))

    def snapshot(self):
        atomic_json(self.folder/'checkpoint.json',
                    dict(fingerprint=self.stamp, records=self.load()), compact=True)

    __call__ = snapshot

    @contextmanager
    def response_lock(self, key):
        with self.lock:
            lock = self.key_locks.setdefault(key, threading.Lock())
        with lock:
            yield

    def get_response(self, key):
        with self.lock:
            row = self.db.execute('SELECT data FROM responses WHERE key=?', (key,)).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None

    def put_response(self, key, value):
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO responses VALUES (?, ?)', (key, encode(value)))

    def close(self):
        self.db.close()


def read_checkpoint(folder):
    """Consistent read-only view for progress tools, including active jobs."""
    folder = Path(folder)
    path = folder/'checkpoint.sqlite3'
    if not path.exists():
        return read_json(folder/'checkpoint.json', {})
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=10) as db:
        db.execute('BEGIN')
        row = db.execute("SELECT value FROM metadata WHERE key='current'").fetchone()
        if not row:
            raise ValueError('작업 저장 데이터베이스의 현재 버전 정보가 없습니다.')
        return dict(fingerprint=row[0], records={key: json.loads(data) for key, data in
            db.execute('SELECT id, data FROM records WHERE stamp=? ORDER BY rowid', (row[0],))})


def persist(save, record):
    """Allow old zero-argument callbacks in integrations and diagnostic tools."""
    if hasattr(save, 'put'):
        save.put(record)
    else:
        save()
