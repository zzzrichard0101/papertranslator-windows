import sqlite3
from pathlib import Path

from papertranslator.checkpoint import Checkpoint, read_checkpoint
from papertranslator.storage import atomic_json, read_json


def test_legacy_import_and_committed_updates_survive_without_a_new_snapshot(tmp_path):
    old=dict(id='one',target='초안',human_approved=False)
    atomic_json(tmp_path/'checkpoint.json',dict(fingerprint='same',records={'one':old}))
    store=Checkpoint(tmp_path,'same')
    assert store.load()=={'one':old}
    edited={**old,'target':'교정문'};store.put(edited);store.close()
    assert read_json(tmp_path/'checkpoint.json')['records']['one']==old
    assert read_checkpoint(tmp_path)['records']=={'one':edited}
    store=Checkpoint(tmp_path,'same');assert store.load()=={'one':edited}
    store.snapshot();store.close()
    assert read_json(tmp_path/'checkpoint.json')['records']=={'one':edited}


def test_removed_failed_record_does_not_resurrect_from_older_json(tmp_path):
    old=dict(id='one',status='failed',target='원문')
    atomic_json(tmp_path/'checkpoint.json',dict(fingerprint='same',records={'one':old}))
    store=Checkpoint(tmp_path,'same');store.remove(['one']);store.close()
    store=Checkpoint(tmp_path,'same');assert store.load()=={};store.close()


def test_new_epoch_never_reuses_other_settings_but_keeps_old_epoch(tmp_path):
    store=Checkpoint(tmp_path,'first');store.put(dict(id='one',target='이전 설정'));store.close()
    store=Checkpoint(tmp_path,'changed');assert store.load()=={};store.close()
    store=Checkpoint(tmp_path,'first');assert store.load()['one']['target']=='이전 설정';store.close()


def test_uncommitted_update_is_rolled_back_on_connection_loss(tmp_path):
    store=Checkpoint(tmp_path,'same');store.put(dict(id='one',target='저장됨'))
    store.db.execute('BEGIN IMMEDIATE')
    store.db.execute('UPDATE records SET data=? WHERE id=?', ('{"id":"one","target":"partial"}','one'))
    store.close()
    assert read_checkpoint(tmp_path)['records']['one']['target']=='저장됨'


def test_abrupt_process_exit_preserves_last_committed_record(tmp_path):
    import subprocess,sys,os,papertranslator
    code='''import os,sys
from papertranslator.checkpoint import Checkpoint
store=Checkpoint(sys.argv[1],'same')
store.put(dict(id='one',target='committed'))
store.db.execute('BEGIN IMMEDIATE')
store.db.execute('UPDATE records SET data=?', ('{"id":"one","target":"uncommitted"}',))
os._exit(17)
'''
    env={**os.environ,'PYTHONPATH':str(Path(papertranslator.__file__).resolve().parents[1])}
    child=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=env)
    assert child.returncode==17
    assert read_checkpoint(tmp_path)['records']['one']['target']=='committed'


def test_corrupt_database_never_silently_falls_back_to_json(tmp_path):
    atomic_json(tmp_path/'checkpoint.json',dict(fingerprint='same',records={}))
    (tmp_path/'checkpoint.sqlite3').write_bytes(b'broken database')
    import pytest
    with pytest.raises(sqlite3.DatabaseError):Checkpoint(tmp_path,'same')


def test_read_only_progress_view_sees_one_consistent_transaction(tmp_path):
    store=Checkpoint(tmp_path,'same')
    for i in range(8):store.put(dict(id=str(i),target='저장된 문단'))
    view=read_checkpoint(tmp_path);assert len(view['records'])==8
    assert all(r['id']==key for key,r in view['records'].items())
    store.close()
