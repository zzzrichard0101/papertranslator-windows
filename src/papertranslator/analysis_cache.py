"""Reuse a complete analysis only for an unchanged input and parser contract."""
from .backend import Cancelled
from .storage import atomic_json,read_json,fingerprint


def get_analysis(folder,identity,analyze,cancel):
    if cancel():raise Cancelled()
    key=fingerprint(dict(version='complete-analysis-1',identity=identity))
    saved=read_json(folder/'analysis-cache.json',{})
    ir=saved.get('ir')
    if (saved.get('key')==key and isinstance(ir,dict) and isinstance(ir.get('blocks'),list)
        and isinstance(ir.get('pages'),list) and saved.get('content_hash')==fingerprint(ir)):
        if cancel():raise Cancelled()
        return ir,True
    ir=analyze()
    if cancel():raise Cancelled()
    atomic_json(folder/'analysis-cache.json',dict(key=key,content_hash=fingerprint(ir),ir=ir),compact=True)
    return ir,False
