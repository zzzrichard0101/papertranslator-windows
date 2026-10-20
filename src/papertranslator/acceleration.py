"""Explicit experimental acceleration for an exactly identified model/runtime."""
from copy import deepcopy

VERSION='mtp-experiment-1'
# Metadata inspected for this published GGUF: qwen35, 65 blocks, one NextN head.
# The backend still verifies the complete file SHA256 before starting inference.
MTP_MODEL_SHA256='f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d'


def capable(model,registry):
    return (model.get('sha256')==MTP_MODEL_SHA256 and
            registry.get('runtime_release')=='b11718' and model.get('adapter')=='qwen')


def apply(model,registry,mode):
    if mode not in {'none','mtp'}:raise ValueError('지원하지 않는 추론 가속 설정')
    result=deepcopy(model)
    if mode=='mtp' and capable(model,registry):
        result['options']={**result.get('options',{}),'speculation':'mtp','spec_tokens':3}
    return result


def arguments(options):
    if options.get('speculation','none')=='none':return []
    return ['--spec-type','draft-mtp','--spec-draft-n-max',str(options.get('spec_tokens',3))]
