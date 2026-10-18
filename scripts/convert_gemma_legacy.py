"""Text-only Ollama legacy Gemma3 -> upstream GGUF migration, weights unchanged.

Follows https://github.com/ollama/ollama/blob/main/compatmigrate/gemma3.go
Uses official gguf reader/writer. Never rewrites the downloaded source model.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

import gguf

from papertranslator.storage import atomic_json,read_json,root,sha256

VERSION='ollama-gemma3-text-migration-1'
TEMPLATE="{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else 'user') + '\\n' + message['content'] + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"


def data_hash(array):
    view=memoryview(array).cast('B');h=hashlib.sha256()
    for start in range(0,len(view),16*1024**2):h.update(view[start:start+16*1024**2])
    return h.hexdigest()


def convert(model):
    folder=root()/'models'/'translategemma27b';manifest=folder/'migration.json'
    previous=read_json(manifest,{})
    if previous.get('version')==VERSION and Path(previous['derived_model']['path']).exists():
        if sha256(previous['derived_model']['path'])==previous['derived_model']['sha256']:
            atomic_json(folder/'verified-model.json',previous['derived_model'])
            return previous['derived_model']
    source=Path(model['path'])
    assert sha256(source)==model['sha256']
    # Preserve exact upstream migration/config evidence without executing it.
    commit='36a6e994d2f6c4ab94b3555185154d069e5dba87'
    upstream=f'https://raw.githubusercontent.com/ollama/ollama/{commit}/compatmigrate/gemma3.go'
    (folder/'migration-reference.go.txt').write_bytes(urllib.request.urlopen(upstream).read())
    reader=gguf.GGUFReader(source)
    tensors=[t for t in reader.tensors if not t.name.startswith(('v.','mm.'))]
    embedding=next(t for t in tensors if t.name=='token_embd.weight')
    vocab=int(embedding.shape[-1])
    target=source.with_name('TranslateGemma-27B-Q4_K_M-llama-text.gguf')
    partial=target.with_suffix('.gguf.converting')
    writer=gguf.GGUFWriter(partial,'gemma3',endianess=reader.endianess)
    replacement={
        'gemma3.attention.layer_norm_rms_epsilon':(1e-6,gguf.GGUFValueType.FLOAT32),
        'gemma3.rope.freq_base':(1000000.0,gguf.GGUFValueType.FLOAT32),
        'gemma3.rope.freq_base_swa':(10000.0,gguf.GGUFValueType.FLOAT32),
        'gemma3.rope.scaling.type':('linear',gguf.GGUFValueType.STRING),
        'gemma3.rope.scaling.factor':(8.0,gguf.GGUFValueType.FLOAT32),
        'tokenizer.chat_template':(TEMPLATE,gguf.GGUFValueType.STRING)}
    copied=set()
    for field in reader.fields.values():
        key=field.name
        if key.startswith('GGUF.') or key=='general.architecture' or key.startswith(('gemma3.vision.','gemma3.mm.')):continue
        if key in {'tokenizer.ggml.add_padding_token','tokenizer.ggml.add_unknown_token'}:continue
        value=field.contents();kind=field.types[0];sub=field.types[-1] if kind==gguf.GGUFValueType.ARRAY else None
        if key=='gemma3.rope.global.freq_base':key='gemma3.rope.freq_base'
        if key=='gemma3.rope.local.freq_base':key='gemma3.rope.freq_base_swa'
        if key in {'tokenizer.ggml.tokens','tokenizer.ggml.scores','tokenizer.ggml.token_type'}:value=value[:vocab]
        writer.add_key_value(key,value,kind,sub_type=sub);copied.add(key)
    for key,(value,kind) in replacement.items():
        if key not in copied:writer.add_key_value(key,value,kind)
    for tensor in tensors:
        writer.add_tensor_info(tensor.name,tensor.data.shape,tensor.data.dtype,tensor.data.nbytes,tensor.tensor_type)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_ti_data_to_file()
    for index,tensor in enumerate(tensors):
        writer.write_tensor_data(tensor.data,tensor_endianess=reader.endianess)
        if index%100==0:print(json.dumps(dict(stage='copying_tensor_bytes',done=index,total=len(tensors))),flush=True)
    writer.close()
    check=gguf.GGUFReader(partial);byname={t.name:t for t in check.tensors};proof=[]
    for tensor in tensors:
        digest=data_hash(tensor.data);assert digest==data_hash(byname[tensor.name].data),'Tensor bytes changed'
        proof.append(dict(name=tensor.name,sha256=digest,bytes=tensor.n_bytes))
    # Release the output mmap before Windows rename; source map remains read-only.
    del byname,check
    import gc;gc.collect()
    partial.replace(target)
    converted={**model,'path':str(target),'sha256':sha256(target),
        'original_asset':dict(path=str(source),sha256=model['sha256']),
        'conversion':dict(version=VERSION,upstream=upstream,text_tensors=len(tensors),omitted_vision_tensors=len(reader.tensors)-len(tensors),
                          token_vocab=vocab,all_text_tensor_bytes_verified=True)}
    atomic_json(manifest,dict(version=VERSION,derived_model=converted,tensor_proof=proof))
    atomic_json(folder/'verified-model.json',converted)
    registry=read_json(root()/'model_registry.json');registry.setdefault('models',{})['translategemma27b']=converted
    atomic_json(root()/'model_registry.json',registry)
    print(json.dumps(dict(status='converted_and_verified',path=str(target))),flush=True)
    return converted


if __name__=='__main__':convert(read_json(root()/'models'/'translategemma27b'/'verified-model.json'))
