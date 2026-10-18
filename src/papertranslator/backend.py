"""Authenticated loopback-only llama.cpp backend, owned by the job worker."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

import httpx

from .storage import read_json, root, sha256
from .models import options_for

OPTIONS = dict(context=8192, parallel=1, kv='f16', threads=12, gpu_layers=99,
               temperature=0.7, top_p=0.6, top_k=20, repeat_penalty=1.05, seed=42)


class Cancelled(Exception):
    pass


class OwnedJob:
    def __init__(self):
        class BASIC(ctypes.Structure):
            _fields_ = [('time1', ctypes.c_int64), ('time2', ctypes.c_int64), ('flags', wintypes.DWORD),
                        ('min', ctypes.c_size_t), ('max', ctypes.c_size_t), ('active', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ['r', 'w', 'o', 'rb', 'wb', 'ob']]
        class EXT(ctypes.Structure):
            _fields_ = [('basic', BASIC), ('io', IO), ('process', ctypes.c_size_t), ('job', ctypes.c_size_t),
                        ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        info = EXT(); info.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.handle or not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close(); raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if getattr(self, 'handle', None):
            self.kernel.CloseHandle(self.handle); self.handle = None


class LlamaBackend:
    def __init__(self, folder, cancel=lambda: False, model=None,log_name='engine.log'):
        self.folder = Path(folder); self.cancel = cancel
        self.log_name=log_name
        self.process = None; self.job = None; self.client = None; self.log = None
        self.registry = read_json(root() / 'model_registry.json')
        if not self.registry:
            raise RuntimeError('모델 설치가 필요합니다. scripts/bootstrap.py를 실행하세요.')
        self.model = model or self.registry['model']
        self.key = secrets.token_urlsafe(32)
        self.options = options_for(self.model)

    def check(self):
        if self.cancel():
            raise Cancelled()

    def start(self, progress=lambda _: None):
        self.check()
        path = Path(self.model['path'])
        if not path.is_file() or sha256(path) != self.model['sha256']:
            raise RuntimeError('모델 파일 SHA256 검증 실패. 설치를 다시 검사하세요.')
        self.check()
        exe = next(Path(self.registry['runtime_dir']).rglob('llama-server.exe'), None)
        if not exe:
            raise RuntimeError('CUDA 추론 런타임이 없습니다.')
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); self.port = s.getsockname()[1]
        self.job = OwnedJob()
        self.log = (self.folder/self.log_name).open('w', encoding='utf-8')
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '0'
        args = [str(exe), '-m', str(path), '--host', '127.0.0.1', '--port', str(self.port),
                '--api-key', self.key, '-ngl', str(self.options['gpu_layers']), '-c', str(self.options['context']),
                '-np', str(self.options['parallel']), '-t', str(self.options['threads']),
                '-ctk',self.options['kv'],'-ctv',self.options['kv'],'-fa',self.options['flash_attention'],
                '--no-webui', '--jinja', '--no-context-shift','--offline']
        if self.options.get('chat_template') and self.options['chat_template']!='model_metadata':
            args+=['--chat-template',self.options['chat_template']]
        if 'enable_thinking' in self.options:
            args+=['--chat-template-kwargs',json.dumps(dict(enable_thinking=self.options['enable_thinking']))]
        self.process = subprocess.Popen(args, cwd=exe.parent, env=env, stdout=self.log, stderr=self.log,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            self.job.assign(self.process)
        except Exception:
            self.process.terminate(); self.close(); raise
        self.client = httpx.Client(base_url=f'http://127.0.0.1:{self.port}',
                                   headers={'Authorization': 'Bearer '+self.key},
                                   timeout=httpx.Timeout(1.0, read=1.0), trust_env=False, follow_redirects=False)
        started = time.monotonic()
        while time.monotonic() - started < 120:
            self.check()
            if self.process.poll() is not None:
                raise RuntimeError('CUDA 엔진 시작 실패. 작업 폴더의 engine.log를 확인하세요.')
            try:
                if self.client.get('/health').status_code == 200:
                    probe='PT_TEMPLATE_CONTRACT_7F2A'
                    rendered=self.client.post('/apply-template',json=dict(messages=[dict(role='user',content=probe)]),timeout=10)
                    rendered.raise_for_status()
                    if probe not in rendered.json().get('prompt',''):
                        raise RuntimeError('채팅 템플릿이 입력 본문을 누락합니다. 추론을 차단했습니다.')
                    progress('모델 준비 완료'); return
            except httpx.HTTPError:
                pass
            time.sleep(.3)
        raise RuntimeError('모델 로드 제한 시간(120초) 초과')

    def translate(self, prompt, max_tokens=2200, json_schema=None):
        self.check()
        req = dict(messages=[dict(role='user', content=prompt)], stream=True,
                   temperature=self.options['temperature'], top_p=self.options['top_p'], top_k=self.options['top_k'],
                   min_p=self.options['min_p'],
                   repeat_penalty=self.options['repeat_penalty'], seed=self.options['seed'], max_tokens=max_tokens,
                   stream_options={'include_usage': True})
        if json_schema:
            req['response_format']=dict(type='json_schema',json_schema=dict(name='review',strict=True,schema=json_schema))
        if 'enable_thinking' in self.options:
            req['chat_template_kwargs']=dict(enable_thinking=self.options['enable_thinking'])
        # Exact runtime tokenizer prevents silent context truncation.
        response = self.client.post('/tokenize', json=dict(content=prompt), timeout=10)
        response.raise_for_status()
        input_tokens=len(response.json()['tokens'])
        if self.model.get('adapter')=='translategemma' and input_tokens>2048:
            raise RuntimeError('TranslateGemma 공식 입력 한도 2K를 초과했습니다.')
        if input_tokens + max_tokens + 256 > self.options['context']:
            raise RuntimeError('문단이 문맥 한도를 초과했습니다. 자동 잘림을 차단했습니다.')
        pieces = []; finish = None; usage = {}; began = time.monotonic(); first = None
        with self.client.stream('POST', '/v1/chat/completions', json=req,
                                timeout=httpx.Timeout(5, read=30)) as result:
            result.raise_for_status()
            for line in result.iter_lines():
                self.check()
                if time.monotonic()-began > 180:
                    raise RuntimeError('문단 번역 제한 시간(180초) 초과')
                if not line.startswith('data: ') or line == 'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('reasoning_content'):
                        raise RuntimeError('사고 채널 출력이 감지되어 번역 채택을 차단했습니다.')
                    if delta.get('content'):
                        first = first or time.monotonic()
                        pieces.append(delta['content'])
                    finish = choice.get('finish_reason') or finish
        if finish != 'stop':
            raise RuntimeError(f'출력 미완료: {finish}')
        text = ''.join(pieces).strip()
        if '<think' in text.lower() or '</think' in text.lower():
            raise RuntimeError('사고 텍스트가 섞여 번역 채택을 차단했습니다.')
        return dict(text=text, seconds=time.monotonic()-began,
                    first_token_seconds=(first-began if first else None), usage=usage,input_tokens=input_tokens)

    def close(self):
        if self.client:
            self.client.close(); self.client = None
        if self.job:
            self.job.close(); self.job = None
        if self.process and self.process.poll() is None:
            self.process.wait(timeout=10)
        if self.log:
            self.log.close(); self.log = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
