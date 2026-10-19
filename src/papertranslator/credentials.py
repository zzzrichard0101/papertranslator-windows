"""API keys in Windows Credential Manager, never settings or job JSON."""
import ctypes
from ctypes import wintypes
import hashlib
from .storage import root


class Credential(ctypes.Structure):
    _fields_=[('Flags',wintypes.DWORD),('Type',wintypes.DWORD),('TargetName',wintypes.LPWSTR),
      ('Comment',wintypes.LPWSTR),('LastWritten',wintypes.FILETIME),('CredentialBlobSize',wintypes.DWORD),
      ('CredentialBlob',ctypes.POINTER(ctypes.c_ubyte)),('Persist',wintypes.DWORD),
      ('AttributeCount',wintypes.DWORD),('Attributes',ctypes.c_void_p),('TargetAlias',wintypes.LPWSTR),('UserName',wintypes.LPWSTR)]


def target():
    suffix=hashlib.sha256(str(root().resolve()).casefold().encode()).hexdigest()[:24]
    return 'PaperTranslator/OpenRouter/'+suffix


def save_key(value):
    value=value.strip()
    if not 10<=len(value)<=512 or any(c.isspace() for c in value):raise ValueError('API 키 형식을 확인하세요.')
    encoded=value.encode('utf-8');blob=(ctypes.c_ubyte*len(encoded)).from_buffer_copy(encoded)
    credential=Credential(Type=1,TargetName=target(),CredentialBlobSize=len(encoded),CredentialBlob=blob,
                          Persist=2,UserName='OpenRouter')
    library=ctypes.WinDLL('Advapi32.dll',use_last_error=True)
    library.CredWriteW.argtypes=[ctypes.POINTER(Credential),wintypes.DWORD]
    if not library.CredWriteW(ctypes.byref(credential),0):raise RuntimeError('Windows 자격 증명 저장 실패')


def load_key():
    library=ctypes.WinDLL('Advapi32.dll',use_last_error=True)
    pointer=ctypes.POINTER(Credential)()
    library.CredReadW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.POINTER(ctypes.POINTER(Credential))]
    library.CredFree.argtypes=[ctypes.c_void_p]
    if not library.CredReadW(target(),1,0,ctypes.byref(pointer)):
        if ctypes.get_last_error()==1168:return None
        raise RuntimeError('Windows 자격 증명을 읽지 못했습니다.')
    try:return ctypes.string_at(pointer.contents.CredentialBlob,pointer.contents.CredentialBlobSize).decode('utf-8')
    finally:library.CredFree(pointer)


def delete_key():
    library=ctypes.WinDLL('Advapi32.dll',use_last_error=True)
    library.CredDeleteW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD]
    if not library.CredDeleteW(target(),1,0) and ctypes.get_last_error()!=1168:
        raise RuntimeError('Windows 자격 증명 삭제 실패')
