"""Install checksum-pinned English OCR data; no document is uploaded."""
from pathlib import Path
import httpx
from papertranslator.ocr_assets import SHA256,SIZE,URL
from papertranslator.storage import sha256


def main():
    folder=Path(__file__).resolve().parents[1]/'work/assets/ocr';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'eng.traineddata'
    if target.exists() and sha256(target)==SHA256:return
    temporary=folder/'eng.traineddata.downloading'
    try:
        with httpx.stream('GET',URL,timeout=60,follow_redirects=False,trust_env=False) as response:
            response.raise_for_status();size=0
            with temporary.open('wb') as stream:
                for chunk in response.iter_bytes():
                    size+=len(chunk)
                    if size>SIZE:raise ValueError('Unexpected OCR data size')
                    stream.write(chunk)
        if temporary.stat().st_size!=SIZE or sha256(temporary)!=SHA256:
            raise ValueError('OCR data checksum mismatch')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    print('Verified English OCR data installed.')


if __name__=='__main__':main()
