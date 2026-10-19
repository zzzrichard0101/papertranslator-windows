"""Pinned English LSTM data for the OCR engine included in MuPDF."""
from pathlib import Path

REVISION='e12c65a915945e4c28e237a9b52bc4a8f39a0cec'
SHA256='8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba'
SIZE=15400601
URL=f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/{REVISION}/eng.traineddata'


def data_directory():
    bundled=Path(__file__).parent/'ocr_models'
    development=Path(__file__).resolve().parents[2]/'work/assets/ocr'
    return next((p for p in [bundled,development] if (p/'eng.traineddata').is_file()),None)
