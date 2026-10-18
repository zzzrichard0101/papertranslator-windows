"""Native Qt event test of PDF -> button -> frozen EXE worker -> output."""
from pathlib import Path
import sys

from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from papertranslator.storage import atomic_json, read_json
from papertranslator.ui import Window, configure_fonts

project=Path(__file__).resolve().parents[1]
app=QApplication(sys.argv); configure_fonts(app)
window=Window(worker_executable=str(project/'dist'/'PaperTranslator'/'PaperTranslator.exe'))
window.show(); outcome={'status':'timed_out'}

def begin():
    window.set_source(str(project/'benchmarks'/'runs'/'Synthetic Windows Study.pdf'))
    window.title.setText('네이티브 UI 통합 시험')
    window.destination=str(project/'test_vault'); window.dest.setText(window.destination)
    QTest.mouseClick(window.start,Qt.LeftButton)
    assert not window.drop.isEnabled()
    window.process.finished.connect(done)

def done(*_):
    window.refresh(); app.processEvents()
    window.history.clear()  # Keep user document metadata out of test artifacts.
    state=read_json(window.folder/'state.json')
    outcome.update(status=state['status'],worker='frozen EXE',job=str(window.folder),
                   output=state.get('output'),buttons_enabled=window.ko_btn.isEnabled(),
                   elapsed_seconds=state['elapsed_seconds'])
    atomic_json(project/'benchmarks'/'runs'/'latest-job.json',dict(folder=str(window.folder),source=state['source']))
    window.grab().save(str(project/'exports'/'app-completed.png'))
    QTimer.singleShot(250,finish)

def finish():
    atomic_json(project/'exports'/'ui-e2e.json',outcome)
    window.close(); app.quit()

QTimer.singleShot(300,begin); QTimer.singleShot(120000,finish)
app.exec()
assert outcome['status']=='completed_with_warnings' and outcome['buttons_enabled'],outcome
print(outcome)
