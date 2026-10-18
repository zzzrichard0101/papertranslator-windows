import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication
from papertranslator.ui import Window, configure_fonts

app=QApplication(sys.argv)
configure_fonts(app)
window=Window(); window.show(); app.processEvents()
project=Path(__file__).resolve().parents[1]
window.set_source(str(project/'benchmarks'/'runs'/'Synthetic Windows Study.pdf'))
window.history.clear()  # Test screenshot does not include user document metadata.
window.grab().save(str(project/'exports'/'app-preview.png'))
assert window.start.isEnabled()
window.close(); app.processEvents()
print('UI initialized, PDF selection and ready state verified; screenshot saved.')
