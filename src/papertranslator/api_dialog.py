"""Explicit API setup; keys stay in Windows Credential Manager."""
from PySide6.QtWidgets import (QDialog,QDialogButtonBox,QVBoxLayout,QFormLayout,QLabel,QLineEdit,
                              QComboBox,QPushButton,QCheckBox,QSpinBox,QMessageBox)
from .cloud import available_models
from .credentials import load_key,save_key,delete_key
from .storage import atomic_json,read_json,root


class ApiDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle('무료 API 검토 설정');self.resize(640,460)
        self.settings=read_json(root()/'settings.json',{});config=self.settings.get('cloud',{})
        layout=QVBoxLayout(self)
        notice=QLabel('선택한 문제 문단의 원문·번역문·짧은 앞뒤 문맥을 OpenRouter와 처리 제공자에게 전송합니다.\n'
                     'PDF 파일·로컬 경로·작업 로그는 전송하지 않습니다. 제공자의 데이터 정책이 적용됩니다.\n'
                     '입력/출력 0원 모델만 확인하며, 비수집 경로가 없거나 무료 한도가 끝나면 로컬 결과를 유지합니다.\n'
                     '무료 API의 실제 번역 품질은 사용자의 키로 별도 확인해야 합니다.')
        notice.setWordWrap(True);layout.addWidget(notice)
        links=QLabel('<a href="https://openrouter.ai/settings/keys">키 만들기</a> · '
                     '<a href="https://openrouter.ai/privacy">데이터 정책</a> · '
                     '<a href="https://openrouter.ai/pricing">무료 한도</a>')
        links.setOpenExternalLinks(True);layout.addWidget(links)
        form=QFormLayout();self.key=QLineEdit();self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText('키는 Windows 자격 증명 관리자에 저장합니다.')
        if load_key():self.key.setPlaceholderText('등록된 키 있음 · 교체하려면 새 키를 입력하세요.')
        form.addRow('OpenRouter API 키',self.key)
        self.model=QComboBox();self.model.setEditable(True)
        self.model.addItem(config.get('model','openai/gpt-oss-120b:free'))
        form.addRow('무료 모델 ID',self.model)
        refresh=QPushButton('현재 0원 모델 목록 새로고침 · 문서 전송 없음');refresh.clicked.connect(self.refresh_models)
        form.addRow('',refresh)
        self.limit=QSpinBox();self.limit.setRange(1,20);self.limit.setValue(config.get('max_requests',20))
        form.addRow('작업당 최대 검토 문단',self.limit);layout.addLayout(form)
        self.enabled=QCheckBox('위 전송 범위를 확인했으며, 혼합 모드에서 무료 API 검토를 사용합니다.')
        self.enabled.setChecked(config.get('enabled',False));layout.addWidget(self.enabled)
        remove=QPushButton('등록된 API 키 삭제');remove.clicked.connect(self.remove_key);layout.addWidget(remove)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.store);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def refresh_models(self):
        try:
            current=self.model.currentText();models=available_models()
            if not models:raise ValueError('현재 0원 모델을 찾지 못했습니다.')
            self.model.clear();self.model.addItems(models)
            if current in models:self.model.setCurrentText(current)
        except Exception:QMessageBox.warning(self,'목록 확인 실패','인터넷 연결과 서비스 상태를 확인하세요. 모델 ID를 직접 입력할 수도 있습니다.')

    def remove_key(self):
        delete_key();self.key.clear();self.key.setPlaceholderText('등록된 키 없음')

    def store(self):
        try:
            model=self.model.currentText().strip()
            if not model.endswith(':free') or '/' not in model or any(c.isspace() for c in model):
                raise ValueError('공급자/모델:free 형식의 무료 모델 ID를 입력하세요.')
            if self.key.text().strip():save_key(self.key.text())
            if self.enabled.isChecked() and not load_key():raise ValueError('API 키를 먼저 등록하세요.')
            self.settings['cloud']=dict(enabled=self.enabled.isChecked(),model=model,max_requests=self.limit.value())
            atomic_json(root()/'settings.json',self.settings);self.key.clear();self.accept()
        except Exception as exc:QMessageBox.warning(self,'설정 확인',str(exc))
