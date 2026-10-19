from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import urllib.parse

import psutil
from PySide6.QtCore import Qt, QLockFile, QProcess, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPushButton, QProgressBar, QVBoxLayout, QWidget)

from . import __version__
from .jobs import create_job
from .storage import atomic_json, read_json, root
from .models import catalog, resolve_model
from .eta import remaining_label


def duration(seconds):
    if seconds is None: return '계산 중'
    seconds=int(max(0,seconds))
    return f'{seconds//60}분 {seconds%60:02d}초'


class DropZone(QPushButton):
    def __init__(self, owner):
        super().__init__('PDF를 여기에 놓거나 클릭해서 선택하세요\n영어 학술 PDF · 로컬 처리')
        self.owner=owner; self.setAcceptDrops(True); self.setObjectName('dropzone')
        self.setMinimumHeight(112); self.clicked.connect(owner.pick_pdf)

    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith('.pdf') for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self,event):
        path=next((u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile().lower().endswith('.pdf')),None)
        if path: self.owner.set_source(path)


class Window(QMainWindow):
    def __init__(self, worker_executable=None):
        super().__init__()
        self.setWindowTitle('논문 번역기 · 로컬 개발판'); self.resize(1120,900); self.setMinimumSize(950,840)
        self.source=None; self.folder=None; self.process=None; self.close_when_done=False
        self.worker_executable=worker_executable
        self.settings=read_json(root()/'settings.json',{})
        default=root()/'results'; default.mkdir(exist_ok=True)
        self.destination=self.settings.get('destination',str(default))
        outer=QWidget(); self.setCentralWidget(outer); layout=QHBoxLayout(outer); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        sidebar=QFrame(); sidebar.setObjectName('sidebar'); sidebar.setFixedWidth(255)
        side=QVBoxLayout(sidebar); side.setContentsMargins(22,28,22,24); side.setSpacing(18)
        brand=QLabel('PAPER\nTRANSLATOR'); brand.setObjectName('brand'); side.addWidget(brand)
        side.addWidget(QLabel('내 컴퓨터에서 읽는 논문'))
        self.new_btn=QPushButton('+ 새 번역'); self.new_btn.clicked.connect(self.new_job); side.addWidget(self.new_btn)
        side.addWidget(QLabel('작업 기록'))
        self.history=QListWidget(); self.history.itemClicked.connect(self.select_job); side.addWidget(self.history,1)
        self.resources=QLabel('GPU 확인 중'); self.resources.setWordWrap(True); side.addWidget(self.resources)
        side.addWidget(QLabel(f'{__version__}  ·  PUBLIC ALPHA'))
        layout.addWidget(sidebar)
        panel=QWidget(); body=QVBoxLayout(panel); body.setContentsMargins(36,26,36,22); body.setSpacing(13)
        kicker=QLabel('영어 → 한국어  /  기본 로컬 · 선택 API'); kicker.setObjectName('kicker'); body.addWidget(kicker)
        heading=QLabel('논문을 한국어로 읽기'); heading.setObjectName('heading'); body.addWidget(heading)
        desc=QLabel('원문과 번역, 영한 대조본을 한 폴더에 저장합니다.'); body.addWidget(desc)
        self.drop=DropZone(self); body.addWidget(self.drop)
        form=QFormLayout(); form.setSpacing(10)
        self.title=QLineEdit(); self.title.setPlaceholderText('파일을 선택하면 제목이 입력됩니다.'); form.addRow('저장 제목',self.title)
        self.page_spec=QLineEdit(); self.page_spec.setPlaceholderText('전체 쪽 · 예: 1-12,15,20-'); form.addRow('번역할 쪽',self.page_spec)
        self.ocr_choice=QComboBox()
        for label,value in [('자동 · 스캔/이미지 속 글자 인식','auto'),('끄기 · PDF 텍스트만 사용','off'),('강제 · 모든 쪽을 이미지로 다시 인식','force')]:self.ocr_choice.addItem(label,value)
        self.ocr_choice.setToolTip('영어 인쇄체용 로컬 CPU OCR. 강제 모드는 기존 텍스트 대신 OCR 결과를 사용합니다.')
        form.addRow('OCR',self.ocr_choice)
        self.profile=QComboBox(); self.profile.addItem('일반 학술','general'); self.profile.addItem('미술치료 · 정신분석','art_therapy'); self.profile.addItem('컴퓨터공학','computer_science'); form.addRow('분야·용어',self.profile)
        self.model_choice=QComboBox()
        registry=read_json(root()/'model_registry.json',{})
        for key,model in catalog(registry).items():
            self.model_choice.addItem(model.get('display_name',model['id']),key)
        preferred=self.settings.get('model_key') or registry.get('default_key','hy7b')
        self.model_choice.setCurrentIndex(max(0,self.model_choice.findData(preferred)))
        self.model_choice.currentIndexChanged.connect(self.model_changed)
        form.addRow('번역 모델',self.model_choice)
        self.review_choice=QCheckBox('다른 계열 모델로 의미 검수 · 추가 시간 소요')
        self.review_choice.hide()
        self.quality_choice=QComboBox()
        for label,value in [('문맥·용어 정리 · 단일 모델','contextual'),('정밀 · 다중 모델 교정 · 추가 시간','ensemble'),
                            ('혼합 · 정밀 + 무료 API 선택 검토','hybrid'),('기본 · 기존 단일 모델','legacy')]:self.quality_choice.addItem(label,value)
        default_quality='ensemble' if any(k in catalog(registry) for k in ['qwen38','qwen30b']) else 'contextual'
        self.quality_choice.setCurrentIndex(max(0,self.quality_choice.findData(self.settings.get('quality_mode',default_quality))))
        self.quality_choice.currentIndexChanged.connect(self.quality_changed)
        quality_row=QHBoxLayout();quality_row.addWidget(self.quality_choice)
        self.api_btn=QPushButton('무료 API 설정');self.api_btn.clicked.connect(self.api_settings);quality_row.addWidget(self.api_btn)
        self.api_btn.setFixedSize(130,36)
        form.addRow('교정 방식',quality_row)
        self.layout_choice=QComboBox()
        self.layout_choice.addItem('원문 위치 대응 · 표·문단 배치 유지','source_position')
        self.layout_choice.addItem('읽기형 · 번역문 순서대로 배치','reflow')
        self.layout_choice.setCurrentIndex(max(0,self.layout_choice.findData(self.settings.get('bilingual_layout','source_position'))))
        self.layout_choice.setToolTip('대조본 오른쪽 번역을 원문 영역에 대응시킵니다. 넘치는 내용은 같은 원문의 보충면에 이어 싣고, 쪽 대응을 확정하지 못하면 표시합니다.')
        form.addRow('대조본 배치',self.layout_choice)
        row=QHBoxLayout(); self.dest=QLineEdit(self.destination); self.dest.setReadOnly(True); row.addWidget(self.dest)
        self.dest_btn=QPushButton('폴더 선택'); self.dest_btn.setFixedHeight(38); self.dest_btn.clicked.connect(self.pick_destination); row.addWidget(self.dest_btn); form.addRow('결과 저장',row)
        body.addLayout(form)
        self.model=QLabel(); self.model.setWordWrap(True); self.model.setObjectName('model'); body.addWidget(self.model)
        actions=QHBoxLayout(); self.start=QPushButton('번역 시작'); self.start.setObjectName('primary'); self.start.clicked.connect(self.start_job)
        self.cancel_btn=QPushButton('취소'); self.cancel_btn.clicked.connect(self.cancel_job); self.cancel_btn.setEnabled(False)
        self.resume=QPushButton('이어하기'); self.resume.clicked.connect(self.resume_job); self.resume.setEnabled(False)
        self.retry=QPushButton('실패 문단 재처리'); self.retry.clicked.connect(lambda: self.resume_job(True)); self.retry.setEnabled(False)
        for w in [self.start,self.cancel_btn,self.resume,self.retry]: actions.addWidget(w)
        body.addLayout(actions)
        progress=QFrame(); progress.setObjectName('progresscard'); box=QVBoxLayout(progress); box.setContentsMargins(18,16,18,16)
        self.stage=QLabel('PDF를 선택하면 시작할 수 있습니다.'); self.stage.setObjectName('stage'); box.addWidget(self.stage)
        self.bar=QProgressBar(); self.bar.setRange(0,100); self.bar.setValue(0); box.addWidget(self.bar)
        self.detail=QLabel('경과 —  /  남은 시간 —'); self.detail.setWordWrap(True); box.addWidget(self.detail)
        self.result=QLabel('작업 기록은 자동 저장됩니다.'); self.result.setWordWrap(True); box.addWidget(self.result)
        body.addWidget(progress)
        results=QHBoxLayout()
        self.open_btn=QPushButton('결과 폴더'); self.open_btn.clicked.connect(lambda: self.open_output(''))
        self.ko_btn=QPushButton('한국어 PDF'); self.ko_btn.clicked.connect(lambda: self.open_output('Korean.pdf'))
        self.dual_btn=QPushButton('영한 대조'); self.dual_btn.clicked.connect(lambda: self.open_output('Bilingual.pdf'))
        self.review_btn=QPushButton('문단 검토'); self.review_btn.clicked.connect(lambda: self.open_output('Review.html'))
        self.obsidian_btn=QPushButton('Obsidian'); self.obsidian_btn.clicked.connect(self.open_obsidian)
        for w in [self.open_btn,self.ko_btn,self.dual_btn,self.review_btn,self.obsidian_btn]: w.setEnabled(False); results.addWidget(w)
        body.addLayout(results)
        body.addStretch(1)
        warning=QLabel('초기 개발판: 스캔·혼합 PDF 로컬 OCR 및 한국어 재배치 지원.\nOCR 인식 오류·복잡한 표/수식은 원문 대조가 필요합니다. 기계 검수 통과는 의미 정확도 보증이 아닙니다.')
        warning.setWordWrap(True); warning.setObjectName('notice'); body.addWidget(warning)
        layout.addWidget(panel,1)
        self.setStyleSheet('''QWidget{font-family:"Malgun Gothic";font-size:12px;color:#1c2d40;background:#f6f8fa}QLabel{background:transparent}#sidebar{background:#142536}#sidebar QLabel{color:#bdcbd9}#brand{font-size:25px;font-weight:700;color:#ffffff;letter-spacing:2px}QListWidget{border:0;background:#142536;color:#e4edf4}QListWidget::item{padding:12px 4px;border-bottom:1px solid #2d4051}QListWidget::item:selected{background:#2a465b}#sidebar QPushButton{background:#274357;color:#f4f9fc;border:1px solid #456277}#heading{font-size:27px;font-weight:700}#kicker{font-size:10px;letter-spacing:1px;color:#397579}QPushButton{padding:10px 12px;border:1px solid #d0d9e0;border-radius:7px;background:#ffffff;font-weight:600}QPushButton:hover{background:#e8f1f1}QPushButton:disabled{color:#a8b1ba;background:#edf0f3}#primary{background:#176d70;color:#fff;border-color:#176d70}#primary:hover{background:#125e60}#primary:disabled{background:#99b5b6}#dropzone{border:2px dashed #93b4bc;background:#edf5f5;font-size:14px;line-height:2;color:#285d67}QLineEdit,QComboBox{padding:8px;border:1px solid #cbd5df;border-radius:5px;background:#fff;min-height:20px}#model{font-size:11px;color:#597080}#progresscard{background:#ffffff;border:1px solid #dce3e9;border-radius:9px}#stage{font-size:15px;font-weight:700}QProgressBar{border:0;border-radius:4px;background:#e8edf1;height:10px;text-align:center;font-size:9px}QProgressBar::chunk{background:#278b87;border-radius:4px}#notice{font-size:11px;color:#84612d;background:#f6efdf;border-radius:6px;padding:12px}''')
        self.timer=QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(750)
        self.gpu=QProcess(self); self.gpu.readyReadStandardOutput.connect(self.gpu_result)
        self.gpu_timer=QTimer(self); self.gpu_timer.timeout.connect(self.poll_gpu); self.gpu_timer.start(3000)
        self.refresh_history(); self.refresh(); self.poll_gpu()

    def active(self):
        return self.process is not None and self.process.state()!=QProcess.NotRunning

    def set_source(self,path):
        if self.active(): return
        self.source=str(Path(path).resolve()); self.title.setText(Path(path).stem)
        self.drop.setText(Path(path).name+'\nPDF 선택됨 · 클릭해서 변경'); self.folder=None; self.refresh()

    def pick_pdf(self):
        path,_=QFileDialog.getOpenFileName(self,'영어 PDF 선택','','PDF (*.pdf)')
        if path: self.set_source(path)

    def pick_destination(self):
        path=QFileDialog.getExistingDirectory(self,'결과 폴더 또는 Obsidian Vault 선택',self.destination)
        if path:
            self.destination=path; self.dest.setText(path)
            self.settings['destination']=path; atomic_json(root()/'settings.json',self.settings)

    def new_job(self):
        if self.active(): return
        self.folder=None; self.source=None; self.title.clear(); self.page_spec.clear()
        self.layout_choice.setCurrentIndex(max(0,self.layout_choice.findData(self.settings.get('bilingual_layout','source_position'))))
        preferred=self.settings.get('quality_mode','ensemble' if any(k in catalog() for k in ['qwen38','qwen30b']) else 'contextual')
        self.quality_choice.setCurrentIndex(max(0,self.quality_choice.findData(preferred)))
        self.drop.setText('PDF를 여기에 놓거나 클릭해서 선택하세요\n영어 학술 PDF · 로컬 처리'); self.refresh()

    def start_job(self):
        try:
            if not self.source: raise ValueError('PDF를 먼저 선택하세요.')
            self.folder=create_job(self.source,self.destination,self.page_spec.text(),self.profile.currentData(),self.title.text(),self.model_choice.currentData(),self.review_choice.isChecked(),ocr_mode=self.ocr_choice.currentData(),quality_mode=self.quality_choice.currentData(),bilingual_layout=self.layout_choice.currentData())
            self.launch()
        except Exception as exc: QMessageBox.warning(self,'시작할 수 없습니다',str(exc))

    def model_changed(self,*_):
        self.settings['model_key']=self.model_choice.currentData()
        atomic_json(root()/'settings.json',self.settings)
        self.refresh()

    def quality_changed(self,*_):
        self.settings['quality_mode']=self.quality_choice.currentData()
        self.settings['bilingual_layout']=self.layout_choice.currentData()
        atomic_json(root()/'settings.json',self.settings)

    def api_settings(self):
        from .api_dialog import ApiDialog
        ApiDialog(self).exec();self.settings=read_json(root()/'settings.json',{})

    def launch(self,retry=False):
        if self.active(): return
        (self.folder/'cancel').unlink(missing_ok=True)
        self.process=QProcess(self); self.process.setWorkingDirectory(str(root()))
        if self.worker_executable or getattr(sys,'frozen',False): args=['--worker',str(self.folder)]
        else: args=['-m','papertranslator','--worker',str(self.folder)]
        if retry: args.append('--retry-failed')
        self.process.finished.connect(self.worker_finished)
        self.process.errorOccurred.connect(self.worker_error)
        self.process.start(self.worker_executable or sys.executable,args); self.refresh()

    def worker_error(self,error):
        if error==QProcess.FailedToStart:
            QMessageBox.warning(self,'작업 실행 실패',self.process.errorString())

    def worker_finished(self,*_):
        state=read_json(self.folder/'state.json',{})
        if state.get('status') not in {'review_required','completed_with_warnings','failed','cancelled'} and not (state.get('pid') and psutil.pid_exists(state['pid'])):
            state.update(status='interrupted',stage='중단됨 · 이어하기 가능',pid=None,eta_seconds=None)
            atomic_json(self.folder/'state.json',state)
        self.refresh_history(); self.refresh()
        if self.close_when_done: self.close()

    def cancel_job(self):
        if self.active() and self.folder:
            (self.folder/'cancel').touch(); self.stage.setText('취소 요청 · 문단 기록 정리 중')

    def resume_job(self,retry=False):
        if self.folder and not self.active(): self.launch(retry)

    def select_job(self,item):
        if self.active(): return
        self.folder=Path(item.data(Qt.UserRole)); state=read_json(self.folder/'state.json',{})
        self.source=state.get('source'); self.title.setText(state.get('title','')); self.page_spec.setText(state.get('page_spec',''))
        self.destination=state.get('destination',self.destination); self.dest.setText(self.destination)
        self.profile.setCurrentIndex(max(0,self.profile.findData(state.get('profile','general'))))
        self.ocr_choice.setCurrentIndex(max(0,self.ocr_choice.findData(state.get('ocr_mode','off'))))
        self.layout_choice.setCurrentIndex(max(0,self.layout_choice.findData(state.get('bilingual_layout','reflow'))))
        self.quality_choice.blockSignals(True)
        self.quality_choice.setCurrentIndex(max(0,self.quality_choice.findData(state.get('quality_policy',{}).get('mode','legacy'))))
        self.quality_choice.blockSignals(False)
        self.drop.setText(state.get('source_name','PDF')+'\n저장된 작업'); self.refresh()

    def refresh_history(self):
        self.history.clear()
        files=sorted((root()/'jobs').glob('*/state.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        labels={'review_required':'검토 필요 · 검사 미통과','completed_with_warnings':'완료 · 확인 필요','failed':'실패','cancelled':'취소','interrupted':'중단'}
        for path in files[:60]:
            state=read_json(path,{})
            item=QListWidgetItem(state.get('title','논문')[:22]+'\n'+labels.get(state.get('status'),state.get('stage','대기')))
            item.setData(Qt.UserRole,str(path.parent)); self.history.addItem(item)

    def refresh(self):
        active=self.active(); state=read_json(self.folder/'state.json',{}) if self.folder else {}
        registry=read_json(root()/'model_registry.json',{})
        chosen=state.get('model') or catalog(registry).get(self.model_choice.currentData(),{})
        reviewer=registry.get('reviewer',{})
        can_review=reviewer.get('qualified_on_small_controls',False) and reviewer.get('adapter')!=chosen.get('adapter','hy')
        self.review_choice.setEnabled(not active and bool(can_review))
        if not can_review:self.review_choice.setChecked(False)
        self.model.setText('로컬 모델  '+chosen.get('id','설치 대기')+'  ·  '+chosen.get('status_ko','비교 중 · 의미 검수 필요'))
        for w in [self.drop,self.title,self.page_spec,self.profile,self.model_choice,self.ocr_choice,self.quality_choice,self.layout_choice,self.api_btn,self.dest_btn,self.new_btn,self.history]: w.setEnabled(not active)
        self.start.setEnabled(not active and self.source is not None and bool(registry))
        self.cancel_btn.setEnabled(active)
        another_worker=bool(state.get('pid') and psutil.pid_exists(state['pid']))
        self.resume.setEnabled(not active and not another_worker and bool(state) and state.get('status') in {'cancelled','interrupted','failed','queued','preparing','analyzing','translating','refining','cloud_reviewing','reviewing','fitting_labels','aligning','typesetting','saving'})
        self.retry.setEnabled(not active and bool(state.get('quality_summary',{}).get('failed')))
        has_output=bool(state.get('output')) and Path(state['output']).is_dir()
        for w in [self.open_btn,self.ko_btn,self.dual_btn,self.review_btn,self.obsidian_btn]: w.setEnabled(has_output)
        if not state:
            self.bar.setRange(0,100); self.bar.setValue(0); self.stage.setText('번역할 PDF를 선택하세요.' if not self.source else '번역 준비됨')
            self.detail.setText('경과 —  /  남은 시간 —'); self.result.setText('결과 폴더에 Papers/<제목>이 새로 만들어집니다.'); return
        self.stage.setText(state.get('stage','대기'))
        total=state.get('total',0); completed=state.get('completed',0)
        if active and state.get('status') in {'preparing','analyzing','refining','cloud_reviewing','reviewing','fitting_labels','aligning','typesetting','saving'}:
            self.bar.setRange(0,0)
        else:
            self.bar.setRange(0,max(total,1)); self.bar.setValue(completed)
        elapsed=state.get('elapsed_seconds',0); age=0
        if active and state.get('updated_at'):
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['updated_at'])).total_seconds(); elapsed+=max(0,age)
        remaining_text=remaining_label(state,elapsed,active)
        current=state.get('current_source_page'); physical=f'원본 {current}쪽 · ' if current else ''
        self.detail.setText(f'{physical}문단 {completed}/{total} · 경과 {duration(elapsed)} · 번역 예상 {remaining_text}')
        if state.get('error'): self.result.setText(state['error'])
        elif state.get('quality_summary'):
            q=state['quality_summary']; self.result.setText(f'번역 {q["translated"]}문단 · 자동 검사 미통과 {q["failed"]}문단 · 경고 {q["warnings"]}문단 · 별도 의미 검수 {q.get("semantic_review_coverage",0):.0%}')
            if q.get('ocr_pages'):self.result.setText(self.result.text()+f'\nOCR {q["ocr_pages"]}쪽 · 낮은 인식 신뢰도 {q.get("ocr_low_confidence_lines",0)}줄 · 이미지 원문 대조 필요')
            if q.get('ocr_unscored_lines'):self.result.setText(self.result.text()+f' · 점수 미제공 {q["ocr_unscored_lines"]}줄')
            if q.get('refinement_processed'):self.result.setText(self.result.text()+f'\n다중 모델 교정 {q["refinement_processed"]}문단 · 후보 반영 {q.get("refinement_accepted",0)}문단 · 인간 승인 없음')
            if q.get('output_layout')=='source_position':
                self.result.setText(self.result.text()+f'\n원문 위치 대응 · 보충면 {q.get("layout_overflow",0)}쪽 · 대응 추정 {q.get("layout_proposed",0)}문단 · 미확정 {q.get("layout_unresolved",0)}문단')
            if (q.get('layout_validation') or {}).get('status')=='review_required':
                self.result.setText(self.result.text()+'\n배치 검사 미통과 · 문단 검토에서 해당 원본 쪽을 확인하세요.')
        else: self.result.setText('문단별 기록 자동 저장 · 재개 시 완료된 문단 재사용')

    def poll_gpu(self):
        if self.gpu.state()!=QProcess.NotRunning: return
        self.gpu.start('nvidia-smi',['--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])

    def gpu_result(self):
        data=bytes(self.gpu.readAllStandardOutput()).decode('utf-8','replace').strip().splitlines()
        if data:
            fields=data[0].split(',')
            if len(fields)==3:
                used,total,load=fields
                self.resources.setText(f'NVIDIA GPU 0\nVRAM {used.strip()} / {total.strip()} MiB\nGPU 사용률 {load.strip()}%\nRAM 여유 {psutil.virtual_memory().available/1024**3:.1f} GiB')

    def open_output(self,name):
        state=read_json(self.folder/'state.json',{})
        if state.get('output'): QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state['output'])/name)))

    def open_obsidian(self):
        state=read_json(self.folder/'state.json',{})
        if state.get('output'):
            note=next(Path(state['output']).glob('*.md'),None)
            if note: QDesktopServices.openUrl(QUrl('obsidian://open?path='+urllib.parse.quote(str(note),safe='')))

    def closeEvent(self,event):
        if self.active():
            self.close_when_done=True; self.cancel_job(); self.stage.setText('기록을 저장하고 종료하는 중'); event.ignore()
            QTimer.singleShot(10000,self.force_close)
        else:
            if self.gpu.state()!=QProcess.NotRunning: self.gpu.kill(); self.gpu.waitForFinished(1000)
            event.accept()

    def force_close(self):
        if self.close_when_done and self.active():
            # Job Object kills only this worker's engine when the worker exits.
            self.process.kill()


def configure_fonts(app):
    # Qt offscreen platform has no system-font discovery on Windows.
    # Load installed OS fonts; do not redistribute Windows font files.
    if 'Malgun Gothic' not in QFontDatabase.families():
        for name in ['malgun.ttf','malgunbd.ttf']:
            path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/name
            if path.exists(): QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont('Malgun Gothic',10))


def main():
    app=QApplication(sys.argv); app.setApplicationName('PaperTranslator'); configure_fonts(app)
    lock=QLockFile(str(root()/'ui.lock')); lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        QMessageBox.information(None,'논문 번역기','앱이 이미 실행 중입니다. 열려 있는 창을 사용하세요.')
        return 0
    window=Window(); window.show()
    return app.exec()
