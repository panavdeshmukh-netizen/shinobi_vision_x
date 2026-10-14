"""SHINOBI VISION X main window."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider,
                               QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from core.config import CONFIG_DIR, load_json, save_json
from core.engine import MODES
from vision.calibration import delete_custom, load_custom
from vision.gesture_classifier import BUILTIN
from vision.sequence_engine import SequenceDef
from vfx.renderer import VFXRenderer
from .workers import VisionWorker

PAGES = ["Dashboard", "Live Studio", "Jutsu Controls", "Gesture Lab", "VFX Library", "Sequence Builder", "Calibration", "Settings", "About"]
_EFFECTS = VFXRenderer().effects
EFFECT_NAMES = list(_EFFECTS)


def lbl(text, name=None):
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(True)
    return w


class MainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.worker = None
        self.setWindowTitle("SHINOBI VISION X")
        self.resize(1440, 860)
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        side = QVBoxLayout()
        side.setSpacing(0)
        side.addWidget(lbl("SHINOBI VISION X", "brand"))
        self.sidebar = QListWidget(objectName="sidebar")
        self.sidebar.addItems(PAGES)
        side.addWidget(self.sidebar)
        row.addLayout(side)

        self.stack = QStackedWidget()
        row.addWidget(self.stack, 1)
        for build in (self._dashboard, self._live, self._jutsu_controls, self._lab, self._library, self._seq_builder,
                      self._calibration, self._settings_page, self._about):
            self.stack.addWidget(build())
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(1)
        row.addWidget(self._right_panel())

        self.s_fps, self.s_cam, self.s_conf, self.s_lat, self.s_mem = (QLabel() for _ in range(5))
        for w in (self.s_fps, self.s_cam, self.s_conf, self.s_lat, self.s_mem):
            self.statusBar().addPermanentWidget(w)
        self._reset_status()
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil.Process()
        except ImportError:
            pass

    # ------------------------------------------------------------ pages
    def _page(self, title, sub=""):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.addWidget(lbl(title, "h1"))
        if sub:
            lay.addWidget(lbl(sub))
        return w, lay

    def _dashboard(self):
        w, lay = self._page("Dashboard", "Webcam hand-sign recognition with real-time VFX. Everything runs locally.")
        card = QFrame(objectName="card")
        cl = QVBoxLayout(card)
        self.dash_state = lbl("Camera is stopped.", "value")
        cl.addWidget(self.dash_state)
        cl.addWidget(lbl("Start the camera in Live Studio. Hold each pose steadily for about a third of a second; "
                         "default sequences are listed in Sequence Builder. Use Sandbox mode or the VFX Library to "
                         "test effects without performing a sequence."))
        b = QPushButton("Open Live Studio and start camera", objectName="primary")
        b.clicked.connect(lambda: (self.sidebar.setCurrentRow(1), self._start_camera()))
        cl.addWidget(b)
        lay.addWidget(card)
        lay.addStretch()
        return w

    def _live(self):
        w, lay = self._page("Live Studio")
        bar = QHBoxLayout()
        self.btn_cam = QPushButton("Start camera", objectName="primary")
        self.btn_cam.clicked.connect(self._toggle_camera)
        self.mode_box = QComboBox()
        self.mode_box.addItems(MODES)
        self.mode_box.setCurrentText(self.settings.get("mode", "practice"))
        self.mode_box.currentTextChanged.connect(self._mode_changed)
        self.chk_overlay = QCheckBox("Tracking overlay")
        self.chk_overlay.setChecked(self.settings.get("show_overlay", True))
        self.chk_overlay.toggled.connect(self._overlay_changed)
        self.chk_mirror = QCheckBox("Mirror")
        self.chk_mirror.setChecked(self.settings.get("mirror", True))
        self.chk_mirror.toggled.connect(self._mirror_changed)
        b_shot, b_full = QPushButton("Screenshot"), QPushButton("Fullscreen")
        self.btn_rec = QPushButton("Record")
        self.btn_rec.setCheckable(True)
        b_shot.clicked.connect(lambda: self.worker and self.worker.snapshot())
        b_full.clicked.connect(self._toggle_fullscreen)
        self.btn_rec.toggled.connect(self._rec_toggled)
        for x in (self.btn_cam, QLabel("Mode"), self.mode_box, self.chk_overlay, self.chk_mirror, b_shot, self.btn_rec, b_full):
            bar.addWidget(x)
        bar.addStretch()
        lay.addLayout(bar)
        self.preview = QLabel("Camera stopped", objectName="preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(480, 270)
        from PySide6.QtWidgets import QSizePolicy
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lay.addWidget(self.preview, 1)
        return w

    def _jutsu_controls(self):
        w, lay = self._page("Jutsu Controls", "Simple hand transitions activate the primary effects. Classic keeps the sequence workflow.")
        self.control_mode_box = QComboBox()
        self.control_mode_box.addItem("Simple Mode", "simple")
        self.control_mode_box.addItem("Classic Mode", "classic")
        self.control_mode_box.setCurrentIndex(max(0, self.control_mode_box.findData(self.settings.get("control_mode", "simple"))))
        self.control_mode_box.currentIndexChanged.connect(self._jutsu_settings_changed)
        row = QHBoxLayout()
        row.addWidget(QLabel("Control mode"), 1)
        row.addWidget(self.control_mode_box)
        lay.addLayout(row)

        self.jutsu_status = lbl("Camera stopped. Right hand -> Rasengan; left hand -> Chidori; face proximity -> Fireball.", "value")
        lay.addWidget(self.jutsu_status)
        self.power_checks = {}
        for key, label, effect in (("rasengan_enabled", "Right hand -> Rasengan", "rasengan"),
                       ("chidori_enabled", "Left hand -> Chidori", "chidori"),
                                   ("fireball_enabled", "Face proximity -> Fireball", "fireball")):
            row = QHBoxLayout()
            check = QCheckBox(label)
            check.setChecked(bool(self.settings.get(key, True)))
            check.toggled.connect(self._jutsu_settings_changed)
            test = QPushButton("Test")
            test.setToolTip(f"Trigger {label.split(' -> ')[-1]} manually")
            test.clicked.connect(lambda _=False, name=effect: self._trigger(name))
            row.addWidget(check, 1)
            row.addWidget(test)
            lay.addLayout(row)
            self.power_checks[key] = check

        self.face_enabled_check = QCheckBox("Enable face-landmark proximity")
        self.face_enabled_check.setChecked(bool(self.settings.get("face_proximity_enabled", True)))
        self.face_enabled_check.toggled.connect(self._jutsu_settings_changed)
        lay.addWidget(self.face_enabled_check)
        self.rasengan_motion_check = QCheckBox("Launch Rasengan when the right hand moves toward camera")
        self.rasengan_motion_check.setChecked(bool(self.settings.get("rasengan_motion_launch", False)))
        self.rasengan_motion_check.toggled.connect(self._jutsu_settings_changed)
        lay.addWidget(self.rasengan_motion_check)
        self.chidori_motion_check = QCheckBox("Discharge Chidori when the left hand moves toward camera")
        self.chidori_motion_check.setChecked(bool(self.settings.get("chidori_motion_discharge", False)))
        self.chidori_motion_check.toggled.connect(self._jutsu_settings_changed)
        lay.addWidget(self.chidori_motion_check)
        self.fireball_proximity_check = QCheckBox("Launch Fireball immediately on face proximity")
        self.fireball_proximity_check.setChecked(bool(self.settings.get("fireball_proximity_only", False)))
        self.fireball_proximity_check.toggled.connect(self._jutsu_settings_changed)
        lay.addWidget(self.fireball_proximity_check)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Mouth radius (face widths)"), 1)
        self.face_threshold = QDoubleSpinBox()
        self.face_threshold.setRange(0.25, 1.5)
        self.face_threshold.setSingleStep(0.05)
        self.face_threshold.setValue(float(self.settings.get("face_proximity_threshold", 0.68)))
        self.face_threshold.valueChanged.connect(self._jutsu_settings_changed)
        threshold_row.addWidget(self.face_threshold)
        lay.addLayout(threshold_row)
        cooldown_row = QHBoxLayout()
        cooldown_row.addWidget(QLabel("Activation cooldown (seconds)"), 1)
        self.jutsu_cooldown = QDoubleSpinBox()
        self.jutsu_cooldown.setRange(0.2, 3.0)
        self.jutsu_cooldown.setSingleStep(0.1)
        self.jutsu_cooldown.setValue(float(self.settings.get("jutsu_cooldown", 0.65)))
        self.jutsu_cooldown.valueChanged.connect(self._jutsu_settings_changed)
        cooldown_row.addWidget(self.jutsu_cooldown)
        lay.addLayout(cooldown_row)
        row = QHBoxLayout()
        reset = QPushButton("Reset tracking states")
        reset.clicked.connect(self._reset_jutsu_tracking)
        row.addWidget(reset)
        row.addStretch()
        lay.addLayout(row)
        lay.addWidget(lbl("Gestures: close then open the right hand for Rasengan; close then open the left hand for Chidori. "
                          "For Fireball, move either hand to the detected mouth region, close it, then open to launch."))
        lay.addStretch()
        return w

    def _lab(self):
        w, lay = self._page("Gesture Lab", "Live per-hand classification. Built-in poses: " + ", ".join(BUILTIN))
        self.lab_text = lbl("Start the camera to see live results.", "value")
        lay.addWidget(self.lab_text)
        lay.addStretch()
        return w

    def _library(self):
        w, lay = self._page("VFX Library", "Trigger any effect manually (works in every mode while the camera is running).")
        for n in EFFECT_NAMES:
            r = QHBoxLayout()
            r.addWidget(QLabel(_EFFECTS[n].label), 1)
            b = QPushButton("Trigger")
            b.clicked.connect(lambda _=False, name=n: self._trigger(name))
            r.addWidget(b)
            lay.addLayout(r)
        lay.addStretch()
        return w

    def _seq_path(self):
        return CONFIG_DIR / "sequences.json"

    def _seq_builder(self):
        w, lay = self._page("Sequence Builder", "Steps are pose names separated by commas.")
        self.seq_list = QListWidget(objectName="plain")
        self.seq_list.itemClicked.connect(self._seq_select)
        lay.addWidget(self.seq_list, 1)
        self.seq_name = QLineEdit(placeholderText="Sequence name")
        self.seq_steps = QLineEdit(placeholderText="e.g. fist, pointing, open_palm")
        self.seq_effect = QComboBox()
        self.seq_effect.addItems(EFFECT_NAMES)
        self.seq_hold, self.seq_timeout, self.seq_cd = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for sp, v in ((self.seq_hold, 0.35), (self.seq_timeout, 4.0), (self.seq_cd, 3.0)):
            sp.setRange(0.05, 30)
            sp.setSingleStep(0.05)
            sp.setValue(v)
        for x in (self.seq_name, self.seq_steps, self.seq_effect):
            lay.addWidget(x)
        r = QHBoxLayout()
        for t, sp in (("Hold s", self.seq_hold), ("Timeout s", self.seq_timeout), ("Cooldown s", self.seq_cd)):
            r.addWidget(QLabel(t))
            r.addWidget(sp)
        lay.addLayout(r)
        r2 = QHBoxLayout()
        b1, b2 = QPushButton("Save / update", objectName="primary"), QPushButton("Delete selected", objectName="danger")
        b1.clicked.connect(self._seq_save)
        b2.clicked.connect(self._seq_delete)
        r2.addWidget(b1)
        r2.addWidget(b2)
        lay.addLayout(r2)
        self.seq_msg = lbl("")
        lay.addWidget(self.seq_msg)
        self._seq_refresh()
        return w

    def _seq_defs(self):
        return (load_json(self._seq_path(), {"sequences": []}) or {}).get("sequences", [])

    def _seq_refresh(self):
        self.seq_list.clear()
        for d in self._seq_defs():
            self.seq_list.addItem(f"{d['name']}  ->  {' > '.join(d['steps'])}  [{d.get('effect')}]")

    def _seq_select(self, item):
        d = self._seq_defs()[self.seq_list.row(item)]
        self.seq_name.setText(d["name"])
        self.seq_steps.setText(", ".join(d["steps"]))
        self.seq_effect.setCurrentText(d.get("effect", "lightning"))
        self.seq_hold.setValue(d.get("hold_time", 0.35))
        self.seq_timeout.setValue(d.get("timeout", 4.0))
        self.seq_cd.setValue(d.get("cooldown", 3.0))

    def _seq_save(self):
        steps = [s.strip() for s in self.seq_steps.text().split(",") if s.strip()]
        valid = set(BUILTIN) | set(load_custom(CONFIG_DIR / "gestures.json"))
        bad = [s for s in steps if s not in valid]
        if not self.seq_name.text().strip() or not steps or bad:
            self.seq_msg.setText("Invalid: need a name, at least one step" + (f"; unknown poses: {bad}" if bad else ""))
            self.seq_msg.setObjectName("warn")
            return
        d = SequenceDef(self.seq_name.text().strip(), steps, self.seq_effect.currentText(),
                        self.seq_hold.value(), self.seq_timeout.value(), self.seq_cd.value()).to_dict()
        defs = [x for x in self._seq_defs() if x["name"] != d["name"]] + [d]
        self._seq_commit(defs, f"Saved '{d['name']}'.")

    def _seq_delete(self):
        row = self.seq_list.currentRow()
        defs = self._seq_defs()
        if row >= 0:
            name = defs.pop(row)["name"]
            self._seq_commit(defs, f"Deleted '{name}'.")

    def _seq_commit(self, defs, msg):
        save_json(self._seq_path(), {"sequences": defs})
        self._seq_refresh()
        self.seq_msg.setObjectName("")
        self.seq_msg.setText(msg)
        if self.worker:
            self.worker.post(lambda p: p.sequences.reload(defs))

    def _calibration(self):
        w, lay = self._page("Calibration", "Capture your own poses (hold the pose with one hand, capture 10-20 samples while "
                                           "varying distance and angle slightly, then save). Camera must be running.")
        self.cal_name = QLineEdit(placeholderText="Pose name, e.g. tiger_seal")
        lay.addWidget(self.cal_name)
        r = QHBoxLayout()
        b1, b2, b3 = QPushButton("Start (Calibration mode)"), QPushButton("Capture sample"), QPushButton("Save pose", objectName="primary")
        b1.clicked.connect(self._cal_start)
        b2.clicked.connect(lambda: self.worker and self.worker.post(lambda p: p.calibration.add_sample(p.last_features)))
        b3.clicked.connect(self._cal_save)
        for b in (b1, b2, b3):
            r.addWidget(b)
        lay.addLayout(r)
        self.cal_info = lbl("Samples: 0")
        lay.addWidget(self.cal_info)
        lay.addWidget(lbl("Saved custom poses:", "h2"))
        self.cal_list = QListWidget(objectName="plain")
        lay.addWidget(self.cal_list, 1)
        b4 = QPushButton("Delete selected pose", objectName="danger")
        b4.clicked.connect(self._cal_delete)
        lay.addWidget(b4)
        self._cal_refresh()
        return w

    def _cal_refresh(self):
        self.cal_list.clear()
        self.cal_list.addItems(sorted(load_custom(CONFIG_DIR / "gestures.json")))

    def _cal_start(self):
        name = self.cal_name.text().strip()
        if not self.worker or not name:
            QMessageBox.information(self, "Calibration", "Start the camera and enter a pose name first.")
            return
        self.mode_box.setCurrentText("calibration")
        self.worker.post(lambda p: p.calibration.start(name))

    def _cal_save(self):
        if not self.worker:
            return

        def save(p):
            ok = p.calibration.save()
            if ok:
                p.classifier.set_custom(load_custom(CONFIG_DIR / "gestures.json"))
            self.worker.message.emit("Pose saved." if ok else "Need at least 5 samples and a name.")
        self.worker.post(save)
        self._cal_refresh_later = True

    def _cal_delete(self):
        it = self.cal_list.currentItem()
        if it:
            delete_custom(CONFIG_DIR / "gestures.json", it.text())
            self._cal_refresh()
            if self.worker:
                self.worker.post(lambda p: p.classifier.set_custom(load_custom(CONFIG_DIR / "gestures.json")))

    def _settings_page(self):
        w, lay = self._page("Settings", "Camera and resolution changes apply the next time the camera starts.")
        self.set_cam = QSpinBox()
        self.set_cam.setValue(self.settings["camera_index"])
        self.set_res = QComboBox()
        self.set_res.addItems(["640x480", "1280x720", "1920x1080"])
        self.set_res.setCurrentText(f"{self.settings['camera_width']}x{self.settings['camera_height']}")
        self.set_proc = QComboBox()
        self.set_proc.addItems(["480", "640", "800"])
        self.set_proc.setCurrentText(str(self.settings["proc_width"]))
        self.set_shake = QCheckBox("Screen shake")
        self.set_shake.setChecked(self.settings["screen_shake"])
        self.set_grade = QCheckBox("Cinematic color grade")
        self.set_grade.setChecked(self.settings["color_grade"])
        self.set_shake.toggled.connect(self._push_vfx)
        self.set_grade.toggled.connect(self._push_vfx)
        for t, x in (("Camera index", self.set_cam), ("Camera resolution", self.set_res), ("Processing width", self.set_proc)):
            r = QHBoxLayout()
            r.addWidget(QLabel(t), 1)
            r.addWidget(x)
            lay.addLayout(r)
        lay.addWidget(self.set_shake)
        lay.addWidget(self.set_grade)
        b = QPushButton("Save settings", objectName="primary")
        b.clicked.connect(self._save_settings)
        lay.addWidget(b)
        lay.addStretch()
        return w

    def _about(self):
        w, lay = self._page("About", "SHINOBI VISION X - real-time hand-sign recognition and procedural VFX.")
        lay.addWidget(lbl("Hand tracking: MediaPipe (Google, Apache-2.0). Rendering: OpenCV + NumPy. UI: PySide6. "
                          "All effects are drawn procedurally; no third-party artwork. Recognition depends on lighting, "
                          "camera quality and hand visibility. Shadow Clones is an approximate frame-echo effect "
                          "(no person segmentation); Cursed Aura is anchored approximately (no face/eye tracking)."))
        lay.addStretch()
        return w

    # ------------------------------------------------------------ right panel
    def _right_panel(self):
        p = QFrame(objectName="panel")
        p.setFixedWidth(310)
        lay = QVBoxLayout(p)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(lbl("TRACKING", "h2"))
        self.v_hands, self.v_gesture, self.v_seq, self.v_state = (lbl("-", "value") for _ in range(4))
        for t, v in (("Hands detected", self.v_hands), ("Current gesture", self.v_gesture),
                     ("Active sequence", self.v_seq), ("State", self.v_state)):
            lay.addWidget(QLabel(t))
            lay.addWidget(v)
        self.prog = QProgressBar()
        self.prog.setFormat("%v / %m")
        lay.addWidget(self.prog)
        lay.addWidget(lbl("EFFECT", "h2"))
        self.eff_box = QComboBox()
        for n in EFFECT_NAMES:
            self.eff_box.addItem(_EFFECTS[n].label, n)
        i = self.eff_box.findData(self.settings.get("selected_effect", "lightning"))
        self.eff_box.setCurrentIndex(max(0, i))
        self.eff_box.currentIndexChanged.connect(lambda _: self.settings.__setitem__("selected_effect", self.eff_box.currentData()))
        lay.addWidget(self.eff_box)
        b = QPushButton("Trigger selected effect", objectName="primary")
        b.clicked.connect(lambda: self._trigger(self.eff_box.currentData()))
        lay.addWidget(b)
        self.sliders = {}
        for key, name, lo, hi in (("intensity", "Effect intensity", 0, 200), ("particle_density", "Particle density", 10, 200),
                                  ("glow", "Glow strength", 0, 150), ("trail", "Motion-trail intensity", 0, 90)):
            lay.addWidget(QLabel(name))
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(int(self.settings[key] * 100))
            s.valueChanged.connect(self._push_vfx)
            self.sliders[key] = s
            lay.addWidget(s)
        lay.addStretch()
        quit_b = QPushButton("Safe shutdown", objectName="danger")
        quit_b.clicked.connect(self.close)
        lay.addWidget(quit_b)
        return p

    # ------------------------------------------------------------ actions
    def _push_vfx(self, *_):
        for k, s in self.sliders.items():
            self.settings[k] = s.value() / 100.0
        self.settings["screen_shake"] = self.set_shake.isChecked()
        self.settings["color_grade"] = self.set_grade.isChecked()
        d = {k: self.settings[k] for k in ("intensity", "particle_density", "glow", "trail", "screen_shake", "color_grade")}

        def apply(p):
            for k, v in d.items():
                setattr(p.renderer.settings, k, v)
        if self.worker:
            self.worker.post(apply)

    def _trigger(self, name):
        if not self.worker:
            QMessageBox.information(self, "VFX", "Start the camera first.")
            return
        self.worker.post(lambda p: p.trigger(name))

    def _jutsu_settings_changed(self, *_):
        if not hasattr(self, "control_mode_box"):
            return
        self.settings["control_mode"] = self.control_mode_box.currentData()
        for key, check in self.power_checks.items():
            self.settings[key] = check.isChecked()
        self.settings["face_proximity_enabled"] = self.face_enabled_check.isChecked()
        self.settings["fireball_proximity_only"] = self.fireball_proximity_check.isChecked()
        self.settings["rasengan_motion_launch"] = self.rasengan_motion_check.isChecked()
        self.settings["chidori_motion_discharge"] = self.chidori_motion_check.isChecked()
        self.settings["face_proximity_threshold"] = self.face_threshold.value()
        mode = self.settings["control_mode"]
        powers = {
            "Right": self.settings["rasengan_enabled"],
            "Left": self.settings["chidori_enabled"],
            "Fireball": self.settings["fireball_enabled"],
        }
        face_enabled = self.settings["face_proximity_enabled"]
        cooldown = self.jutsu_cooldown.value()
        proximity_only = self.settings["fireball_proximity_only"]
        rasengan_motion = self.settings["rasengan_motion_launch"]
        chidori_motion = self.settings["chidori_motion_discharge"]
        face_threshold = self.settings["face_proximity_threshold"]
        if self.worker:
            def apply(p):
                p.set_control_mode(mode)
                for name, enabled in powers.items():
                    p.set_power_enabled(name, enabled)
                p.set_face_enabled(face_enabled)
                p.simple_controls.cooldown = cooldown
                p.fireball_proximity_only = proximity_only
                p.rasengan_motion_launch = rasengan_motion
                p.chidori_motion_discharge = chidori_motion
                p.face_proximity.threshold = face_threshold
            self.worker.post(apply)

    def _reset_jutsu_tracking(self):
        if self.worker:
            self.worker.post(lambda p: p.reset_tracking_states())
        self.jutsu_status.setText("Tracking state reset.")

    def _mode_changed(self, m):
        self.settings["mode"] = m
        if self.worker:
            self.worker.post(lambda p: p.set_mode(m))

    def _overlay_changed(self, v):
        self.settings["show_overlay"] = v
        if self.worker:
            self.worker.post(lambda p: setattr(p, "show_overlay", v))

    def _mirror_changed(self, v):
        self.settings["mirror"] = v
        if self.worker:
            self.worker.post(lambda p: setattr(p, "mirror", v))

    def _rec_toggled(self, v):
        if self.worker:
            self.worker.recording = v
        elif v:
            self.btn_rec.setChecked(False)

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _save_settings(self):
        self.settings["camera_index"] = self.set_cam.value()
        w, h = self.set_res.currentText().split("x")
        self.settings["camera_width"], self.settings["camera_height"] = int(w), int(h)
        self.settings["proc_width"] = int(self.set_proc.currentText())
        self._push_vfx()
        self.settings.save()
        self.statusBar().showMessage("Settings saved.", 4000)

    def _toggle_camera(self):
        self._stop_camera() if self.worker else self._start_camera()

    def _start_camera(self):
        if self.worker:
            return
        self._save_settings_quiet()
        self.worker = VisionWorker(self.settings)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status.connect(self._on_status)
        self.worker.failed.connect(self._on_failed)
        self.worker.message.connect(lambda m: self.statusBar().showMessage(m, 5000))
        self.worker.start()
        self.btn_cam.setText("Stop camera")
        self.s_cam.setText("Camera: starting")
        self.dash_state.setText("Camera running.")

    def _save_settings_quiet(self):
        self.settings["camera_index"] = self.set_cam.value()
        w, h = self.set_res.currentText().split("x")
        self.settings["camera_width"], self.settings["camera_height"] = int(w), int(h)
        self.settings["proc_width"] = int(self.set_proc.currentText())

    def _stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(4000)
            self.worker = None
        self.btn_rec.setChecked(False)
        self.btn_cam.setText("Start camera")
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Camera stopped")
        self.dash_state.setText("Camera is stopped.")
        self._reset_status()

    def _reset_status(self):
        self.s_fps.setText("FPS: -  ")
        self.s_cam.setText("Camera: off  ")
        self.s_conf.setText("Confidence: -  ")
        self.s_lat.setText("Latency: -  ")
        self.s_mem.setText("Memory: n/a  ")

    def _on_failed(self, msg):
        self._stop_camera()
        self.s_cam.setText("Camera: error  ")
        QMessageBox.warning(self, "SHINOBI VISION X", msg)

    def _on_frame(self, img):
        pm = QPixmap.fromImage(img)
        self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def _on_status(self, st):
        sq = st["sequence"]
        self.v_hands.setText(str(st["hands"]))
        self.v_gesture.setText(st["gesture"])
        self.v_seq.setText(sq["name"])
        self.prog.setMaximum(max(1, sq["total"]))
        self.prog.setValue(sq["done"])
        state = "cooldown %.1fs" % sq["cooldown"] if sq["cooldown"] > 0 else ("effect: " + ", ".join(st["active_effects"]) if st["active_effects"] else "ready")
        self.v_state.setText(state)
        self.s_fps.setText("FPS: %.0f  " % st["fps"])
        self.s_cam.setText("Camera: live  ")
        self.s_conf.setText("Confidence: %.2f  " % st["confidence"])
        self.s_lat.setText("Latency: %.0f ms (infer %.0f)  " % (st["latency_ms"], st["infer_ms"]))
        if self._psutil:
            self.s_mem.setText("Memory: %.0f MB  " % (self._psutil.memory_info().rss / 1e6))
        self.lab_text.setText("\n".join(f"{lab}: {n} ({c:.2f})" for lab, n, c in st["per_hand"]) or "No hands detected")
        if hasattr(self, "jutsu_status"):
            hs = st.get("hand_status", {})
            right = hs.get("Right", {})
            left = hs.get("Left", {})
            face = st.get("face_status", {})
            if not face.get("available"):
                face_text = "unavailable (model/setup)"
            elif not face.get("tracking"):
                face_text = "paused"
            elif any(face.get("near", {}).values()):
                face_text = "near mouth"
            else:
                face_text = "face found" if face.get("face_found") else "searching"
            self.jutsu_status.setText(
                f"Right: {'tracking' if right.get('present') else 'not seen'} / {right.get('pose', 'none')} "
                f"({right.get('cooldown_remaining', 0):.1f}s cooldown)\n"
                f"Left: {'tracking' if left.get('present') else 'not seen'} / {left.get('pose', 'none')} "
                f"({left.get('cooldown_remaining', 0):.1f}s cooldown)\n"
                f"Face proximity: {face_text} | Control mode: {st.get('control_mode', 'simple')}\n"
                f"Events: {', '.join(st.get('events', [])) or 'none'}")
        self.cal_info.setText(f"Samples: {st['calib_samples']}")
        if "Pose saved" in self.statusBar().currentMessage():
            self._cal_refresh()

    def closeEvent(self, e):
        self._stop_camera()
        self._save_settings_quiet()
        self._push_vfx()
        try:
            self.settings.save()
        except OSError:
            pass
        e.accept()
