QSS = """
* { font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; color: #d7deea; }
QMainWindow, QWidget#root { background: #14161a; }
QListWidget#sidebar { background: #0f1115; border: none; padding: 8px 0; outline: 0; min-width: 170px; max-width: 170px; }
QListWidget#sidebar::item { padding: 11px 18px; margin: 2px 8px; border-radius: 8px; color: #8d99ad; }
QListWidget#sidebar::item:selected { background: #17253d; color: #5ab0ff; border-left: 3px solid #2f9bff; }
QListWidget#sidebar::item:hover:!selected { background: #181c24; }
QLabel#brand { font-size: 15px; font-weight: 700; letter-spacing: 2px; color: #2f9bff; padding: 16px 18px 6px 18px; background:#0f1115; }
QFrame#panel { background: #101a2b; border-left: 1px solid #1d2b45; }
QFrame#card { background: #121d31; border: 1px solid #1d2b45; border-radius: 10px; }
QLabel#h1 { font-size: 20px; font-weight: 700; color: #ffffff; }
QLabel#h2 { font-size: 11px; font-weight: 700; letter-spacing: 1.5px; color: #6f86ad; margin-top: 8px; }
QLabel#value { font-size: 15px; font-weight: 600; color: #5ab0ff; }
QLabel#preview { background: #05070a; border: 1px solid #1d2b45; border-radius: 8px; color: #50607d; }
QLabel#warn { color: #ff6b6b; }
QPushButton { background: #1a2a47; border: 1px solid #27406b; border-radius: 7px; padding: 7px 14px; color: #dbe7ff; }
QPushButton:hover { background: #223a63; border-color: #2f9bff; }
QPushButton:pressed { background: #2f9bff; color: #08101c; }
QPushButton#primary { background: #2f9bff; color: #06101d; font-weight: 700; border: none; }
QPushButton#primary:hover { background: #55b0ff; }
QPushButton#danger { background: #3b1a28; border-color: #8b2f4f; color: #ff9fb6; }
QPushButton:checked { background: #3a2160; border-color: #8a5cff; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: #0c1424; border: 1px solid #27406b; border-radius: 6px; padding: 5px 8px; }
QComboBox QAbstractItemView { background: #0c1424; selection-background-color: #1e3a66; }
QSlider::groove:horizontal { height: 4px; background: #27406b; border-radius: 2px; }
QSlider::handle:horizontal { background: #2f9bff; width: 14px; margin: -6px 0; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #2f9bff; border-radius: 2px; }
QProgressBar { background: #0c1424; border: 1px solid #27406b; border-radius: 6px; text-align: center; height: 16px; }
QProgressBar::chunk { background: #8a5cff; border-radius: 5px; }
QStatusBar { background: #0f1115; color: #8d99ad; border-top: 1px solid #1d2b45; }
QListWidget#plain { background: #0c1424; border: 1px solid #27406b; border-radius: 6px; }
QCheckBox { spacing: 8px; }
"""
