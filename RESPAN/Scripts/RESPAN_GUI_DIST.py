# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:16:58 2023

"""
__title__     = 'RESPAN'
__version__   = '1.5.01'
__date__      = "20 September, 2026"
__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2025 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'



respan_paper =  ("\nPlease cite the following paper when using RESPAN:"
                    "\nA deep learning pipeline for accurate and automated restoration, segmentation, and quantification of dendritic spines"
                    "\nSergio Bernal-Garcia, Alexa P. Schlotter, Daniela Pereira, Franck Polleux, Luke A. Hammond"
                    "\ndoi: https://doi.org/10.1101/2024.06.06.597812\n")

self_net_paper =  ("\nPlease cite the following paper when using Self-Net:"
                    "\nDeep self-learning enables fast, high-fidelity isotropic resolution restoration for volumetric fluorescence microscopy"
                    "\nKefu Ning, Bolin Lu, Xiaojun Wang, Xiaoyu Zhang, Shuo Nie, Tao Jiang, Anan Li, Guoqing Fan, Xiaofeng Wang, "
                    "\nQingming Luo, Hui Gong & Jing Yuan "
                    "\nLight: Science & Applications volume 12, Article number: 204 (2023)\n")

care_paper = ("\nPlease cite the following paper when using CARE:"
                    "\nContent-Aware Image Restoration: Pushing the Limits of Fluorescence Microscopy. "
                    "\nMartin Weigert, Uwe Schmidt, Tobias Boothe, Andreas Müller, Alexandr Dibrov, Akanksha Jain, "
                    "\nBenjamin Wilhelm, Deborah Schmidt, Coleman Broaddus, Siân Culley, Mauricio Rocha-Martins, "
                    "\nFabián Segovia-Miranda, Caren Norden, Ricardo Henriques, Marino Zerial, Michele Solimena, "
                    "\nJochen Rink, Pavel Tomancak, Loic Royer, Florian Jug, and Eugene W. Myers. "
                    "\nNature Methods 15.12 (2018): 1090–1097.\n")

elastix_paper =  ("\nPlease cite the following paper when using spine tracking:"
                    "\nelastix: a toolbox for intensity based medical image registration"
                    "\nS. Klein, M. Staring, K. Murphy, M.A. Viergever, J.P.W. Pluim"
                    "\nIEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, January 2010\n")

nnUNet_paper = ("Please cite the following paper when using nnU-Net:"
                    "\nIsensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring"
                    "\nnetwork for medical image segmentation. Nature methods, 18(2), 203-211.\n")

about_text = ("\nWe developed RESPAN to address a need for a comprehensive, accurate, automated analysis pipeline that utilizes state-of-the-art tools in a single user-friendly package."
                       "\nTypically, employing multiple tools in pipelines like this can be complex, making it difficult to remember all the necessary steps and parameters for optimal results. "
                       "Managing different environments, tracking inputs and outputs from various software tools, and navigating different GUIs or command line interfaces often requires extensive "
                       "troubleshooting, consuming days to weeks of time."
                        "\n\nIn addition to the main analysis pipeline, RESPAN includes intuitive GUI's to streamline the training and validation of a variety of state-of-the-art neural network based appraoches. These include tabs for training both tensorflow and pytorch models for restoration and segmentation, as well as a GUI for validating segmentation results. "
                        "We hope that through this effort, RESPAN will not only facilitate neuron and spine quantification, but broadly empower you to consider these approaches when facing other challenging research questions."
                       "\n "
                       "\nIf RESPAN contributes to your research, please acknowledge it by citing:\n\n"
                       "RESPAN: an accurate, unbiased and automated pipeline for analysis of dendritic morphology and dendritic spine mapping\n"
                       "Authors: Sergio B. Garcia, Alexa P. Schlotter, Daniela Pereira, Franck Polleux, Luke A. Hammond\n"
                       "DOI: https://doi.org/10.1101/2024.06.06.597812"
                       "\n\n")



import multiprocessing
import sys

# Force UTF-8 console output. Prevents Windows cp1252 logging crashes
# when log messages contain non-ASCII chars (mu, arrows, etc.). No-op
# when stdout is None (bundled --windowed exe) or already UTF-8.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

import os
import pickle
import logging
import shutil
from datetime import datetime
import subprocess
import argparse
from types import SimpleNamespace
import json
import threading
import re
import pynvml
import yaml
import importlib
import ctypes

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QTabWidget,
                             QPushButton, QCheckBox, QLabel, QLineEdit, QComboBox,
                             QMessageBox, QTextEdit, QWidget, QFileDialog,
                             QHBoxLayout, QGroupBox, QProgressBar, QSplashScreen,QFrame, QStyleFactory)
from PyQt5.QtCore import Qt, pyqtSlot, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QTextCursor, QPixmap, QPainter, QColor, QFont, QPalette, QIcon

from pathlib import Path




####### GLOBAL PARAMS
FROZEN = getattr(sys, "frozen", False)
DEV_ENV  = "respan99"                      # development environmet

additional_logging_dev = False
#Base Dir
APP_DIR  = Path(sys.executable).parent if FROZEN \
           else Path(__file__).resolve().parent            # …\Scripts

if FROZEN:
    VENV_DIR = APP_DIR / "_internal" / "respan"
    PY_EXE   = VENV_DIR / "python.exe"
    SRC_DIR = APP_DIR / "_internal" / "nnUNet_install"
    INSTALL_DIR = Path(sys.executable).parent
    INTERNAL_DIR = APP_DIR / "_internal"
    SELFNET_TRAINING_SCRIPT = INTERNAL_DIR / "SelfNet_Model_Training.py"
    SELFNET_INFERENCE_SCRIPT = INTERNAL_DIR / "SelfNet_Inference.py"
    clean_launcher = INTERNAL_DIR / "clean_launcher.py"

    global_GUI_app = True
    global_respan_env = 'respan'
    global_env_path = str(INTERNAL_DIR)  # Set this immediately
    global_conda_path = str(APP_DIR)

else:

    active_env = Path(sys.base_prefix)           # respandev
    conda_root = active_env.parent.parent        # …\anaconda3

    # Search for DEV_ENV — may be under a different conda root than active env
    _candidate_roots = [
        conda_root / "envs",                              # sibling of current env
        Path.home() / "anaconda3" / "envs",                # default anaconda
        Path.home() / "miniconda3" / "envs",               # default miniconda
    ]
    VENV_DIR = None
    for _root in _candidate_roots:
        _candidate = _root / DEV_ENV
        if (_candidate / "python.exe").exists():
            VENV_DIR = _candidate
            break
    if VENV_DIR is None:
        raise RuntimeError(
            f"Dev env '{DEV_ENV}' not found. Searched: "
            + ", ".join(str(r / DEV_ENV) for r in _candidate_roots))
    PY_EXE = VENV_DIR / "python.exe"
    INSTALL_DIR = Path.home()
    SELFNET_TRAINING_SCRIPT = APP_DIR / "SelfNet_Model_Training.py"
    SELFNET_INFERENCE_SCRIPT = APP_DIR / "SelfNet_Inference.py"
    clean_launcher = APP_DIR / "clean_launcher.py"
    global_GUI_app = False
    global_respan_env = 'respan99'
    global_env_path = str(VENV_DIR.parent)
    global_conda_path = str(VENV_DIR.parent.parent)

nnunet_predict_bat = VENV_DIR / "Scripts" / "nnUNetv2_predict.bat"
nnunet_plan_bat = VENV_DIR / "Scripts" / "nnUNetv2_plan_and_preprocess.bat"
nnunet_train_bat = VENV_DIR / "Scripts" / "nnUNetv2_train.bat"

_SENTINEL = APP_DIR / ".nnunet_ok"


def _settings_dirs():
    """Folders the per-tab GUI settings pickles are saved to, in priority order.

    Next to the exe (frozen) / this script (source) first, so dropping a
    settings pickle beside the executable pre-fills the GUI; then the user's
    local app-data folder, which stays writable when the app is installed
    under Program Files.
    """
    dirs = [str(APP_DIR)]
    local = os.environ.get('LOCALAPPDATA')
    if local:
        dirs.append(os.path.join(local, 'RESPAN'))
    return dirs


def _load_settings(name):
    """Return the most recently saved settings dict for pickle `name`, or None.

    Candidates are the save folders only (next to the exe, then app-data).
    The current working directory, where releases before v1.5 wrote the
    pickle, is deliberately NOT consulted: unpickling a file from whatever
    folder the app happens to be launched from would execute that file's
    contents. The newest readable file wins, so a save that had to fall back
    to app-data is what the next start restores; an unreadable or corrupt
    file is skipped, never fatal. These pickles are RESPAN's own GUI-state
    files written by this application into its own folders (pre-existing
    on-disk format, kept for compatibility).
    """
    candidates = []
    for base in _settings_dirs():
        path = os.path.join(base, name)
        try:
            if os.path.isfile(path):
                candidates.append((os.path.getmtime(path), path))
        except OSError:
            continue
    for _, path in sorted(candidates, reverse=True):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            continue
    return None


def _save_settings(name, variables_dict, log_display=None):
    """Persist a tab's settings dict as pickle `name`. Never raises.

    The settings pickle used to be written with a bare relative filename, i.e.
    into the current working directory. Launching the packaged app from a
    shell whose cwd is not writable made the Run button's save raise
    PermissionError inside a Qt slot, which PyQt turns into a silent abort of
    the whole application before the analysis even starts. A failed save is
    now a warning line, never a crash.
    """
    for base in _settings_dirs():
        try:
            os.makedirs(base, exist_ok=True)
            path = os.path.join(base, name)
            with open(path, 'wb') as f:
                pickle.dump(variables_dict, f)
            return path
        except OSError:
            continue
    warning = f"  Warning: could not save GUI settings ({name}); continuing without saving them."
    if sys.__stderr__ is not None:
        try:
            sys.__stderr__.write(warning + '\n')
        except Exception:
            pass
    if log_display is not None:
        # Deferred: every Run button calls get_variables() and then
        # run_function(), whose first act is log_display.clear(). A zero-delay
        # timer appends after that, so the warning is actually seen.
        try:
            QTimer.singleShot(0, lambda: log_display.append(warning))
        except Exception:
            pass
    return None


def _install_crash_reporter():
    """Route uncaught exceptions to a log file and a dialog instead of a silent abort.

    PyQt5 calls qFatal() when a Python exception escapes a slot or a QThread.run
    and sys.excepthook is still the default, which kills the windowed
    application with exception code 0xc0000409 in Qt5Core.dll and no message
    anywhere (the user sees the window vanish). With a custom hook installed
    PyQt calls the hook instead and the application keeps running.
    """
    import traceback

    def hook(exc_type, exc, tb):
        text = ''.join(traceback.format_exception(exc_type, exc, tb))
        now = datetime.now()
        entry = (f"\n===== {now.strftime('%Y-%m-%d %H:%M:%S')}  RESPAN {__version__}  "
                 f"unhandled exception =====\n{text}")
        path = None
        for base in _settings_dirs():
            try:
                os.makedirs(base, exist_ok=True)
                candidate = os.path.join(base, 'RESPAN_crash_' + now.strftime('%Y-%m-%d') + '.log')
                # Append: several reports in one second, or from several
                # instances, must never overwrite each other.
                with open(candidate, 'a', encoding='utf-8') as f:
                    f.write(entry)
                path = candidate
                break
            except OSError:
                continue
        stream = sys.__stderr__
        if stream is not None:
            try:
                stream.write(text)
            except Exception:
                pass
        try:
            from PyQt5.QtCore import QThread
            from PyQt5.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app is not None and QThread.currentThread() is app.thread():
                QMessageBox.critical(
                    None, "RESPAN error",
                    "An unexpected error occurred. RESPAN will keep running, but the current action did not complete.\n\n"
                    + (f"Details appended to:\n{path}\n\n" if path else "")
                    + text[-1800:])
        except Exception:
            pass

    sys.excepthook = hook

NNUNET_INSTALL = "nnUNet_install"                # name of the nnUNet install dir

if getattr(sys, "frozen", False):
    # …/RESPAN_v0_9_97/_internal/nnUNet_install
    NNUNET_SRC = (Path(sys.executable).parent / "_internal" / NNUNET_INSTALL).resolve()
else:
    # running from source#
    NNUNET_SRC = (INSTALL_DIR / NNUNET_INSTALL).resolve()

print(f"NNUNET_SRC: {NNUNET_SRC}")
print(f"VENV_DIR: {VENV_DIR}")
print(f"PY_EXE: {PY_EXE}")
print(f"APP_DIR: {APP_DIR}")
###

####

def add_folders_to_path(base_path, folder_names):
    for folder_name in folder_names:
        folder_path = os.path.abspath(os.path.join(base_path, folder_name))
        if os.path.exists(folder_path):
            os.environ['PATH'] = folder_path + os.pathsep + os.environ['PATH']
        else:
            print(f"Warning: {folder_path} does not exist")


if getattr(sys, 'frozen', False):
    global_GUI_app = True #set to True when built as app
    global_respan_env = 'respan'
    base_path = sys._MEIPASS
    print(f"Running in frozen environment. Base path: {base_path}")

    elastix_path =  os.path.abspath(os.path.join(base_path, "elastix_5_2/"))
    elastix_params =  os.path.abspath(os.path.join(base_path, "Elastix_params/"))

    icon_path = os.path.join(base_path, 'spine_icon_v2.ico')

    #os.environ['LD_LIBRARY_PATH'] = internal_path + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    cuda_path = os.path.abspath(os.path.join(base_path, 'cuda'))
    cuda_bin = os.path.join(cuda_path, 'bin')
    os.environ['CUDA_PATH'] = cuda_path
    # 2026-09-19: the bundled CUDA DLLs live in cuda/bin ONLY since the v1.5
    # slim spec (v1.0 carried a second copy in cuda/). NVRTC loads its
    # nvrtc-builtins64_*.dll through the ordinary DLL search, i.e. PATH, so
    # cuda/bin must be on PATH or every CuPy kernel compile fails with
    # "failed to open nvrtc-builtins64_112.dll" (seen on the first GUI run of
    # the v1.5 build; invisible when the exe inherits a shell PATH that
    # already carries a CUDA bin folder). add_dll_directory covers loads made
    # through Python's own LoadLibraryEx search as well.
    os.environ['PATH'] = cuda_bin + os.pathsep + cuda_path + os.pathsep + os.environ['PATH']
    if hasattr(os, 'add_dll_directory') and os.path.isdir(cuda_bin):
        try:
            os.add_dll_directory(cuda_bin)
        except OSError:
            pass
    print(f"CUDA_PATH: {os.environ['CUDA_PATH']}")
    #print("Current PATH: ", os.environ['PATH'])

    folders_to_add = ['scipy', 'skimage', 'cupy']

    # Add specified folders to PATH
    add_folders_to_path(base_path, folders_to_add)

else:
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(os.path.dirname(base_path), 'Scripts')
    print(f"Running in development environment. Base path: {base_path}")
    global_GUI_app = False
    global_respan_env = 'respan99'

    # CUDA setup — dev machines may not have the CUDA toolkit installed system-wide.
    # Search bundled dist DLLs so TensorFlow (CARE) and CuPy can find CUDA 11.x libs.
    _cuda_search_dirs = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin"),
        Path("D:/RESPAN_v1_0_00/_internal/cuda/bin"),
        Path("D:/RESPAN_v1_0_00/_internal/cuda"),
    ]
    for _d in _cuda_search_dirs:
        if _d.is_dir():
            os.environ.setdefault("CUDA_PATH", str(_d.parent if _d.name == "bin" else _d))
            os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")
            print(f"CUDA_PATH: {os.environ['CUDA_PATH']}")
            break

    elastix_path = "C:/Program Files/elastix_5_2/"
    elastix_params = str((Path(__file__).resolve().parent.parent / "Elastix_params"))



## supress windows with subprocess
startupinfo = subprocess.STARTUPINFO()
startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
startupinfo.wShowWindow = subprocess.SW_HIDE


def log_output(pipe, logger):
    for line in iter(pipe.readline, ''):
        logger.info(line.strip())

def run_process_with_logging(cmd, logger):
    """Run a subprocess with logging. cmd can be a list (recommended) or string."""
    use_shell = isinstance(cmd, str)

    startupinfo = None
    if os.name == 'nt' and not use_shell:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=use_shell, universal_newlines=True, text=True,
                               startupinfo=startupinfo)

    # Start threads to read stdout and stderr
    stdout_thread = threading.Thread(target=log_output, args=(process.stdout, logger))
    stderr_thread = threading.Thread(target=log_output, args=(process.stderr, logger))

    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to complete
    process.wait()

    # Wait for the output threads to finish
    stdout_thread.join()
    stderr_thread.join()

    return process.returncode

class QtHandler(logging.Handler, QObject):
    log_generated = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_generated.emit(msg)

class Logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.qt_handler = QtHandler()
        self.qt_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        self.qt_handler.setFormatter(formatter)
        self.logger.addHandler(self.qt_handler)

        self.file_handler = None

    def get_logger(self):
        return self.logger

    def clear_log(self, log_path):
        # Open the log file in write mode to clear it
        with open(log_path, 'w'):
            pass

    def info(self, message):
        self.logger.info(message)

    def error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)

    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

    def set_log_file(self, file_path):
        # If there was a previous FileHandler, remove it
        if self.file_handler is not None:
            self.file_handler.close()  # Added to release resources
            self.logger.removeHandler(self.file_handler)

        # Set up the file handler
        self.file_handler = logging.FileHandler(file_path, encoding='utf-8') #added utf-8
        self.file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

### Script runner fucntions

def run_external_script(script_path: str, args: dict[str, str], logger):

    cmd = [str(PY_EXE), "-u", str(script_path)]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]

    print("Executing: %s", " ".join(cmd))

    with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            logger.info(line.rstrip())
        if proc.wait():
            raise RuntimeError(f"{script_path} failed (exit {proc.returncode})")

def run_external_script_prev(script_path: str,
                        conda_prefix: str,
                        env_name: str,
                        args: dict[str, str],
                        logger):

    env_path   = os.path.join(conda_prefix, "envs", env_name)
    python_exe = os.path.join(env_path, "python.exe")

    # build the command as a list → no shell quoting headaches
    cmd = [python_exe, "-u", script_path]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]

    print("Executing command: %s", " ".join(cmd))

    # run and stream output
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        for line in proc.stdout:
            logger.info(line.rstrip())
        proc.wait()

    if proc.returncode:
        print("training script exited with code %s", proc.returncode)


def nnunet_sentinel_ok() -> bool:
    """Check if sentinel file exists and contains the correct app directory"""
    try:
        return _SENTINEL.read_text(encoding="utf-8").strip() == str(APP_DIR)
    except FileNotFoundError:
        return False

def ensure_nnunet_installed(logger):
    """Install nnUNet using the embedded Python if not already done"""
    if nnunet_sentinel_ok():
        return  # Already installed and path matches

    #logger.info("Installing nnUNet in embedded environment...")
    logger.info(f"First time running RESPAN, please allow a few minutes to complete installation...")

    # Subprocess env: override PIP_REQUIRE_VIRTUALENV in case the host
    # machine has it set (pip will otherwise refuse with "Could not find
    # an activated virtualenv"). The bundled respan env is fully self-
    # contained and doesn't need to appear "activated" to pip.
    import os as _os
    _pip_env = _os.environ.copy()
    _pip_env['PIP_REQUIRE_VIRTUALENV'] = 'false'
    # Also tell pip to NOT read user/global pip config files that might
    # impose additional constraints (e.g. --require-venv, index-url).
    _pip_env['PIP_CONFIG_FILE'] = _os.devnull

    try:
        # Simple pip install using the embedded Python
        cmd = [
            str(PY_EXE), "-m", "pip",
            "install", "-e", str(SRC_DIR),
            "--no-deps", "--force-reinstall"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=_pip_env,
        )

        # Mark as installed with current APP_DIR path
        _SENTINEL.write_text(str(APP_DIR), encoding="utf-8")


    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install nnUNet: {e.stderr}")
        raise


class EnvSetup(QThread):
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, respan_env):
        super().__init__()
        self.respan_env = respan_env
        self.logger = Logger()

    def run(self):
        try:
            if FROZEN:
                # For frozen app, check if the embedded environment exists
                if VENV_DIR.exists() and PY_EXE.exists():
                    # Ensure nnUNet is installed (checks sentinel)
                    self._ensure_nnunet_installed()
                    self.finished_signal.emit(True, f"RESPAN environment ready at {VENV_DIR}")
                    self.finished_signal.emit(True,"\nInitialization complete.")
                else:
                    self.finished_signal.emit(False, "Embedded environment not found")
            else:
                # For development, just check if the environment exists
                if VENV_DIR.exists() and PY_EXE.exists():
                    self.finished_signal.emit(True, f"RESPAN environment found at {VENV_DIR}")
                    self.finished_signal.emit(True,"\nInitialization complete.")
                else:
                    self.finished_signal.emit(False, f"Development environment '{DEV_ENV}' not found")

        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}")

    def _sentinel_ok(self) -> bool:
        return nnunet_sentinel_ok()

    def _ensure_nnunet_installed(self):
        ensure_nnunet_installed(self.logger)


class Splash(QSplashScreen):
    def __init__(self, text, time_to_show):
        pixmap = QPixmap(500, 300)  # Set your pixmap's size.
        pixmap.fill(Qt.transparent)  # You can set the background color here.

        # QPainter for drawing the text on the pixmap.
        painter = QPainter(pixmap)
        painter.setFont(QFont("Arial", 30))  # Set the font, size.
        painter.setPen(QColor(Qt.white))  # Set the color.
        painter.drawText(pixmap.rect(), Qt.AlignCenter, text)  # Draw the text.
        painter.end()

        super().__init__(pixmap)

        # QTimer to close the splash screen after 'time_to_show' milliseconds.
        QTimer.singleShot(time_to_show, self.close)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.logger = Logger()
        # Create the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create the tab widget
        tab_widget = QTabWidget()


        self.setCentralWidget(tab_widget)


        # Create instances of the tab widgets
        widget1 = RESPANAnalysis()
        widget2 = RESPANValidation()
        UNetwidget = UNet()
        CAREwidget = CARE()
        SNwidget = SelfNet()
        Aboutwidget = About()

        # Add as tabs
        tab_widget.addTab(widget1, "RESPAN Analysis")
        tab_widget.addTab(widget2, "Analysis Validation")
        tab_widget.addTab(UNetwidget, "Train nnU-Net Model")
        tab_widget.addTab(CAREwidget, "Train CARE Model")
        tab_widget.addTab(SNwidget, "Train SelfNet Model")
        tab_widget.addTab(Aboutwidget, "About RESPAN")

        tab_widget.setStyleSheet("QTabBar::tab { font-size: 10pt; width: 200px;}")

        # Create and add the log display widget
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)

        # Connect the logger signal to the update log display slot
        #self.logger.qt_handler.log_generated.connect(self.update_log_display)




class RESPANAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.log_display.append("Initializing RESPAN and checking environment...\n")

        # Log app location and warn about potential path issues
        app_path = str(APP_DIR)
        self.log_display.append(f"RESPAN location: {app_path}")
        if ' ' in app_path:
            self.log_display.append("  WARNING: RESPAN path contains spaces, which may cause issues with some operations.")
            self.log_display.append("  Consider moving RESPAN to a path without spaces (e.g., C:\\RESPAN\\)\n")
        if 'OneDrive' in app_path:
            self.log_display.append("  WARNING: RESPAN appears to be running from a OneDrive-synced folder.")
            self.log_display.append("  Cloud sync can interfere with processing. Consider using a local folder.\n")

        #if global_GUI_app == True:
            #self.load_dlls()
        self.check_gpu_availability()
        self.check_env_and_setup()




    def load_dlls(self):

        import ctypes

        try:
            dll_dir = os.path.join(base_path, 'cuda')
            ctypes.windll.kernel32.AddDllDirectory(dll_dir)
            ctypes.CDLL(os.path.join(base_path, '_internal', 'cusolver64_11.dll'))
            #self.log_display.append("Successfully loaded cusolver64_11.dll")
        except OSError as e:
            self.log_display.append(f"Error loading DLL: {e}")
        except Exception as e:
            self.log_display.append(f"dll_dir: {dll_dir}")
            self.log_display.append(f"path: {os.environ.get('PATH', 'Not set')}")
            self.log_display.append(f"cuda path: {os.environ.get('CUDA_PATH', 'Not set')}")
            self.log_display.append(f"ld lib path: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
            self.log_display.append(f"Failed to load cusolver64_11.dll: {str(e)}")



    def initUI(self):
            layout = QVBoxLayout(self)

            self.logger = Logger()

            # TitleFont
            titlefont = QFont()
            titlefont.setBold(True)
            titlefont.setPointSize(9)

            dir_options = QGroupBox("Image Data and Model Selection")
            dir_options.setStyleSheet("QGroupBox::title {"
                                            "subcontrol-origin: margin;"
                                            "subcontrol-position: top left;"
                                            "padding: 2px;"
                                            "color: black;"
                                            "}")
            dir_options.setFont(titlefont)
            dir_options_layout = QVBoxLayout()
            dir_options.setLayout(dir_options_layout)

            if global_GUI_app == False:
                self.RESPANdir_label = QLabel("No RESPAN directory selected.")
                self.RESPANdir_button = QPushButton("Select RESPAN directory")
                self.RESPANdir_button.clicked.connect(self.get_RESPANdir)

            self.directory_label = QLabel("No data directory selected.")
            self.directory_button = QPushButton("Select data directory")
            self.directory_button.clicked.connect(self.get_datadir)


            self.model_directory_label = QLabel("No nnU-Net model directory selected.")
            self.model_directory_button = QPushButton("Select nnU-Net model directory")
            self.model_directory_button.clicked.connect(self.get_modeldir)

            self.line = QFrame()
            self.line.setFrameShape(QFrame.HLine)
            self.line.setFrameShadow(QFrame.Sunken)
            #create string with underlines
            self.info_label = QLabel("*RESPAN is optimized for batch processing. Images need to be organized in subfolders \nof the selected data directory. E.g. a subfolder for each animal or condition.")
            self.info_label2 = QLabel("Ensure the correct Analysis_Settings.yaml file is included in each subfolder.")
            self.line2 = QFrame()
            self.line2.setFrameShape(QFrame.HLine)
            self.line2.setFrameShadow(QFrame.Sunken)

            if global_GUI_app == False:
                self.RESPANdir_button.setFixedWidth(300)
            self.directory_button.setFixedWidth(300)
            self.model_directory_button.setFixedWidth(300)

            self.line3 = QFrame()
            self.line3.setFrameShape(QFrame.HLine)
            self.line3.setFrameShadow(QFrame.Sunken)
            self.model_info_label = QLabel("nnU-Net model type used for analysis:")
            self.model_type = QComboBox()
            model_options = ["Spines, Dendrites, and Soma (original)", "Dendrites and Soma Only", "Spines, Necks, Dendrites, and Soma", "Spines, Necks, Dendrites, Soma, and Axons (2025)"]
            for option in model_options:
                self.model_type.addItem(option)

            # 2026-09-20: start on the CURRENT scheme, not the oldest one. A
            # QComboBox defaults to index 0, which left every fresh install on
            # "Spines, Dendrites, and Soma (original)" - three model generations
            # behind the bundled Dataset226. Looked up by name so reordering
            # model_options cannot silently change the default. A saved
            # <tab>GUI.pkl still wins: restore_variables runs later and calls
            # setCurrentIndex with the user's stored choice.
            self.model_type.setCurrentIndex(
                model_options.index("Spines, Necks, Dendrites, Soma, and Axons (2025)"))

            self.model_type.setFixedWidth(400)


            if global_GUI_app == False:
                dir_options_layout.addWidget(self.RESPANdir_button)
                dir_options_layout.addWidget(self.RESPANdir_label)
            dir_options_layout.addWidget(self.directory_button)
            dir_options_layout.addWidget(self.directory_label)
            dir_options_layout.addWidget(self.info_label)
            dir_options_layout.addWidget(self.info_label2)
            dir_options_layout.addWidget(self.model_directory_button)
            dir_options_layout.addWidget(self.model_directory_label)
            dir_options_layout.addWidget(self.line3)
            dir_options_layout.addWidget(self.model_info_label)
            dir_options_layout.addWidget(self.model_type)

            dir_options_layout.addWidget(self.line)


            #dir_options_layout.addWidget(self.line2)

            res_options = QGroupBox("Image Data and Model Resolution")
            res_options.setStyleSheet("QGroupBox::title {"
                                            "subcontrol-origin: margin;"
                                            "subcontrol-position: top left;"
                                            "padding: 2px;"
                                            "color: black;"
                                            "}")
            res_options.setFont(titlefont)
            res_options_layout = QVBoxLayout()
            res_options.setLayout(res_options_layout)

            self.inputdata_xy_label = QLabel("Image voxel size XY (µm):")
            self.inputdata_xy = QLineEdit("0.102")
            self.inputdata_z_label = QLabel("Image voxel size Z (µm):")
            self.inputdata_z = QLineEdit("1")
            self.inputdata_xy.setFixedWidth(90)
            self.inputdata_z.setFixedWidth(90)
            horizontal_input = QHBoxLayout()
            horizontal_input.addWidget(self.inputdata_xy_label)
            horizontal_input.addWidget(self.inputdata_xy)
            horizontal_input.addWidget(self.inputdata_z_label)
            horizontal_input.addWidget(self.inputdata_z)
            horizontal_input.addStretch(1)
            horizontal_input.setAlignment(Qt.AlignLeft)

            self.modeldata_xy_label = QLabel("Model voxel size XY (µm):")
            self.modeldata_xy = QLineEdit("0.102")
            self.modeldata_z_label = QLabel(" Model voxel size Z (µm):")
            self.modeldata_z = QLineEdit("1")
            self.modeldata_xy.setFixedWidth(90)
            self.modeldata_z.setFixedWidth(90)
            self.res_opt = QCheckBox("Use voxel sizes in analysis_settings.yaml")
            self.res_opt.setChecked(False)
            self.res_opt.stateChanged.connect(self.toggle_res)

            horizontal_model = QHBoxLayout()
            horizontal_model.addWidget(self.modeldata_xy_label)
            horizontal_model.addWidget(self.modeldata_xy)
            horizontal_model.addWidget(self.modeldata_z_label)
            horizontal_model.addWidget(self.modeldata_z)
            horizontal_model.addStretch(1)
            horizontal_model.setAlignment(Qt.AlignLeft)

            res_options_layout.addLayout(horizontal_input)
            res_options_layout.addLayout(horizontal_model)
            res_options_layout.addWidget(self.res_opt)


            dir_options.setFixedWidth(590)
            res_options.setFixedWidth(590)
            res_options.setFixedHeight(150)


            #input_data_and_res = QHBoxLayout()
            #input_data_and_res.addLayout(dir_options_layout)
            #input_data_and_res.addLayout(res_options_layout)

            '''
            options_group = QGroupBox("Cell Analysis Options")
            options_layout = QVBoxLayout()
            options_group.setLayout(options_layout)
            self.analyze1 = QCheckBox("Analyze Channel 1")
            self.analyze1.setChecked(True) 
            self.analyze2 = QCheckBox("Analyze Channel 2")
            self.analyze2.setChecked(True) 
            self.analyze3 = QCheckBox("Analyze Channel 3")
            self.analyze3.setChecked(True) 
            self.analyze4 = QCheckBox("Analyze Channel 4")
            self.analyze4.setChecked(True) 
            options_layout.addWidget(self.analyze1)
            options_layout.addWidget(self.analyze2)
            options_layout.addWidget(self.analyze3)
            options_layout.addWidget(self.analyze4)
            '''


            options_group1 = QGroupBox("Spine and Dendrite Detection")
            options_group1.setStyleSheet("QGroupBox::title {"
                                            "subcontrol-origin: margin;"
                                            "subcontrol-position: top left;"
                                            "padding: 2px;"
                                            "color: black;"
                                            "}")
            options_group1.setFont(titlefont)
            options_layout1 = QVBoxLayout()
            options_group1.setLayout(options_layout1)
            self.neuron_channel_label = QLabel("Channel containing neuron/dendrite signal:")
            self.neuron_channel_input = QLineEdit("1")
            h_neuron_ch = QHBoxLayout()
            h_neuron_ch.addWidget(self.neuron_channel_label)
            h_neuron_ch.addWidget(self.neuron_channel_input)

            self.float_label_1 = QLabel("Minimum dendrite volume (µm<sup>3</sup>, dendrites smaller than this will be ignored):")
            self.float_input_1 = QLineEdit("15")
            h_dend_min = QHBoxLayout()
            h_dend_min.addWidget(self.float_label_1)
            h_dend_min.addWidget(self.float_input_1)


            self.float_label_2 = QLabel("Spine volume filter (min, max volume in µm<sup>3</sup>):")
            self.float_input_2 = QLineEdit("0.03,15")
            h_spine_vol = QHBoxLayout()
            h_spine_vol.addWidget(self.float_label_2)
            h_spine_vol.addWidget(self.float_input_2)

            self.float_label_3 = QLabel("Spine distance filter (max distance from dendrite in µm):")
            self.float_input_3 = QLineEdit("4")
            h_spine_dist = QHBoxLayout()
            h_spine_dist.setAlignment(Qt.AlignLeft)
            h_spine_dist.addWidget(self.float_label_3)
            h_spine_dist.addWidget(self.float_input_3)


            #self.analysis_method_label = QLabel("Select analysis method:")
            #self.analysis_method = QComboBox()
            #analysis_options = ["Dendrite Specific", "Whole Neuron"]
            #for option in analysis_options:
            #    self.analysis_method.addItem(option)
            #h_an_opt = QHBoxLayout()
            #h_an_opt.addWidget(self.analysis_method_label)
            #h_an_opt.addWidget(self.analysis_method)

            self.neuron_channel_input.setFixedWidth(90)
            self.float_input_1.setFixedWidth(90)
            self.float_input_2.setFixedWidth(90)
            self.float_input_3.setFixedWidth(90)
            #self.analysis_method.setFixedWidth(150)

            options_layout1.addLayout(h_neuron_ch)
            options_layout1.addLayout(h_dend_min)
            options_layout1.addLayout(h_spine_vol)
            options_layout1.addLayout(h_spine_dist)
            #options_layout1.addLayout(h_an_opt)
            options_layout1.setAlignment(Qt.AlignTop)

            # -----------------------------------------------------------------
            # Spine Thresholding — spurious-neck detection + head-keep option
            # -----------------------------------------------------------------
            options_thresholding = QGroupBox("Spine Thresholding (optional quality filter)")
            options_thresholding.setStyleSheet("QGroupBox::title {"
                                               "subcontrol-origin: margin;"
                                               "subcontrol-position: top left;"
                                               "padding: 2px;"
                                               "color: black;"
                                               "}")
            options_thresholding.setFont(titlefont)
            options_thresh_layout = QVBoxLayout()
            options_thresholding.setLayout(options_thresh_layout)

            self.spurious_length_label = QLabel("Max neck path length (µm) before a spine is eligible for spurious-flagging:")
            self.spurious_length_input = QLineEdit("1.0")
            self.spurious_length_input.setFixedWidth(90)
            self.spurious_length_input.setToolTip(
                "Only spines whose measured neck path length EXCEEDS this value in "
                "µm become candidates for the spurious filter. Real biological "
                "spine necks are typically 0.2-1.5 µm; pathfinder-synthesized necks "
                "through noise are commonly longer. Lower → more aggressive, higher "
                "→ more permissive. Default 1.0 µm. The AND-gate with nnU-Net "
                "support protects real necks with genuine segmentation from being "
                "dropped purely on length."
            )
            h_spu_len = QHBoxLayout()
            h_spu_len.addWidget(self.spurious_length_label)
            h_spu_len.addWidget(self.spurious_length_input)

            self.spurious_support_label = QLabel("Minimum nnU-Net neck support (fraction 0-1):")
            self.spurious_support_input = QLineEdit("0.2")
            self.spurious_support_input.setFixedWidth(90)
            self.spurious_support_input.setToolTip(
                "Fraction of the assigned neck voxels that must coincide with the "
                "original nnU-Net neck prediction. Spines BELOW this threshold AND "
                "above the length threshold are flagged. 0.2 means necks with "
                "<20% nnU-Net backing are candidates. Lower → allow more "
                "pathfinder-synthesized necks, higher → stricter."
            )
            h_spu_sup = QHBoxLayout()
            h_spu_sup.addWidget(self.spurious_support_label)
            h_spu_sup.addWidget(self.spurious_support_input)

            # Drop-filter master toggle — post-bridge cleanup filters
            # (drop_disconnected_neck_fragments, drop_wrong_direction_neck_fragments)
            # are YAML-only advanced options, loaded from SpuriousNeck block with
            # default True. They operate on `connected_necks` per-label and do not
            # affect filopodia recovery (which is gated by a separate checkbox in
            # the "Additional Options" group).
            self.drop_spurious_necks = QCheckBox(
                "Drop spurious necks (AND-gate: length + low nnU-Net support)"
            )
            self.drop_spurious_necks.setChecked(True)
            self.drop_spurious_necks.setToolTip(
                "When ON (default): spines flagged by the AND-gate above have their "
                "NECK voxels removed. The head is retained and reclassified as "
                "partial-spine (kept if within distance to dendrite, dropped if "
                "'Keep partial spines' is unchecked). When OFF: the flag is recorded "
                "to neck_path_confidence.csv but no voxels are removed. "
                "To drop BOTH neck and head (aggressive mode), set YAML "
                "SpuriousNeck.exclude_mode to 'both'."
            )

            self.keep_head_if_flagged = QCheckBox(
                "Keep spine head even when neck can't be confidently generated"
            )
            self.keep_head_if_flagged.setChecked(False)
            self.keep_head_if_flagged.setToolTip(
                "When a neck fails the spurious-neck AND-gate above and the user "
                "has set YAML SpuriousNeck.exclude_mode=both (aggressive drop), "
                "this toggle downgrades the behaviour to 'neck_only' — the head "
                "stays in detected_spines.csv with neck metrics zeroed, rather "
                "than being removed entirely. No effect under the default "
                "exclude_mode=neck_only because heads are already retained. "
                "Advanced reviewer option."
            )

            options_thresh_layout.addLayout(h_spu_len)
            options_thresh_layout.addLayout(h_spu_sup)
            options_thresh_layout.addWidget(self.drop_spurious_necks)
            options_thresh_layout.addWidget(self.keep_head_if_flagged)
            options_thresh_layout.setAlignment(Qt.AlignTop)

            '''
            options_layout1.addWidget(self.neuron_channel_label)
            options_layout1.addWidget(self.neuron_channel_input)
            options_layout1.addWidget(self.float_label_1)
            options_layout1.addWidget(self.float_input_1)
            options_layout1.addWidget(self.float_label_2)
            options_layout1.addWidget(self.float_input_2)
            options_layout1.addWidget(self.float_label_3)
            options_layout1.addWidget(self.float_input_3)
            options_layout1.addWidget(self.analysis_method_label)
            options_layout1.addWidget(self.analysis_method)
            options_layout1.setAlignment(Qt.AlignTop)
            '''
            options_tracking = QGroupBox("Spine Tracking and Temporal Analysis")
            options_tracking.setStyleSheet("QGroupBox::title {"
                                         "subcontrol-origin: margin;"
                                         "subcontrol-position: top left;"
                                         "padding: 2px;"
                                         "color: black;"
                                         "}")
            options_tracking.setFont(titlefont)
            options_tracking_layout = QVBoxLayout()
            options_tracking.setLayout(options_tracking_layout)
            self.Track = QCheckBox("Spine tracking (temporal analysis of spines)")
            self.Track_label = QLabel(
                "Each subfolder should contain data from a single dendrite region with separate files/volumes\nfor each time point.")
            self.Track.setChecked(False)
            #self.HistMatch = QCheckBox("Histogram matching (matches image histograms to first image in the series)")
            #self.HistMatch.setChecked(False)
            self.reg_label = QLabel("Select registration method for aligning volumes across time:")
            self.reg_method = QComboBox()
            reg_options = ["Rigid", "Elastic"]
            for option in reg_options:
                self.reg_method.addItem(option)

            self.reg_method.setFixedWidth(150)

            options_tracking_layout.addWidget(self.Track)
            options_tracking_layout.addWidget(self.Track_label)
            #options_tracking_layout.addWidget(self.HistMatch)
            options_tracking_layout.addWidget(self.reg_label)
            options_tracking_layout.addWidget(self.reg_method)


            options_group2 = QGroupBox("Additional Options")
            options_group2.setStyleSheet("QGroupBox::title {"
                                            "subcontrol-origin: margin;"
                                            "subcontrol-position: top left;"
                                            "padding: 2px;"
                                            "color: black;"
                                            "}")
            options_group2.setFont(titlefont)
            options_layout2 = QVBoxLayout()
            options_group2.setLayout(options_layout2)


            self.save_validation = QCheckBox("Save validation data (2D MIPs for confirming dendrites and spines)")
            self.save_validation.setChecked(True)
            self.neck_generation = QCheckBox("Perform spine neck generation")
            self.neck_generation.setChecked(True)
            self.recover_filopodia = QCheckBox(
                "Recover filopodia from orphan nnU-Net neck fragments")
            self.recover_filopodia.setChecked(False)
            self.recover_filopodia.setToolTip(
                "Preserve nnU-Net-predicted necks that have no associated spine head "
                "as filopodia. Carves a small head from the neck tip (no synthesis). "
                "Off by default — enables new behaviour on top of normal neck "
                "assignment. Filopodia are marked with spine_type='filopodia' in the CSV.")
            self.dendrite_repair = QCheckBox(
                "Dendrite repair (reconnect fragments without dilating)")
            self.dendrite_repair.setChecked(False)
            self.dendrite_repair.setToolTip(
                "Bridge dendrite fragments that were broken by discontinuous nnU-Net "
                "predictions. Operates in XY only and only fills background voxels "
                "along the bridge path — original dendrite morphology is preserved. "
                "Merged fragments adopt the parent (largest CC) label.")
            self.dendrite_repair_max_dist_um = QLineEdit("2.0")
            self.dendrite_repair_max_dist_um.setFixedWidth(60)
            self.dendrite_repair_max_dist_um.setToolTip(
                "Maximum XY gap (microns) to bridge between dendrite fragments. "
                "Default 2.0 µm.")
            self.detect_multi_head_spines = QCheckBox(
                "Multi-head spine detection (detects multi-headed spines, "
                "analyzes heads as sub features of these spines)")
            self.detect_multi_head_spines.setChecked(False)
            self.detect_multi_head_spines.setToolTip(
                "Group physically connected spine heads under one parent ID. "
                "Two heads belong to the same multi-head group when their "
                "territories (head + assigned neck + nnU-Net neck) are "
                "26-connected. Adds parent_spine_id, head_index, "
                "multi_head_group_size columns to Detected_spines.csv, plus a "
                "Multi_Head_Spine_Groups.csv aggregate file and a 5th channel "
                "in the validation MIP showing the group ID.")
            self.keep_partial_spines = QCheckBox(
                "Keep partial spines (head detected but neck path occluded by a "
                "nearby spine)")
            self.keep_partial_spines.setChecked(True)
            self.keep_partial_spines.setToolTip(
                "Spines where insufficient information exists to complete the neck "
                "path to the dendrite (e.g. path occluded by a nearby spine). The "
                "head is kept in the output with distance metrics; neck metrics are "
                "zeroed and spine_type='partial-spine'. Uncheck to drop these spines "
                "entirely from the output.")

            self.save_intermediate = QCheckBox("Save intermediate data (3D spine volumes, MIPs, meshes for detailed inspection)")
            self.save_intermediate.setChecked(False)
            self.nnUNet_patching = QCheckBox("Enable patching for nnUNet (recommended for datasets >1GB)")
            self.nnUNet_patching.setChecked(False)
            #self.second_pass = QCheckBox("Enable second pass for more accurate spine and neck annotations")
            #self.second_pass.setChecked(False)

            self.image_restore_opt = QCheckBox("Use image restoration (*requires trained CARE models)")
            self.image_restore_opt.setChecked(False)
            self.axial_restore_opt = QCheckBox("Use axial restoration (*requires trained SelfNet model)")
            self.axial_restore_opt.setChecked(False)

            self.swc_gen = QCheckBox("Generate SWC files for dendrite/neuron (via Vaa3D APP2 plugin)")
            self.swc_gen.setChecked(False)
            self.dask_enabled = QCheckBox("Enable Dask parallelization (in development)")
            self.dask_enabled.setChecked(False)

            #save_options = QHBoxLayout()
            options_layout2.addWidget(self.save_validation)
            options_layout2.addWidget(self.neck_generation)
            options_layout2.addWidget(self.recover_filopodia)
            # Dendrite repair: checkbox + max-distance input on a single row
            _dr_row = QHBoxLayout()
            _dr_row.addWidget(self.dendrite_repair)
            _dr_row.addWidget(QLabel("max gap (µm):"))
            _dr_row.addWidget(self.dendrite_repair_max_dist_um)
            _dr_row.addStretch(1)
            _dr_wrap = QWidget(); _dr_wrap.setLayout(_dr_row)
            options_layout2.addWidget(_dr_wrap)
            options_layout2.addWidget(self.detect_multi_head_spines)
            options_layout2.addWidget(self.keep_partial_spines)
            options_layout2.addWidget(self.save_intermediate)
            options_layout2.addWidget(self.nnUNet_patching)

            options_layout2.addWidget(self.image_restore_opt)
            options_layout2.addWidget(self.axial_restore_opt)
            options_layout2.addWidget(self.swc_gen)
            options_layout2.addWidget(self.dask_enabled)

            #options_layout2.addWidget(self.second_pass)

            #options_layout2.addLayout(save_options)


            options_group1.setFixedWidth(590)
            #options_group1.setFixedHeight(200)
            options_group2.setFixedWidth(590)

            input_data_and_res = QHBoxLayout()


            left_side_options = QVBoxLayout()
            #input_data_and_res.addWidget(dir_options)
            left_side_options.addWidget(dir_options, alignment=Qt.AlignTop)
            left_side_options.addWidget(options_group1, alignment=Qt.AlignTop)
            left_side_options.addWidget(options_thresholding, alignment=Qt.AlignTop)

            right_side_options = QVBoxLayout()
            right_side_options.addWidget(res_options, alignment=Qt.AlignTop)
            right_side_options.addWidget(options_tracking, alignment=Qt.AlignTop)
            right_side_options.addWidget(options_group2, alignment=Qt.AlignTop)


            main_options = QHBoxLayout()
            main_options.addLayout(left_side_options)
            main_options.addLayout(right_side_options)



            run_cancel_layout = QHBoxLayout()
            self.run_button = QPushButton("Run")
            self.run_button.clicked.connect(self.get_variables)
            self.run_button.clicked.connect(self.run_function)
            run_cancel_layout.addWidget(self.run_button)
            self.cancel_button = QPushButton("Close")
            self.cancel_button.clicked.connect(self.close)
            run_cancel_layout.addWidget(self.cancel_button)

            self.run_button.setStyleSheet("background-color: lightblue;")
            #self.cancel_button.setStyleSheet("background-color: lightcoral;")

            self.progress = QProgressBar()
            self.progress.setVisible(False)

            #self.run_button.setFixedWidth(300)
            #self.cancel_button.setFixedWidth(300)

            #FINAL Layout

            layout.addLayout(input_data_and_res)

            layout.addLayout(main_options)

            layout.addLayout(run_cancel_layout)

            layout.addWidget(self.progress)

            self.log_display = QTextEdit()
            layout.addWidget(self.log_display)

            self.logger.qt_handler.log_generated.connect(self.update_log_display)

            self.setLayout(layout)

            # Retrieve individual variables from the dictionary
            try:
                variables_dict = _load_settings('parametersanalysisGUI.pkl')
                if variables_dict is None:
                    raise FileNotFoundError('parametersanalysisGUI.pkl')  # handled below exactly as before

                # retreive
                if global_GUI_app == False:
                    spine_dir = variables_dict.get('spine_dir', None)

                data_dir = variables_dict.get('data_dir', None)
                model_dir = variables_dict.get('model_dir', None)
                inputxy = variables_dict.get('inputxy', None)
                inputz = variables_dict.get('inputz', None)
                modelxy = variables_dict.get('modelxy', None)
                modelz = variables_dict.get('modelz', None)

                neuron_ch = variables_dict.get('neuron_ch', None)
                min_dend = variables_dict.get('min_dend', None)
                spine_vol = variables_dict.get('spine_vol', None)
                spine_dist = variables_dict.get('spine_dist', None)
                spurious_length_min_um = variables_dict.get('spurious_length_min_um', "1.0")
                spurious_nnunet_support_min = variables_dict.get('spurious_nnunet_support_min', "0.2")
                spurious_keep_head_if_flagged = variables_dict.get('spurious_keep_head_if_flagged', False)
                drop_spurious_necks = variables_dict.get('drop_spurious_necks', True)
                #analysis_meth = variables_dict.get('analysis_meth', None)

                image_restore = variables_dict.get('image_restore', None)
                neck_generation = variables_dict.get('neck_generation', None)
                recover_filopodia = variables_dict.get('recover_filopodia', False)
                # Accept legacy 'intelligent_dendrite_repair' key for backward compat with old pkl state files.
                dendrite_repair = variables_dict.get(
                    'dendrite_repair',
                    variables_dict.get('intelligent_dendrite_repair', False))
                dendrite_repair_max_dist_um = variables_dict.get('dendrite_repair_max_dist_um', "2.0")
                detect_multi_head_spines = variables_dict.get('detect_multi_head_spines', False)
                keep_partial_spines = variables_dict.get('keep_partial_spines', True)
                axial_restore = variables_dict.get('axial_restore', None)
                swc_gen = variables_dict.get('swc_gen', None)
                dask_enabled = variables_dict.get('dask_enabled', None)
                nnUNet_patching = variables_dict.get('nnUNet_patching', None)
                save_val_data = variables_dict.get('save_val_data', None)
                save_int_data = variables_dict.get('save_int_data', None)
                #hist_match = variables_dict.get('hist_match', None)
                spine_track = variables_dict.get('spine_track', None)
                reg_meth = variables_dict.get('reg_meth', None)

                #second_pass = variables_dict.get('second_pass', None)
                model_type = variables_dict.get('model_type', None)

                # udpate GUI:
                if global_GUI_app == False:
                    self.RESPANdir_label.setText(f"Selected directory: {spine_dir}")
                self.directory_label.setText(f"Selected directory: {data_dir}")
                self.model_directory_label.setText(f"Selected directory: {model_dir}")

                self.inputdata_xy.setText(str(inputxy))
                self.inputdata_z.setText(str(inputz))
                self.modeldata_xy.setText(str(modelxy))
                self.modeldata_z.setText(str(modelz))

                self.neuron_channel_input.setText(str(neuron_ch))
                self.float_input_1.setText(str(min_dend))
                self.float_input_2.setText(str(spine_vol))
                self.float_input_3.setText(str(spine_dist))
                self.spurious_length_input.setText(str(spurious_length_min_um))
                self.spurious_support_input.setText(str(spurious_nnunet_support_min))
                self.keep_head_if_flagged.setChecked(bool(spurious_keep_head_if_flagged))
                self.drop_spurious_necks.setChecked(bool(drop_spurious_necks))
                #self.analysis_method.setCurrentIndex(int(analysis_meth))

                # Legacy pkl state files may have None for checkboxes added in
                # later versions. Default to the current widget's checked-state
                # so missing keys don't crash the GUI on load.
                def _bool_or_keep(val, widget):
                    return bool(val) if val is not None else widget.isChecked()
                self.image_restore_opt.setChecked(_bool_or_keep(image_restore, self.image_restore_opt))
                self.axial_restore_opt.setChecked(_bool_or_keep(axial_restore, self.axial_restore_opt))
                self.neck_generation.setChecked(_bool_or_keep(neck_generation, self.neck_generation))
                self.recover_filopodia.setChecked(_bool_or_keep(recover_filopodia, self.recover_filopodia))
                self.dendrite_repair.setChecked(_bool_or_keep(dendrite_repair, self.dendrite_repair))
                self.dendrite_repair_max_dist_um.setText(str(dendrite_repair_max_dist_um))
                self.detect_multi_head_spines.setChecked(_bool_or_keep(detect_multi_head_spines, self.detect_multi_head_spines))
                self.keep_partial_spines.setChecked(_bool_or_keep(keep_partial_spines, self.keep_partial_spines))
                self.swc_gen.setChecked(_bool_or_keep(swc_gen, self.swc_gen))
                self.dask_enabled.setChecked(_bool_or_keep(dask_enabled, self.dask_enabled))
                self.nnUNet_patching.setChecked(_bool_or_keep(nnUNet_patching, self.nnUNet_patching))
                self.save_validation.setChecked(_bool_or_keep(save_val_data, self.save_validation))
                self.save_intermediate.setChecked(_bool_or_keep(save_int_data, self.save_intermediate))
                #self.HistMatch.setChecked(hist_match)
                self.Track.setChecked(spine_track)
                self.reg_method.setCurrentIndex(int(reg_meth))

                #self.second_pass.setChecked(second_pass)
                self.model_type.setCurrentIndex(int(model_type))


            except (FileNotFoundError, EOFError, pickle.PickleError):
                pass  # If can't load the pickle file, just ignore the error

    def check_gpu_availability(self):

        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.log_display.append("GPU available:")
                for gpu in gpus:
                    # Attempt to print detailed information about each GPU
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        self.log_display.append(f"     {gpu_details['device_name']}")
                        try:
                            pynvml.nvmlInit()
                            device_count = pynvml.nvmlDeviceGetCount()
                            if device_count > 0:
                                for i in range(device_count):
                                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                                    #gpu_info = pynvml.nvmlDeviceGetName(handle)
                                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                    #self.log_display.append(f"GPU {i}: {gpu_info.decode('utf-8')}")
                                    self.log_display.append(f"     Total Memory: {memory_info.total / 1024 ** 2:.2f} MB")
                                    self.log_display.append(f"     Free Memory: {memory_info.free / 1024 ** 2:.2f} MB")
                                    self.log_display.append(f"     TensorFlow version: {tf.__version__}")
                                    self.log_display.append(f"     CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
                                    self.log_display.append(f"     cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
                                    self.log_display.append(f"     CUDA Path: {os.environ.get('CUDA_PATH', 'Not set')}\n")
                                    #self.log_display.append(f"Used Memory: {memory_info.used / 1024 ** 2:.2f} MB")
                        except Exception as e:
                            self.log_display.append(f"Failed to retrieve GPU information: {str(e)}")
                    except RuntimeError as e:
                        self.log_display.append(f"Could not set GPU: {str(e)}")
                    except Exception as e:
                        self.log_display.append(f"Error retrieving GPU details: {str(e)}")
            else:
                self.log_display.append("GPU not available. Using CPU.")
        except Exception as e:
            self.log_display.append("TensorFlow not installed. Using CPU.")

    def check_env_and_setup(self):
        self.env_setup_thread = EnvSetup(global_respan_env)
        self.env_setup_thread.finished_signal.connect(self.on_env_setup_finished)
        self.env_setup_thread.start()

    @pyqtSlot(bool, str)
    def on_env_setup_finished(self, success, message):
        if success:
            # Enable all previously disabled UI components
            #self.enableUIComponents()
            self.log_display.append(message)
        else:
            QMessageBox.critical(self, "Environment Setup Failed", message)




    def toggle_res(self, state):
        # Disable or enable LineEdits based on the checkbox state
        isInvisible = self.res_opt.isChecked()

        # Set text color to white to make it 'invisible' against a white background
        invisibleStyleSheet = "QLineEdit { color: white; }"

        # Reset to default (assumed black text on white background)
        defaultStyleSheet = "QLineEdit { color: black; background-color: white; }"

        if isInvisible:
            self.inputdata_xy.setStyleSheet(invisibleStyleSheet)
            self.inputdata_z.setStyleSheet(invisibleStyleSheet)
            self.modeldata_xy.setStyleSheet(invisibleStyleSheet)
            self.modeldata_z.setStyleSheet(invisibleStyleSheet)
        else:
            self.inputdata_xy.setStyleSheet(defaultStyleSheet)
            self.inputdata_z.setStyleSheet(defaultStyleSheet)
            self.modeldata_xy.setStyleSheet(defaultStyleSheet)
            self.modeldata_z.setStyleSheet(defaultStyleSheet)


    @pyqtSlot()
    def get_RESPANdir(self):
         RESPANdir = QFileDialog.getExistingDirectory(self, 'Select RESPAN directory')
         if RESPANdir:
             self.RESPANdir_label.setText(f"Selected directory: {RESPANdir}")
             #self.RESPANdir_label.setFixedWidth(300)
             #self.RESPANdir_labe.setWordWrap(True)

    @pyqtSlot()
    def get_datadir(self):
         datadir = QFileDialog.getExistingDirectory(self, 'Select data directory')
         if datadir:
             self.directory_label.setText(f"Selected directory: {datadir}")
             #self.directory_label.setWordWrap(True)

    @pyqtSlot()
    def get_modeldir(self):
         modeldir = QFileDialog.getExistingDirectory(self, 'Select model directory')
         if modeldir:
             self.model_directory_label.setText(f"Selected directory: {modeldir}")
             #self.model_directory_label.setMaximumWidth(300)
             #self.model_directory_label.setWordWrap(True)


    @pyqtSlot()
    def get_variables(self):

        if global_GUI_app == False:
            spinedir_text = self.RESPANdir_label.text()
            spine_dir = spinedir_text.split(": ")[-1]
        else:
            spine_dir = "none"

        dirlabel_text = self.directory_label.text()
        modellabel_text = self.model_directory_label.text()

        data_dir = dirlabel_text.split(": ")[-1]
        model_dir = modellabel_text.split(": ")[-1]

        inputxy = str(self.inputdata_xy.text())
        inputz = str(self.inputdata_z.text())
        modelxy = str(self.modeldata_xy.text())
        modelz = str(self.modeldata_z.text())

        neuron_ch = str(self.neuron_channel_input.text())
        min_dend = str(self.float_input_1.text())
        spine_vol = str(self.float_input_2.text())
        spine_dist = str(self.float_input_3.text())
        #analysis_meth = self.analysis_method.currentIndex()

        # Spine thresholding (optional spurious-neck filter)
        spurious_length_min_um = str(self.spurious_length_input.text())
        spurious_nnunet_support_min = str(self.spurious_support_input.text())
        spurious_keep_head_if_flagged = self.keep_head_if_flagged.isChecked()
        drop_spurious_necks = self.drop_spurious_necks.isChecked()

        image_restore = self.image_restore_opt.isChecked()
        axial_restore = self.axial_restore_opt.isChecked()
        swc_gen = self.swc_gen.isChecked()
        neck_generation = self.neck_generation.isChecked()
        recover_filopodia = self.recover_filopodia.isChecked()
        dendrite_repair = self.dendrite_repair.isChecked()
        dendrite_repair_max_dist_um = str(self.dendrite_repair_max_dist_um.text())
        detect_multi_head_spines = self.detect_multi_head_spines.isChecked()
        # 2026-09-18: was missing, so every GUI Run click raised NameError in
        # this slot (the dict below references it) and PyQt aborted the app.
        keep_partial_spines = self.keep_partial_spines.isChecked()
        save_int_data = self.save_intermediate.isChecked()
        dask_enabled = self.dask_enabled.isChecked()
        nnUNet_patching = self.nnUNet_patching.isChecked()
        save_val_data = self.save_validation.isChecked()
        #hist_match = self.HistMatch.isChecked()
        spine_track = self.Track.isChecked()
        reg_meth = self.reg_method.currentIndex()

        #second_pass = self.second_pass.isChecked()
        model_type = self.model_type.currentIndex()


        variables_dict = {
            'spine_dir': spine_dir,
            'data_dir': data_dir,
            'model_dir': model_dir,
            'inputxy': inputxy,
            'inputz': inputz,
            'modelxy': modelxy,
            'modelz': modelz,

            'neuron_ch': neuron_ch,
            'min_dend': min_dend,
            'spine_vol': spine_vol,
            'spine_dist': spine_dist,
            'spurious_length_min_um': spurious_length_min_um,
            'spurious_nnunet_support_min': spurious_nnunet_support_min,
            'spurious_keep_head_if_flagged': spurious_keep_head_if_flagged,
            'drop_spurious_necks': drop_spurious_necks,
            #'analysis_meth': analysis_meth,
            'neck_generation': neck_generation,
            'recover_filopodia': recover_filopodia,
            'dendrite_repair': dendrite_repair,
            'dendrite_repair_max_dist_um': dendrite_repair_max_dist_um,
            'detect_multi_head_spines': detect_multi_head_spines,
            'keep_partial_spines': keep_partial_spines,
            'image_restore': image_restore,
            'axial_restore': axial_restore,
            'swc_gen': swc_gen,
            'save_val_data': save_val_data,
            'save_int_data': save_int_data,
            'dask_enabled': dask_enabled,
            'nnUNet_patching': nnUNet_patching,

            #'hist_match': hist_match,
            'spine_track': spine_track,
            'reg_meth': reg_meth,

            #'second_pass': second_pass,
            'model_type': model_type
        }


        # Save the dictionary to a pickle file
        _save_settings('parametersanalysisGUI.pkl', variables_dict, getattr(self, 'log_display', None))



    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode

        if global_GUI_app == False:
            RESPAN = self.RESPANdir_label.text().replace("Selected directory: ", "")
            if RESPAN == "No directory selected.":
                QMessageBox.critical(self, "Error", "No directory selected.")
                self.progress.setVisible(False)
                return

        directory = self.directory_label.text().replace("Selected directory: ", "")
        if directory == "No directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return

        model_dir = self.model_directory_label.text().replace("Selected directory: ", "")
        if model_dir == "No directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return

        try:
            inputxy =  float(self.inputdata_xy.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for image xy.")
            self.progress.setVisible(False)
            return
        try:
            inputz =  float(self.inputdata_z.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for image z.")
            self.progress.setVisible(False)
            return
        try:
            modelxy =  float(self.modeldata_xy.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for model xy.")
            self.progress.setVisible(False)
            return
        try:
            modelz =  float(self.modeldata_z.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for model z.")
            self.progress.setVisible(False)
            return
        try:
            neuron_ch =  float(self.neuron_channel_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for neuron channel.")
            self.progress.setVisible(False)
            return

        try:
            min_dendrite_vol =  float(self.float_input_1.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for minimum dendrite volume.")
            self.progress.setVisible(False)
            return

        try:
            spine_vol = list(map(float, self.float_input_2.text().split(',')))
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for spine volume.")
            self.progress.setVisible(False)
            return

        try:
            spine_dist =  float(self.float_input_3.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for spine distance to dendrite.")
            self.progress.setVisible(False)
            return

        # Spine thresholding (optional — parsed for worker to attach to settings)
        try:
            spurious_length_min_um = float(self.spurious_length_input.text())
            spurious_nnunet_support_min = float(self.spurious_support_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input in Spine Thresholding section.")
            self.progress.setVisible(False)
            return
        spurious_keep_head_if_flagged = self.keep_head_if_flagged.isChecked()
        drop_spurious_necks = self.drop_spurious_necks.isChecked()

        reg_method =  str(self.reg_method.currentText())
        analysis_method = "Whole Neuron"# str(self.analysis_method.currentText())

        neck_generation = self.neck_generation.isChecked()
        recover_filopodia = self.recover_filopodia.isChecked()
        dendrite_repair = self.dendrite_repair.isChecked()
        try:
            dendrite_repair_max_dist_um = float(self.dendrite_repair_max_dist_um.text())
        except (ValueError, AttributeError):
            dendrite_repair_max_dist_um = 2.0
        detect_multi_head_spines = self.detect_multi_head_spines.isChecked()
        keep_partial_spines = self.keep_partial_spines.isChecked()
        image_restore = self.image_restore_opt.isChecked()
        axial_restore = self.axial_restore_opt.isChecked()
        save_intermediate = self.save_intermediate.isChecked()
        dask_enabled = self.dask_enabled.isChecked()
        nnUNet_patching = self.nnUNet_patching.isChecked()
        save_validation = self.save_validation.isChecked()
        #HistMatch = self.HistMatch.isChecked()
        Track = self.Track.isChecked()
        use_yaml_res = self.res_opt.isChecked()
        swc_gen = self.swc_gen.isChecked()
        #second_pass = self.second_pass.isChecked()
        second_pass = False
        model_type = str(self.model_type.currentText())

        directory =  directory + "/"

        # --- Path robustness checks ---
        # Spaces in the MODEL folder are only logged: nnU-Net
        # is launched with the path quoted and runs fine from e.g. "G:/My Drive";
        # the data folder keeps the dialog because outputs and tools are written
        # and launched under it.
        if ' ' in model_dir:
            self.log_display.append(f"  Note: model directory path contains spaces: {model_dir}")
        for label, path in [("Data directory", directory)]:
            if ' ' in path:
                reply = QMessageBox.warning(self, "Path Contains Spaces",
                    f"{label} path contains spaces:\n{path}\n\n"
                    "Spaces in file paths can cause issues with some processing steps.\n"
                    "Consider moving data to a path without spaces (e.g., C:\\RESPAN_Data\\).\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.progress.setVisible(False)
                    return

            if 'OneDrive' in path:
                reply = QMessageBox.warning(self, "Cloud Sync Path Detected",
                    f"{label} appears to be in a OneDrive folder:\n{path}\n\n"
                    "OneDrive file syncing can interfere with RESPAN processing.\n"
                    "We recommend using a local folder (e.g., C:\\RESPAN_Data\\).\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.progress.setVisible(False)
                    return

        # Check data directory has valid subfolders
        try:
            dir_contents = os.listdir(directory)
            has_subfolders = any(os.path.isdir(os.path.join(directory, d)) for d in dir_contents)
        except OSError:
            has_subfolders = False

        if not has_subfolders:
            QMessageBox.critical(self, "Invalid Data Directory",
                "The selected data directory contains no subfolders.\n\n"
                "RESPAN expects datasets organized as:\n"
                "  data_directory/\n"
                "    dataset_1/\n"
                "      Analysis_Settings.yaml\n"
                "      [image files]\n"
                "    dataset_2/\n"
                "      Analysis_Settings.yaml\n"
                "      [image files]\n\n"
                "Please see the RESPAN user guide for dataset preparation instructions.")
            self.progress.setVisible(False)
            return

        if global_GUI_app == False:
            RESPAN = RESPAN +"/"
        else:
            RESPAN = "none"



        #channel_options = [self.analyze1.isChecked(), self.analyze2.isChecked(), self.analyze3.isChecked(),self.analyze4.isChecked()]
        #other_options = [self.save_intermediate.isChecked()]

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        #self.worker = AnalysisWorker(RESPAN, directory, other_options, min_dendrite_vol, spine_vol, spine_dist, HistMatch, Track, reg_method, self.logger)
        self.worker = AnalysisWorker(RESPAN, directory, model_dir, neck_generation, save_intermediate, dask_enabled, nnUNet_patching, save_validation,
                                     inputxy, inputz, modelxy, modelz, neuron_ch, analysis_method,
                                     image_restore, axial_restore, swc_gen,
                                     min_dendrite_vol, spine_vol, spine_dist, #HistMatch,
                                     Track, reg_method, use_yaml_res, second_pass, model_type, self.logger,
                                     spurious_length_min_um=spurious_length_min_um,
                                     spurious_nnunet_support_min=spurious_nnunet_support_min,
                                     spurious_keep_head_if_flagged=spurious_keep_head_if_flagged,
                                     drop_spurious_necks=drop_spurious_necks,
                                     recover_filopodia=recover_filopodia,
                                     dendrite_repair=dendrite_repair,
                                     dendrite_repair_max_dist_um=dendrite_repair_max_dist_um,
                                     detect_multi_head_spines=detect_multi_head_spines,
                                     keep_partial_spines=keep_partial_spines)


        self.worker.task_done.connect(self.on_task_done)
        self.worker.start()


    @pyqtSlot(str)
    def update_log_display(self, message):
        self.log_display.append(message)
        self.log_display.moveCursor(QTextCursor.End)


    @pyqtSlot(str)
    def on_task_done(self, message):
        self.logger.info(message)
        self.progress.setValue(self.progress.maximum())
        self.progress.setVisible(False)


def apply_run_parameters(settings, locations, p, logger):
    """Wire one run's parameters onto settings/locations and log them.

    Shared by the GUI (AnalysisWorker passes itself as p) and the headless
    --cli path (a SimpleNamespace carrying the same attribute names), so both
    entry points configure the pipeline from one piece of code. Module-level
    globals (elastix paths, bundled env paths, PY_EXE, base_path) are read
    directly, exactly as the worker did before this block was extracted.
    """
    #update locations for elastix
    locations.elastix_path = elastix_path
    locations.elastix_params = elastix_params
    locations.elastix_temp = locations.nnUnet_input + "/temp_elastix"
    
    
    #Modify specific parameters and settings:
    settings.save_intermediate_data = p.save_intermediate
    settings.dask_enabled = p.dask_enabled
    settings.save_val_data = p.save_validation
    settings.neuron_seg_model_path = p.model_dir
    
    if p.use_yaml_res == False:
        settings.input_resXY = p.inputxy
        settings.input_resZ = p.inputz
        settings.model_resXY = p.modelxy
        settings.model_resZ = p.modelz
    
    settings.nnUnet_type = '3d_fullres'
    settings.nnUnet_conda_path = global_conda_path
    settings.nnUnet_env_path = global_env_path
    settings.nnUnet_env = global_respan_env
    settings.selfnet_inference_script = str(SELFNET_INFERENCE_SCRIPT)
    settings.selfnet_training_script = str(SELFNET_TRAINING_SCRIPT)
    settings.clean_launcher = str(clean_launcher)
    settings.internal_py_path = str(PY_EXE)
    settings.nnunet_predict_bat = nnunet_predict_bat
    
    settings.neck_generation = p.neck_generation
    settings.recover_filopodia = p.recover_filopodia
    settings.dendrite_repair = bool(p.dendrite_repair)
    settings.dendrite_repair_max_dist_um = float(p.dendrite_repair_max_dist_um)
    settings.detect_multi_head_spines = bool(p.detect_multi_head_spines)
    settings.keep_partial_spines = p.keep_partial_spines
    
    #basepath for selfnet
    settings.basepath = base_path
    
    settings.image_restore = p.image_restore
    settings.axial_restore = p.axial_restore
    settings.neuron_channel = int(p.neuron_ch)
    settings.analysis_method = "Whole Neuron"
    settings.dask_block = (64, 256, 256)
    settings.dask_halo = (8, 8, 8)
    settings.Vaa3d = p.swc
    #settings.spine_roi_volume_size = 4 #in microns in x, y, z - approx 50px for 0.3 resolution data
    settings.min_dendrite_vol = round(p.min_dendrite_vol / settings.input_resXY/settings.input_resXY/settings.input_resZ, 0)
    settings.neuron_spine_size = [round(x / (settings.input_resXY*settings.input_resXY*settings.input_resZ),0) for x in p.spine_vol]
    settings.neuron_spine_dist = round(p.spine_dist / (settings.input_resXY),2)
    # Override YAML spurious-neck thresholds with GUI values.
    settings.spurious_path_length_min_um = float(p.spurious_length_min_um)
    settings.spurious_nnunet_support_min = float(p.spurious_nnunet_support_min)
    settings.spurious_keep_head_if_flagged = bool(p.spurious_keep_head_if_flagged)
    # GUI master toggle: checked = neck_only (head retained,
    # reclassified as partial-spine). unchecked = flag_only
    # (no voxel drops, CSV-only flag). YAML override for 'both'.
    settings.spurious_exclude_mode = (
        'neck_only' if bool(p.drop_spurious_necks) else 'flag_only')
    # drop_disconnected_neck_fragments and
    # drop_wrong_direction_neck_fragments are YAML-only advanced
    # options — loaded in Main.py SpuriousNeck block with default
    # True. GUI does not override; users can disable via YAML.
    settings.HistMatch = False #p.HistMatch
    settings.Track = p.Track
    settings.reg_method = p.reg_method
    # Logging and memory tracking are always on (fast, console only).
    # save_intermediate_data controls disk-heavy spine array/MIP exports.
    settings.additional_logging = True
    settings.checkmem = True
    settings.additional_logging_dev = additional_logging_dev
    #settings.second_pass = p.second_pass
    settings.second_pass = False
    
    settings.patch_for_nnunet = p.nnUNet_patching
    settings.nnunet_patch_size = (256, 512, 512)
    settings.nnunet_stride = tuple(int(p*0.75) for p in settings.nnunet_patch_size)
    
    settings.use_vox_measurements = False
    if p.dask_enabled:
        settings.resave_omezarr = True
        settings.zarr_threads = None
        settings.zarr_show_progress = True
    else:
        settings.resave_omezarr = False
    
    if p.model_type == "Spines, Dendrites, and Soma (original)":
        settings.model_type = 1
    elif p.model_type == "Dendrites and Soma Only":
        settings.model_type = 2
    elif p.model_type == "Spines, Necks, Dendrites, and Soma":
        settings.model_type = 3
    elif p.model_type == "Spines, Necks, Dendrites, Soma, and Axons (2025)":
        settings.model_type = 4
    
    logger.info(" RESPAN Parameters:")
    logger.info(f"  Image resolution: {settings.input_resXY}µm XY, {settings.input_resZ}µm Z")
    logger.info(f"  Model used: {settings.neuron_seg_model_path}")
    logger.info(f"  Model type: {p.model_type}")
    logger.info(f"  Model resolution: {settings.model_resXY}µm XY, {settings.model_resZ}µm Z")
    logger.info(
        f"  Conda Path {settings.nnUnet_conda_path} and Environment Path {settings.nnUnet_env_path}")
    logger.info(f"  Environment: {settings.nnUnet_env}")
    logger.info(f"  Base path: {settings.basepath}\n")
    logger.info(f"  Dendrite volume set to: {p.min_dendrite_vol} µm, {settings.min_dendrite_vol} voxels")
    logger.info(f"  Spine volume set to: {p.spine_vol[0]} to {p.spine_vol[1]} µm³, {settings.neuron_spine_size[0]} to {settings.neuron_spine_size[1]} voxels")
    logger.info(f"  Spine distance filter set to: {p.spine_dist} µm, {settings.neuron_spine_dist} pixels")
    logger.info(f"  Analysis method: {p.analysis_method}")
    #logger.info(f" Second pass enabled: {p.second_pass}")
    logger.info(f"  GPU block size set to: {settings.GPU_block_size[0]},{settings.GPU_block_size[1]},{settings.GPU_block_size[1]}")
    logger.info(f"  Tracking set to: {settings.Track}, using {settings.reg_method} registration.")
    #logger.info(f"  Histogram matching set to: {settings.HistMatch}")
    logger.info(f"  Image restoration set to {p.image_restore} and axial restoration set to {p.axial_restore}")
    logger.info(f"  Neck generation {p.neck_generation}")
    logger.info(f"  Dask parallelization set to: {p.dask_enabled}")
    # log dask block and halo
    if p.dask_enabled:
        logger.info(f"  Dask block size set to: {settings.dask_block[0]},{settings.dask_block[1]},{settings.dask_block[2]}")
        logger.info(f"  Dask halo size set to: {settings.dask_halo[0]},{settings.dask_halo[1]},{settings.dask_halo[2]}")
    if settings.patch_for_nnunet:
        logger.info(f"  nnUNet patching enabled.Patch size set to: {settings.nnunet_patch_size[0]},{settings.nnunet_patch_size[1]},{settings.nnunet_patch_size[2]}")
    else:
        logger.info(f"  nnUNet patching disabled.")


class AnalysisWorker(QThread):
    task_done = pyqtSignal(str)


    def __init__(self, RESPAN, directory, model_dir, neck_generation, save_intermediate, dask_enabled, nnUNet_patching, save_validation,
                                     inputxy, inputz, modelxy, modelz, neuron_ch, analysis_method,
                                     image_restore, axial_restore, swc,
                                     min_dendrite_vol, spine_vol, spine_dist, #HistMatch,
                                     Track, reg_method, use_yaml_res, second_pass, model_type, logger,
                                     spurious_length_min_um=1.0,
                                     spurious_nnunet_support_min=0.2,
                                     spurious_keep_head_if_flagged=False,
                                     drop_spurious_necks=True,
                                     recover_filopodia=False,
                                     dendrite_repair=False,
                                     dendrite_repair_max_dist_um=2.0,
                                     detect_multi_head_spines=False,
                                     keep_partial_spines=True):


        super().__init__()
        self.RESPAN = RESPAN
        self.directory = directory
        self.model_dir = model_dir
        self.neck_generation = neck_generation
        self.recover_filopodia = recover_filopodia
        self.dendrite_repair = dendrite_repair
        self.dendrite_repair_max_dist_um = dendrite_repair_max_dist_um
        self.detect_multi_head_spines = detect_multi_head_spines
        self.keep_partial_spines = keep_partial_spines
        self.save_intermediate = save_intermediate
        self.dask_enabled = dask_enabled
        self.nnUNet_patching = nnUNet_patching
        self.save_validation = save_validation
        self.image_restore = image_restore
        self.axial_restore = axial_restore
        self.swc = swc
        self.inputxy = inputxy
        self.inputz = inputz
        self.modelxy = modelxy
        self.modelz = modelz
        self.neuron_ch = neuron_ch
        self.analysis_method = analysis_method
        self.min_dendrite_vol = min_dendrite_vol
        self.spine_vol = spine_vol
        self.spine_dist = spine_dist
        #self.HistMatch = HistMatch
        self.Track = Track
        self.reg_method = reg_method
        self.logger = logger
        self.use_yaml_res = use_yaml_res
        #self.second_pass = second_pass
        self.model_type = model_type
        # Spine thresholding — passed into settings before pipeline run
        self.spurious_length_min_um = spurious_length_min_um
        self.spurious_nnunet_support_min = spurious_nnunet_support_min
        self.spurious_keep_head_if_flagged = spurious_keep_head_if_flagged
        self.drop_spurious_necks = drop_spurious_necks

    def run(self):
        self.is_running = True

        try:

            if global_GUI_app == False:
                sys.path.append(self.RESPAN)

            from RESPAN.Environment import main, imgan, timer, strk, sr, io
            main.check_gpu()

            if self.directory == "No data directory selected." or not self.directory:
                self.logger.info("No directory has been selected. Please select a valid data directory first.")
                self.task_done.emit("No directory selected.")
                return
            if not os.path.exists(self.directory):
                self.logger.info(f"Directory does not exist: {self.directory}")
                self.task_done.emit("Directory does not exist.")
                return
            if len(os.listdir(self.directory)) == 0:
                self.logger.info(
                    "Selected directory contains no subfolders! \nPlease ensure your datasets are stored in subfolders nested within the selected directory.")
                self.task_done.emit("Empty directory.")
                return

            processed_count = 0
            skipped_subfolders = []

            for subfolder in os.listdir(self.directory):
                subfolder_path = os.path.join(self.directory, subfolder)
                if os.path.isdir(subfolder_path):
                    subfolder_path = subfolder_path +"/"

                    # Check for Analysis_Settings.yaml before attempting to load
                    settings_path = os.path.join(subfolder_path, "Analysis_Settings.yaml")
                    if not os.path.exists(settings_path):
                        self.logger.info(f"  Skipping '{subfolder}': No Analysis_Settings.yaml found.")
                        skipped_subfolders.append(subfolder)
                        continue

                    processed_count += 1
                    #Load in experiment parameters and analysis settings
                    settings, locations = main.initialize_RESPAN(subfolder_path)

                    #import RESPAN.Environment

                    log_path = subfolder_path +'RESPAN_Log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'

                    self.logger.set_log_file(log_path)

                    self.logger.clear_log(log_path)

                    self.logger.info("RESPAN Version: "+__version__)
                    self.logger.info("Release Date: "+__date__)
                    self.logger.info("Created by: "+__author__+"")
                    self.logger.info("Department of Neurology, The Ohio State University\n")

                    apply_run_parameters(settings, locations, self, self.logger)

                    #Processing
                    self.logger.info(f"\nProcessing folder: {subfolder_path}\n")

                    if os.path.exists(f'{global_env_path}/{global_respan_env}/'):

                        log = sr.restore_and_segment(settings, locations, self.logger)

                        if settings.Track == False and log == 0:
                            n_failed = imgan.analyze_spines(settings, locations, log, self.logger) or 0
                            if n_failed:
                                self.logger.error(
                                    f"{n_failed} image(s) FAILED in this folder. Check the errors above "
                                    f"before using any results from it.")

                        if settings.Track == True and log == 0:
                            strk.track_spines(settings, locations, log, self.logger)
                            imgan.analyze_spines_4D(settings, locations, log, self.logger)
                            self.logger.info(
                                "    *Please note that spine tracking does not currently provide all the metrics available when analyzing static volumes.")
                        if log != 0:
                            self.logger.info("Error with restoration and segmentation.")
                            self.logger.info(
                                "-----------------------------------------------------------------------------------------------------")
                        if log == 0:

                            self.logger.info("-----------------------------------------------------------------------------------------------------")
                            self.logger.info("RESPAN Version: "+__version__)
                            self.logger.info("Release Date: "+__date__+"")
                            self.logger.info("Created by: "+__author__+"")
                            self.logger.info("Department of Neurology, The Ohio State University")
                            self.logger.info("-----------------------------------------------------------------------------------------------------")
                            self.logger.info(respan_paper)
                            self.logger.info(nnUNet_paper)
                            if self.image_restore == True:
                                self.logger.info(care_paper)
                            if self.axial_restore == True:
                                self.logger.info(self_net_paper)
                            if settings.Track == True and log == 0:
                                self.logger.info(elastix_paper)

                            self.logger.info(
                                "-----------------------------------------------------------------------------------------------------")
                    else:
                        self.logger.info("Error: nnUNet environment not created or set correctly.")
                        self.logger.info(" Path set to:" + global_env_path + '/' + global_respan_env + '/')
                        self.logger.info(" Please check installation, or that paths set correctly in Analysis_Settings.yaml")

            if processed_count == 0:
                self.logger.info("\nNo valid dataset subfolders were found in the selected directory.")
                self.logger.info("RESPAN expects each dataset to be in its own subfolder, containing an Analysis_Settings.yaml file.")
                self.logger.info("\nExpected structure:")
                self.logger.info("  selected_directory/")
                self.logger.info("    dataset_1/")
                self.logger.info("      Analysis_Settings.yaml")
                self.logger.info("      [image files]")

            if skipped_subfolders:
                self.logger.info(f"\nSkipped {len(skipped_subfolders)} subfolder(s) due to missing Analysis_Settings.yaml:")
                for name in skipped_subfolders:
                    self.logger.info(f"  - {name}")

            self.task_done.emit("")
            self.is_running = False

        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")


class RESPANValidation(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)


        self.logger = Logger()

        # TitleFont
        titlefont = QFont()
        titlefont.setBold(True)
        titlefont.setPointSize(9)

        dir_options = QGroupBox("Ground Truth and Analysis Output Selection")
        dir_options.setStyleSheet("QGroupBox::title {"
                                        "subcontrol-origin: margin;"
                                        "subcontrol-position: top left;"
                                        "padding: 2px;"
                                        "color: black;"
                                        "}")
        dir_options.setFont(titlefont)
        dir_options_layout = QVBoxLayout()
        dir_options.setLayout(dir_options_layout)

        if global_GUI_app == False:
            self.RESPANdir_label = QLabel("No RESPAN directory selected.")
            self.RESPANdir_button = QPushButton("Select RESPAN directory")
            self.RESPANdir_button.clicked.connect(self.get_RESPANdir)

        self.ground_truth_dir_label = QLabel("No ground truth data directory selected.")
        self.ground_truth_dir_button = QPushButton("Select ground truth data directory")
        self.ground_truth_dir_button.clicked.connect(self.get_gtdir)

        self.directory_label = QLabel("No analysis output directory selected.")
        self.directory_button = QPushButton("Select data analysis output directory")
        self.directory_button.clicked.connect(self.get_outputdir)



        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        #self.info_label2 = QLabel("Ensure the correct Analysis_Settings.yaml file is included in the analysis output subfolder.")
        self.line2 = QFrame()
        self.line2.setFrameShape(QFrame.HLine)
        self.line2.setFrameShadow(QFrame.Sunken)

        if global_GUI_app == False:
            self.RESPANdir_button.setFixedWidth(300)

        self.directory_button.setFixedWidth(300)
        self.ground_truth_dir_button.setFixedWidth(300)

        if global_GUI_app == False:
            dir_options_layout.addWidget(self.RESPANdir_button)
            dir_options_layout.addWidget(self.RESPANdir_label)

        dir_options_layout.addWidget(self.directory_button)
        dir_options_layout.addWidget(self.directory_label)
        dir_options_layout.addWidget(self.ground_truth_dir_button)
        dir_options_layout.addWidget(self.ground_truth_dir_label)

        dir_options_layout.addWidget(self.line)

        #dir_options_layout.addWidget(self.info_label2)
        dir_options_layout.addWidget(self.line2)


        options_group1 = QGroupBox("Spine and Dendrite Detection")
        options_group1.setStyleSheet("QGroupBox::title {"
                                        "subcontrol-origin: margin;"
                                        "subcontrol-position: top left;"
                                        "padding: 2px;"
                                        "color: black;"
                                        "}")
        options_group1.setFont(titlefont)
        options_layout1 = QVBoxLayout()
        options_group1.setLayout(options_layout1)

        self.inputdata_xy_label = QLabel("Image voxel size XY (µm):")
        self.inputdata_xy = QLineEdit("0.102")
        self.inputdata_z_label = QLabel("Image voxel size Z (µm):")
        self.inputdata_z = QLineEdit("1")
        horizontal_input = QHBoxLayout()
        horizontal_input.addWidget(self.inputdata_xy_label)
        horizontal_input.addWidget(self.inputdata_xy)
        horizontal_input.addWidget(self.inputdata_z_label)
        horizontal_input.addWidget(self.inputdata_z)

        self.neuron_channel_label = QLabel("Channel containing neuron/dendrite signal:")
        self.neuron_channel_input = QLineEdit("1")
        self.float_label_1 = QLabel("Minimum dendrite size in µm (dendrites smaller than this will be ignored):")
        self.float_input_1 = QLineEdit("15")
        self.float_label_2 = QLabel("Spine volume filter (min, max volume in µm<sup>3</sup>):")
        self.float_input_2 = QLineEdit("0.03,15")
        self.float_label_3 = QLabel("Spine distance filter (max distance from dendrite in µm):")
        self.float_input_3 = QLineEdit("4")

        self.neuron_channel_input.setFixedWidth(90)
        self.float_input_1.setFixedWidth(90)
        self.float_input_2.setFixedWidth(90)
        self.float_input_3.setFixedWidth(90)

        options_layout1.addWidget(self.neuron_channel_label)
        options_layout1.addWidget(self.neuron_channel_input)
        options_layout1.addWidget(self.float_label_1)
        options_layout1.addWidget(self.float_input_1)
        options_layout1.addWidget(self.float_label_2)
        options_layout1.addWidget(self.float_input_2)
        options_layout1.addWidget(self.float_label_3)
        options_layout1.addWidget(self.float_input_3)
        options_layout1.addLayout(horizontal_input)
        options_layout1.setAlignment(Qt.AlignTop)

        options_group1.setFixedWidth(500)

        main_options = QHBoxLayout()
        main_options.addWidget(dir_options, alignment=Qt.AlignTop)
        main_options.addWidget(options_group1, alignment=Qt.AlignTop)


        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.get_variables)
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.run_button.setStyleSheet("background-color: lightblue;")
        #self.cancel_button.setStyleSheet("background-color: lightcoral;")

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        #FINAL Layout

        layout.addLayout(main_options)
        layout.addLayout(run_cancel_layout)

        layout.addWidget(self.progress)

        #layout.addWidget(self.logger.log_display)

        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)


        self.logger.qt_handler.log_generated.connect(self.update_log_display)
        self.setLayout(layout)

        try:
            variables_dict = _load_settings('parametesrvalGUI.pkl')
            if variables_dict is None:
                raise FileNotFoundError('parametesrvalGUI.pkl')  # handled below exactly as before

            #retreive
            if global_GUI_app == False:
                spine_dir = variables_dict.get('spine_dir', None)

            data_dir = variables_dict.get('data_dir', None)
            gt_dir = variables_dict.get('gt_dir', None)

            inputxy = variables_dict.get('inputxy', None)
            inputz = variables_dict.get('inputz', None)

            neuron_ch = variables_dict.get('neuron_ch', None)
            min_dend = variables_dict.get('min_dend', None)
            spine_vol = variables_dict.get('spine_vol', None)
            spine_dist = variables_dict.get('spine_dist', None)



            #udpate GUI:

            self.directory_label.setText(f"Selected directory: {data_dir}")
            self.ground_truth_dir_label.setText(f"Selected directory: {gt_dir}")

            if global_GUI_app == False:
                self.RESPANdir_label.setText(f"Selected directory: {spine_dir}")

            self.inputdata_xy.setText(str(inputxy))
            self.inputdata_z.setText(str(inputz))

            self.neuron_channel_input.setText(str(neuron_ch))
            self.float_input_1.setText(str(min_dend))
            self.float_input_2.setText(str(spine_vol))
            self.float_input_3.setText(str(spine_dist))



        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the pickle file, just ignore the error

    @pyqtSlot()
    def get_RESPANdir(self):
         RESPANdir = QFileDialog.getExistingDirectory(self, 'Select RESPAN directory')
         if RESPANdir:
             self.RESPANdir_label.setText(f"Selected directory: {RESPANdir}")

    @pyqtSlot()
    def get_outputdir(self):
         datadir = QFileDialog.getExistingDirectory(self, 'Select data analysis output directory')
         if datadir:
             self.directory_label.setText(f"Selected directory: {datadir}")

    @pyqtSlot()
    def get_gtdir(self):
         gtdir = QFileDialog.getExistingDirectory(self, 'Select ground truth data directory')
         if gtdir:
             self.ground_truth_dir_label.setText(f"Selected directory: {gtdir}")

    @pyqtSlot()
    def get_variables(self):


        dirlabel_text = self.directory_label.text()
        gtlabel_text = self.ground_truth_dir_label.text()

        if global_GUI_app == False:
            spinedir_text = self.RESPANdir_label.text()
            spine_dir = spinedir_text.split(": ")[-1]
        else:
            spine_dir = "none"

        data_dir = dirlabel_text.split(": ")[-1]
        gt_dir = gtlabel_text.split(": ")[-1]



        inputxy = str(self.inputdata_xy.text())
        inputz = str(self.inputdata_z.text())

        neuron_ch = str(self.neuron_channel_input.text())
        min_dend = str(self.float_input_1.text())
        spine_vol = str(self.float_input_2.text())
        spine_dist = str(self.float_input_3.text())


        variables_dict = {
            'spine_dir': spine_dir,
            'data_dir': data_dir,
            'gt_dir': gt_dir,

            'inputxy': inputxy,
            'inputz': inputz,

            'neuron_ch': neuron_ch,
            'min_dend': min_dend,
            'spine_vol': spine_vol,
            'spine_dist': spine_dist,


        }


        # Save the dictionary to a pickle file
        _save_settings('parametesrvalGUI.pkl', variables_dict, getattr(self, 'log_display', None))


    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode


        if global_GUI_app == False:
            RESPAN = self.RESPANdir_label.text().replace("Selected directory: ", "")
            if RESPAN == "No RESPAN directory selected.":
                QMessageBox.critical(self, "Error", "No directory selected.")
                self.progress.setVisible(False)
                return


        ground_truth = self.ground_truth_dir_label.text().replace("Selected directory: ", "")
        if ground_truth == "No ground truth directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return


        directory = self.directory_label.text().replace("Selected directory: ", "")
        if directory == "No analysis output directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return

        try:
            min_dendrite_vol =  float(self.float_input_1.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for minimum dendrite volume.")
            self.progress.setVisible(False)
            return

        try:
            spine_vol = list(map(float, self.float_input_2.text().split(',')))
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for spine volume.")
            self.progress.setVisible(False)
            return

        try:
            spine_dist =  float(self.float_input_3.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for spine distance to dendrite.")
            self.progress.setVisible(False)
            return
        try:
            inputxy =  float(self.inputdata_xy.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for image xy.")
            self.progress.setVisible(False)
            return
        try:
            inputz =  float(self.inputdata_z.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for image z.")
            self.progress.setVisible(False)
            return
        try:
            neuron_ch =  float(self.neuron_channel_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for neuron channel.")
            self.progress.setVisible(False)
            return

        if global_GUI_app == False:
            RESPAN = RESPAN +"/"
        else:
            RESPAN = "none"

        ground_truth = ground_truth +"/"

        analysis_output =  directory + "/"


        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        #self.worker = Worker(RESPAN, directory, channel_options, integers, self.logger.get_logger())
        #self.worker = Worker(RESPAN, ground_truth, analysis_output, self.logger)
        self.worker = ValidationWorker(RESPAN, ground_truth, analysis_output, neuron_ch, min_dendrite_vol, spine_vol, spine_dist, inputxy, inputz, self.logger)
        self.worker.task_done.connect(self.on_task_done)
        self.worker.start()


    @pyqtSlot(str)
    def update_log_display(self, message):
        self.log_display.append(message)
        self.log_display.moveCursor(QTextCursor.End)


    @pyqtSlot(str)
    def on_task_done(self, message):
        self.logger.info(message)
        self.progress.setValue(self.progress.maximum())
        self.progress.setVisible(False)
        # Disconnect the signals when done
        #self.worker.task_done.disconnect(self.on_task_done)
        #self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


class ValidationWorker(QThread):
    task_done = pyqtSignal(str)

    #def __init__(self, RESPAN, ground_truth, analysis_output, logger):
    def __init__(self, RESPAN, ground_truth, analysis_output, neuron_ch, min_dendrite_vol, spine_vol, spine_dist, inputxy, inputz, logger):
        super().__init__()
        self.RESPAN = RESPAN
        self.ground_truth = ground_truth
        self.analysis_output = analysis_output
        self.neuron_ch = neuron_ch
        self.min_dendrite_vol = min_dendrite_vol
        self.spine_vol = spine_vol
        self.spine_dist = spine_dist
        self.logger = logger
        self.inputxy = inputxy
        self.inputz = inputz

    def run(self):
        self.is_running = True

        try:

            if global_GUI_app == False:
                sys.path.append(self.RESPAN)
            from RESPAN.Environment import main, val
            main.check_gpu()
            #Load in experiment parameters and analysis settings
            #settings, locations = main.initialize_RESPAN(self.analysis_output)

            import RESPAN.Environment

            log_path = self.analysis_output +'RESPAN_Validation_Log' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            self.logger.set_log_file(log_path)  # Note: we're not passing a log_display, because it's not thread-safe


            self.logger.info("RESPAN Validation Tool - Version: "+__version__)
            self.logger.info("Release Date: "+__date__)
            self.logger.info("Created by: "+__author__+"")
            self.logger.info("Department of Neurology, The Ohio State University")


            #Load in experiment parameters and analysis settings
            #self.analysis_output is two directories deeper than the analysis yml

            main_dir = Path(self.analysis_output).resolve().parents[1]

            settings, locations = main.initialize_RESPAN_validation(main_dir)


            settings.input_resXY = self.inputxy
            settings.input_resZ = self.inputz
            settings.neuron_channel = self.neuron_ch
            settings.min_dendrite_vol = round(self.min_dendrite_vol / settings.input_resXY/settings.input_resXY/settings.input_resZ, 0)
            settings.neuron_spine_size = [round(x / (settings.input_resXY*settings.input_resXY*settings.input_resZ),0) for x in self.spine_vol]
            settings.neuron_spine_dist = round(self.spine_dist / (settings.input_resXY),2)
            settings.Track = False

            self.logger.info("RESPAN Validation Parameters:")
            self.logger.info(f" Image resolution: {settings.input_resXY}um XY, {settings.input_resZ}um Z")
            self.logger.info(f" Spine volume set to: {self.spine_vol[0]} to {self.spine_vol[1]} um<sup>3</sup>, {settings.neuron_spine_size[0]} to {settings.neuron_spine_size[1]} voxels.")
            self.logger.info(f" Spine distance filter set to: {self.spine_dist} um, {settings.neuron_spine_dist} pixels")
            self.logger.info(f" GPU block size set to: {settings.GPU_block_size[0]},{settings.GPU_block_size[1]},{settings.GPU_block_size[1]}")
            self.logger.info("")


            #Processing
            self.logger.info("Processing folder: "+self.analysis_output)
            val.validate_analysis(self.ground_truth, self.analysis_output, settings, locations, self.logger)



            self.logger.info("-----------------------------------------------------------------------------------------------------")
            self.logger.info("RESPAN Validation Tool - Version: "+__version__)
            self.logger.info("Release Date: "+__date__+"")
            self.logger.info("Created by: "+__author__+"")
            self.logger.info("Department of Neurology, The Ohio State University")
            self.logger.info("-----------------------------------------------------------------------------------------------------")

            self.task_done.emit("")

            self.is_running = False

        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")


class UNet(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.logger = Logger()

        # TitleFont
        titlefont = QFont()
        titlefont.setBold(True)
        titlefont.setPointSize(9)

        dir_options = QGroupBox("3D nnU-Net Training Data and Model Output Selection")
        dir_options.setStyleSheet("QGroupBox::title {"
                                  "subcontrol-origin: margin;"
                                  "subcontrol-position: top left;"
                                  "padding: 2px;"
                                  "color: black;"
                                  "}")
        dir_options.setFont(titlefont)
        dir_options_layout = QVBoxLayout()
        dir_options.setLayout(dir_options_layout)

        if global_GUI_app == False:
            self.RESPANdir_label = QLabel("No RESPAN directory selected.")
            self.RESPANdir_button = QPushButton("Select RESPAN directory")
            self.RESPANdir_button.clicked.connect(self.get_RESPANdir)

        self.ground_truth_dir_label = QLabel("No training data directory selected.")
        self.ground_truth_dir_button = QPushButton("Select training data directory")
        self.ground_truth_dir_button.clicked.connect(self.get_gtdir)

        self.directory_label = QLabel("No model output directory selected.")
        self.directory_button = QPushButton("Select model output directory")
        self.directory_button.clicked.connect(self.get_outputdir)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.info_label2 = QLabel("nnUNet input folder should contain imagesTr and labelsTr subfolders - refer to guide for more information.")
        self.line2 = QFrame()
        self.line2.setFrameShape(QFrame.HLine)
        self.line2.setFrameShadow(QFrame.Sunken)


        if global_GUI_app == False:
            self.RESPANdir_button.setFixedWidth(300)
        self.directory_button.setFixedWidth(300)
        self.ground_truth_dir_button.setFixedWidth(300)

        if global_GUI_app == False:
            dir_options_layout.addWidget(self.RESPANdir_button)
            dir_options_layout.addWidget(self.RESPANdir_label)
        dir_options_layout.addWidget(self.ground_truth_dir_button)
        dir_options_layout.addWidget(self.ground_truth_dir_label)
        dir_options_layout.addWidget(self.directory_button)
        dir_options_layout.addWidget(self.directory_label)

        dir_options_layout.addWidget(self.line)

        dir_options_layout.addWidget(self.info_label2)
        dir_options_layout.addWidget(self.line2)

        options_group1 = QGroupBox("Additional Parameters")
        options_group1.setStyleSheet("QGroupBox::title {"
                                     "subcontrol-origin: margin;"
                                     "subcontrol-position: top left;"
                                     "padding: 2px;"
                                     "color: black;"
                                     "}")
        options_group1.setFont(titlefont)
        options_layout1 = QVBoxLayout()
        options_group1.setLayout(options_layout1)

        # Dataset ID
        self.datasetID_label = QLabel("Dataset ID:")
        self.datasetID_input = QLineEdit()
        self.datasetID_input.setText('100')

        # Ignore
        self.ignore_label = QLabel("Use ignore label during training:")
        self.ignore_input = QCheckBox()
        self.ignore_input.setChecked(False)

        horizontal_input1 = QHBoxLayout()
        horizontal_input1.addWidget(self.datasetID_label)
        horizontal_input1.addWidget(self.datasetID_input)
        horizontal_input1.addWidget(self.ignore_label)
        horizontal_input1.addWidget(self.ignore_input)
        horizontal_input1.addStretch(1)


        options_layout1.addLayout(horizontal_input1)
        #options_layout1.addLayout(horizontal_input2)

        # options_group1.setFixedWidth(600)
        input_data_and_opt = QVBoxLayout()
        dir_options.setFixedWidth(800)
        options_group1.setFixedWidth(800)
        input_data_and_opt.addWidget(dir_options)
        input_data_and_opt.addWidget(options_group1)

        main_options = QHBoxLayout()
        main_options.addLayout(input_data_and_opt)
        main_options.addStretch(1)# , alignment=Qt.AlignTop)
        #main_options.addWidget(options_group2)  # , alignment=Qt.AlignLeft)
        # main_options.addWidget(options_group2, alignment=Qt.AlignTop)

        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.get_variables)
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.run_button.setStyleSheet("background-color: lightblue;")
        # self.cancel_button.setStyleSheet("background-color: lightcoral;")

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # FINAL Layout
        layout.addLayout(main_options)
        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)
        # layout.addWidget(self.logger.log_display)

        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

        self.logger.qt_handler.log_generated.connect(self.update_log_display)
        self.setLayout(layout)

        try:
            variables_dict = _load_settings('nnUNETparametersGUI.pkl')
            if variables_dict is None:
                raise FileNotFoundError('nnUNETparametersGUI.pkl')  # handled below exactly as before

            # Retrieve
            if global_GUI_app == False:
                respan_dir = variables_dict.get('respan_dir', None)

            gt_dir = variables_dict.get('gt_dir', None)
            data_dir = variables_dict.get('data_dir', None)

            datasetID = variables_dict.get('datasetID', None)
            ignore = variables_dict.get('ignore', None)

            # Update GUI
            if global_GUI_app == False:
                self.RESPANdir_label.setText(f"Selected directory: {respan_dir}")

            self.directory_label.setText(f"Selected directory: {data_dir}")
            self.ground_truth_dir_label.setText(f"Selected directory: {gt_dir}")

            self.datasetID_input.setText(datasetID)
            self.ignore_input.setChecked(ignore)


        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the pickle file, just ignore the error

    @pyqtSlot()
    def get_RESPANdir(self):
        RESPANdir = QFileDialog.getExistingDirectory(self, 'Select RESPAN directory')
        if RESPANdir:
            self.RESPANdir_label.setText(f"Selected directory: {RESPANdir}")

    @pyqtSlot()
    def get_gtdir(self):

        gtdir = QFileDialog.getExistingDirectory(self, 'Select training data directory')
        if gtdir:
            self.ground_truth_dir_label.setText(f"Selected directory: {gtdir}")

    @pyqtSlot()
    def get_outputdir(self):
        datadir = QFileDialog.getExistingDirectory(self, 'Select model output directory')
        if datadir:
            self.directory_label.setText(f"Selected directory: {datadir}")


    @pyqtSlot()
    def get_variables(self):
        try:
            if global_GUI_app == False:
                respanlabel_text = self.RESPANdir_label.text()
                respan_dir = respanlabel_text.split(": ")[-1]
            else:
                respan_dir = "none"

            gtlabel_text = self.ground_truth_dir_label.text()
            dirlabel_text = self.directory_label.text()


            gt_dir = gtlabel_text.split(": ")[-1]
            data_dir = dirlabel_text.split(": ")[-1]

            datasetID = str(self.datasetID_input.text())
            ignore = self.ignore_input.isChecked()


            variables_dict = {
                'respan_dir': respan_dir,
                'gt_dir': gt_dir,
                'data_dir': data_dir,

                'datasetID': datasetID,
                'ignore': ignore,

            }

            # Save the dictionary to a pickle file
            _save_settings('nnUNETparametersGUI.pkl', variables_dict, getattr(self, 'log_display', None))
        except Exception as e:
            self.log_display.append(f"An error occurred: {e}")


    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode


        if global_GUI_app == False:

            RESPAN = self.RESPANdir_label.text().replace("Selected directory: ", "")
            if RESPAN == "No RESPAN directory selected.":
                QMessageBox.critical(self, "Error", "No directory selected.")
                self.progress.setVisible(False)
                return

        ground_truth = self.ground_truth_dir_label.text().replace("Selected directory: ", "")
        if ground_truth == "No training data directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return

        directory = self.directory_label.text().replace("Selected directory: ", "")
        if directory == "No model output directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return


        try:
            ignore = self.ignore_input.isChecked()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for augmentation.")
            self.progress.setVisible(False)
            return

        try:
            datasetID = int(self.datasetID_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for datasetID.")
            self.progress.setVisible(False)
            return

        input_dir = ground_truth + "/"
        model_output = directory + "/"
        if global_GUI_app == False:
            RESPAN = RESPAN + "/"
        else:
            RESPAN = "none"

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        self.worker = UNetWorker(RESPAN, input_dir, model_output, datasetID, ignore,
                                 self.logger)
        self.worker.task_done.connect(self.on_task_done)
        self.worker.start()

    @pyqtSlot(str)
    def update_log_display(self, message):
        self.log_display.append(message)
        self.log_display.moveCursor(QTextCursor.End)

    @pyqtSlot(str)
    def on_task_done(self, message):
        self.logger.info(message)
        self.progress.setValue(self.progress.maximum())
        self.progress.setVisible(False)
        # Disconnect the signals when done
        # self.worker.task_done.disconnect(self.on_task_done)
        # self.logger.qt_handler.log_generated.disconnect(self.update_log_display)

class UNetWorker(QThread):
    task_done = pyqtSignal(str)

    # def __init__(self, RESPAN, ground_truth, analysis_output, logger):
    def __init__(self, RESPAN, input_dir, model_output, datasetID, ignore,
                                  logger):
        super().__init__()
        self.RESPAN = RESPAN
        self.input_dir = input_dir
        self.model_output = model_output
        self.datasetID = datasetID
        self.ignore = ignore

        self.logger = logger

    def run(self):
        self.is_running = True

        try:

            if global_GUI_app == False:
                sys.path.append(self.RESPAN)

            from RESPAN.Environment import main, mt
            main.check_gpu()
            # Load in experiment parameters and analysis settings
            # settings, locations = main.initialize_RESPAN(self.analysis_output)



            log_path = self.input_dir + 'nnU-Net_Training_Log' + datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S") + '.log'
            self.logger.set_log_file(log_path)  # Note: we're not passing a log_display, because it's not thread-safe

            self.logger.info("nnU-Net Training GUI | RESPAN Version: " + __version__)
            self.logger.info("Release Date: " + __date__)
            self.logger.info("Created by: " + __author__ + "")
            self.logger.info("Department of Neurology, The Ohio State University")
            self.logger.info(
                "-----------------------------------------------------------------------------------------------------")
            self.logger.info(nnUNet_paper)
            self.logger.info(
                "-----------------------------------------------------------------------------------------------------")
            # Load in experiment parameters and analysis settings
            #settings, locations = main.initialize_RESPAN_validation(self.analysis_output)

            # generate additional variables
            parent_dir = os.path.dirname(os.path.dirname(self.input_dir))
            # make sub dir "nnunet_preprocessed" in parent dir
            preprocessed_folder = os.path.join(parent_dir, 'nnunet_preprocessed')
            if os.path.exists(global_env_path + '/' + global_respan_env + '/'):
                os.makedirs(preprocessed_folder, exist_ok=True)

                self.logger.info("\nCreating JSON file for nnU-Net training...")

                # Generate JSON file
                json_data = mt.generate_nnunet_json(self.input_dir, use_ignore=self.ignore)

                self.logger.info(" JSON file created.\n")
                # Convert dictionary to pretty-printed string
                json_str = json.dumps(json_data, indent=4)

                # Create and display the dialog
                self.logger.info(" Displaying nnU-Net data information for confirmation:\n" + json_str)

                self.logger.info("\n")

                # initialize and run nnunet
                #self.logger.info("\n Training nnU-Net. Please allow 12-24 hours depending on available GPU resources...")

                mt.train_nnUNet(parent_dir, preprocessed_folder, self.model_output, self.datasetID, str(PY_EXE), clean_launcher, nnunet_plan_bat,
                                nnunet_train_bat , self.logger)


                self.logger.info(
                    "-----------------------------------------------------------------------------------------------------")
                self.logger.info("nnU-Net Training GUI | RESPAN Version: " + __version__)
                self.logger.info("Release Date: " + __date__ + "")
                self.logger.info("Created by: " + __author__ + "")
                self.logger.info("Department of Neurology, The Ohio State University")
                self.logger.info(
                    "-----------------------------------------------------------------------------------------------------")
                self.logger.info(respan_paper)
                self.logger.info(nnUNet_paper)
                self.logger.info(
                    "-----------------------------------------------------------------------------------------------------")
            else:
                    self.logger.info("Environment not found. Please check the path and environment name.")

            self.task_done.emit("")

            self.is_running = False

        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")

class CARE(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.logger = Logger()

        # TitleFont
        titlefont = QFont()
        titlefont.setBold(True)
        titlefont.setPointSize(9)

        dir_options = QGroupBox("CARE Training Data and Model Output Selection")
        dir_options.setStyleSheet("QGroupBox::title {"
                                  "subcontrol-origin: margin;"
                                  "subcontrol-position: top left;"
                                  "padding: 2px;"
                                  "color: black;"
                                  "}")
        dir_options.setFont(titlefont)
        dir_options_layout = QVBoxLayout()
        dir_options.setLayout(dir_options_layout)

        if global_GUI_app == False:
            self.RESPANdir_label = QLabel("No RESPAN directory selected.")
            self.RESPANdir_button = QPushButton("Select RESPAN directory")
            self.RESPANdir_button.clicked.connect(self.get_RESPANdir)

        self.ground_truth_dir_label = QLabel("No training data directory selected.")
        self.ground_truth_dir_button = QPushButton("Select select training data directory")
        self.ground_truth_dir_button.clicked.connect(self.get_gtdir)
        self.info_labeltraing = QLabel("Folder should contain subfolders lowSNR and highSNR3D contained matched datasets.")

        self.directory_label = QLabel("No model output directory selected.")
        self.directory_button = QPushButton("Select model output directory")
        self.directory_button.clicked.connect(self.get_outputdir)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.info_label2 = QLabel("Note: 3D images should have at least 8 z slices.")
        self.line2 = QFrame()
        self.line2.setFrameShape(QFrame.HLine)
        self.line2.setFrameShadow(QFrame.Sunken)

        if global_GUI_app == False:
            self.RESPANdir_button.setFixedWidth(300)
        self.directory_button.setFixedWidth(300)
        self.ground_truth_dir_button.setFixedWidth(300)

        if global_GUI_app == False:
            dir_options_layout.addWidget(self.RESPANdir_button)
            dir_options_layout.addWidget(self.RESPANdir_label)

        dir_options_layout.addWidget(self.ground_truth_dir_button)
        dir_options_layout.addWidget(self.ground_truth_dir_label)
        dir_options_layout.addWidget(self.directory_button)
        dir_options_layout.addWidget(self.directory_label)

        dir_options_layout.addWidget(self.line)

        dir_options_layout.addWidget(self.info_label2)
        dir_options_layout.addWidget(self.line2)

        options_group1 = QGroupBox("Model Training Options")
        options_group1.setStyleSheet("QGroupBox::title {"
                                     "subcontrol-origin: margin;"
                                     "subcontrol-position: top left;"
                                     "padding: 2px;"
                                     "color: black;"
                                     "}")
        options_group1.setFont(titlefont)
        options_layout1 = QVBoxLayout()
        options_group1.setLayout(options_layout1)

        # Model Name
        self.model_name_label = QLabel("Model Name:")
        self.model_name_input = QLineEdit()
        self.model_name_input.setText("example_model_name_dendrites_60x_150nmXY_300nmZ_60epochs")
        self.model_name_input.setFixedWidth(400)
        horizontal_input1 = QHBoxLayout()
        horizontal_input1.addWidget(self.model_name_label)
        horizontal_input1.addWidget(self.model_name_input)
        horizontal_input1.addStretch(1)


        # Augmentation Checkbox
        self.augmentation_label = QLabel("Augmentation:")
        self.augmentation_input = QCheckBox()
        horizontal_input = QHBoxLayout()
        self.augmentation_input.setChecked(True)
        horizontal_input.addWidget(self.augmentation_label)
        horizontal_input.addWidget(self.augmentation_input)


        # Dimension Selection Dropdown
        self.dimension_label = QLabel("Data type:")
        self.dimension_select = QComboBox()
        self.dimension_select.addItems(["2D", "3D"])
        self.dimension_select.setCurrentText("3D")
        horizontal_input.addWidget(self.dimension_label)
        horizontal_input.addWidget(self.dimension_select)
        horizontal_input.addStretch(1)

        options_layout1.addLayout(horizontal_input1)
        options_layout1.addLayout(horizontal_input)

        #options_group1.setFixedWidth(600)
        input_data_and_opt = QVBoxLayout()
        dir_options.setFixedWidth(600)
        options_group1.setFixedWidth(600)
        input_data_and_opt.addWidget(dir_options)
        input_data_and_opt.addWidget(options_group1)

        # Additional Parameters
        self.patch_size_input = QLineEdit("8, 64, 64")
        self.patch_size_input.setFixedWidth(90)
        self.num_patches_per_img_input = QLineEdit("100")
        self.num_patches_per_img_input.setFixedWidth(90)
        self.epochs_input = QLineEdit("150")
        self.epochs_input.setFixedWidth(90)
        self.unet_kern_size_input = QLineEdit("3")
        self.unet_kern_size_input.setFixedWidth(90)
        self.unet_n_depth_input = QLineEdit("3")
        self.unet_n_depth_input.setFixedWidth(90)
        self.batch_size_input = QLineEdit("16")
        self.batch_size_input.setFixedWidth(90)
        self.steps_per_epoch_input = QLineEdit("400")
        self.steps_per_epoch_input.setFixedWidth(90)
        self.pct_validation_input = QLineEdit("0.10")
        self.pct_validation_input.setFixedWidth(90)


        options_group2 = QGroupBox("Additonal Options")
        options_group2.setStyleSheet("QGroupBox::title {"
                                     "subcontrol-origin: margin;"
                                     "subcontrol-position: top left;"
                                     "padding: 2px;"
                                     "color: black;"
                                     "}")
        options_group2.setFont(titlefont)
        options_layout2 = QVBoxLayout()
        options_group2.setLayout(options_layout2)
        #additional_params_layout = QVBoxLayout()

        for label, widget in [
            ("Patch Size:", self.patch_size_input),
            ("Num Patches per Image:", self.num_patches_per_img_input),
            ("Epochs:", self.epochs_input),
            ("UNet Kernel Size:", self.unet_kern_size_input),
            ("UNet Depth:", self.unet_n_depth_input),
            ("Batch Size:", self.batch_size_input),
            ("Steps per Epoch:", self.steps_per_epoch_input),
            ("Percent Validation:", self.pct_validation_input)]:

            #form_row = QFormLayout()
            form_row = QHBoxLayout()
            label = QLabel(label)
            form_row.addWidget(label)
            form_row.addWidget(widget)
            #form_row.addRow(QLabel(label), widget)
            options_layout2.addLayout(form_row)

        options_group2.setFixedWidth(300)

        main_options = QHBoxLayout()
        main_options.addLayout(input_data_and_opt)#, alignment=Qt.AlignTop)
        main_options.addWidget(options_group2) #, alignment=Qt.AlignLeft)
        #main_options.addWidget(options_group2, alignment=Qt.AlignTop)
        main_options.addStretch(1)

        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.get_variables)
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.run_button.setStyleSheet("background-color: lightblue;")
        # self.cancel_button.setStyleSheet("background-color: lightcoral;")

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # FINAL Layout

        # layout.addLayout(input_data_and_res)

        layout.addLayout(main_options)
        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)

        # layout.addWidget(self.logger.log_display)

        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

        self.logger.qt_handler.log_generated.connect(self.update_log_display)
        self.setLayout(layout)

        try:
            variables_dict = _load_settings('CAREparametersGUI.pkl')
            if variables_dict is None:
                raise FileNotFoundError('CAREparametersGUI.pkl')  # handled below exactly as before

            # Retrieve
            if global_GUI_app == False:
                respan_dir = variables_dict.get('respan_dir', None)
            gt_dir = variables_dict.get('gt_dir', None)
            data_dir = variables_dict.get('data_dir', None)

            patch_size = variables_dict.get('patch_size', None)
            num_patches_per_img = variables_dict.get('num_patches_per_img', None)
            epochs = variables_dict.get('epochs', None)
            unet_kern_size = variables_dict.get('unet_kern_size', None)
            unet_n_depth = variables_dict.get('unet_n_depth', None)
            batch_size = variables_dict.get('batch_size', None)
            steps_per_epoch = variables_dict.get('steps_per_epoch', None)
            pct_validation = variables_dict.get('pct_validation', None)

            model_name = variables_dict.get('model_name', None)
            augmentation = variables_dict.get('augmentation', None)
            dimension = variables_dict.get('dimension', None)

            # Update GUI
            if global_GUI_app == False:
                self.RESPANdir_label.setText(f"Selected directory: {respan_dir}")
            self.directory_label.setText(f"Selected directory: {data_dir}")
            self.ground_truth_dir_label.setText(f"Selected directory: {gt_dir}")

            self.patch_size_input.setText(str(patch_size))
            self.num_patches_per_img_input.setText(str(num_patches_per_img))
            self.epochs_input.setText(str(epochs))
            self.unet_kern_size_input.setText(str(unet_kern_size))
            self.unet_n_depth_input.setText(str(unet_n_depth))
            self.batch_size_input.setText(str(batch_size))
            self.steps_per_epoch_input.setText(str(steps_per_epoch))
            self.pct_validation_input.setText(str(pct_validation))

            self.model_name_input.setText(model_name)
            self.augmentation_input.setChecked(augmentation)
            self.dimension_select.setCurrentText(dimension)

        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the pickle file, just ignore the error

    @pyqtSlot()
    def get_RESPANdir(self):
        RESPANdir = QFileDialog.getExistingDirectory(self, 'Select RESPAN directory')
        if RESPANdir:
            self.RESPANdir_label.setText(f"Selected directory: {RESPANdir}")


    @pyqtSlot()
    def get_gtdir(self):
        gtdir = QFileDialog.getExistingDirectory(self, 'Select ground truth data directory')
        if gtdir:
            self.ground_truth_dir_label.setText(f"Selected directory: {gtdir}")

    @pyqtSlot()
    def get_outputdir(self):
        datadir = QFileDialog.getExistingDirectory(self, 'Select data directory')
        if datadir:
            self.directory_label.setText(f"Selected directory: {datadir}")

    @pyqtSlot()
    def get_variables(self):

        if global_GUI_app == False:
            respanlabel_text = self.RESPANdir_label.text()
            respan_dir = respanlabel_text.split(": ")[-1]
        else:
            respan_dir = "none"

        gtlabel_text = self.ground_truth_dir_label.text()
        dirlabel_text = self.directory_label.text()

        gt_dir = gtlabel_text.split(": ")[-1]
        data_dir = dirlabel_text.split(": ")[-1]

        patch_size = str(self.patch_size_input.text())
        num_patches_per_img = str(self.num_patches_per_img_input.text())
        epochs = str(self.epochs_input.text())
        unet_kern_size = str(self.unet_kern_size_input.text())
        unet_n_depth = str(self.unet_n_depth_input.text())
        batch_size = str(self.batch_size_input.text())
        steps_per_epoch = str(self.steps_per_epoch_input.text())
        pct_validation = str(self.pct_validation_input.text())

        model_name = str(self.model_name_input.text())
        augmentation = self.augmentation_input.isChecked()
        dimension = str(self.dimension_select.currentText())

        variables_dict = {
            'respan_dir': respan_dir,
            'gt_dir': gt_dir,
            'data_dir': data_dir,

            'patch_size': patch_size,
            'num_patches_per_img': num_patches_per_img,
            'epochs': epochs,
            'unet_kern_size': unet_kern_size,
            'unet_n_depth': unet_n_depth,
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch,
            'pct_validation': pct_validation,

            'model_name': model_name,
            'augmentation': augmentation,
            'dimension': dimension
        }

        # Save the dictionary to a pickle file
        _save_settings('CAREparametersGUI.pkl', variables_dict, getattr(self, 'log_display', None))

    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode

        # RESPAN = self.RESPANdir_label.text().replace("Selected directory: ", "")
        # if RESPAN == "No RESPAN directory selected.":
        #    QMessageBox.critical(self, "Error", "No directory selected.")
        #    self.progress.setVisible(False)
        #    return

        if global_GUI_app == False:
            RESPAN = self.RESPANdir_label.text().replace("Selected directory: ", "")
            if RESPAN == "No directory selected.":
                QMessageBox.critical(self, "Error", "No directory selected.")
                self.progress.setVisible(False)
                return

        ground_truth = self.ground_truth_dir_label.text().replace("Selected directory: ", "")
        if ground_truth == "No ground truth directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return

        directory = self.directory_label.text().replace("Selected directory: ", "")
        if directory == "No analysis output directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return

        try:
            model_name = str(self.model_name_input.text())
            if not model_name:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for model name.")
            self.progress.setVisible(False)
            return

        try:
            augmentation = self.augmentation_input.isChecked()
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for augmentation.")
            self.progress.setVisible(False)
            return

        try:
            data_type = str(self.dimension_select.currentText())
            if data_type not in ["2D", "3D"]:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for data type.")
            self.progress.setVisible(False)
            return

        try:
            patch_size = list(map(int, self.patch_size_input.text().split(',')))
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for patch size.")
            self.progress.setVisible(False)
            return

        try:
            num_patches_per_img = int(self.num_patches_per_img_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for number of patches per image.")
            self.progress.setVisible(False)
            return

        try:
            epochs = int(self.epochs_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for epochs.")
            self.progress.setVisible(False)
            return

        try:
            unet_kern_size = int(self.unet_kern_size_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for UNet kernel size.")
            self.progress.setVisible(False)
            return

        try:
            unet_n_depth = int(self.unet_n_depth_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for UNet depth.")
            self.progress.setVisible(False)
            return

        try:
            batch_size = int(self.batch_size_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for batch size.")
            self.progress.setVisible(False)
            return

        try:
            steps_per_epoch = int(self.steps_per_epoch_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for steps per epoch.")
            self.progress.setVisible(False)
            return

        try:
            pct_validation = float(self.pct_validation_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for percent validation.")
            self.progress.setVisible(False)
            return

        input_dir = ground_truth + "/"

        model_output = directory + "/"

        if global_GUI_app == False:
            RESPAN = RESPAN + "/"
        else:
            RESPAN = "none"

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)

        # self.worker = Worker(RESPAN, directory, channel_options, integers, self.logger.get_logger())
        # self.worker = Worker(RESPAN, ground_truth, analysis_output, self.logger)
        self.worker = CAREWorker(RESPAN, input_dir, model_output, model_name, augmentation, data_type,
                                 patch_size, num_patches_per_img, epochs, unet_kern_size, unet_n_depth, batch_size,
                                 steps_per_epoch, pct_validation,  self.logger)
        self.worker.task_done.connect(self.on_task_done)
        self.worker.start()

    @pyqtSlot(str)
    def update_log_display(self, message):
        self.log_display.append(message)
        self.log_display.moveCursor(QTextCursor.End)

    @pyqtSlot(str)
    def on_task_done(self, message):
        self.logger.info(message)
        self.progress.setValue(self.progress.maximum())
        self.progress.setVisible(False)
        # Disconnect the signals when done
        # self.worker.task_done.disconnect(self.on_task_done)
        # self.logger.qt_handler.log_generated.disconnect(self.update_log_display)

class CAREWorker(QThread):
    task_done = pyqtSignal(str)

    def __init__(self, RESPAN, input_dir, model_output, model_name, augmentation, data_type,
                                 patch_size, num_patches_per_img, epochs, unet_kern_size, unet_n_depth, batch_size,
                                 steps_per_epoch, pct_validation,  logger):

        super().__init__()

        self.RESPAN = RESPAN
        self.input_dir = input_dir
        self.model_output = model_output
        self.model_name = model_name
        self.augmentation = augmentation
        self.data_type = data_type
        self.patch_size = patch_size
        self.num_patches_per_img = num_patches_per_img
        self.epochs = epochs
        self.unet_kern_size = unet_kern_size
        self.unet_n_depth = unet_n_depth
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.pct_validation = pct_validation
        self.logger = logger

    def run(self):
        self.is_running = True

        try:

            if global_GUI_app == False:
                sys.path.append(self.RESPAN)
            from RESPAN.Environment import main, mt
            main.check_gpu()

            if not os.path.isdir(os.path.join(self.input_dir, "highSNR")) or not os.path.isdir(os.path.join(self.input_dir, "lowSNR")):
                self.logger.info(
                    "Selected training data folder is missing 'highSNR' or 'lowSNR' subfolders. \nPlease ensure both subfolders are present within the selected directory.\nRefer to guide as needed regarding data preparation.")

            else:

                # print(f"Processing subfolder: {subfolder_path}")
                # Load in experiment parameters and analysis settings
                #settings, locations = main.initialize_RESPAN(subfolder_path)

                # import RESPAN.Environment

                log_path = self.input_dir + 'RESPAN_CARE_Training_Log' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
                self.logger.set_log_file(log_path)

                self.logger.info("CARE Training GUI | RESPAN Version: " + __version__)
                self.logger.info("Release Date: " + __date__)
                self.logger.info("Created by: " + __author__ + "")
                self.logger.info("Department of Neurology, The Ohio State University")
                self.logger.info(
                    "-----------------------------------------------------------------------------------------------------")
                self.logger.info(
                    care_paper)
                self.logger.info(
                    "-----------------------------------------------------------------------------------------------------")

                self.logger.info("CARE Parameters:")
                self.logger.info(f" Low SNR Data Directory: {os.path.join(self.input_dir, 'lowSNR')}")
                self.logger.info(f" Low SNR Data Directory: {os.path.join(self.input_dir, 'highSNR')}")
                self.logger.info(f" Model Name: {self.model_name}")
                self.logger.info(f" Model Output Directory: {self.model_output}")
                self.logger.info(f" Augmentation: {self.augmentation}")
                self.logger.info(f" Data Type: {self.data_type}")
                self.logger.info(f" Patch Size: {self.patch_size}")
                self.logger.info(f" Number of Patches per Image: {self.num_patches_per_img}")
                self.logger.info(f" Epochs: {self.epochs}")
                self.logger.info(f" U-Net Kernel Size: {self.unet_kern_size}")
                self.logger.info(f" U-Net Depth: {self.unet_n_depth}")
                self.logger.info(f" Batch Size: {self.batch_size}")
                self.logger.info(f" Steps per Epoch: {self.steps_per_epoch}")
                self.logger.info(f" Percent Validation: {self.pct_validation}")

                self.logger.info("")

                # Processing
                self.logger.info("Preparing to train CARE Model... ")


                mt.train_care(self.input_dir, self.model_output, self.model_name, self.augmentation, self.data_type, self.patch_size,
                               self.num_patches_per_img, self.epochs, self.unet_kern_size, self.unet_n_depth,
                               self.batch_size, self.steps_per_epoch, self.pct_validation, global_GUI_app, self.logger)


                self.logger.info("-----------------------------------------------------------------------------------------------------")
                self.logger.info("RESPAN Version: " + __version__)
                self.logger.info("Release Date: " + __date__ + "")
                self.logger.info(
                    "-----------------------------------------------------------------------------------------------------")
                self.logger.info(respan_paper)
                self.logger.info(care_paper)
                self.logger.info(
                    "-----------------------------------------------------------------------------------------------------")
            #else:
              #  self.logger.info("Error: CARE failed to train.")
                #self.logger.info(
                 #   " Path set to:" + settings.nnUnet_conda_path + '/envs/' + settings.respan_env + '/')
                #self.logger.info(
                 #   " Please check that paths are set correctly")

            self.task_done.emit("")
            self.is_running = False

        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")

class SelfNet(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.logger = Logger()

        # TitleFont
        titlefont = QFont()
        titlefont.setBold(True)
        titlefont.setPointSize(9)

        dir_options = QGroupBox("Self-Net Training Data and Model Output Selection")
        dir_options.setStyleSheet("QGroupBox::title {"
                                  "subcontrol-origin: margin;"
                                  "subcontrol-position: top left;"
                                  "padding: 2px;"
                                  "color: black;"
                                  "}")
        dir_options.setFont(titlefont)
        dir_options_layout = QVBoxLayout()
        dir_options.setLayout(dir_options_layout)

        if global_GUI_app == False:
            self.RESPANdir_label = QLabel("No RESPAN directory selected.")
            self.RESPANdir_button = QPushButton("Select RESPAN directory")
            self.RESPANdir_button.clicked.connect(self.get_RESPANdir)

        self.ground_truth_dir_label = QLabel("No training data directory selected.")
        self.ground_truth_dir_button = QPushButton("Select training data directory")
        self.ground_truth_dir_button.clicked.connect(self.get_gtdir)

        self.directory_label = QLabel("No model output directory selected.")
        self.directory_button = QPushButton("Select model output directory")
        self.directory_button.clicked.connect(self.get_outputdir)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.info_label2 = QLabel("Self-Net input folder should contain at least one .tif image - refer to guide for more information.")
        self.line2 = QFrame()
        self.line2.setFrameShape(QFrame.HLine)
        self.line2.setFrameShadow(QFrame.Sunken)

        if global_GUI_app == False:
            self.RESPANdir_button.setFixedWidth(300)
        self.directory_button.setFixedWidth(300)
        self.ground_truth_dir_button.setFixedWidth(300)

        if global_GUI_app == False:
            dir_options_layout.addWidget(self.RESPANdir_button)
            dir_options_layout.addWidget(self.RESPANdir_label)
        dir_options_layout.addWidget(self.ground_truth_dir_button)
        dir_options_layout.addWidget(self.ground_truth_dir_label)
        dir_options_layout.addWidget(self.directory_button)
        dir_options_layout.addWidget(self.directory_label)

        dir_options_layout.addWidget(self.line)

        dir_options_layout.addWidget(self.info_label2)
        dir_options_layout.addWidget(self.line2)

        options_group1 = QGroupBox("Additional Parameters")
        options_group1.setStyleSheet("QGroupBox::title {"
                                     "subcontrol-origin: margin;"
                                     "subcontrol-position: top left;"
                                     "padding: 2px;"
                                     "color: black;"
                                     "}")
        options_group1.setFont(titlefont)
        options_layout1 = QVBoxLayout()
        options_group1.setLayout(options_layout1)

        #model name
        self.model_name_label = QLabel("Model Name:")
        self.model_name_input = QLineEdit()
        self.model_name_input.setFixedWidth(400)
        self.model_name_input.setText("example_model_name_Self-Net_dendrites_60x_150nmXY_300nmZ")


        # minV
        self.minv_label = QLabel("Minimum intensity:")
        self.minv_input = QLineEdit()
        self.minv_input.setText('0')

        # maxV
        self.maxv_label = QLabel("Maximum intensity:")
        self.maxv_input = QLineEdit()
        self.maxv_input.setText('65535')

        # mimum background intensity for patches
        self.minb_label = QLabel("Minimum background intensity (for patch selection):")
        self.minb_input = QLineEdit()
        self.minb_input.setText('1000')

        #scale
        self.scale_label = QLabel("XY sampling (um) / Z step (um) (eg 75/150 = 0.5) :")
        self.scale_input = QLineEdit()
        self.scale_input.setText('0.5')

        #xy_int
        self.xy_int_label = QLabel("XY interval:")
        self.xy_int_input = QLineEdit()
        self.xy_int_input.setText('4')

        #xz_int
        self.xz_int_label = QLabel("XZ interval:")
        self.xz_int_input = QLineEdit()
        self.xz_int_input.setText('8')

        #batch size
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_input = QLineEdit()
        self.batch_size_input.setText('8')

        #epochs
        self.epochs_label = QLabel("Epochs:")
        self.epochs_input = QLineEdit()
        self.epochs_input.setText('40')

        #log interval
        self.log_interval_label = QLabel("Log Interval:")
        self.log_interval_input = QLineEdit()
        self.log_interval_input.setText('1')

        #imshow interval
        self.imshow_interval_label = QLabel("Imshow Interval:")
        self.imshow_interval_input = QLineEdit()
        self.imshow_interval_input.setText('100')


        # nnU-Net Environment
        #self.conda_path_label = QLabel("Conda Path:")
        #self.conda_path_input = QLineEdit()
        #self.conda_path_input.setFixedWidth(400)
        #self.conda_path_input.setText('<path to Anaconda3>')

        #self.respan_env_label = QLabel("Environment:")
        #self.respan_env_input = QLineEdit()
        #self.respan_env_input.setText('respan')

        horizontal_input1 = QHBoxLayout()
        horizontal_input1.addWidget(self.model_name_label)
        horizontal_input1.addWidget(self.model_name_input)
        horizontal_input1.addStretch(1)

        horizontal_input2 = QHBoxLayout()
        horizontal_input2.addWidget(self.minv_label)
        horizontal_input2.addWidget(self.minv_input)
        horizontal_input2.addWidget(self.maxv_label)
        horizontal_input2.addWidget(self.maxv_input)
        horizontal_input2.addWidget(self.minb_label)
        horizontal_input2.addWidget(self.minb_input)
        horizontal_input2.addStretch(1)

        horizontal_input3 = QHBoxLayout()
        horizontal_input3.addWidget(self.scale_label)
        horizontal_input3.addWidget(self.scale_input)
        horizontal_input3.addWidget(self.xy_int_label)
        horizontal_input3.addWidget(self.xy_int_input)
        horizontal_input3.addWidget(self.xz_int_label)
        horizontal_input3.addWidget(self.xz_int_input)
        #horizontal_input3.addStretch(1)

        horizontal_input4 = QHBoxLayout()
        horizontal_input4.addWidget(self.batch_size_label)
        horizontal_input4.addWidget(self.batch_size_input)
        horizontal_input4.addWidget(self.epochs_label)
        horizontal_input4.addWidget(self.epochs_input)
        horizontal_input4.addWidget(self.log_interval_label)
        horizontal_input4.addWidget(self.log_interval_input)
        horizontal_input4.addWidget(self.imshow_interval_label)
        horizontal_input4.addWidget(self.imshow_interval_input)

        #horizontal_input5 = QHBoxLayout()
        #horizontal_input5.addWidget(self.conda_path_label)
        #horizontal_input5.addWidget(self.conda_path_input)
        #horizontal_input5.addWidget(self.respan_env_label)
        #horizontal_input5.addWidget(self.respan_env_input)
        #horizontal_input4.addStretch(1)


        options_layout1.addLayout(horizontal_input1)
        options_layout1.addLayout(horizontal_input2)
        options_layout1.addLayout(horizontal_input3)
        options_layout1.addLayout(horizontal_input4)
        #options_layout1.addLayout(horizontal_input5)

        # options_group1.setFixedWidth(600)
        input_data_and_opt = QVBoxLayout()
        dir_options.setFixedWidth(600)
        options_group1.setFixedWidth(800)
        input_data_and_opt.addWidget(dir_options)
        input_data_and_opt.addWidget(options_group1)


        main_options = QHBoxLayout()
        main_options.addLayout(input_data_and_opt)
        main_options.addStretch(1) # , alignment=Qt.AlignTop)
        #main_options.addWidget(options_group2)  # , alignment=Qt.AlignLeft)
        # main_options.addWidget(options_group2, alignment=Qt.AlignTop)

        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.get_variables)
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.run_button.setStyleSheet("background-color: lightblue;")
        # self.cancel_button.setStyleSheet("background-color: lightcoral;")

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # FINAL Layout
        layout.addLayout(main_options)
        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)
        # layout.addWidget(self.logger.log_display)

        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

        self.logger.qt_handler.log_generated.connect(self.update_log_display)
        self.setLayout(layout)

        try:
            variables_dict = _load_settings('SelfNetparametersGUI.pkl')
            if variables_dict is None:
                raise FileNotFoundError('SelfNetparametersGUI.pkl')  # handled below exactly as before

            # Retrieve
            if global_GUI_app == False:
                respan_dir = variables_dict.get('respan_dir', None)
            gt_dir = variables_dict.get('gt_dir', None)
            data_dir = variables_dict.get('data_dir', None)

            model_name = variables_dict.get('model_name', None)
            minv = variables_dict.get('minv', None)
            maxv = variables_dict.get('maxv', None)
            minb = variables_dict.get('minb', None)
            scale = variables_dict.get('scale', None)
            xy_int = variables_dict.get('xy_int', None)
            xz_int = variables_dict.get('xz_int', None)
            batch_size = variables_dict.get('batch_size', None)
            epochs = variables_dict.get('epochs', None)
            log_interval = variables_dict.get('log_interval', None)
            imshow_interval = variables_dict.get('imshow_interval', None)

            #conda_path = variables_dict.get('conda_path', None)
            #respan_env = variables_dict.get('respan_env', None)

            # Update GUI
            if global_GUI_app == False:
                self.RESPANdir_label.setText(f"Selected directory: {respan_dir}")
            self.directory_label.setText(f"Selected directory: {data_dir}")
            self.ground_truth_dir_label.setText(f"Selected directory: {gt_dir}")

            self.model_name_input.setText(model_name)
            self.minv_input.setText(minv)
            self.maxv_input.setText(maxv)
            self.minb_input.setText(minb)
            self.scale_input.setText(scale)
            self.xy_int_input.setText(xy_int)
            self.xz_int_input.setText(xz_int)
            self.batch_size_input.setText(batch_size)
            self.epochs_input.setText(epochs)
            self.log_interval_input.setText(log_interval)
            self.imshow_interval_input.setText(imshow_interval)

            #self.conda_path_input.setText(conda_path)
            #self.respan_env_input.setText(respan_env)

        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the pickle file, just ignore the error

    @pyqtSlot()
    def get_RESPANdir(self):
        RESPANdir = QFileDialog.getExistingDirectory(self, 'Select RESPAN directory')
        if RESPANdir:
            self.RESPANdir_label.setText(f"Selected directory: {RESPANdir}")

    @pyqtSlot()
    def get_gtdir(self):

        gtdir = QFileDialog.getExistingDirectory(self, 'Select training data directory')
        if gtdir:
            self.ground_truth_dir_label.setText(f"Selected directory: {gtdir}")

    @pyqtSlot()
    def get_outputdir(self):
        datadir = QFileDialog.getExistingDirectory(self, 'Select model output directory')
        if datadir:
            self.directory_label.setText(f"Selected directory: {datadir}")


    @pyqtSlot()
    def get_variables(self):
        try:
            if global_GUI_app == False:
                respanlabel_text = self.RESPANdir_label.text()
                respan_dir = respanlabel_text.split(": ")[-1]
            else:
                respan_dir = "none"

            gtlabel_text = self.ground_truth_dir_label.text()
            dirlabel_text = self.directory_label.text()


            gt_dir = gtlabel_text.split(": ")[-1]
            data_dir = dirlabel_text.split(": ")[-1]

            model_name = str(self.model_name_input.text())
            minv = str(self.minv_input.text())
            maxv = str(self.maxv_input.text())
            minb = str(self.minb_input.text())
            scale = str(self.scale_input.text())
            xy_int = str(self.xy_int_input.text())
            xz_int = str(self.xz_int_input.text())
            batch_size = str(self.batch_size_input.text())
            epochs = str(self.epochs_input.text())
            log_interval = str(self.log_interval_input.text())
            imshow_interval = str(self.imshow_interval_input.text())

            #conda_path = str(self.conda_path_input.text())
            #respan_env = str(self.respan_env_input.text())

            variables_dict = {
                'respan_dir': respan_dir,
                'gt_dir': gt_dir,
                'data_dir': data_dir,

                'model_name': model_name,
                'minv': minv,
                'maxv': maxv,
                'minb': minb,
                'scale': scale,
                'xy_int': xy_int,
                'xz_int': xz_int,
                'batch_size': batch_size,
                'epochs': epochs,
                'log_interval': log_interval,
                'imshow_interval': imshow_interval,

                #'conda_path': conda_path,
                #'respan_env': respan_env
            }

            # Save the dictionary to a pickle file
            _save_settings('SelfNetparametersGUI.pkl', variables_dict, getattr(self, 'log_display', None))
        except Exception as e:
            self.log_display.append(f"An error occurred: {e}")


    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode

        try:
            # Validate directory selections
            if global_GUI_app == False:
                RESPAN = self.RESPANdir_label.text().replace("Selected directory: ", "")
            ground_truth = self.ground_truth_dir_label.text().replace("Selected directory: ", "")
            directory = self.directory_label.text().replace("Selected directory: ", "")

            for dir_path, label in [(ground_truth, "training data directory"),
                                    (directory, "model output directory")]:
                if not os.path.isdir(dir_path):
                    raise ValueError(f"No valid {label} selected.")

                # Validate model name
                model_name = self.model_name_input.text()

                # Validate numeric inputs
                minv = int(self.minv_input.text())
                maxv = int(self.maxv_input.text())
                minb = int(self.minb_input.text())
                scale = float(self.scale_input.text())
                xy_int = int(self.xy_int_input.text())
                xz_int = int(self.xz_int_input.text())
                batch_size = int(self.batch_size_input.text())
                epochs = int(self.epochs_input.text())
                log_interval = int(self.log_interval_input.text())
                imshow_interval = int(self.imshow_interval_input.text())

                # Validate Conda path and environment
                #conda_path = self.conda_path_input.text()
                #respan_env = self.respan_env_input.text()
                if not os.path.exists(global_env_path):
                    raise ValueError("Invalid Conda path.")
                # Assuming a function check_conda_env_exists that checks if the environment exists
                #if not self.check_conda_env_exists(conda_path, respan_env):
                #    raise ValueError("Specified nnU-Net environment does not exist.")

            input_dir = ground_truth + "/"
            model_output = directory + "/"

            if global_GUI_app == False:
                RESPAN = RESPAN + "/"
            else:
                RESPAN = "none"

            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.task_done.disconnect(self.on_task_done)
                self.logger.qt_handler.log_generated.disconnect(self.update_log_display)

            self.worker = SelfNetWorker(RESPAN, input_dir, model_output, model_name,
                                       minv, maxv, minb, scale, xy_int, xz_int, batch_size, epochs, log_interval, imshow_interval,
                                        global_env_path, global_respan_env, self.logger)
            self.worker.task_done.connect(self.on_task_done)
            self.worker.start()
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            self.progress.setVisible(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
            self.progress.setVisible(False)

    @pyqtSlot(str)
    def update_log_display(self, message):
        self.log_display.append(message)
        self.log_display.moveCursor(QTextCursor.End)

    @pyqtSlot(str)
    def on_task_done(self, message):
        self.logger.info(message)
        self.progress.setValue(self.progress.maximum())
        self.progress.setVisible(False)
        # Disconnect the signals when done
        # self.worker.task_done.disconnect(self.on_task_done)
        # self.logger.qt_handler.log_generated.disconnect(self.update_log_display)

class SelfNetWorker(QThread):
    task_done = pyqtSignal(str)

    def __init__(self, RESPAN, input_dir, model_dir, model_name,
                 min_v, max_v, bg_threshold, scale, xy_int, xz_int, batch_size, epochs, log_interval, imshow_interval,
                 conda_path, respan_env, logger):
        super().__init__()

        self.RESPAN = RESPAN
        self.input_dir = input_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.min_v = min_v
        self.max_v = max_v
        self.bg_threshold = bg_threshold
        self.scale = scale
        self.xy_int = xy_int
        self.xz_int = xz_int
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.imshow_interval = imshow_interval

        self.conda_path = conda_path
        self.respan_env = respan_env
        self.logger = logger

    def run(self):
        self.is_running = True

        try:
            if global_GUI_app == False:
                sys.path.append(self.RESPAN)

            from RESPAN.Environment import main, mt

            args_dict = {
                'input_dir': self.input_dir,
                'model_path': self.model_dir,
                'model_name': self.model_name,
                'min_v': self.min_v,
                'max_v': self.max_v,
                'bg_threshold': self.bg_threshold,
                'scale': self.scale,
                'xy_int': self.xy_int,
                'xz_int': self.xz_int,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'log_interval': self.log_interval,
                'imshow_interval': self.imshow_interval

            }

            log_path = self.input_dir + 'RESPAN_Self-Net_Training_Log' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            self.logger.set_log_file(log_path)

            self.logger.info("Self-Net Training GUI | RESPAN Version: " + __version__)
            self.logger.info("Release Date: " + __date__)
            self.logger.info("Created by: " + __author__ + "")
            self.logger.info("Department of Neurology, The Ohio State University")
            self.logger.info(
                "-----------------------------------------------------------------------------------------------------")
            self.logger.info(self_net_paper)
            self.logger.info(
                "-----------------------------------------------------------------------------------------------------")

            self.logger.info("Self-Net Parameters:")
            self.logger.info(f" Using conda environment: {global_env_path}/{global_respan_env}")
            self.logger.info(f" Using conda path: {global_conda_path}")
            self.logger.info(f" Input Directory: {self.input_dir}")
            self.logger.info(f" Model Path: {self.model_dir}")
            self.logger.info(f" Model Name: {self.model_name}")
            self.logger.info(f" Minimum Volume: {self.min_v}")
            self.logger.info(f" Maximum Volume: {self.max_v}")
            self.logger.info(f" Background Threshold: {self.bg_threshold}")
            self.logger.info(f" Scale: {self.scale}")
            self.logger.info(f" XY Intensity: {self.xy_int}")
            self.logger.info(f" XZ Intensity: {self.xz_int}")
            self.logger.info(f" Batch Size: {self.batch_size}")
            self.logger.info(f" Epochs: {self.epochs}")
            self.logger.info(f" Log Interval: {self.log_interval}")
            self.logger.info(f" Imshow Interval: {self.imshow_interval}")

            self.logger.info("\nTraining Self-Net model... please allow 30min - 2 hours for training to complete.\n")

            if os.path.exists(global_env_path + '/' + global_respan_env + '/'):
                #log = imgan.restore_and_segment(settings, locations, self.logger)
                #_run([str(PY_EXE), "-u", str(base_path + "\SelfNet_Model_Training.py"), *args_dict], self.logger)
                run_external_script( SELFNET_TRAINING_SCRIPT, args_dict, self.logger)
                #run_external_script(
                   # base_path + "\SelfNet_Model_Training.py", global_conda_path,
                 #   global_respan_env, args_dict, self.logger)
                self.logger.info("-----------------------------------------------------------------------------------------------------")
                self.logger.info("RESPAN Version: " + __version__)
                self.logger.info("Release Date: " + __date__ + "")

                self.logger.info("-----------------------------------------------------------------------------------------------------")
                self.logger.info(respan_paper)
                self.logger.info(self_net_paper)
                self.logger.info("-----------------------------------------------------------------------------------------------------")
            else:
                self.logger.info("Error: Self-Net failed to train.")
                #self.logger.info(
                #" Path set to:" + settings.nnUnet_conda_path + '/envs/' + settings.respan_env + '/')
                #self.logger.info(
                #" Please check installation, or that paths set correctly in Analysis_Settings.yaml")

            self.task_done.emit("")
            self.is_running = False

        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")

class About(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Title font setup
        titleFont = QFont()
        titleFont.setBold(False)
        titleFont.setPointSize(12)  # Make the font size larger for the title
        bodyFont = QFont()
        bodyFont.setPointSize(10)

        # Group box for RESPAN information
        respan_info = QGroupBox("Restoration Enhanced Neuron and SPine Analysis (RESPAN)")
        respan_info.setFont(titleFont)
        respan_info_layout = QVBoxLayout(respan_info)
        respan_info.setAlignment(Qt.AlignCenter)
        # Text description as required
        description_label = QLabel(about_text)


        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setFont(bodyFont)

        # Adding widgets to the layout
        respan_info_layout.addWidget(description_label)
        layout.addWidget(respan_info)

        # Logger Display - Text Edit
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)

        # Set the main layout
        self.setLayout(layout)
        '''
        self.logger = Logger()

        # TitleFont
        titlefont = QFont()
        titlefont.setBold(True)
        titlefont.setPointSize(9)

        dir_options = QGroupBox("Restoration Enhanced Neuron and SPine Analysis (RESPAN)")
        dir_options.setStyleSheet("QGroupBox::title {"
                                  "subcontrol-origin: margin;"
                                  "subcontrol-position: top left;"
                                  "padding: 2px;"
                                  "color: black;"
                                  "}")
        dir_options.setFont(titlefont)
        dir_options_layout = QVBoxLayout()
        dir_options.setLayout(dir_options_layout)


        self.info_label2 = QLabel("Self-Net input folder should contain at least one .tif image - refer to guide for more information.")
        self.line2 = QFrame()
        self.line2.setFrameShape(QFrame.HLine)
        self.line2.setFrameShadow(QFrame.Sunken)
        dir_options_layout.addWidget(self.line)


        dir_options.setFixedWidth(600)


        self.run_button.setStyleSheet("background-color: lightblue;")
        # self.cancel_button.setStyleSheet("background-color: lightcoral;")

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # FINAL Layout
        layout.addLayout(main_options)

        layout.addWidget(self.progress)
        # layout.addWidget(self.logger.log_display)

        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

        self.logger.qt_handler.log_generated.connect(self.update_log_display)
        self.setLayout(layout)


    @pyqtSlot(str)
    def update_log_display(self, message):
        self.log_display.append(message)
        self.log_display.moveCursor(QTextCursor.End)

    @pyqtSlot(str)
    def on_task_done(self, message):
        self.logger.info(message)
        self.progress.setValue(self.progress.maximum())
        self.progress.setVisible(False)
        # Disconnect the signals when done
        # self.worker.task_done.disconnect(self.on_task_done)
        # self.logger.qt_handler.log_generated.disconnect(self.update_log_display)

'''
# ---------------------------------------------------------------------------
# Headless command-line mode (--cli)
# ---------------------------------------------------------------------------
# Runs the same initialize -> restore_and_segment -> analyze_spines chain the
# GUI's Run button drives, on every dataset subfolder of --data-dir, without
# creating a Qt window. Exists so the packaged application can be exercised
# end-to-end from a shell (bundled nnU-Net environment included) and return a
# meaningful exit code: 0 = every dataset completed, 1 = a dataset failed or
# none were found, 2 = unhandled exception.
#
# The distribution is a windowed executable with no console of its own, so a
# parent console is attached when one exists (cmd / PowerShell / Windows
# Terminal). Under a pipe-based terminal (Git Bash) there is nothing to attach
# to; pass --log-file to capture the whole run. Each dataset also gets its
# usual <dataset>/RESPAN_Log_<timestamp>.log.

MODEL_TYPE_LABELS = {
    1: "Spines, Dendrites, and Soma (original)",
    2: "Dendrites and Soma Only",
    3: "Spines, Necks, Dendrites, and Soma",
    4: "Spines, Necks, Dendrites, Soma, and Axons (2025)",
}


def _attach_parent_console():
    """Windowed builds may start with sys.stdout and/or sys.stderr = None.

    Attach the launching console when there is one (cmd / PowerShell /
    Windows Terminal) and point each missing stream at it. Whatever is still
    None afterwards (no console, no redirection of that stream) gets a
    discard stream, because argparse's usage/error printing cannot handle a
    None stream on Python 3.9. Streams are opened UTF-8 with replacement so a
    non-encodable character in a diagnostic can never raise.
    """
    if os.name != 'nt':
        return
    if sys.stdout is None or sys.stderr is None:
        try:
            import ctypes
            if ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
                con = open('CONOUT$', 'w', encoding='utf-8', errors='replace', buffering=1)
                if sys.stdout is None:
                    sys.stdout = con
                if sys.stderr is None:
                    sys.stderr = con
        except Exception:
            pass
    for name in ('stdout', 'stderr'):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, 'w', encoding='utf-8', errors='replace'))


class CliLogger(Logger):
    """The GUI Logger with the Qt handler replaced by console / file streams."""

    def __init__(self, log_file=None):
        self.logger = logging.getLogger("respan_cli")
        self.logger.setLevel(logging.DEBUG)
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
        self.file_handler = None
        fmt = logging.Formatter('%(message)s')
        if sys.stdout is not None:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)
        if log_file:
            gh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            gh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(gh)


def _min_max_pair(text):
    """argparse type for --spine-vol: 'min,max' -> [min, max] as floats."""
    try:
        lo, hi = (float(x) for x in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected two numbers 'min,max', got {text!r}")
    return [lo, hi]


def _build_cli_parser():
    ap = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]) + " --cli",
        description="RESPAN headless mode. Processes every dataset subfolder of "
                    "--data-dir that contains an Analysis_Settings.yaml, exactly as "
                    "the GUI Run button does. Defaults match the GUI defaults.")
    ap.add_argument('--cli', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--data-dir', required=True,
                    help="Folder whose subfolders are datasets (each with Analysis_Settings.yaml)")
    ap.add_argument('--model-dir', required=True,
                    help="nnU-Net model folder (the Dataset###_... directory)")
    ap.add_argument('--model-type', type=int, choices=sorted(MODEL_TYPE_LABELS), required=True,
                    help="1=" + MODEL_TYPE_LABELS[1] + "; 2=" + MODEL_TYPE_LABELS[2]
                         + "; 3=" + MODEL_TYPE_LABELS[3] + "; 4=" + MODEL_TYPE_LABELS[4])
    ap.add_argument('--input-xy', type=float, help="Image XY voxel size in um")
    ap.add_argument('--input-z', type=float, help="Image Z step in um")
    ap.add_argument('--model-xy', type=float, help="Model XY voxel size in um")
    ap.add_argument('--model-z', type=float, help="Model Z step in um")
    ap.add_argument('--use-yaml-res', action='store_true',
                    help="Take the four resolutions from Analysis_Settings.yaml instead")
    ap.add_argument('--neuron-channel', type=int, default=1)
    ap.add_argument('--min-dendrite-vol', type=float, default=15.0, help="um^3 (GUI default 15)")
    ap.add_argument('--spine-vol', type=_min_max_pair, default="0.03,15",
                    help="min,max spine volume in um^3 (GUI default 0.03,15)")
    ap.add_argument('--spine-dist', type=float, default=4.0, help="max spine-to-dendrite distance in um (GUI default 4)")
    ap.add_argument('--spurious-length-min-um', type=float, default=1.0)
    ap.add_argument('--spurious-support-min', type=float, default=0.2)
    ap.add_argument('--keep-head-if-flagged', action='store_true')
    ap.add_argument('--no-drop-spurious-necks', action='store_true')
    ap.add_argument('--no-neck-generation', action='store_true')
    ap.add_argument('--recover-filopodia', action='store_true')
    ap.add_argument('--dendrite-repair', action='store_true')
    ap.add_argument('--dendrite-repair-max-dist-um', type=float, default=2.0)
    ap.add_argument('--detect-multi-head-spines', action='store_true')
    ap.add_argument('--no-keep-partial-spines', action='store_true')
    ap.add_argument('--image-restore', action='store_true', help="CARE restoration (model path from YAML)")
    ap.add_argument('--axial-restore', action='store_true', help="SelfNet axial restoration (model path from YAML)")
    ap.add_argument('--save-intermediate', action='store_true')
    ap.add_argument('--no-save-validation', action='store_true')
    ap.add_argument('--nnunet-patching', action='store_true', help="Tile large images for nnU-Net inference")
    ap.add_argument('--track', action='store_true', help="4D spine tracking across timepoints")
    ap.add_argument('--reg-method', choices=["Rigid", "Elastic"], default="Rigid")
    ap.add_argument('--swc', action='store_true')
    ap.add_argument('--dask', action='store_true', help=argparse.SUPPRESS)  # experimental, GUI-hidden semantics
    ap.add_argument('--log-file', help="Also write the whole headless run to this file")
    ap.add_argument('--respan-dir', help=argparse.SUPPRESS)  # source runs only: repo root holding the RESPAN package
    return ap


def _cli_params(a, parser):
    """Build the parameter namespace apply_run_parameters() expects."""
    spine_vol = list(a.spine_vol)  # already validated by _min_max_pair
    if not a.use_yaml_res and None in (a.input_xy, a.input_z, a.model_xy, a.model_z):
        parser.error("give --input-xy --input-z --model-xy --model-z, or pass --use-yaml-res")
    return SimpleNamespace(
        model_dir=a.model_dir,
        use_yaml_res=a.use_yaml_res,
        inputxy=a.input_xy, inputz=a.input_z, modelxy=a.model_xy, modelz=a.model_z,
        neuron_ch=a.neuron_channel,
        analysis_method="Whole Neuron",
        min_dendrite_vol=a.min_dendrite_vol,
        spine_vol=spine_vol,
        spine_dist=a.spine_dist,
        spurious_length_min_um=a.spurious_length_min_um,
        spurious_nnunet_support_min=a.spurious_support_min,
        spurious_keep_head_if_flagged=a.keep_head_if_flagged,
        drop_spurious_necks=not a.no_drop_spurious_necks,
        neck_generation=not a.no_neck_generation,
        recover_filopodia=a.recover_filopodia,
        dendrite_repair=a.dendrite_repair,
        dendrite_repair_max_dist_um=a.dendrite_repair_max_dist_um,
        detect_multi_head_spines=a.detect_multi_head_spines,
        keep_partial_spines=not a.no_keep_partial_spines,
        image_restore=a.image_restore,
        axial_restore=a.axial_restore,
        swc=a.swc,
        save_intermediate=a.save_intermediate,
        save_validation=not a.no_save_validation,
        dask_enabled=a.dask,
        nnUNet_patching=a.nnunet_patching,
        Track=a.track,
        reg_method=a.reg_method,
        second_pass=False,
        model_type=MODEL_TYPE_LABELS[a.model_type],
    )


def run_cli(argv):
    """Entry point for --cli. Returns the process exit code."""
    _attach_parent_console()
    parser = _build_cli_parser()
    args = parser.parse_args(argv)          # usage errors exit 2 via argparse
    p = _cli_params(args, parser)
    try:
        logger = CliLogger(args.log_file)
    except OSError as e:
        # Unwritable --log-file: still run, but say so where it can be seen.
        sys.stderr.write(f"Cannot open --log-file {args.log_file!r}: {e}; continuing without it\n")
        logger = CliLogger(None)
    try:
        return _run_cli(args, p, logger)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Headless run aborted: {e}", exc_info=True)
        return 2


def _run_cli(args, p, logger):
    logger.info(f"RESPAN {__version__} ({__date__}) headless mode")
    logger.info(f"  APP_DIR: {APP_DIR}")
    logger.info(f"  Frozen build: {FROZEN}")
    if FROZEN:
        if not (VENV_DIR.exists() and PY_EXE.exists()):
            logger.error(f"Embedded environment not found at {VENV_DIR}")
            return 1
        ensure_nnunet_installed(logger)   # first-run editable install, sentinel-gated
        logger.info(f"  Bundled nnU-Net environment ready at {VENV_DIR}")

    if not FROZEN:
        # Running from source: make the RESPAN package importable, as the GUI
        # does when the user selects the repository directory (dev mode only;
        # the frozen build bundles the package).
        repo_root = Path(args.respan_dir) if args.respan_dir else APP_DIR.parent.parent
        sys.path.append(str(repo_root))
    from RESPAN.Environment import main, imgan, strk, sr
    main.check_gpu()

    directory = os.path.abspath(args.data_dir).replace(os.sep, "/") + "/"
    if not os.path.isdir(directory):
        logger.error(f"Directory does not exist: {directory}")
        return 1
    if not os.path.isdir(p.model_dir):
        logger.error(f"Model directory does not exist: {p.model_dir}")
        return 1
    for label, path in (("data directory", directory), ("model directory", p.model_dir)):
        if " " in path:
            logger.warning(f"  Warning: {label} path contains spaces (known to break some tools): {path}")
        if "onedrive" in path.lower():
            logger.warning(f"  Warning: {label} appears to be inside a OneDrive-synced folder: {path}")

    processed = 0
    failed = []
    skipped = []
    for subfolder in sorted(os.listdir(directory)):
        subfolder_path = os.path.join(directory, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        subfolder_path = subfolder_path.replace(os.sep, "/") + "/"
        if not os.path.exists(os.path.join(subfolder_path, "Analysis_Settings.yaml")):
            logger.info(f"  Skipping '{subfolder}': No Analysis_Settings.yaml found.")
            skipped.append(subfolder)
            continue

        processed += 1
        # Everything dataset-specific sits inside one handler: a malformed
        # Analysis_Settings.yaml or a bad parameter in ONE dataset is recorded
        # as that dataset's failure and the batch continues (exit code 1).
        try:
            settings, locations = main.initialize_RESPAN(subfolder_path)

            log_path = subfolder_path + 'RESPAN_Log_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            logger.set_log_file(log_path)
            logger.clear_log(log_path)

            logger.info("RESPAN Version: " + __version__)
            logger.info("Release Date: " + __date__)
            logger.info("Created by: " + __author__)
            logger.info("Department of Neurology, The Ohio State University\n")

            apply_run_parameters(settings, locations, p, logger)

            logger.info(f"\nProcessing folder: {subfolder_path}\n")

            if not os.path.exists(f'{global_env_path}/{global_respan_env}/'):
                logger.error("Error: nnUNet environment not created or set correctly.")
                logger.error(" Path set to:" + global_env_path + '/' + global_respan_env + '/')
                failed.append(subfolder)
                continue

            log = sr.restore_and_segment(settings, locations, logger)
            if log == 0:
                n_failed = 0
                if settings.Track:
                    strk.track_spines(settings, locations, log, logger)
                    imgan.analyze_spines_4D(settings, locations, log, logger)
                else:
                    n_failed = imgan.analyze_spines(settings, locations, log, logger) or 0
                if n_failed:
                    logger.error(f"RESPAN analysis finished with errors: {subfolder} ({n_failed} image(s) failed)")
                    failed.append(subfolder)
                else:
                    logger.info(f"RESPAN analysis complete: {subfolder}")
            else:
                logger.error("Error with restoration and segmentation.")
                failed.append(subfolder)
        except Exception as e:
            logger.error(f"An error occurred processing '{subfolder}': {e}", exc_info=True)
            failed.append(subfolder)

    if processed == 0:
        logger.error("\nNo valid dataset subfolders were found in the selected directory.")
        logger.error("RESPAN expects each dataset to be in its own subfolder, containing an Analysis_Settings.yaml file.")
        return 1
    if skipped:
        logger.info(f"\nSkipped {len(skipped)} subfolder(s) due to missing Analysis_Settings.yaml: " + ", ".join(skipped))
    done = processed - len(failed)
    logger.info(f"\nHeadless run finished: {done} of {processed} dataset(s) completed"
                + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def RESPAN_gui():
    _install_crash_reporter()
    app = QApplication([])

    #app.setStyleSheet(qss)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(230, 230, 230))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(204, 204, 204))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(204, 204, 204))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(0, 102, 153))
    palette.setColor(QPalette.Highlight, QColor(0, 102, 153))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(palette)


    splash = Splash("Loading RESPAN...", 2000)
    splash.move(600, 600)

    splash.show()

    # Ensures that the application is fully up and running before closing the splash screen
    app.processEvents()

    window = MainWindow()
    window.setWindowTitle(f' RESPAN - Version: {__version__}')
    window.setGeometry(100, 100, 1100, 1200)
    if global_GUI_app == True:
        icon = QIcon(icon_path)
        window.setWindowIcon(icon)
    window.show()


    sys.exit(app.exec_())

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support() # required to allow multithreading internally
    if '--cli' in sys.argv[1:]:
        sys.exit(run_cli(sys.argv[1:]))
    RESPAN_gui()