# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:16:58 2023

"""
__title__     = 'SpinePipe'
__version__   = '0.9.0'
__date__      = "25 July, 2023"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2023 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'

import sys
#import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QPushButton, QCheckBox, QLabel, QLineEdit, 
                             QMessageBox, QTextEdit, QWidget, QFileDialog, 
                             QGridLayout,QHBoxLayout, QGroupBox, QProgressBar, QSplashScreen,QFrame)
from PyQt5.QtCore import Qt, pyqtSlot, QTime, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QTextCursor, QPixmap, QPainter, QColor, QFont, QPalette

import pickle
import logging
from datetime import datetime
#import time


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
        
        # Set up the handler for the Qt logging
        self.qt_handler = QtHandler()
        self.qt_handler.setLevel(logging.DEBUG)  # You can adjust the level if needed
        formatter = logging.Formatter('%(message)s')
        self.qt_handler.setFormatter(formatter)
        self.logger.addHandler(self.qt_handler)
        
        # No file handler at this point
        self.file_handler = None

    def get_logger(self):
        return self.logger

    def info(self, message):
        self.logger.info(message)

    def error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)

    def set_log_file(self, file_path):
        # If there was a previous FileHandler, remove it
        if self.file_handler is not None:
            self.logger.removeHandler(self.file_handler)
        
        # Set up the file handler
        self.file_handler = logging.FileHandler(file_path)
        self.file_handler.setLevel(logging.DEBUG)  # You can adjust the level if needed
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

class Worker(QThread):
    task_done = pyqtSignal(str)

    #def __init__(self, spinepipe, ground_truth, analysis_output, logger):
    def __init__(self, ground_truth, analysis_output, logger):
        super().__init__()
        #self.spinepipe = spinepipe
        self.ground_truth = ground_truth
        self.analysis_output = analysis_output
        self.logger = logger
        

    def run(self):
        self.is_running = True
        
        try:
            
            #sys.path.append(self.spinepipe)
            from spinepipe.Environment import main, val
            main.check_gpu()
            #Load in experiment parameters and analysis settings   
            #settings, locations = main.initialize_spinepipe(self.analysis_output)
        
            #import spinepipe.Environment
            
            log_path = self.analysis_output +'SpinePipe_Validation_Log' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            self.logger.set_log_file(log_path)  # Note: we're not passing a log_display, because it's not thread-safe
            
        
            self.logger.info("SpinePipe Validation Tool - Version: "+__version__)
            self.logger.info("Release Date: "+__date__) 
            self.logger.info("Created by: "+__author__+"")
            self.logger.info("Zuckerman Institute, Columbia University\n")
              
           
            #Load in experiment parameters and analysis settings   
            settings, locations = main.initialize_spinepipe_validation(self.analysis_output)
            
            
            self.logger.info("Processing folder: "+self.analysis_output)
            self.logger.info(f" Image resolution: {settings.input_resXY}um XY, {settings.input_resZ}um Z")
            self.logger.info(f" Model used: {settings.neuron_seg_model_path}")    
            self.logger.info(f" Model resolution: {settings.neuron_seg_model_res[0]}um XY, {settings.neuron_seg_model_res[2]}um Z")    
            self.logger.info(f" Spine volume set to: {settings.neuron_spine_size[0]} to {settings.neuron_spine_size[1]} voxels.") 
            self.logger.info(f" GPU block size set to: {settings.GPU_block_size[0]},{settings.GPU_block_size[1]},{settings.GPU_block_size[1]}") 
            
            #Processing
            
            val.validate_analysis(self.ground_truth, self.analysis_output, settings, locations, self.logger)
            
          
            
            self.logger.info("Validation complete.")
            self.logger.info("SpinePipe Validation Tool - Version: "+__version__)
            self.logger.info("Release Date: "+__date__+"") 
            self.logger.info("Created by: "+__author__+"") 
            self.logger.info("Zuckerman Institute, Columbia University\n") 
            self.logger.info("--------------------------------------------------------------------")
            
            self.task_done.emit("")
            
            self.is_running = False
            

        
        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")

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
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)


        widget = QWidget()
        layout = QVBoxLayout() 
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.logger = Logger()

        #self.spinepipedir_label = QLabel("No SpinePipe directory selected.")
        #self.spinepipedir_button = QPushButton("Select SpinePipe directory")
        #self.spinepipedir_button.clicked.connect(self.get_spinepipedir)        

        self.ground_truth_dir_label = QLabel("No ground truth data directory selected.")
        self.ground_truth_dir_button = QPushButton("Select ground truth data directory")
        self.ground_truth_dir_button.clicked.connect(self.get_ground_truth_dir)
        
        self.directory_label = QLabel("No analysis output directory selected.")
        self.directory_button = QPushButton("Select data analysis output directory")
        self.directory_button.clicked.connect(self.get_directories)
        
        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
 
        self.integer_label = QLabel("Please ensure the Analysis_Settings.csv is in the folder of analysis outputs you wish to validate.")
        
        
        

        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setVisible(False)


        self.ground_truth_dir_button.setFixedWidth(300)
        self.directory_button.setFixedWidth(300)

        #layout.addWidget(self.spinepipedir_label)
        #layout.addWidget(self.spinepipedir_button)
        layout.addWidget(self.ground_truth_dir_button)
        layout.addWidget(self.ground_truth_dir_label)
        layout.addWidget(self.directory_button)
        layout.addWidget(self.directory_label)

        layout.addWidget(self.line)

        layout.addWidget(self.integer_label)
        #layout.addWidget(options_group)

        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)
        #layout.addWidget(self.logger.log_display)
        
        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

 
        self.logger.qt_handler.log_generated.connect(self.update_log_display)

        #try:
        #    with open('last_dir2.pickle', 'rb') as f:
        #        last_dir2 = pickle.load(f)
        #        self.spinepipedir_label.setText(f"Selected directory: {last_dir2}")
        #except (FileNotFoundError, EOFError, pickle.PickleError):
        #    pass  # If we can't load the last directory2 path, just ignore the error

        try:
            with open('last_dir3.pickle', 'rb') as f:
                last_dir3 = pickle.load(f)
                self.ground_truth_dir_label.setText(f"Selected directory: {last_dir3}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory2 path, just ignore the error

        
        try:
            with open('last_dir.pickle', 'rb') as f:
                last_dir = pickle.load(f)
                self.directory_label.setText(f"Selected directory: {last_dir}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory path, just ignore the error
            


    @pyqtSlot()
    def get_spinepipedir(self):
        spinepipedir = QFileDialog.getExistingDirectory(self, 'Select SpinePipe directory')
        if spinepipedir:
            self.spinepipedir_label.setText(f"Selected directory: {spinepipedir}")
    
            # Save the selected directory2 path
            with open('last_dir2.pickle', 'wb') as f:
                pickle.dump(spinepipedir, f)
                
    @pyqtSlot()
    def get_ground_truth_dir(self):
        ground_truth_dir = QFileDialog.getExistingDirectory(self, 'Select ground truth data directory')
        if ground_truth_dir:
            self.ground_truth_dir_label.setText(f"Selected directory: {ground_truth_dir}")
    
            # Save the selected directory2 path
            with open('last_dir3.pickle', 'wb') as f:
                pickle.dump(ground_truth_dir, f)
                
    @pyqtSlot()
    def get_directories(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select data analysis output directory')
        if directory:
            self.directory_label.setText(f"Selected directory: {directory}")

            # Save the selected directory path
            with open('last_dir.pickle', 'wb') as f:
                pickle.dump(directory, f)
    
    

    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode
        
        
        #spinepipe = self.spinepipedir_label.text().replace("Selected directory: ", "")
        #if spinepipe == "No SpinePipe directory selected.":
        #    QMessageBox.critical(self, "Error", "No directory selected.")
        #    self.progress.setVisible(False)
        #    return

        
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
        
        #spinepipe = spinepipe +"/"
        
        ground_truth = ground_truth +"/"
        
        analysis_output =  directory + "/"
        
        
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        #self.worker = Worker(spinepipe, directory, channel_options, integers, self.logger.get_logger())
        #self.worker = Worker(spinepipe, ground_truth, analysis_output, self.logger)
        self.worker = Worker(ground_truth, analysis_output, self.logger)
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

            
qss = '''
QWidget {
    color: #b1b1b1;
    background-color: #323232;
}
QPushButton {
    background-color: #5a5a5a;
    border: none;
    color: white;
}
QPushButton:hover {
    background-color: #727272;
}
QTextEdit {
    background-color: #242424;
}
QPushButton {
    font-family: "Helvetica";
    font-size: 16px;
    min-height: 30px;

}
QLabel {
    font-family: "Helvetica";
    font-size: 16px;
    min-height: 30px;
}
'''

    
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


splash = Splash("SpinePipe is loading...", 3000)
splash.show()

# Ensures that the application is fully up and running before closing the splash screen
app.processEvents()

window = MainWindow()
window.setWindowTitle(f'SpinePipe Validation - Version: {__version__}')
window.setGeometry(100, 100, 1200, 800)  
window.show()
sys.exit(app.exec_())