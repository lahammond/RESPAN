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
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QPushButton, QCheckBox, QLabel, QLineEdit, 
                             QMessageBox, QTextEdit, QWidget, QFileDialog, 
                             QGridLayout,QHBoxLayout, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSlot, QTime, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QTextCursor

import pickle
import logging
from datetime import datetime
import time

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

    def __init__(self, spinepipe, directory, other_options, GPU_block, spine_vol, logger):
        super().__init__()
        self.spinepipe = spinepipe
        self.directory = directory
        self.other_options = other_options
        self.GPU_block = GPU_block
        self.spine_vol = spine_vol
        self.logger = logger
        

    def run(self):
        self.is_running = True
        
        try:
            
            sys.path.append(self.spinepipe)
            from spinepipe.Environment import main, imgan, timer
            main.check_gpu()
            #Load in experiment parameters and analysis settings   
            settings, locations = main.initialize_spinepipe(self.directory)
        
            #import spinepipe.Environment
            
            log_path = self.directory +'SpinePipe_Log' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            self.logger.set_log_file(log_path)  # Note: we're not passing a log_display, because it's not thread-safe
            
        
            self.logger.info("SpinePipe Version: "+__version__)
            self.logger.info("Release Date: "+__date__) 
            self.logger.info("Created by: "+__author__+"")
            self.logger.info("Zuckerman Institute, Columbia University\n")
              
           
            #Load in experiment parameters and analysis settings   
            settings, locations = main.initialize_spinepipe(self.directory)
            
            #Modify specific parameters and settings:    
            settings.save_intermediate_data = True
            #settings.spine_roi_volume_size = 4 #in microns in x, y, z - approx 50px for 0.3 resolution data
            #settings.GPU_block_size = (150,500,500) #dims used for processing images in block for cell extraction. Reduce if recieving out of memory errors
            settings.GPU_block_size = self.GPU_block
            settings.neuron_spine_size = self.spine_vol


            self.logger.info("Processing folder: "+self.directory)
            self.logger.info(f" Image resolution: {settings.input_resXY}um XY, {settings.input_resZ}um Z")
            self.logger.info(f" Model used: {settings.neuron_seg_model_path}")    
            self.logger.info(f" Model resolution: {settings.neuron_seg_model_res[0]}um XY, {settings.neuron_seg_model_res[2]}um Z")    
            self.logger.info(f" Spine volume set to: {settings.neuron_spine_size[0]} to {settings.neuron_spine_size[1]} voxels.") 
            self.logger.info(f" GPU block size set to: {settings.GPU_block_size[0]},{settings.GPU_block_size[1]},{settings.GPU_block_size[1]}") 
            
            self.logger.info(f" {settings.neuron_seg_model_path}um Z")
            #Processing
            
            log = imgan.restore_and_segment(self.directory, settings, locations, self.logger)
            
            imgan.analyze_spines(settings, locations, log, self.logger)
            
            self.logger.info("SpinePipe analysis complete.")
            self.logger.info("\nSpinePipe Version: "+__version__)
            self.logger.info("Release Date: "+__date__+"") 
            self.logger.info("Created by: "+__author__+"") 
            self.logger.info("Zuckerman Institute, Columbia University\n") 
            self.logger.info("--------------------------------------------------------------------")
            
            self.task_done.emit("")
            
            self.is_running = False
            

        
        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        widget = QWidget()
        layout = QVBoxLayout() 
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.logger = Logger()


        self.spinepipedir_label = QLabel("No SpinePipe directory selected.")
        self.spinepipedir_button = QPushButton("Select SpinePipe directory")
        self.spinepipedir_button.clicked.connect(self.get_spinepipedir)
        
        self.directory_label = QLabel("No directory selected.")
        self.directory_button = QPushButton("Select data directory")
        self.directory_button.clicked.connect(self.get_directories)
        


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

        options_group2 = QGroupBox("Options")
        options_layout2 = QVBoxLayout()
        options_group2.setLayout(options_layout2)
        self.save_intermediate = QCheckBox("Save intermediate data")
        self.save_intermediate.setChecked(True) 
        self.integer_label = QLabel("GPU block size (decrease block size if processing fails):")
        self.integer_input = QLineEdit("150,500,500")
        self.integer_label_2 = QLabel("Spine volume filter (min, max volume in voxels):")
        self.integer_input_2 = QLineEdit("3,1500")

        options_layout2.addWidget(self.save_intermediate)

        

        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        

        layout.addWidget(self.spinepipedir_label)
        layout.addWidget(self.spinepipedir_button)
        layout.addWidget(self.directory_label)
        layout.addWidget(self.directory_button)
        layout.addWidget(options_group2)
        layout.addWidget(self.integer_label)
        layout.addWidget(self.integer_input)
        layout.addWidget(self.integer_label_2)
        layout.addWidget(self.integer_input_2)
       
        #layout.addWidget(options_group)
 

        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)
        #layout.addWidget(self.logger.log_display)
        
        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

 
        self.logger.qt_handler.log_generated.connect(self.update_log_display)


        
        try:
            with open('last_dir.pickle', 'rb') as f:
                last_dir = pickle.load(f)
                self.directory_label.setText(f"Selected directory: {last_dir}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory path, just ignore the error
            
        try:
            with open('last_dir2.pickle', 'rb') as f:
                last_dir2 = pickle.load(f)
                self.spinepipedir_label.setText(f"Selected directory: {last_dir2}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory2 path, just ignore the error



    @pyqtSlot()
    def get_directories(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select data directory')
        if directory:
            self.directory_label.setText(f"Selected directory: {directory}")

            # Save the selected directory path
            with open('last_dir.pickle', 'wb') as f:
                pickle.dump(directory, f)
    
    @pyqtSlot()
    def get_spinepipedir(self):
        spinepipedir = QFileDialog.getExistingDirectory(self, 'Select spinepipe directory')
        if spinepipedir:
            self.spinepipedir_label.setText(f"Selected directory: {spinepipedir}")

            # Save the selected directory2 path
            with open('last_dir2.pickle', 'wb') as f:
                pickle.dump(spinepipedir, f)

    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode
        
        directory = self.directory_label.text().replace("Selected directory: ", "")
        if directory == "No directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return

        spinepipe = self.spinepipedir_label.text().replace("Selected directory: ", "")
        if spinepipe == "No directory selected.":
            QMessageBox.critical(self, "Error", "No directory selected.")
            self.progress.setVisible(False)
            return
        
        try:
            GPU_block = list(map(int, self.integer_input.text().split(',')))
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for GPU block.")
            self.progress.setVisible(False)
            return
        
        try:
            spine_vol = list(map(int, self.integer_input_2.text().split(',')))
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for spine volume.")
            self.progress.setVisible(False)
            return
        
        
        directory =  directory + "/"
        
        spinepipe = spinepipe +"/"
        
        #channel_options = [self.analyze1.isChecked(), self.analyze2.isChecked(), self.analyze3.isChecked(),self.analyze4.isChecked()]
        other_options = [self.save_intermediate.isChecked()]
        
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        #self.worker = Worker(spinepipe, directory, channel_options, integers, self.logger.get_logger())
        self.worker = Worker(spinepipe, directory, other_options, GPU_block, spine_vol, self.logger)
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
        
    
app = QApplication([])
window = MainWindow()
window.setWindowTitle(f'SpinePipe - Version: {__version__}')
window.setGeometry(100, 100, 1200, 800)  
window.show()
sys.exit(app.exec_())