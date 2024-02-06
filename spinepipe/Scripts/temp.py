# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:16:58 2023

"""
__title__     = 'SpinePipe'
__version__   = '0.9.5'
__date__      = "6 December, 2023"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2023 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/spinepipe'

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QTabWidget,
                             QPushButton, QCheckBox, QLabel, QLineEdit, QComboBox, 
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
    
    def clear_log(self, log_path):
        # Open the log file in write mode to clear it
        with open(log_path, 'w'):
            pass

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

class AnalysisWorker(QThread):
    task_done = pyqtSignal(str)

    def __init__(self, spinepipe, directory, other_options, min_dendrite_vol, spine_vol, spine_dist, HistMatch, Track, reg_method, logger):
        
        super().__init__()
        self.spinepipe = spinepipe
        self.directory = directory
        self.other_options = other_options
        self.min_dendrite_vol = min_dendrite_vol
        self.spine_vol = spine_vol
        self.spine_dist = spine_dist
        self.HistMatch = HistMatch
        self.Track = Track
        self.reg_method = reg_method
        self.logger = logger
        

    def run(self):
        self.is_running = True
        
        try:
            
            sys.path.append(self.spinepipe)
            from spinepipe.Environment import main, imgan, timer, strk
            main.check_gpu()
            
            if len(os.listdir(self.directory)) == 0:
                self.logger.info("Selected directory contains no subfolders! \nPlease ensure your datasets are stored in subfolders nested within the selected directory.")
            
            for subfolder in os.listdir(self.directory):
                subfolder_path = os.path.join(self.directory, subfolder)
                if os.path.isdir(subfolder_path): 
                    subfolder_path = subfolder_path +"/"    
                    #print(f"Processing subfolder: {subfolder_path}")
                    #Load in experiment parameters and analysis settings   
                    settings, locations = main.initialize_spinepipe(subfolder_path)
                
                    #import spinepipe.Environment
                    
                    log_path = subfolder_path +'SpinePipe_Log' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
                    
                    self.logger.set_log_file(log_path)  # Note: we're not passing a log_display, because it's not thread-safe
                    
                    self.logger.clear_log(log_path)
                
                    self.logger.info("SpinePipe Version: "+__version__)
                    self.logger.info("Release Date: "+__date__) 
                    self.logger.info("Created by: "+__author__+"")
                    self.logger.info("Zuckerman Institute, Columbia University\n")
                      
                   
                                   
                    #Modify specific parameters and settings:    
                    settings.save_intermediate_data = False
                    #settings.spine_roi_volume_size = 4 #in microns in x, y, z - approx 50px for 0.3 resolution data
                    settings.min_dendrite_vol = self.min_dendrite_vol #dims used for processing images in block for cell extraction. Reduce if recieving out of memory errors
                    settings.neuron_spine_size = [round(x / (settings.input_resXY*settings.input_resXY*settings.input_resZ),0) for x in self.spine_vol] 
                    settings.neuron_spine_dist = round(self.spine_dist / (settings.input_resXY),2)
                    settings.HistMatch = self.HistMatch
                    settings.Track = self.Track
                    settings.reg_method = self.reg_method
        
        
                    self.logger.info("Processing folder: "+subfolder_path)
                    self.logger.info(f" Image resolution: {settings.input_resXY}um XY, {settings.input_resZ}um Z")
                    self.logger.info(f" Model used: {settings.neuron_seg_model_path}")    
                    self.logger.info(f" Model resolution: {settings.neuron_seg_model_res[0]}um XY, {settings.neuron_seg_model_res[2]}um Z")    
                    self.logger.info(f" Spine volume set to: {self.spine_vol[0]} to {self.spine_vol[1]} um3, {settings.neuron_spine_size[0]} to {settings.neuron_spine_size[1]} voxels.") 
                    self.logger.info(f" Spine distance filter set to: {self.spine_dist} um, {settings.neuron_spine_dist} pixels") 
                    self.logger.info(f" GPU block size set to: {settings.GPU_block_size[0]},{settings.GPU_block_size[1]},{settings.GPU_block_size[1]}") 
                    self.logger.info(f" Tracking set to: {settings.Track}, using {settings.reg_method} registration.") 
                    self.logger.info(f" {settings.neuron_seg_model_path}um Z")
                    self.logger.info("")
                    #Processing
            
            
                    
                    log = imgan.restore_and_segment(subfolder_path, settings, locations, self.logger)
            
                    imgan.analyze_spines(settings, locations, log, self.logger)
                    
                    if settings.Track == True:
                        strk.track_spines(settings, locations, log, self.logger)
                        imgan.analyze_spines_4D(settings, locations, log, self.logger)
                    
                    self.logger.info("")
                    self.logger.info("SpinePipe analysis complete.")
                    self.logger.info("SpinePipe Version: "+__version__)
                    self.logger.info("Release Date: "+__date__+"") 
                    self.logger.info("Created by: "+__author__+"") 
                    self.logger.info("Department of Neurology, The Ohio State University")
                    self.logger.info("Zuckerman Institute, Columbia University\n") 
                    self.logger.info("--------------------------------------------------------------------")
            
            self.task_done.emit("")
            
            self.is_running = False
            

        
        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")

class ValidationWorker(QThread):
    task_done = pyqtSignal(str)

    #def __init__(self, spinepipe, ground_truth, analysis_output, logger):
    def __init__(self, ground_truth, analysis_output, min_dendrite_vol, spine_vol, spine_dist, logger):
        super().__init__()
        #self.spinepipe = spinepipe
        self.ground_truth = ground_truth
        self.analysis_output = analysis_output
        self.logger = logger
        self.min_dendrite_vol = min_dendrite_vol
        self.spine_vol = spine_vol
        self.spine_dist = spine_dist
        

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
            
            settings.min_dendrite_vol = self.min_dendrite_vol #dims used for processing images in block for cell extraction. Reduce if recieving out of memory errors
            settings.neuron_spine_size = [round(x / (settings.input_resXY*settings.input_resXY*settings.input_resZ),0) for x in self.spine_vol] 
            settings.neuron_spine_dist = round(self.spine_dist / (settings.input_resXY),2)
            
            
            self.logger.info("Processing folder: "+self.analysis_output)
            self.logger.info(f" Image resolution: {settings.input_resXY}um XY, {settings.input_resZ}um Z")
            self.logger.info(f" Model used: {settings.neuron_seg_model_path}")    
            self.logger.info(f" Model resolution: {settings.neuron_seg_model_res[0]}um XY, {settings.neuron_seg_model_res[2]}um Z")    
            self.logger.info(f" Spine volume set to: {self.spine_vol[0]} to {self.spine_vol[1]} um3, {settings.neuron_spine_size[0]} to {settings.neuron_spine_size[1]} voxels.") 
            self.logger.info(f" Spine distance filter set to: {self.spine_dist} um, {settings.neuron_spine_dist} pixels") 
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
    def __init__(self):
        super().__init__()
        
        # Create the tab widget
        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)

        # Create instances of the complex widgets
        widget1 = SpinePipeAnalysis()
        widget2 = SpinePipeValidation()

        # Add the complex widgets as tabs
        tab_widget.addTab(widget1, "SpinePipe Analysis")
        tab_widget.addTab(widget2, "Analysis Validation")


        #self.setWindowTitle("SpinePipe")
        #self.setGeometry(100, 100, 800, 600)

class SpinePipeValidation(QWidget):
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
      
        
        self.logger = Logger()
        
        self.spinepipedir_label = QLabel("No SpinePipe directory selected.")
        self.spinepipedir_button = QPushButton("Select SpinePipe directory")
        self.spinepipedir_button.clicked.connect(self.get_spinepipedir)
        
        self.ground_truth_dir_label = QLabel("No ground truth data directory selected.")
        self.ground_truth_dir_button = QPushButton("Select ground truth data directory")
        self.ground_truth_dir_button.clicked.connect(self.get_ground_truth_dir)
        
        self.directory_label = QLabel("No analysis output directory selected.")
        self.directory_button = QPushButton("Select data analysis output directory")
        self.directory_button.clicked.connect(self.get_directories)
        
        
 
        self.integer_label = QLabel("Please ensure the Analysis_Settings.csv is in the folder of analysis outputs you wish to validate.")
        
        
        options_group2 = QGroupBox("Analysis Options:")
        options_layout2 = QVBoxLayout()
        options_group2.setLayout(options_layout2)
        self.save_intermediate = QCheckBox("Save Intermediate Data")
        self.save_intermediate.setChecked(False) 
        self.float_label_1 = QLabel("Minimum dendrite size in um3 (dendrites smaller than this will be ignored):")
        self.float_input_1 = QLineEdit("15")
        self.float_label_2 = QLabel("Spine volume range (min, max volume in um3):")
        self.float_input_2 = QLineEdit("0.03,15")
        self.float_label_3 = QLabel("Spine distance threshold (max distance from dendrite in um):")
        self.float_input_3 = QLineEdit("4")
        
        

        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        

        #layout.addWidget(self.spinepipedir_label)
        #layout.addWidget(self.spinepipedir_button)
        layout.addWidget(self.spinepipedir_button)
        layout.addWidget(self.spinepipedir_label)
        layout.addWidget(self.ground_truth_dir_button)
        layout.addWidget(self.ground_truth_dir_label)
        layout.addWidget(self.directory_button)
        layout.addWidget(self.directory_label)
        layout.addWidget(self.integer_label)
        #layout.addWidget(options_group)
 
        options_layout2.addWidget(self.float_label_1)
        options_layout2.addWidget(self.float_input_1)
        options_layout2.addWidget(self.float_label_2)
        options_layout2.addWidget(self.float_input_2)
        options_layout2.addWidget(self.float_label_3)
        options_layout2.addWidget(self.float_input_3)  
 
        layout.addWidget(options_group2)
        
        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)
        #layout.addWidget(self.logger.log_display)
        
        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

 
        self.logger.qt_handler.log_generated.connect(self.update_log_display)

        try:
            with open('spinepipe_dir.pickle', 'rb') as f:
                spinepipe_dir = pickle.load(f)
                self.spinepipedir_label.setText(f"Selected directory: {spinepipe_dir}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory2 path, just ignore the error

        try:
            with open('gt_dir.pickle', 'rb') as f:
                gt_dir = pickle.load(f)
                self.ground_truth_dir_label.setText(f"Selected directory: {gt_dir}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory2 path, just ignore the error

        
        try:
            with open('analysis_output_dir.pickle', 'rb') as f:
                analysis_output_dir = pickle.load(f)
                self.directory_label.setText(f"Selected directory: {analysis_output_dir}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory path, just ignore the error
            
        self.setLayout(layout)

    @pyqtSlot()
    def get_spinepipedir(self):
        spinepipedir = QFileDialog.getExistingDirectory(self, 'Select SpinePipe directory')
        if spinepipedir:
            self.spinepipedir_label.setText(f"Selected directory: {spinepipedir}")
    
            # Save the selected directory2 path
            with open('spinepipe_dir.pickle', 'wb') as f:
                pickle.dump(spinepipedir, f)
                
    @pyqtSlot()
    def get_ground_truth_dir(self):
        ground_truth_dir = QFileDialog.getExistingDirectory(self, 'Select ground truth data directory')
        if ground_truth_dir:
            self.ground_truth_dir_label.setText(f"Selected directory: {ground_truth_dir}")
    
            # Save the selected directory2 path
            with open('gt_dir.pickle', 'wb') as f:
                pickle.dump(ground_truth_dir, f)
                
    @pyqtSlot()
    def get_directories(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select data analysis output directory')
        if directory:
            self.directory_label.setText(f"Selected directory: {directory}")

            # Save the selected directory path
            with open('analysis_output_dir.pickle', 'wb') as f:
                pickle.dump(directory, f)
    
    

    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode
        
        
        spinepipe = self.spinepipedir_label.text().replace("Selected directory: ", "")
        if spinepipe == "No SpinePipe directory selected.":
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
        
        #spinepipe = spinepipe +"/"
        
        ground_truth = ground_truth +"/"
        
        analysis_output =  directory + "/"
        
        
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        #self.worker = Worker(spinepipe, directory, channel_options, integers, self.logger.get_logger())
        #self.worker = Worker(spinepipe, ground_truth, analysis_output, self.logger)
        self.worker = ValidationWorker(ground_truth, analysis_output, min_dendrite_vol, spine_vol, spine_dist, self.logger)
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
        
        
        
        
        
        
    
        
class SpinePipeAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        
        
        self.logger = Logger()
        

        self.spinepipedir_label = QLabel("No SpinePipe directory selected.")
        self.spinepipedir_button = QPushButton("Select SpinePipe directory")
        self.spinepipedir_button.clicked.connect(self.get_spinepipedir)
        
        self.directory_label = QLabel("No directory selected.")
        self.directory_button = QPushButton("Select data directory")
        self.directory_button.clicked.connect(self.get_directories)
        
        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

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

        options_group2 = QGroupBox("Analysis Options:")
        options_layout2 = QVBoxLayout()
        options_group2.setLayout(options_layout2)
        
        self.save_intermediate = QCheckBox("Save Intermediate Data")
        self.save_intermediate.setChecked(False) 
        self.float_label_1 = QLabel("Minimum dendrite size in um3 (dendrites smaller than this will be ignored):")
        self.float_input_1 = QLineEdit("15")
        self.float_label_2 = QLabel("Spine volume range (min, max volume in um3):")
        self.float_input_2 = QLineEdit("0.03,15")
        self.float_label_3 = QLabel("Spine distance threshold (max distance from dendrite in um):")
        self.float_input_3 = QLineEdit("4")
        self.HistMatch = QCheckBox("Histogram Matching (matches image histograms to first image in the series)")
        self.HistMatch.setChecked(False) 
        self.Track = QCheckBox("Spine Tracking (track spines over time, subfolders should contain sperate volumes for each timepoint)")
        self.Track.setChecked(False) 
        self.reg_label = QLabel("Select registration method for aligning volumes across time:")
        self.reg_method = QComboBox()
        reg_options = ["Rigid", "Elastic"]
        for option in reg_options:
            self.reg_method.addItem(option)
        
        
        
        options_layout2.addWidget(self.save_intermediate)

        options_layout2.addWidget(self.HistMatch)
        options_layout2.addWidget(self.Track)
        options_layout2.addWidget(self.reg_label)
        options_layout2.addWidget(self.reg_method)
        
        options_layout2.addWidget(self.float_label_1)
        options_layout2.addWidget(self.float_input_1)
        options_layout2.addWidget(self.float_label_2)
        options_layout2.addWidget(self.float_input_2)
        options_layout2.addWidget(self.float_label_3)
        options_layout2.addWidget(self.float_input_3)

        

        run_cancel_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_function)
        run_cancel_layout.addWidget(self.run_button)
        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self.close)
        run_cancel_layout.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        
        #self.ground_truth_dir_button.setFixedWidth(300)
        self.directory_button.setFixedWidth(300)
        #self.run_button.setFixedWidth(300)
        #self.cancel_button.setFixedWidth(300)


        layout.addWidget(self.spinepipedir_button)
        layout.addWidget(self.spinepipedir_label)
        layout.addWidget(self.directory_button)
        layout.addWidget(self.directory_label)
        layout.addWidget(self.line)

        layout.addWidget(options_group2)
    

       
        #layout.addWidget(options_group)
 

        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)
        #layout.addWidget(self.logger.log_display)
        
        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

 
        self.logger.qt_handler.log_generated.connect(self.update_log_display)
        
        self.setLayout(layout)

        
        try:
            with open('analysis_dir.pickle', 'rb') as f:
                analysis_dir = pickle.load(f)
                self.directory_label.setText(f"Selected directory: {analysis_dir}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory path, just ignore the error
            
        try:
            with open('spinepipe_dir.pickle', 'rb') as f:
                spinepipe_dir = pickle.load(f)
                self.spinepipedir_label.setText(f"Selected directory: {spinepipe_dir}")
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the last directory2 path, just ignore the error




    @pyqtSlot()
    def get_directories(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select data directory')
        if directory:
            self.directory_label.setText(f"Selected directory: {directory}")

            # Save the selected directory path
            with open('analysis_dir.pickle', 'wb') as f:
                pickle.dump(directory, f)
    
    @pyqtSlot()
    def get_spinepipedir(self):
        spinepipedir = QFileDialog.getExistingDirectory(self, 'Select spinepipe directory')
        if spinepipedir:
            self.spinepipedir_label.setText(f"Selected directory: {spinepipedir}")

            # Save the selected directory2 path
            with open('spinepipe_dir.pickle', 'wb') as f:
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
        """
        try:
            GPU_block = list(map(int, self.integer_input.text().split(',')))
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid input for GPU block.")
            self.progress.setVisible(False)
            return
        """
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
        
        reg_method =  str(self.reg_method.currentText())
        
        
        HistMatch = self.HistMatch.isChecked()
        Track = self.Track.isChecked()
        
        directory =  directory + "/"
        
        spinepipe = spinepipe +"/"
        
        #channel_options = [self.analyze1.isChecked(), self.analyze2.isChecked(), self.analyze3.isChecked(),self.analyze4.isChecked()]
        other_options = [self.save_intermediate.isChecked()]
        
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        #self.worker = Worker(spinepipe, directory, channel_options, integers, self.logger.get_logger())
        self.worker = AnalysisWorker(spinepipe, directory, other_options, min_dendrite_vol, spine_vol, spine_dist, HistMatch, Track, reg_method, self.logger)
       
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
        
        
        
        
        #layout = QVBoxLayout(self)
        #self.button = QPushButton("Run Program 1", self)
        ##self.button.clicked.connect(self.run_program_1)
        #layout.addWidget(self.button)

#    def run_program_1(self):
 #       # Code to run the first program
  #      print("Running Program 1")

class TabTwo(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.button = QPushButton("Run Program 2", self)
        self.button.clicked.connect(self.run_program_2)
        layout.addWidget(self.button)

    def run_program_2(self):
        # Code to run the second program
        print("Running Program 2")


#class MainWindow(QMainWindow):
 #   def __init__(self, *args, **kwargs):
 #       super(MainWindow, self).__init__(*args, **kwargs)

        
    
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


splash = Splash("SpinePipe loading...", 3000)
splash.show()

# Ensures that the application is fully up and running before closing the splash screen
app.processEvents()

window = MainWindow()
window.setWindowTitle(f' SpinePipe - Version: {__version__}')
window.setGeometry(100, 100, 1200, 1200)  
window.show()
sys.exit(app.exec_())