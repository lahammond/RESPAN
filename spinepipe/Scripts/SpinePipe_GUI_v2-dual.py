# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:16:58 2023

"""
__title__     = 'SpinePipe'
__version__   = '0.9.7'
__date__      = "2 February, 2024"
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

       
    def __init__(self, spinepipe, directory, model_dir, save_intermediate, save_validation,
                                     inputxy, inputz, modelxy, modelz, neuron_ch, analysis_method,
                                     image_restore, axial_restore,
                                     min_dendrite_vol, spine_vol, spine_dist, HistMatch,
                                     Track, reg_method, use_yaml_res, logger):

        
        super().__init__()
        self.spinepipe = spinepipe
        self.directory = directory
        self.model_dir = model_dir
        self.save_intermediate = save_intermediate
        self.save_validation = save_validation
        self.image_restore = image_restore
        self.axial_restore = axial_restore
        self.inputxy = inputxy
        self.inputz = inputz
        self.modelxy = modelxy
        self.modelz = modelz
        self.neuron_ch = neuron_ch
        self.analysis_method = analysis_method
        self.min_dendrite_vol = min_dendrite_vol
        self.spine_vol = spine_vol
        self.spine_dist = spine_dist
        self.HistMatch = HistMatch
        self.Track = Track
        self.reg_method = reg_method
        self.logger = logger
        self.use_yaml_res = use_yaml_res
        

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
                    settings.save_intermediate_data = self.save_intermediate
                    settings.save_val_data = self.save_validation
                    settings.neuron_seg_model_path = self.model_dir
                    if self.use_yaml_res == False:
                        settings.input_resXY = self.inputxy
                        settings.input_resZ = self.inputz      
                        settings.model_resXY = self.modelxy
                        settings.model_resZ = self.modelz
                    
                    settings.image_restore = self.image_restore
                    settings.axial_restore = self.axial_restore
                    settings.neuron_channel = int(self.neuron_ch)
                    settings.analysis_method = self.analysis_method
                    #settings.spine_roi_volume_size = 4 #in microns in x, y, z - approx 50px for 0.3 resolution data
                    settings.min_dendrite_vol = round(self.min_dendrite_vol / settings.input_resXY/settings.input_resXY/settings.input_resZ, 0)
                    settings.neuron_spine_size = [round(x / (settings.input_resXY*settings.input_resXY*settings.input_resZ),0) for x in self.spine_vol] 
                    settings.neuron_spine_dist = round(self.spine_dist / (settings.input_resXY),2)
                    settings.HistMatch = self.HistMatch
                    settings.Track = self.Track
                    settings.reg_method = self.reg_method
                    
        
                    self.logger.info("Processing folder: "+subfolder_path)
                    self.logger.info(f" Image resolution: {settings.input_resXY}um XY, {settings.input_resZ}um Z")
                    self.logger.info(f" Model used: {settings.neuron_seg_model_path}")    
                    self.logger.info(f" Model resolution: {settings.model_resXY}um XY, {settings.model_resZ}um Z")
                    self.logger.info(f" Dendrite volume set to: {self.min_dendrite_vol} um, {settings.min_dendrite_vol} voxels") 
                    self.logger.info(f" Spine volume set to: {self.spine_vol[0]} to {self.spine_vol[1]} um3, {settings.neuron_spine_size[0]} to {settings.neuron_spine_size[1]} voxels.") 
                    self.logger.info(f" Spine distance filter set to: {self.spine_dist} um, {settings.neuron_spine_dist} pixels") 
                    self.logger.info(f" Analysis method: {self.analysis_method}") 
                    self.logger.info(f" GPU block size set to: {settings.GPU_block_size[0]},{settings.GPU_block_size[1]},{settings.GPU_block_size[1]}") 
                    self.logger.info(f" Tracking set to: {settings.Track}, using {settings.reg_method} registration.") 
                    self.logger.info(f" Image restoration set to {self.image_restore} and axial restoration set to {self.axial_restore}")
                    self.logger.info("")
                    #Processing
            
            
                    
                    log = imgan.restore_and_segment(settings, locations, self.logger)
            
                    imgan.analyze_spines(settings, locations, log, self.logger)
                    
                    if settings.Track == True:
                        strk.track_spines(settings, locations, log, self.logger)
                        imgan.analyze_spines_4D(settings, locations, log, self.logger)
                    
                    self.logger.info("SpinePipe analysis complete.")
                    self.logger.info("SpinePipe Version: "+__version__)
                    self.logger.info("Release Date: "+__date__+"") 
                    self.logger.info("Created by: "+__author__+"") 
                    self.logger.info("Department of Neurology, The Ohio State University")
                    self.logger.info("Zuckerman Institute, Columbia University\n") 
                    self.logger.info("-----------------------------------------------------------------------------------------------------")
            
            self.task_done.emit("")
            
            self.is_running = False
            

        
        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            self.task_done.emit("An error occurred.")

class ValidationWorker(QThread):
    task_done = pyqtSignal(str)

    #def __init__(self, spinepipe, ground_truth, analysis_output, logger):
    def __init__(self, spinepipe, ground_truth, analysis_output, neuron_ch, min_dendrite_vol, spine_vol, spine_dist, inputxy, inputz, logger):
        super().__init__()
        self.spinepipe = spinepipe
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
            
            sys.path.append(self.spinepipe)
            from spinepipe.Environment import main, val
            main.check_gpu()
            #Load in experiment parameters and analysis settings   
            settings, locations = main.initialize_spinepipe(self.analysis_output)
        
            import spinepipe.Environment
            
            log_path = self.analysis_output +'SpinePipe_Validation_Log' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.log'
            self.logger.set_log_file(log_path)  # Note: we're not passing a log_display, because it's not thread-safe
            
        
            self.logger.info("SpinePipe Validation Tool - Version: "+__version__)
            self.logger.info("Release Date: "+__date__) 
            self.logger.info("Created by: "+__author__+"")
            self.logger.info("Zuckerman Institute, Columbia University\n")
              
           
            #Load in experiment parameters and analysis settings   
            settings, locations = main.initialize_spinepipe_validation(self.analysis_output)
            
            
            settings.input_resXY = self.inputxy
            settings.input_resZ = self.inputz
            settings.neuron_channel = self.neuron_ch
            settings.min_dendrite_vol = self.min_dendrite_vol #dims used for processing images in block for cell extraction. Reduce if recieving out of memory errors
            settings.neuron_spine_size = [round(x / (settings.input_resXY*settings.input_resXY*settings.input_resZ),0) for x in self.spine_vol] 
            settings.neuron_spine_dist = round(self.spine_dist / (settings.input_resXY),2)
            
            
            self.logger.info("Processing folder: "+self.analysis_output)
            self.logger.info(f" Image resolution: {settings.input_resXY}um XY, {settings.input_resZ}um Z")
   
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
      
        tab_widget.setStyleSheet("QTabBar::tab { font-size: 10pt; width: 200px;}")



        #self.setWindowTitle("SpinePipe")
        #self.setGeometry(100, 100, 800, 600)

class SpinePipeValidation(QWidget):
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
        
        self.spinepipedir_label = QLabel("No SpinePipe directory selected.")
        self.spinepipedir_button = QPushButton("Select SpinePipe directory")
        self.spinepipedir_button.clicked.connect(self.get_spinepipedir)
        
        self.ground_truth_dir_label = QLabel("No ground truth data directory selected.")
        self.ground_truth_dir_button = QPushButton("Select ground truth data directory")
        self.ground_truth_dir_button.clicked.connect(self.get_gtdir)
        
        self.directory_label = QLabel("No analysis output directory selected.")
        self.directory_button = QPushButton("Select data analysis output directory")
        self.directory_button.clicked.connect(self.get_outputdir)
        
        
 
        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.info_label2 = QLabel("Ensure the correct Analysis_Settings.yaml file is included in each subfolder.")
        self.line2 = QFrame()
        self.line2.setFrameShape(QFrame.HLine)
        self.line2.setFrameShadow(QFrame.Sunken)
        
        self.spinepipedir_button.setFixedWidth(300)
        self.directory_button.setFixedWidth(300)
        self.ground_truth_dir_button.setFixedWidth(300)
        
        
        dir_options_layout.addWidget(self.spinepipedir_button)
        dir_options_layout.addWidget(self.spinepipedir_label)
        dir_options_layout.addWidget(self.directory_button)
        dir_options_layout.addWidget(self.directory_label)
        dir_options_layout.addWidget(self.ground_truth_dir_button)
        dir_options_layout.addWidget(self.ground_truth_dir_label)

        dir_options_layout.addWidget(self.line)
  
        dir_options_layout.addWidget(self.info_label2)
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
        
        self.inputdata_xy_label = QLabel("Image voxel size XY (um):")
        self.inputdata_xy = QLineEdit("0.102")
        self.inputdata_z_label = QLabel("Image voxel size Z (um):")
        self.inputdata_z = QLineEdit("1")
        horizontal_input = QHBoxLayout()
        horizontal_input.addWidget(self.inputdata_xy_label)
        horizontal_input.addWidget(self.inputdata_xy)
        horizontal_input.addWidget(self.inputdata_z_label)
        horizontal_input.addWidget(self.inputdata_z)  
        
        self.neuron_channel_label = QLabel("Neuron/dendrite channel:")
        self.neuron_channel_input = QLineEdit("1") 
        self.float_label_1 = QLabel("Minimum dendrite size in um3 (dendrites smaller than this will be ignored):")
        self.float_input_1 = QLineEdit("15")
        self.float_label_2 = QLabel("Spine volume filter (min, max volume in um3):")
        self.float_input_2 = QLineEdit("0.03,15")
        self.float_label_3 = QLabel("Spine distance filter (max distance from dendrite in um):")
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
        
        options_group1.setFixedWidth(600)
        
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

 
        layout.addLayout(run_cancel_layout)
        layout.addWidget(self.progress)
        #layout.addWidget(self.logger.log_display)
        
        self.log_display = QTextEdit()
        layout.addWidget(self.log_display)

 
        self.logger.qt_handler.log_generated.connect(self.update_log_display)
        self.setLayout(layout)

        try:
            with open('parametesrvalGUI.pkl', 'rb') as f:
                variables_dict = pickle.load(f)
                
            #retreive
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
            self.spinepipedir_label.setText(f"Selected directory: {spine_dir}")
      
            self.inputdata_xy.setText(str(inputxy))
            self.inputdata_z.setText(str(inputz))
            
            self.neuron_channel_input.setText(str(neuron_ch))
            self.float_input_1.setText(str(min_dend))
            self.float_input_2.setText(str(spine_vol))
            self.float_input_3.setText(str(spine_dist))
            
               
            
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the pickle file, just ignore the error
        
    @pyqtSlot()
    def get_spinepipedir(self):
         spinepipedir = QFileDialog.getExistingDirectory(self, 'Select spinepipe directory')
         if spinepipedir:
             self.spinepipedir_label.setText(f"Selected directory: {spinepipedir}")
             
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
        spinedir_text = self.spinepipedir_label.text()
        
        
        data_dir = dirlabel_text.split(": ")[-1]
        gt_dir = gtlabel_text.split(": ")[-1]
        spine_dir = spinedir_text.split(": ")[-1]
        

        inputxy = str(self.inputdata_xy.text())
        inputz = str(self.inputdata_z.text())
        
        neuron_ch = str(self.neuron_channel_input.text())
        min_dend = float(self.float_input_1.text())
        spine_vol = float(self.float_input_2.text())
        spine_dist = float(self.float_input_3.text())
        
       
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
        with open('parametesrvalGUI.pkl', 'wb') as f:
            pickle.dump(variables_dict, f)
            

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
        
        spinepipe = spinepipe +"/"
        
        ground_truth = ground_truth +"/"
        
        analysis_output =  directory + "/"
        
        
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        #self.worker = Worker(spinepipe, directory, channel_options, integers, self.logger.get_logger())
        #self.worker = Worker(spinepipe, ground_truth, analysis_output, self.logger)
        self.worker = ValidationWorker(spinepipe, ground_truth, analysis_output, neuron_ch, min_dendrite_vol, spine_vol, spine_dist, inputxy, inputz, self.logger)
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

        self.spinepipedir_label = QLabel("No SpinePipe directory selected.")
        self.spinepipedir_button = QPushButton("Select SpinePipe directory")
        self.spinepipedir_button.clicked.connect(self.get_spinepipedir)
        
        self.directory_label = QLabel("No directory selected.")
        self.directory_button = QPushButton("Select data directory")
        self.directory_button.clicked.connect(self.get_datadir)
        

        self.model_directory_label = QLabel("No model directory selected.")
        self.model_directory_button = QPushButton("Select model directory")
        self.model_directory_button.clicked.connect(self.get_modeldir)
        
        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.info_label = QLabel("SpinePipe will process data stored in subfolders of the data directory.")
        self.info_label2 = QLabel("Ensure the correct Analysis_Settings.yaml file is included in each subfolder.")
        self.line2 = QFrame()
        self.line2.setFrameShape(QFrame.HLine)
        self.line2.setFrameShadow(QFrame.Sunken)
        
        self.spinepipedir_button.setFixedWidth(300)
        self.directory_button.setFixedWidth(300)
        self.model_directory_button.setFixedWidth(300)
        
        dir_options_layout.addWidget(self.spinepipedir_button)
        dir_options_layout.addWidget(self.spinepipedir_label)
        dir_options_layout.addWidget(self.directory_button)
        dir_options_layout.addWidget(self.directory_label)
        dir_options_layout.addWidget(self.model_directory_button)
        dir_options_layout.addWidget(self.model_directory_label)

        dir_options_layout.addWidget(self.line)
        dir_options_layout.addWidget(self.info_label)
        dir_options_layout.addWidget(self.info_label2)
        dir_options_layout.addWidget(self.line2)

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
        
        self.inputdata_xy_label = QLabel("Image voxel size XY (um):")
        self.inputdata_xy = QLineEdit("0.102")
        self.inputdata_z_label = QLabel("Image voxel size Z (um):")
        self.inputdata_z = QLineEdit("1")
        horizontal_input = QHBoxLayout()
        horizontal_input.addWidget(self.inputdata_xy_label)
        horizontal_input.addWidget(self.inputdata_xy)
        horizontal_input.addWidget(self.inputdata_z_label)
        horizontal_input.addWidget(self.inputdata_z)
        
        self.modeldata_xy_label = QLabel("Model voxel size XY (um):")
        self.modeldata_xy = QLineEdit("0.102")
        self.modeldata_z_label = QLabel("Model voxel size Z (um):")
        self.modeldata_z = QLineEdit("1")
        self.res_opt = QCheckBox("Use voxel sizes in analysis_settings.yaml")
        self.res_opt.setChecked(False)
        self.res_opt.stateChanged.connect(self.toggle_res)
        
        horizontal_model = QHBoxLayout()
        horizontal_model.addWidget(self.modeldata_xy_label)
        horizontal_model.addWidget(self.modeldata_xy)
        horizontal_model.addWidget(self.modeldata_z_label)
        horizontal_model.addWidget(self.modeldata_z)
        
        
        
        res_options_layout.addLayout(horizontal_input)
        res_options_layout.addLayout(horizontal_model)
        res_options_layout.addWidget(self.res_opt)
        
        
        dir_options.setFixedWidth(600)
        res_options.setFixedWidth(600)
        res_options.setFixedHeight(150)
        
        input_data_and_res = QHBoxLayout()
        input_data_and_res.addWidget(dir_options)
        input_data_and_res.addWidget(res_options, alignment=Qt.AlignTop)
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
        self.neuron_channel_label = QLabel("Neuron/dendrite channel:")
        self.neuron_channel_input = QLineEdit("1")        
        self.float_label_1 = QLabel("Minimum dendrite size in um3 (dendrites smaller than this will be ignored):")
        self.float_input_1 = QLineEdit("15")
        self.float_label_2 = QLabel("Spine volume filter (min, max volume in um3):")
        self.float_input_2 = QLineEdit("0.03,15")
        self.float_label_3 = QLabel("Spine distance filter (max distance from dendrite in um):")
        self.float_input_3 = QLineEdit("4")
        self.analysis_method_label = QLabel("Select analysis method for analyzing spines:")
        self.analysis_method = QComboBox()
        analysis_options = ["Dendrite Specific", "Whole Neuron"]
        for option in analysis_options:
            self.analysis_method.addItem(option)
        
        
        self.neuron_channel_input.setFixedWidth(90)
        self.float_input_1.setFixedWidth(90)
        self.float_input_2.setFixedWidth(90)
        self.float_input_3.setFixedWidth(90)
        self.analysis_method.setFixedWidth(150)

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
        self.image_restore_opt = QCheckBox("Use image restoration (requires trained models)")
        self.image_restore_opt.setChecked(False)
        self.axial_restore_opt = QCheckBox("Use axial restoration (requires trained models)")
        self.axial_restore_opt.setChecked(False)
        self.save_validation = QCheckBox("Save validation data")
        self.save_validation.setChecked(True)
        self.save_intermediate = QCheckBox("Save intermediate data")
        self.save_intermediate.setChecked(False) 
        self.HistMatch = QCheckBox("Histogram matching (matches image histograms to first image in the series)")
        self.HistMatch.setChecked(False) 
        self.Track = QCheckBox("Spine tracking (temporal analysis of spines)")
        self.Track_label = QLabel("Each subfolder should contain a single neuron with sperate volumes for each timepoint.")
        self.Track.setChecked(False) 
        self.reg_label = QLabel("Select registration method for aligning volumes across time:")
        self.reg_method = QComboBox()
        reg_options = ["Rigid", "Elastic"]
        for option in reg_options:
            self.reg_method.addItem(option)
        
        self.reg_method.setFixedWidth(150)
        
        save_options = QHBoxLayout()
        save_options.addWidget(self.save_validation)
        save_options.addWidget(self.save_intermediate)
        
        options_layout2.addLayout(save_options)
        options_layout2.addWidget(self.image_restore_opt)
        options_layout2.addWidget(self.axial_restore_opt)
        
        options_layout2.addWidget(self.HistMatch)
        options_layout2.addWidget(self.Track)
        options_layout2.addWidget(self.Track_label)
        options_layout2.addWidget(self.reg_label)
        options_layout2.addWidget(self.reg_method)
        
        
        options_group1.setFixedWidth(600)
        #options_group1.setFixedHeight(200)
        options_group2.setFixedWidth(600)
        
        main_options = QHBoxLayout()
        main_options.addWidget(options_group1, alignment=Qt.AlignTop)
        main_options.addWidget(options_group2, alignment=Qt.AlignTop)
        

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
            with open('parametersscriptGUI.pkl', 'rb') as f:
                variables_dict = pickle.load(f)
                
            #retreive
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
            analysis_meth = variables_dict.get('analysis_meth', None)
            
            image_restore = variables_dict.get('image_restore', None)
            axial_restore = variables_dict.get('axial_restore', None)
            save_val_data = variables_dict.get('save_val_data', None)
            save_int_data = variables_dict.get('save_int_data', None)
            hist_match = variables_dict.get('hist_match', None)
            spine_track = variables_dict.get('spine_track', None)
            reg_meth = variables_dict.get('reg_meth', None)
            
            
            #udpate GUI:                               
            self.spinepipedir_label.setText(f"Selected directory: {spine_dir}")
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
            self.analysis_method.setCurrentIndex(int(analysis_meth))
            
            self.image_restore_opt.setChecked(image_restore)
            self.axial_restore_opt.setChecked(axial_restore)
            self.save_validation.setChecked(save_val_data)
            self.save_intermediate.setChecked(save_int_data)
            self.HistMatch.setChecked(hist_match)
            self.Track.setChecked(spine_track)
            self.reg_method.setCurrentIndex(int(reg_meth))

            
        except (FileNotFoundError, EOFError, pickle.PickleError):
            pass  # If we can't load the pickle file, just ignore the error
      
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
    def get_spinepipedir(self):
         spinepipedir = QFileDialog.getExistingDirectory(self, 'Select spinepipe directory')
         if spinepipedir:
             self.spinepipedir_label.setText(f"Selected directory: {spinepipedir}")

    @pyqtSlot()
    def get_datadir(self):
         datadir = QFileDialog.getExistingDirectory(self, 'Select data directory')
         if datadir:
             self.directory_label.setText(f"Selected directory: {datadir}")
 
    @pyqtSlot()
    def get_modeldir(self):
         modeldir = QFileDialog.getExistingDirectory(self, 'Select model directory')
         if modeldir:
             self.model_directory_label.setText(f"Selected directory: {modeldir}")
         
      
    @pyqtSlot()
    def get_variables(self):
        
        spinedir_text = self.spinepipedir_label.text()
        
        dirlabel_text = self.directory_label.text()
        
        modellabel_text = self.model_directory_label.text()
        
        
        spine_dir = spinedir_text.split(": ")[-1]
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
        analysis_meth = self.analysis_method.currentIndex()
        
        image_restore = self.image_restore_opt.isChecked()
        axial_restore = self.axial_restore_opt.isChecked()
        save_int_data = self.save_intermediate.isChecked()
        save_val_data = self.save_validation.isChecked()
        hist_match = self.HistMatch.isChecked()
        spine_track = self.Track.isChecked()
        reg_meth = self.reg_method.currentIndex()

        
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
            'analysis_meth': analysis_meth,
            
            'image_restore': image_restore,
            'axial_restore': axial_restore,
            'save_val_data': save_val_data,
            'save_int_data': save_int_data,
            'hist_match': hist_match,
            'spine_track': spine_track,
            'reg_meth': reg_meth
        }
        
        
        # Save the dictionary to a pickle file
        with open('parametersscriptGUI.pkl', 'wb') as f:
            pickle.dump(variables_dict, f)
            
            

    @pyqtSlot()
    def run_function(self):
        self.log_display.clear()  # Clear the log display
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Set to busy mode
        
        spinepipe = self.spinepipedir_label.text().replace("Selected directory: ", "")
        if spinepipe == "No directory selected.":
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
        
        reg_method =  str(self.reg_method.currentText())
        analysis_method =  str(self.analysis_method.currentText())
        
        image_restore = self.image_restore_opt.isChecked()
        axial_restore = self.axial_restore_opt.isChecked()
        save_intermediate = self.save_intermediate.isChecked()
        save_validation = self.save_validation.isChecked()
        HistMatch = self.HistMatch.isChecked()
        Track = self.Track.isChecked()
        use_yaml_res = self.res_opt.isChecked()
        
        directory =  directory + "/"
        
        spinepipe = spinepipe +"/"
        
        #channel_options = [self.analyze1.isChecked(), self.analyze2.isChecked(), self.analyze3.isChecked(),self.analyze4.isChecked()]
        #other_options = [self.save_intermediate.isChecked()]
        
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.task_done.disconnect(self.on_task_done)
            self.logger.qt_handler.log_generated.disconnect(self.update_log_display)


        #self.worker = AnalysisWorker(spinepipe, directory, other_options, min_dendrite_vol, spine_vol, spine_dist, HistMatch, Track, reg_method, self.logger)
        self.worker = AnalysisWorker(spinepipe, directory, model_dir, save_intermediate, save_validation,
                                     inputxy, inputz, modelxy, modelz, neuron_ch, analysis_method,
                                     image_restore, axial_restore,
                                     min_dendrite_vol, spine_vol, spine_dist, HistMatch,
                                     Track, reg_method, use_yaml_res, self.logger)
       
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

'''
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


splash = Splash("SpinePipe loading...", 3000)
splash.show()

# Ensures that the application is fully up and running before closing the splash screen
app.processEvents()

window = MainWindow()
window.setWindowTitle(f' SpinePipe - Version: {__version__}')
window.setGeometry(100, 100, 1200, 1200)  
window.show()
sys.exit(app.exec_())