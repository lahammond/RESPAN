# -*- coding: utf-8 -*-
"""
Timer functions
==========


"""
__title__     = 'spinpipe'
__version__   = '0.1.0'
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'


import time

from dataclasses import dataclass, field
from typing import Any, ClassVar
import functools


##############################################################################
# Timer Functions
##############################################################################

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Processing time: {elapsed_time:0.4f} seconds.")
        return value
    return wrapper_timer


##############################################################################
# Time Classes
##############################################################################



class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer:
    
    timers: ClassVar = {} #dictionary to store accumulating times with different names
    name: Any = None
    text: Any = "Processing time: {:0.1f} seconds."
    #logger: Any = print
    _start_time: Any = field(default=None, init=False, repr=False)
   
    def __post_init__(self):
        """Initialization: add timer to dict of timers"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

            
    """
    timers = {} #dictionary to store accumulating times with different names
    def __init__(
        self, 
        name = None,
        text="Processing time: {:0.1f} seconds.",
        #logger = print
    ):
        self._start_time = None
        self.name = name
        self.text = text
        #self.logger = logger
        
        # Add new named timers to dictionary of timers
        if name:
            self.timers.setdefault(name, 0)        
    """
    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        
        #print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        #if self.logger:
        #    self.logger(self.text.format(elapsed_time))
        
        print(self.text.format(elapsed_time))

        if self.name:
            self.timers[self.name] += elapsed_time
            
        return elapsed_time

    #added to allow Context Manager
    #use by:
        #with Timer():
            #function
    
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()

    def __call__(self, func):
        """Support using Timer as a decorator"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper_timer

