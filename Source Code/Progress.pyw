import logging
from tkinter import ttk
from tkinter import *
from threading import Thread


"""
Main pupose: initialize each base parameter to classification operation usage
"""
class Progress():
	"""
	Main pupose: start processing bar view
	@param	self	this object
	"""
	def start_progress(self):
		self.progress_bar.start()
		self.parent.pack()
		
		
	"""
	Main pupose: end processing bar view
	@param	self	this object
	"""
	def stop_progress(self):
		self.progress_bar.stop()
		
		
	"""
	Main pupose: initialize new processing view to its parent
	@param	self	this object
	"""
	def __init__(self,parent):
		self.parent = parent
		self.progress_bar = ttk.Progressbar(self.parent,length = 550,maximum = 100,takefocus = True,orient = 'horizontal',mode = 'indeterminate')
		self.progress_bar.step(1.0)
		self.progress_bar.pack(fill = BOTH, expand = False, padx = 5, pady = 5)
		self.start_progress()
			
			
			