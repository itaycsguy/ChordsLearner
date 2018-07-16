import logging
import time
import itertools as it
import threading
import tkinter.messagebox as tkMes
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from ImgLearner import *
from ExtractIMGS import *
from Progress import *
#import openpyxl


"""
Main pupose: Management the whole GUI application and preparing the initializations of classification requests 
"""
class GUILearner():
	"""
	Main pupose: initialize each base parameter to GUI usage
	@param	self	this object
	"""
	def __init__(self):
		self._OUTTER_EVENT_ = 0 # outter interrupt
		self._INNER_EVENT_ = 1 # trap from that software
		self.lock = threading.Lock()
		self.isFinish = False
		
		self.root = ""
		self.path = ""
		self.defaultNeighbors = "3" #Constants value
		self.logger_name = ""
		
		self.files_amount = 1 # files' creation counter - DO NOT  reset it all the way it is running
		self.percentage = 33 # recommanded by other researcher and consider as good start parameter from Machine Learning reviews
		self.algorithm = -1
		self.neighborsRange = 3 # KNN algorithm usage
		self.focusOn = 1 # notes' focus bydefault
		self.algoLabelIdx = 3 # grid layout
		
		#root window initialize:
		self.root = Tk()
		self.root.protocol("WM_DELETE_WINDOW",self.close)
		self.root.wm_title("Notes Learner")
		self.root.configure(background = 'black')
		#self.root.wm_geometry("550x300") ### doesn't fit to other configurations
		self.root.resizable(width = False, height = False)

		#myPercentagePane:
		self.algo_label = StringVar()
		self.algo_label.set("None ")
		self.def_algo = StringVar()
		self.def_algo.set("")
		
		#main frame:
		self.myframe = Frame(self.root,width = 550, height = 300, background = "gray")
		self.myframe.pack(fill = BOTH, expand = False, padx = 5, pady = 5)
		
		INNER_HEIGHT = 30
		INNER_WIDTH = 500
		#menubar and PanedWindow:
		self.menuBar = Menu(self.myframe)
		self.optionsMenu = Menu(self.menuBar, tearoff = 0)
		self.algosMenu = Menu(self.menuBar, tearoff = 0)
		self.focusMenu = Menu(self.menuBar, tearoff = 0)
		self.myLabelsPane = PanedWindow(self.myframe,width = INNER_WIDTH)
		self.percText = StringVar() # createPercentagesInfo
		self.neighbors = StringVar()
		self.myPercentagePane = PanedWindow(self.myframe,width = INNER_WIDTH)
		self.myXcorrPane = PanedWindow(self.myframe,width = INNER_WIDTH, height = INNER_HEIGHT)
		self.myLearnPane = PanedWindow(self.myframe,width = INNER_WIDTH, height = INNER_HEIGHT)
		self.myProgressPane = PanedWindow(self.root,width = INNER_WIDTH)
		self.menuBar.add_cascade(label = "Options", menu = self.optionsMenu)
		self.optionsMenu.add_command(label = "Database",command = self.browse,state="disabled")
		self.optionsMenu.add_separator()
		self.optionsMenu.add_command(label = "Clear",command = self.reset)
		self.optionsMenu.add_separator()
		self.optionsMenu.add_command(label = "About",command = self.about)

		self.root.config(menu = self.menuBar)
		
		# toplevel:
		self.topKNN = None # represent: "how many neighbors would you like to choose for knn algorithm?"
		self.topMUSTALGO = None # alert which says that the user was not pick some algorithm

		#run GUI:
		self.progress = Progress(self.myProgressPane)
		loading_thread = Thread(target = self.preloading) # loading images and data structure under that action
		loading_thread.start()
		self.root.update_idletasks()
		self.root.update()
		
		self.root.mainloop() # starting the real GUI running
		
	
	"""
	Main purpose: destroy all UI
	@param	self	this object
	"""
	def close(self):
		self.root.destroy()
		
	
	"""
	Main purpose: show details of the developer and that programming plain
	@param	self	this object
	"""
	def about(self):
		about_thread = Thread(target = tkMes.showinfo("ABOUT:\n",
		"Invented&Developed By: Itay Guy\nAt: RBD-Music-lab\nSubject: Notes' writers classification\nInstitution: University Of Haifa\nPublishing Date: xx.xx.2017\n"))
		about_thread.start()
		
	
	"""
	Main purpose: invoke to cross correlation calculation
	@param	self	this object
	@param	event	button click
	"""
	def  xCorrButton(self,event):
		correlation_thread = Thread(target = ImgLearner,args = (self.root,self.percentage,0,self.images,self.files_amount,self.neighborsRange,self.focusOn,))
		correlation_thread.start()
		
	
	"""
	Main purpose: create the cross correlation button
	@param	self	this object
	"""
	def createCrossCorrelationButton(self):
		xCorr = Button(self.myXcorrPane,text='Cross Correlation View',fg = "black",font = ("Ariel", 12,"bold"),relief = RAISED,cursor="hand2")
		xCorr.bind("<ButtonRelease-1>",self.xCorrButton)
		xCorr.pack()
		self.myXcorrPane.add(xCorr)
		self.myXcorrPane.pack(fill = BOTH,padx = 2,pady = 2)
		
	
	"""
	Main purpose: create the Learn operation button
	@param	self	this object
	"""
	def createLearnButton(self):
		if len(self.myLearnPane.winfo_children()) > 0:
			return
		textPred = ""
		if self.focusOn == 1:
			textPred = "Notes"
		elif self.focusOn == 2:
			textPred = "Pages"
		learn = Button(self.myLearnPane,text='Learn&Predict '+textPred+"!",fg = "black",font = ("Ariel", 12,"bold"),relief = RAISED,cursor="hand2")
		learn.bind("<ButtonRelease-1>",self.learnButton)
		learn.pack()
		self.myLearnPane.add(learn)
		self.myLearnPane.pack(fill = BOTH,padx = 2,pady = 2)
		
	
	"""
	Main purpose: create the database label
	@param	self	this object
	"""
	def createLabelsInfo(self):
		if len(self.myLabelsPane.winfo_children()) > 0:
			return
		file_path = str(self.path)
		filepath = StringVar()
		selectednum = StringVar()
		path = Label(self.myLabelsPane,textvariable = filepath,fg = "black",font = ("Ariel", 12,"bold"))
		filepath.set('DB: '+file_path)#('From: '+file_path)
		path.pack()
		self.myLabelsPane.pack(fill = BOTH,padx = 2,pady = 2)	
		
	
	"""
	Main purpose: define the percentage of the training set option after occurring some change
	@param	self	this object
	@param	event	mouse click
	"""
	def define_distribution(self,event):
		#can do int(event.char) to get an integer if it really is,otherwise there is an error occur
		curr_entry = self.myPercentagePane.winfo_children()[2]
		if int(event.type) == 5:
			curr_entry.config(state = 'normal')
		if event.char == '\r': #ENTER occur
			lockByError = False
			try:
				perc = int(self.percText.get())
			except:
				logger.warning("Could Not Handle Of Object Which is different than INT.")
				lockByError = True
			if lockByError == False and perc != self.percentage and perc > 0 and perc < 100:
				self.percentage = perc
				self.percText.set(str(perc))
				curr_entry.config(state = 'disabled')
			elif lockByError == False:
				self.percText.set(str(self.percentage)) #last percentage parameter
				curr_entry.config(state = 'disabled')
			if lockByError == True:
				self.percText.set(str(self.percentage)) #last percentage parameter
				curr_entry.config(state = 'disabled')
		
		
	"""
	Main purpose: create the container which shows the training set and algorithms details
	@param	self	this object
	"""
	def createPercentagesInfo(self):
		meaningLabel = StringVar()
		trainingLabel = Label(self.myPercentagePane,textvariable = meaningLabel,fg = "black",font = ("Ariel", 12,"bold"))
		meaningLabel.set(" Training Set Size:") # the space is programmed hard coded but there are no any problems with it
		percLabel = StringVar()
		label = Label(self.myPercentagePane,textvariable = percLabel,fg = "black",font = ("Ariel", 12,"bold"))
		percLabel.set('%')
		defaultPerc = Entry(self.myPercentagePane, textvariable = self.percText,fg = "black",font=("Ariel", 12,"bold"),width = 2)
		defaultPerc.focus_set()
		self.percText.set(str(self.percentage))
		defaultPerc.config(state = 'disabled')
		trainingLabel.pack(side=LEFT)
		defaultPerc.pack(side=LEFT)
		label.pack(side=LEFT)
		my_algo_def = Label(self.myPercentagePane,textvariable = self.algo_label,fg = "black",font = ("Ariel", 12,"bold"))
		my_algo_def.pack(side=RIGHT)
		
		# defAlgo_label is moved to this place because of the logic order placement between its and my_algo_def:
		defAlgo_label = Label(self.myPercentagePane,textvariable = self.def_algo,fg = "black",font = ("Ariel", 12,"bold"))
		self.def_algo.set("Algorithm:")
		defAlgo_label.pack(side=RIGHT)
		
		defaultPerc.bind("<KeyRelease>",self.define_distribution)
		defaultPerc.bind("<ButtonRelease-1>",self.define_distribution)
		self.myPercentagePane.pack(fill = BOTH,padx = 2,pady = 2)
	
	
	"""
	Main purpose: fetch all pictures and metadata from sqlite server,browse and show progressbar
	@param	self	this object
	"""
	def preloading(self):
		self.root.config(cursor="wait")
		tmp_thread = Thread(target = self.progress.start_progress)
		tmp_thread.start()
		
		self.images = ExtractIMGS()
		self.images.server_fetch()
		self.browse()
		self.progress.stop_progress()
		self.myProgressPane.pack_forget()
		self.root.config(cursor="")
		
	
	"""
	Main purpose: draw the selected algorithm
	@param	self	this object
	@param	excepof	clear the color of existing algorithm selection
	"""
	def clearSelectedAlgos(self,exceptOf):
		if exceptOf != 0:
			self.algosMenu.entryconfig(0, background = "")
		if exceptOf != 2:	
			self.algosMenu.entryconfig(2, background = "")
		if exceptOf != 4:	
			self.algosMenu.entryconfig(4, background = "")
		if exceptOf != 6:	
			self.algosMenu.entryconfig(6, background = "")
		if exceptOf != 8:	
			self.algosMenu.entryconfig(8, background = "")
		if exceptOf != 10:	
			self.algosMenu.entryconfig(10, background = "")
			
	
	"""
	Main purpose: print the neighbors from KNN algorithm to percentage container where the algorithm is written
	@param	self	this object
	@param	neighborNum	number of neighbors - bigger than 1 and less than 10
	"""
	def getNeighbors(self,neighborNum):
		neigh = int(self.defaultNeighbors)
		if neighborNum.char == '\r': #enter is pressed
			try:
				neigh = int(self.neighbors.get())
				if neigh > 10 or neigh < 1:
					print("Too much neighbors range is to be selected.")
					self.neighborsRange = int(self.defaultNeighbors)
					self.neighbors.set(self.defaultNeighbors)
					self.clearSelectedAlgos(-1) #clear all
				else:
					self.neighborsRange = neigh
					self.algo_label = StringVar()
					self.myPercentagePane.winfo_children()[self.algoLabelIdx].config(textvariable = self.algo_label)
					if self.comboType == 0:
						self.algo_label.set("KNN : N = "+str(neigh)+" ")
					elif self.comboType == 1:
						self.algo_label.set("Combo1 : N = "+str(neigh)+" ")
					elif self.comboType == 2:
						self.algo_label.set("Combo2 : N = "+str(neigh)+" ")
					self.topKNN.destroy() # close toplevel object (little window)
			except:
				print("No integer value is allowed.")
				self.neighborsRange = int(self.defaultNeighbors)
				self.neighbors.set(self.defaultNeighbors)
				self.clearSelectedAlgos(-1) #clear all
				
		
	"""
	Main purpose: pop up which shows an neighbor selection option window
	@param	self	this object
	@param	comboType	there is 2 combo's options (1 or 2 - show docomentation), 0 as not picked
	"""	
	def determineNeighborsNum(self,comboType=0):
		self.comboType = comboType
		self.topKNN = Toplevel()
		self.topKNN.wm_geometry("250x90")
		self.topKNN.resizable(width = False, height = False)
		self.topKNN.title("KNN Range")
		msg = Label(self.topKNN,text = "Determine the neighbor's range \n you want to apply.\n",fg = "black",font = ("Ariel", 10,"bold"))
		msg.pack()
		neigh = Entry(self.topKNN,textvariable = self.neighbors,fg = "black",font=("Ariel", 10,"bold"),width = 4)
		self.neighbors.set(self.defaultNeighbors)
		neigh.focus_set()
		neigh.bind("<KeyRelease>",self.getNeighbors)
		neigh.pack()

	
	"""
	Main purpose: mark and print the selected algorithm name
	@param	self	this object
	@param	algonum	algorithm number - 1,2,4,6,8,10 by the GUI view order
	"""	
	def markAlgo(self,algonum):
		self.algorithm = algonum
		if algonum == 1:
			self.algosMenu.entryconfig(0, background = "lightgray") #SVM
			self.clearSelectedAlgos(0)
			
			#view this selection on GUI:
			label = "" #optional object
			self.algo_label = StringVar()
			try:
				self.myPercentagePane.winfo_children()[self.algoLabelIdx].config(textvariable = self.algo_label)
				self.algo_label.set("SVM ")
			except:
				label = Label(self.myPercentagePane,textvariable = self.algo_label,fg = "black",font = ("Ariel", 12,"bold"))
				self.algo_label.set('SVM ')
				#label.grid(row = 0, column = self.algoLabelIdx) #exchanged by pack manager:
				label.pack(side=RIGHT)
			self.myPercentagePane.pack(fill = BOTH,padx = 2,pady = 2)
		elif algonum == 2:
			self.algosMenu.entryconfig(2, background = "lightgray") #Random forest
			self.clearSelectedAlgos(2)
			label = "" #optional object
			self.algo_label = StringVar()
			try:
				self.myPercentagePane.winfo_children()[self.algoLabelIdx].config(textvariable = self.algo_label)
				self.algo_label.set("Random Forest ")
			except:
				label = Label(self.myPercentagePane,textvariable = self.algo_label,fg = "black",font = ("Ariel", 12,"bold"))
				self.algo_label.set("Random Forest ")
				#label.grid(row = 0, column = self.algoLabelIdx) #exchanged by pack manager:
				label.pack(side=RIGHT)
			self.myPercentagePane.pack(fill = BOTH,padx = 2,pady = 2)
		elif algonum == 4:
			self.algosMenu.entryconfig(4, background = "lightgray") #Random forest
			self.clearSelectedAlgos(4)
			label = "" #optional object
			self.algo_label = StringVar()
			try:
				self.myPercentagePane.winfo_children()[self.algoLabelIdx].config(textvariable = self.algo_label)
				self.algo_label.set("KNN ")
			except:
				label = Label(self.myPercentagePane,textvariable = self.algo_label,fg = "black",font = ("Ariel", 12,"bold"))
				self.algo_label.set("KNN ")
				#label.grid(row = 0, column = self.algoLabelIdx) #exchanged by pack manager:
				label.pack(side=RIGHT)
			self.myPercentagePane.pack(fill = BOTH,padx = 2,pady = 2)
			self.determineNeighborsNum()
		elif algonum == 6:
			self.algosMenu.entryconfig(6, background = "lightgray") #KNN
			self.clearSelectedAlgos(6)
			label = "" #optional object
			self.algo_label = StringVar()
			try:
				self.myPercentagePane.winfo_children()[self.algoLabelIdx].config(textvariable = self.algo_label)
				self.algo_label.set("Multinomial LR ") #Logistic Regression
			except:
				label = Label(self.myPercentagePane,textvariable = self.algo_label,fg = "black",font = ("Ariel", 12,"bold"))
				self.algo_label.set("Multinomial LR ")
				#label.grid(row = 0, column = self.algoLabelIdx) #exchanged by pack manager:
				label.pack(side=RIGHT)
			self.myPercentagePane.pack(fill = BOTH,padx = 2,pady = 2)
		elif algonum == 8:
			self.algosMenu.entryconfig(8, background = "lightgray") #Logistic regression
			self.clearSelectedAlgos(8)
			label = "" #optional object
			self.algo_label = StringVar()
			try:
				self.myPercentagePane.winfo_children()[self.algoLabelIdx].config(textvariable = self.algo_label)
				self.algo_label.set("Combo1 ")
			except:
				label = Label(self.myPercentagePane,textvariable = self.algo_label,fg = "black",font = ("Ariel", 12,"bold"))
				self.algo_label.set("Combo1 ")
				#label.grid(row = 0, column = self.algoLabelIdx) #exchanged by pack manager:
				label.pack(side=RIGHT)
			self.myPercentagePane.pack(fill = BOTH,padx = 2,pady = 2)
			self.determineNeighborsNum(1)
		elif algonum == 10:
			self.algosMenu.entryconfig(10, background = "lightgray")
			self.clearSelectedAlgos(10)
			label = "" #optional object
			self.algo_label = StringVar()
			try:
				self.myPercentagePane.winfo_children()[self.algoLabelIdx].config(textvariable = self.algo_label)
				self.algo_label.set("Combo2 ")
			except:
				label = Label(self.myPercentagePane,textvariable = self.algo_label,fg = "black",font = ("Ariel", 12,"bold"))
				self.algo_label.set("Combo2 ")
				#label.grid(row = 0, column = self.algoLabelIdx) #exchanged by pack manager:
				label.pack(side=RIGHT)
			self.myPercentagePane.pack(fill = BOTH,padx = 2,pady = 2)
			self.determineNeighborsNum(2)

		
	"""
	Main purpose: create the algorithm options in the menu bar
	@param	self	this object
	"""	
	def addAlgorithmOptions(self):
		self.algosMenu = Menu(self.menuBar, tearoff = 0)
		self.menuBar.add_cascade(label = "Algorithms", menu = self.algosMenu)
		self.algosMenu.add_command(label = "SVM",command = lambda: self.markAlgo(1))
		self.root.config(menu = self.menuBar)
		self.algosMenu.add_separator()
		self.algosMenu.add_command(label = "Random Forest",command = lambda: self.markAlgo(2))
		self.algosMenu.add_separator()
		self.algosMenu.add_command(label = "KNN",command = lambda: self.markAlgo(4))
		self.algosMenu.add_separator()
		self.algosMenu.add_command(label = "Multinomial LR",command = lambda: self.markAlgo(6))
		self.algosMenu.add_separator()
		self.algosMenu.add_command(label = "Combo 1",command = lambda: self.markAlgo(8))
		self.algosMenu.add_separator()
		self.algosMenu.add_command(label = "Combo 2",command = lambda: self.markAlgo(10))

	
	"""
	Main purpose: clear the draw from the selected focus method
	@param	self	this object
	#param	exceptOf	one drawn object is left
	"""	
	def clearFocus(self,exceptOf):
		if exceptOf != 0:
			self.focusMenu.entryconfig(0, background = "")
		if exceptOf != 2:	
			self.focusMenu.entryconfig(2, background = "")

			
	"""
	Main purpose: draw the selected focus method
	@param	self	this object
	#param	focus	pick the focus you want to use with
	"""	
	def markFocus(self,focus):
		self.focusOn = focus
		textPred = ""
		if focus == 1:
			textPred = "Notes"
			self.focusMenu.entryconfig(0, background = "lightgray") #per note
			self.clearFocus(0)
		elif focus == 2:
			textPred = "Pages"
			self.focusMenu.entryconfig(2, background = "lightgray") #per page
			self.clearFocus(2)
		try:
			self.myLearnPane.winfo_children()[0].config(text = "Learn&Predict "+textPred+"!")
		except:
			print("No Learn&Predict Button is to be exist.")
			
		
	"""
	Main purpose: create the focus functionality menu UI
	@param	self	this object
	"""	
	def addFocusFunctionality(self):
		self.focusMenu = Menu(self.menuBar, tearoff = 0)
		self.menuBar.add_cascade(label = "Focus On", menu = self.focusMenu)
		self.focusMenu.add_command(label = "Notes",command = lambda: self.markFocus(1))
		self.root.config(menu = self.menuBar)
		self.focusMenu.add_separator()
		self.focusMenu.add_command(label = "Pages",command = lambda: self.markFocus(2))
		self.focusMenu.entryconfig(0, background = "lightgray") #per note default mark
		self.clearFocus(0)
	
	
	"""
	Main purpose: build DB path name and could be manager of browsing new DB from some FS location
	@param	self	this object
	"""	
	def browse(self):
		out = False
		tmp_filePath = ""
		if self.path != "":
			pass # temporary use!
			#fixing change to new path DB: - the Button is already disabled!
			#tmp_filePath = filedialog.askopenfilename(initialdir = "C:\\",title = "Choose your database",filetypes = (("db files","*.db"),("all files","*.*")))
		else:
			tmp_filePath = self.images.file_name
		if not tmp_filePath: #close the "askOpenFileName" function without choosing
			if not self.path:
				out = True
			else:
				return 1
		else:
			slash = -1
			for i in range(0,len(tmp_filePath)):
				if tmp_filePath[i] == "/":
					slash = i
			if slash != -1:
				before_slash = tmp_filePath[:slash+1]
				after_slash = tmp_filePath[slash+1:]
				self.images.path = before_slash
				self.images.notes_path = before_slash+"//Notes\\"
				self.images.file_name = after_slash
				self.path = self.images.file_name
			else:
				self.path = tmp_filePath
		if out == False:
			self.addAlgorithmOptions()
			self.addFocusFunctionality()
			self.createLabelsInfo()
			self.createPercentagesInfo()
			self.createCrossCorrelationButton()
			self.createLearnButton()
		self.finish_browse = True
		
		
	"""
	Main purpose: make a competible classification correspond to initialize parameters from the UI - new worker thread is built
	@param	self	this object
	"""	
	def learnImgs(self):
		img_thread = Thread(target = ImgLearner,args = (self.root,self.percentage,self.algorithm,self.images,self.files_amount,self.neighborsRange,self.focusOn,))
		img_thread.start()
		
		
	"""
	Main purpose: destroy the "no algotirhm is picked" pop up object
	@param	self	this object
	@param	event	close event
	"""	
	def closeMustAlgoNotify(self,event): # close notify window~
		self.topMUSTALGO.destroy()
		
		
	"""
	Main purpose: validation the classification request.
					if no algorithm is picked will alert will appear, otherwise classify and reset the to basic GUI view
	@param	self	this object
	@param	event	button click
	"""	
	def learnButton(self,event):
		if self.algorithm == -1:
			self.topMUSTALGO = Toplevel()
			self.topMUSTALGO.wm_geometry("250x90")
			self.topMUSTALGO.resizable(width = False, height = False)
			self.topMUSTALGO.title("Notification")
			msg = Label(self.topMUSTALGO,text = "\nAn algorithm is must to be selected!\n",fg = "black",font = ("Ariel", 10,"bold"))
			msg.pack()
			butt = Button(self.topMUSTALGO,text="OK",fg = "black",font=("Ariel", 10,"bold"))
			butt.bind("<ButtonRelease-1>",self.closeMustAlgoNotify)
			butt.pack()
			return
		self.learnImgs()

		# continue from here after it is classified:
		self.lock.acquire()
		self.files_amount = self.files_amount+1 # shared data
		self.lock.release()
		self.isFinish = True
		self.reset(self._INNER_EVENT_)
		
		
	"""
	Main purpose: reset (destroy and rebuild) each GUI and primitive object to achieve a basic GUI view for each classification
					action
	@param	self	this object
	@param	event	"clear" would do that explicitly UI request, otherwise 0 will operate for inside reset actions
	"""		
	def reset(self,event=0):
		if event == self._OUTTER_EVENT_:
			self.isFinish = True
		self.path = ""
		self.percentage = 33
		self.algorithm = -1
		self.focusOn = 1
		self.neighborsRange = 3
		
		self.menuBar.delete(2,6) #delete algorithm menubar section
		self.algosMenu.delete(0,10) #delete all algorithms names
		self.focusMenu.delete(0,4) #delete all focus options
		
		labelsWidget = self.myLabelsPane.winfo_children() #return all childrens of this object!
		if labelsWidget:
			labelPath = labelsWidget[0]
			labelPath.pack_forget()
			labelPath.destroy()
			self.myLabelsPane.pack_forget()
		percWidget = self.myPercentagePane.winfo_children() #return all childrens of this object!
		if percWidget:
			percLabel = percWidget[0]
			percLabel.pack_forget()
			percLabel.destroy()
			perc = percWidget[1]
			perc.pack_forget()
			perc.destroy()
			text = percWidget[2]
			text.pack_forget()
			text.destroy()
			algotext = percWidget[3]
			algotext.pack_forget()
			algotext.destroy()
			algo = percWidget[4]
			algo.pack_forget()
			algo.destroy()
			self.algo_label.set("None")
			self.myPercentagePane.pack_forget()
		crossWidget = self.myXcorrPane.winfo_children()
		if crossWidget:
			cross= crossWidget[0]
			cross.pack_forget()
			cross.destroy()
			self.myXcorrPane.pack_forget()
		learnWidget = self.myLearnPane.winfo_children() #return all childrens of this object!
		if learnWidget:
			learn = learnWidget[0]
			learn.pack_forget()
			learn.destroy()
			self.myLearnPane.pack_forget()
			
		if self.isFinish == True:
			self.browse()
			self.isFinish = False
		
		