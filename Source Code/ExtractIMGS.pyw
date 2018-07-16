import os
import logging
import io
import sqlite3
from PIL import Image
from glob import glob
import numpy as np
import cv2
import random
import shutil
import stat
from pathlib import Path
#import scipy.fftpack


class ExtractIMGS():
	"""
	Main pupose: initialize each base parameter to classification objecs fetching
	@param	self	this object
	"""
	def __init__(self):
		self.max_width_size = 0 # holds the maximum size of width or height of an image.
		self.writersMap = {}
		self.notesTopagesMap = {}
		self.writers_inv_map = {}#{v: k for k, v in self.writersMap.items()}
		self.attrs = []
		self.realNotesNumInPage = {} #count number of notes foreach phisical page
		self.realWritersNumInPage = [] #after some inside transformation -> <page => number of writers>	
		#self.access_img_path = "D://Itay_Guy//Desktop//NL-Lab//Notes"
		self.access_img_path = "file:///C://Users//Itay_Guy//SharePoint//Robotics%20-%20OMR%20-%20%D7%9E%D7%A1%D7%9E%D7%9B%D7%99%D7%9D//Exported%20Databases//Notes//"
		self.general_db_path = "C://Users//Itay_Guy//SharePoint//Robotics//Exported Databases"
		#self.notes_folder = "D://Itay_Guy//Desktop//NL-Lab//Notes"
		self.notes_folder = "C://Users//Itay_Guy//SharePoint//Robotics//Exported Databases//Notes"
		#self.pred_path = "D://Itay_Guy//Desktop//NL-Lab//Predictions"
		self.pred_path = "C://Users//Itay_Guy//SharePoint//Robotics//Exported Databases//Predictions"
		self.file_name = '25.9.db'
		conn = sqlite3.connect(self.general_db_path+"//"+self.file_name)
		self.c = conn.cursor()
		
	
	"""
	Main pupose: getter
	@param	self	this object
	@return	database local path
	"""
	def getDBPath(self):
		return self.general_db_path

		
	"""
	Main pupose: getter
	@param	self	this object
	@return	writers mapping
	"""
	def getMap(self):
		return self.writersMap
	
	
	"""
	Main pupose: getter
	@param	self	this object
	@return	object attributes for classification usage
	"""
	def getImgAttrVec(self):
		return self.attrs
	
	
	"""
	Main pupose: getter
	@param	self	this object
	@return	inverse writers mapping
	"""
	def getiMap(self):
		return self.writers_inv_map
	
	
	"""
	Main pupose: getter
	@param	self	this object
	@return	pages mapping
	"""
	def getPagesMap(self):
		return self.notesTopagesMap
		
	
	"""
	Main pupose: return the number of different classes at all in vec
	@param	self	this object
	@param	vec	vector of classes (categories)
	@return	number of different class into vec
	"""
	def __numOfDiffClasses__(self,vec):
		classes = []
		for elem in vec:
			if elem not in classes:
				classes.append(elem)
		return classes
	
	
	"""
	Main pupose: fetching all desired data from sqlite server
	@param	self	this object
	"""
	def server_fetch(self):
		i = 0
		if Path(self.pred_path).is_dir():
			shutil.rmtree(self.pred_path)
		os.makedirs(self.pred_path)
		os.chmod(self.pred_path,stat.S_IWRITE|stat.S_IEXEC|stat.S_IREAD)
		if Path(self.notes_folder).is_dir():
			shutil.rmtree(self.notes_folder)
		os.makedirs(self.notes_folder)
		os.chmod(self.notes_folder,stat.S_IWRITE|stat.S_IEXEC|stat.S_IREAD)

		pageName = []
		for pageRow in self.c.execute('SELECT `Hash`,`Name` FROM `Pages`'):
			pageName.append(list(pageRow))
				
		for row in self.c.execute('SELECT `Author`,`Original`,`ID`,`PageHash` FROM `Notes`'):
			author_img = row[0]
			hist_img = Image.open(io.BytesIO(row[1])).histogram()
			real_img = Image.open(io.BytesIO(row[1]))
			id = row[2]
			
			currPage = [] #[0 = hash,1 = name]
			currPageHash = row[3]
			currPage.append(currPageHash) # useless - use pageName instead
			
			for data in pageName:
				if data[0] == currPageHash:
					correctPageName = data[1]
					currPage.append(correctPageName)
					
			needToAppend = False
			for page in self.realWritersNumInPage:
				if page[0] == currPage[1]:
					needToAppend = True
					page.append(author_img)
			if needToAppend == False or len(self.realWritersNumInPage) == 0:
				self.realWritersNumInPage.append([currPage[1],author_img])
			
			try:
				self.realNotesNumInPage[currPage[1]] += 1
			except:
				self.realNotesNumInPage[currPage[1]] = 1
				
				
			np_arr = np.fromstring(row[1],np.uint8)	# get byte array from the database          
			img_dec = cv2.imdecode(np_arr, -1) # convert byte array to a numpy matrix (flag -1 means "as is", no color conversion)
			img_pixels = cv2.cvtColor(img_dec,cv2.COLOR_BGR2RGB) # convert to the correct color space
			if self.max_width_size < max(img_pixels.shape):
				self.max_width_size = max(img_pixels.shape)
			dir_path = self.notes_folder+"//"+str(id)
			if Path(dir_path).is_dir():
				shutil.rmtree(dir_path)
			os.makedirs(dir_path)
			os.chmod(dir_path,stat.S_IWRITE|stat.S_IEXEC|stat.S_IREAD)
			
			cv2.imwrite(self.notes_folder+"//"+str(id)+"//"+str(id)+".png",img_dec)
			self.attrs.append([author_img,img_pixels,real_img,id,currPage[1]])
			key_found = True
			for key in self.writersMap.keys():
				if key == author_img:
					key_found = False
					break
			if key_found:
				self.writersMap[author_img] = i
				self.writers_inv_map[i] = author_img
				i = i+1
		
		
		numWritersForeachPage = {}
		for page in self.realWritersNumInPage:
			numWritersForeachPage[page[0]] = self.__numOfDiffClasses__(page[1:])
			
		self.realWritersNumInPage = numWritersForeachPage # pairs: <page => number of writers>

		# scale all images from buffer to a uniform dimensions:
		i = 0
		for item in self.attrs: #item[1] = img_pixels
			resized_img = cv2.resize(item[1],(self.max_width_size,self.max_width_size))
			
			#resized_img = cv2.Laplacian(resized_img, cv2.CV_64F).var() #shows the laplacian of that image
			#resized_img = cv2.Canny(resized_img,self.max_width_size,self.max_width_size) #shows edge image
			#resized_img = scipy.fftpack.dct(resized_img, norm='ortho') #shows the dct transformation of that image
			#resized_img = cv2.fastNlMeansDenoisingColored(resized_img,None,10,10,7,21) #denoising operations
			
			#take histogram oriented gradient of that image:
			#resized_img = self.deskew(resized_img)
			#resized_img = self.hog(resized_img)
			
			### until that point there is no other method which can improve those images -> solution: large the data set without writer number 1.
			### we can see clearely that the system learned writer 1 style well!
			
			item[1] = resized_img
			
			
			# maping: N -> page hash:
			exitHash = False
			for k in self.notesTopagesMap.values():
				if k == item[4]:
					exitHash = True
					break
			if not exitHash:
				self.notesTopagesMap[i] = item[4]
				i = i+1
		arr = []
		nums = []
		for key,value in zip(list(self.notesTopagesMap.keys()),list(self.notesTopagesMap.values())):
			nums.append(key)
			arr.append([key,value])
		random.shuffle(nums)
		self.notesTopagesMap.clear()
		for num in nums:
			for elem in arr:
				if num == elem[0]:
					self.notesTopagesMap[num] = elem[1]
					
					