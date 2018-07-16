import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
from PIL import Image


"""
Main pupose: Management the whole preprocessing operation
"""
class Preprocess():
	"""
	Main pupose: initialize each base parameter to preprocess usage action
	@param	self	this object
	@param	imgs	images object for the classifiction object access
	@param	maps	mapping object between writers and series number fomr 0 to n
	@param	perc	training set percentage
	@param	img_size	image object constant size
	@param	per_note_flag	focus on flag - True for notes otherwise for pages
	"""
	def __init__(self,imgs,maps,perc,img_size,per_note_flag=True):
		self._perNote = per_note_flag
		self._img_width_size = img_size
		self._percentage = perc
		self._writer_maps = maps
		random.shuffle(imgs)
		self._images = imgs
		self._token_dict = []
		self._tfidf = []
		self._tfidf_vec = []
		self._test = []
		self._test_vec = []
		self._cat_vec = []
		self._test_imgs = []
		self._cat_tests = []
		self._identifiers = []
		self._groups = {} # contains page at each index
		self._train_amount = 0
		self._pagesToTrain = 0
		
		if self._perNote == True:
			self._train_amount = int(len(self._images)*self._percentage/100)
			self.__trainTestSelectionToNotes__()
		else: #pages
			self.__seperateNotesToPageBulks__()
			self.__trainTestSelectionToPages__()
			
	
	"""
	Main pupose: initialize each base parameter to preprocess usage action
	@param	self	this object
	@param	set	set of writers mapping key and their real value (real name)
	@return	shuffled set
	"""
	def myShuffle(self,set):
		arr = []
		hashs = []
		for key,value in zip(list(set.keys()),list(set.values())):
			hashs.append(key)
			arr.append([key,value])
		random.shuffle(hashs)
		set.clear()
		for hash in hashs:
			for item in arr:
				if hash == item[0]:
					set[hash] = item[1]
		return set
		
	
	"""
	Main pupose: check is there is different value into a vector [class items]
	@param	self	this object
	@param	arr	array of classes taggin for each writer object
	@return	True of existance a different otherwise False
	"""
	def isDiffValExist(self,arr):
		lastVal = arr[0]
		for i in range(1,len(arr)):
			if lastVal == arr[i]:
				return True
		return False
		
		
	"""
	Main pupose: make the preprocessing action for given input image objects under tf-idf technique
	@param	self	this object
	@return	tf-idf matrix of the training set
	"""
	def preprocessing(self):
		self._tfidf = TfidfVectorizer(lowercase = False,tokenizer = None,use_idf = True,smooth_idf = True)
		self._tfidf_vec = self._tfidf.fit_transform(self._token_dict)
		self._test_vec = self._tfidf.transform(self._test)
		return self._tfidf_vec
		
		
	"""
	Main pupose: find page groups keys
	@param	self	this object
	@return	keys
	"""
	def getGroupsKeys(self):
		if not self._groups:
			self.__seperateNotesToPageBulks__()
		return self._groups.keys()
		
		
	"""
	Main pupose: find page groups values
	@param	self	this object
	@return	values
	"""		
	def getGroupsValues(self):
		if not self._groups:
			self.__seperateNotesToPageBulks__()
		return self._groups.values()
		
	
	"""
	Main pupose: getter
	@param	self	this object
	@return	category vector of the training set
	"""
	def getCatVec(self):
		return self._cat_vec

		
	"""
	Main pupose: getter
	@param	self	this object
	@return	test feature matrix after transform method of the validation set
	"""		
	def getTestVec(self):
		return self._test_vec
		
		
	"""
	Main pupose: getter
	@param	self	this object
	@return	raw validation set objects
	"""
	def getTestImgs(self):
		return self._test_imgs
		
	
	"""
	Main pupose: getter
	@param	self	this object
	@return	validation set objects category vector
	"""	
	def getCatTestVec(self):
		return self._cat_tests
		
		
	"""
	Main pupose: getter
	@param	self	this object
	@return	notes id numbers vector
	"""			
	def getIds(self):
		return self._identifiers
		
		
	"""
	Main pupose: getter
	@param	self	this object
	@return	traning set tf-idf matrix
	"""			
	def getTfIdfVec(self):
		return self._tfidf
		
		
	"""
	Main pupose: seperate the incoming notes to page bulk units
	@param	self	this object
	"""				
	def __seperateNotesToPageBulks__(self):
		for img in self._images:
			if self.__isExistElement__(self._groups,img[4]) == True:
				self._groups[img[4]].append(img)
			else:
				self._groups[img[4]] = [img]
		if len(self._groups) > 0:
			self._groups = self.myShuffle(self._groups)
			self._pagesToTrain = int(np.round(len(self._groups)*(self._percentage/100))) #foreach index there is a page hash
			
	
	"""
	Main pupose: seperate the incoming notes data set to training set and validation set
	@param	self	this object
	"""	
	def __trainTestSelectionToNotes__(self):
		classCountVec = []
		i = 0
		for img in self._images:
			if i > self._train_amount:
				self._test.append(str(np.sqrt(img[1]))) #skretch the artifact's image using sqrt to emphesize the writer's manner
				self._test_imgs.append(img[2]) #byte array of that note image from the server!
				self._cat_tests.append(str(self._writer_maps[img[0]])) #we care about which collected pages we predict more  successfull
				self._identifiers.append(img[3])
			else:
				self._token_dict.append(str(np.sqrt(img[1]))) #skretch the artifact's image using sqrt to emphesize the writer's manner
				classType = str(self._writer_maps[img[0]])
				if not self.__isExistElement__(classCountVec,classType):
					#print("classType="+str(classType)+"->"+"classCountVec="+str(classCountVec))
					classCountVec.append(classType)
				self._cat_vec.append(classType) #indicate the real category of which is this image belog to
			i = i+1
		if len(classCountVec) < 2:
			self.__collectRandImgAndCat__(classCountVec) #use to prevent one class case in the classifier
			
			
	"""
	Main pupose: seperate the incoming pages data set to training set and validation set
	@param	self	this object
	"""		
	def __trainTestSelectionToPages__(self):
		classCountVec = []
		i = 0
		for key,values in zip(list(self._groups.keys()),list(self._groups.values())):
			if i >= self._pagesToTrain:
				for value in values:
					self._test.append(str(value[1]))
					self._test_imgs.append([value[2],value[4]]) #byte array of that note image!
					#self._test_imgs.append(value[4]) #append the page hash too
					self._cat_tests.append(str(self._writer_maps[value[0]])) #we care about which collected pages we predict more  successfull
					self._identifiers.append(value[3])
			else:
				for value in values: #foreach key go through all its values
					self._token_dict.append(str(value[1]))
					classType = str(self._writer_maps[value[0]])
					if not self.__isExistElement__(classCountVec,classType):
						classCountVec.append(classType)
					self._cat_vec.append(classType) #indicate the real category of which is this image belog to
			i += 1
		if len(classCountVec) < 2:
			self.__collectRandImgAndCat__(classCountVec) #use to prevent one class case in the classifier

		
	"""
	Main pupose: add new not existance category class number to the category vector and new random gray scale matrix using naural numbers between 0 to 255
	@param	self	this object
	@param	classes	training set category vector
	"""			
	def __collectRandImgAndCat__(self,classes):
		try:
			max = max(classes)
			max = int(max)
		except:
			max = 1
		for i in range(0,max+1):
			if not self.__isExistElement__(self._cat_vec,str(i)):
				self._cat_vec.append(str(i))
				self._token_dict.append(self.__buildRandImg__(self._img_width_size,self._img_width_size))
	
		
	"""
	Main pupose: building random gray scale image using naural numbers into 0 to 255 range
	@param	self	this object
	@param	img_width	image width size
	@param	img_height	image height size
	@return built random new matrix
	"""		
	def __buildRandImg__(self,img_width,img_height):
		matrix = []
		for i in range(img_height):
			vec = []
			for j in range(img_width):
				vec.append(random.randint(0,255))
			matrix.append(vec)
		return str(matrix)
	
	
	"""
	Main pupose: check is key exists into element
	@param	self	this object
	@param	element	vector of items
	@param	key	item we wish to find into element
	@return	True for existance key into element otherwise False
	"""		
	def __isExistElement__(self,element,key):
		for item in element:
			if item == key:
				return True
		return False