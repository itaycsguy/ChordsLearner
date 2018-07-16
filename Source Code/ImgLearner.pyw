import logging
from tkinter import *
from Preprocess import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import TruncatedSVD #LSA


"""
Main purpose: Management the whole classification and cross correlation process for each request
"""
class ImgLearner():
	"""
	Main purpose: initialize each base parameter to classification operation usage
	@param	self	this object
	@param	root	changing the GUI view as correspond to the classification status (thinking view at most)
	@param	perc_num	training set percentage to take
	@param	algo	choosen algoritm to classify with
	@param	imgs	image's object for the preprocessing time which require that and next classification usage
	@param	files_amount	instance counter for tagging each classification with its count number
	@param	neigh	neighbors to KNN, combo1 and combo2 usage - 3 by default
	@param	focusOn	[1 = notes] and [2 = pages]
	"""
	def __init__(self,root,perc_num,algo,imgs,files_amount,neigh=3,focusOn=1): #focusOn=1 -> notes classification
		root.config(cursor="wait") # GUI waiting view
		self._crossCorrelationAns = [] # hold on the classification solutions to show the crosscorrelation function
		self._perc = perc_num
		self._algorithm = algo
		self._images = imgs
		self._files_amount = files_amount
		self._neigh = neigh # default value
		if self._neigh < 3:
			self._neigh = 3
		self._notesFocus = True
		if focusOn == 2: # focus on pages
			self._notesFocus = False
		self._CLASSIFY_UPPER_SHRESHOLD = 0.8
		
		# preprocess time on images set using all top parameters:
		self._IMGPreprocess = Preprocess(self._images.getImgAttrVec(),self._images.getMap(),self._perc,self._images.max_width_size,self._notesFocus)
		self._feature_matrix = self._IMGPreprocess.preprocessing()
		self._category_class_indecies = self._IMGPreprocess.getCatVec()
		self._test_features_transform = self._IMGPreprocess.getTestVec()
		self._test_imgs = self._IMGPreprocess.getTestImgs()
		self._cat_tests = self._IMGPreprocess.getCatTestVec()
		self._identifiers = self._IMGPreprocess.getIds()
		
		# pages output html:
		self._HTML_PAGES_HEAD = """<!DOCTYPE html>
									<html lang="en">
										<head>
											<meta http-equiv="Content-Type" content="text/html"/>
											<meta charset="utf-8"/>
											<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1"/>
											<meta name="viewport" content="width=device-width, initial-scale=1,user-scalable=yes"/>
											<meta name="HandheldFriendly" content="true"/>
											<meta name="MobileOptimized" content="320"/>
											<meta name="description" content="patient's parent questions table"/>
											<meta name="Itay_Guy" content="B.Sc Student project in University Of Haifa - 2017"/>
											<link id="csslink" href="..\\styleForm.css" rel="stylesheet"/>
											<link id="csslink" href="..\\researcherForm.css" rel="stylesheet"/>
											<title>Pages' Active Learning</title>
												<style type="text/css"></style>
											<script type="text/javascript" src="researcherForm.js"></script>
										</head>
											<body>
												<div id="layout-top"></div>
												<div id="layout-middle">
													<div id="layout-main">
														<div class="layout-center-0">
														<div class="layout-center-1">
														<div class="layout-center-2">
														<div class="layout-center-3">
															<div id="section page_to_be_nowrap">
																<table id="table">
																	<tr class="stright_right">
																		<strong><h3 id="title">System Refactoring And Data View:</h3></strong>
																	</tr>
																</table>
																<table>
																	<thead>
																		<th class="h_style">
																			<label class="style adjacent">Num.</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Page Id.</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Real Writers Amount</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Predicted Writers Amount</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Prob.</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Who Predicted</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Who Real</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Mistake?</label>
																		</th>
																	</thead>
																	<tbody>"""
		# notes output html:															
		self._HTML_NOTES_HEAD = """<!DOCTYPE html>
									<html lang="en">
										<head>
											<meta http-equiv="Content-Type" content="text/html"/>
											<meta charset="utf-8"/>
											<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1"/>
											<meta name="viewport" content="width=device-width, initial-scale=1,user-scalable=yes"/>
											<meta name="HandheldFriendly" content="true"/>
											<meta name="MobileOptimized" content="320"/>
											<meta name="description" content="patient's parent questions table"/>
											<meta name="Itay_Guy" content="B.Sc Student project in University Of Haifa - 2017"/>
											<link id="csslink" href="..\\styleForm.css" rel="stylesheet"/>
											<link id="csslink" href="..\\researcherForm.css" rel="stylesheet"/>
											<title>Notes' Active Learning</title>
												<style type="text/css"></style>
											<script type="text/javascript" src="researcherForm.js"></script>
										</head>
											<body>
												<div id="layout-top"></div>
												<div id="layout-middle">
													<div id="layout-main">
														<div class="layout-center-0">
														<div class="layout-center-1">
														<div class="layout-center-2">
														<div class="layout-center-3">
															<div id="section page_to_be_nowrap">
																<table id="table">
																	<tr class="stright_right">
																		<strong><h3 id="title">System Refactoring And Data View:</h3></strong>
																	</tr>
																</table>
																<table>
																	<thead>
																		<th class="h_style">
																			<label class="style adjacent">Num.</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Id.</label>
																		</th>
																		<th class="h_style">
																			<label class="style adjacent">Image</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Page Id.</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Real Writer</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Predicted Writer</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Prob.</label>
																		</th>
																		<th class="h_style fit_sizes">
																			<label class="style adjacent">Mistake?</label>
																		</th>
																	</thead>
																	<tbody>"""
		self.HTML_TAIL = """</tbody>
							</table><!-- .end outter table -->
							<table>
								<tr>
									<td>
										<div class="continue_place">
											<input id="inp_continue" type="button" onclick="Form.ajax();" value="Send"/>
										</div>
									</td>
								</tr>
							</table><!-- .end outter table -->
						</div><!-- .end outter div section -->
					</div><!-- .layout-center-3 -->
					</div><!-- .layout-center-2 -->
					</div><!-- .layout-center-1 -->
					</div><!-- .layout-center-0 -->
				</div><!-- #layout-main -->
			</div><!-- #layout-middle -->
			<div id="layout-bottom"></div>
			<script type="text/javascript"></script>
		</body></html>"""
		

		if self._notesFocus == True and self._algorithm == 0: # need cross correlation calculation:
			self.showCrossCorrelationChart()
		elif self._notesFocus == False and self._algorithm == 0:
			root.config(cursor="") # GUI end-thinking
			return
		else:
			self.predict()
			
		root.config(cursor="") # GUI end-thinking


	"""
	Main purpose: retrieve cross correlation data for each algonum and corresponding to vecId
	@param	self	this object
	@param	algonum	algorithm number
	@param	vecId	index location of solution vector into _crossCorrelationAns class matrix
	@return	cross correlation vector
	"""
	def __getCrossCorrelationData__(self,algonum,vecId):
		crossCorrDataVec = []
		for datums in self._crossCorrelationAns[algonum]:
			if vecId == 0:
				crossCorrDataVec.append(str(datums[0]))
			else:
				crossCorrDataVec.append(str(datums[1]))
		return crossCorrDataVec
		
		
	"""
	Main purpose: build cross correlation matrix relative to algorithm invokes and its results
	@param	self	this object
	"""
	def __buildCrossCorrelationMatrix__(self):
		self._algorithm = 1
		self.show_results_for_each(self.SVM(),crossCorrMode = True)
		self._algorithm = 2
		self.show_results_for_each(self.RandomForest(),crossCorrMode = True)
		self._algorithm = 4
		self.show_results_for_each(self.KNN(),crossCorrMode = True)
		self._algorithm = 6
		self.show_results_for_each(self.LogisticRegression(),crossCorrMode = True)
		self._algorithm = 8
		self.show_results_for_each(self.Combo_SecondClassify(1,self.SVM()),crossCorrMode = True)
		self._algorithm = 10
		self.show_results_for_each(self.Combo_SecondClassify(2,self.LogisticRegression()),crossCorrMode = True)
		
		
	"""
	Main purpose: build cross correlation subplot relative to writers, values and color
	@param	self	this object
	@param	fig	figure object which would show the plot object
	@param	writers	classified writers
	@param	values	classification results
	@param	color	that classification unique color for identification
	"""
	def __buildCrossCorrelationSubplot__(self,fig,writers,values,color):
		ax = fig.add_subplot(111)
		writers = np.array(writers,dtype=float)
		values = np.array(values,dtype=float)
		ax.xcorr(writers, values, usevlines=False, normed=True, lw=2,color=color)
		ax.grid(True)
		ax.set_xlim(1,15)
		ax.set_ylim(0,1)
		
		
	"""
	Main purpose: show up the cross correlation chart using image saving the plot object
	@param	self	this object
	"""
	def showCrossCorrelationChart(self):
		self.__buildCrossCorrelationMatrix__()
		correctWriters = self.__getCrossCorrelationData__(0,0)
		correctVal = (np.ones(len(correctWriters)))
		for item in correctVal:
			item = float(item)
		
		fig = plt.figure()
		
		self.__buildCrossCorrelationSubplot__(fig,correctVal,correctVal,"red")
		
		svmId = np.array(self.__getCrossCorrelationData__(0,0),dtype=np.uint32)
		svmSol = np.array(self.__getCrossCorrelationData__(0,1),dtype=float)
		self.__buildCrossCorrelationSubplot__(fig,svmSol,correctVal,"green")
			
			
		randomForestId = np.array(self.__getCrossCorrelationData__(1,0),dtype=np.uint32)
		randomForestSol = np.array(self.__getCrossCorrelationData__(1,1),dtype=float)
		self.__buildCrossCorrelationSubplot__(fig,randomForestSol,correctVal,"yellow")
		
		
		knnId = np.array(self.__getCrossCorrelationData__(2,0),dtype=np.uint32)
		knnSol = np.array(self.__getCrossCorrelationData__(2,1),dtype=float)
		self.__buildCrossCorrelationSubplot__(fig,knnSol,correctVal,"blue")#,knnId,knnSol)
		
		
		logisticRegressionId = np.array(self.__getCrossCorrelationData__(3,0),dtype=np.uint32)
		logisticRegressionSol = np.array(self.__getCrossCorrelationData__(3,1),dtype=float)	
		self.__buildCrossCorrelationSubplot__(fig,logisticRegressionSol,correctVal,"purple")
		
		combo1Id = np.array(self.__getCrossCorrelationData__(4,0),dtype=np.uint32)
		combo1Sol = np.array(self.__getCrossCorrelationData__(4,1),dtype=float)	
		
		correctForComboLenVal = (np.ones(len(combo1Sol)))
		
		self.__buildCrossCorrelationSubplot__(fig,combo1Sol,correctForComboLenVal,"gray")
		
		combo2Id = np.array(self.__getCrossCorrelationData__(5,0),dtype=np.uint32)
		combo2Sol = np.array(self.__getCrossCorrelationData__(5,1),dtype=float)
		
		correctForComboLenVal = (np.ones(len(combo2Sol)))
		
		self.__buildCrossCorrelationSubplot__(fig,combo2Sol,correctForComboLenVal,"black")
		
		plt.savefig(self._images.pred_path + "//" + "Cross_Correlation_View_"+ str(self._files_amount) + ".png") #save the image but that is a compromise
		#plt.show() #cannot block it self at this moment because of we are running through interactive mode, no any solution from: subprocesses and Threads.
		
		
	"""
	Main purpose: classify using random forest algorithm
	@param	self	this object
	@return	predictred vector
	"""
	def RandomForest(self):
		classifier = RandomForestClassifier().fit(self._feature_matrix, self._category_class_indecies)
		predicted = classifier.predict(self._test_features_transform)
		return predicted
	
	
	#explain: next  i am going to append each predicted result line which his score over 0.80 - than return to the test featue matrix and filter it and than make inverse function to re-learn
	"""
	Main purpose: classify using combo1 or combo2 algorithms
	@param	self	this object
	@param	comboType	[1  = combo1 and this is: svm->knn], [2 = combo2 and this is: logistic regression->knn]
	@param	firstPredicted	svm or logistic regression predicted solutions vector
	@return	predicted matrix
	"""
	def Combo_SecondClassify(self,comboType,firstPredicted): # 0.80 is a thrashold here
		next_features_idxs = []
		# run through top values and select under the threshold:
		for i in range(0,len(firstPredicted)):
			try:
				if firstPredicted[i][int(self._cat_tests[i])] >= self._CLASSIFY_UPPER_SHRESHOLD: #0.8
					next_features_idxs.append(i)
			except:
				idx = self.__get_max_fit_index__(firstPredicted[i])
				if firstPredicted[i][idx] >= self._CLASSIFY_UPPER_SHRESHOLD:
					next_features_idxs.append(i)
		invFeaturesTrans = self._IMGPreprocess.getTfIdfVec().inverse_transform(self._test_features_transform) # make an inverse transformation to restart the classification
		
		# build input structure:
		new_feature_matrix = []
		for attr in self._images.getImgAttrVec():
			equal = False
			i = 0
			while i < len(next_features_idxs):
				if attr[3] == self._identifiers[int(next_features_idxs[i])]:
					equal = True
				i += 1
			if equal:
				new_feature_matrix.append(attr)
		
		# preprocessing:
		newPreprocessObj = Preprocess(new_feature_matrix,self._images.getMap(),self._perc,self._images.max_width_size,self._notesFocus)
		feature_matrix = newPreprocessObj.preprocessing()
		category_class_indecies = newPreprocessObj.getCatVec()
		test_features_transform = newPreprocessObj.getTestVec()
		test_imgs = newPreprocessObj.getTestImgs()
		cat_tests = newPreprocessObj.getCatTestVec()
		identifiers = newPreprocessObj.getIds()

		
		#second predication:
		predicted = self.KNN(feature_matrix,category_class_indecies,test_features_transform)
		self._IMGPreprocess = newPreprocessObj
		self._feature_matrix = feature_matrix
		self._category_class_indecies = category_class_indecies
		
		self._test_features_transform = test_features_transform
		self._test_imgs = test_imgs
		self._cat_tests = cat_tests
		self._identifiers = identifiers
		return predicted
	
	
	"""
	Main purpose: classify using logistic regression algorithm (softmax regression - implementation of multi logistic regression)
	@param	self	this object
	@return predicted matrix
	"""
	def LogisticRegression(self):
		classifier = LogisticRegression()
		classifier.fit(self._feature_matrix,self._category_class_indecies)
		predicted = classifier.predict_proba(self._test_features_transform)
		return predicted
	
	
	"""
	Main purpose: classify using KNN algorithm
	@param	self	this object
	@param	feature_matrix	feature matrix, not None value when using combo1 or combo2
	@param	category_class_indecies	categories indicator vector, not None value when using combo1 or combo2
	@param	test_features_transform	feature matrix of the test objects, not None value when using combo1 or combo2
	@return predictred matrix
	"""
	def KNN(self,feature_matrix=None,category_class_indecies=None,test_features_transform=None): #params for second classification
		neigh = KNeighborsClassifier(n_neighbors=self._neigh)
		predicted = None
		if feature_matrix == None:
			neigh.fit(self._feature_matrix,self._category_class_indecies)
			predicted = neigh.predict_proba(self._test_features_transform)
		else:
			neigh.fit(feature_matrix,category_class_indecies)
			predicted = neigh.predict_proba(test_features_transform)
		return predicted
		
		
	"""
	Main purpose: classify using svm (more than one class) algorithm
	@param	self	this object
	@return predicted matrix
	"""
	def SVM(self):
		classifier = svm.SVC(probability = True).fit(self._feature_matrix,self._category_class_indecies)
		predicted = classifier.predict_proba(self._test_features_transform)#unsupervised
		return predicted
		
		
	"""
	Main purpose: find the maximum predicted value for a vector of classification results
	@param	self	this object
	@param	predVec	vector with predictions
	@return	index with the maximum value
	"""		
	def __get_max_fit_index__(self,predVec):
		max = -1
		idx = -1
		for i in range(0,len(predVec)):
			if max < predVec[i]:
				max = predVec[i]
				idx = i
		return idx
	
	
	"""
	Main purpose: find the number of classes in some series notes
	@param	self	this object
	@param	notes	series note objects
	@return	number of class into 'notes'
	"""		
	def __getNumClassesOf__(self,notes):
		classes = {}
		for writer in notes:
			try:
				classes[writer] += 1
			except:
				classes[writer] = 1
		sum = 0
		for key in classes.keys():
			sum += 1
		return sum
			
			
	
	"""
	Main purpose: output all page classification results to html file using the name Page_Pred_? [? = files_amount count value]
	@param	self	this object
	@param	predicted	predicted solutions vector from a specific classification
	"""	
	def show_results_for_pages(self,predicted):
		results = open(self._images.pred_path+"//Page_Pred_"+str(self._files_amount)+".txt","w+") 
		title = "Classified By: "
		type = ""
		if self._algorithm == 1:
			type = "SVM"
		elif self._algorithm == 2:
			type = "Random Forest"
		elif self._algorithm == 4:
			type = "KNN Using Neighbors = "+str(self._neigh)
		elif self._algorithm == 6:
			type = "Multinomial LR"
		elif self._algorithm == 8:
			type = "Combo1 Using Neighbors = "+str(self._neigh)
		elif self._algorithm == 10:
			type = "Combo2 Using Neighbors = "+str(self._neigh)
		perc = " With Training-Set Percentage Of "+str(self._perc)+"% \n"
		results.write(title+type+perc)
		local_html_content = self._HTML_PAGES_HEAD
		keys = self._IMGPreprocess.getGroupsKeys()
		amounts = {}
		writerProbabilities = {}
		for key in keys:
			for i in range(0,len(self._test_imgs)):
				if self._test_imgs[i][1] == key:
					id = self._identifiers[i]
					try:
						probabilities = predicted[i][id]
						try:
							writerProbabilities[id] += probabilities
						except:
							writerProbabilities[id] = probabilities
						
						amounts[key].append(id)
					except:
						amounts[key] = [id]
		
		
		"""### output debugging issue!
		for key,value in zip(list(amounts.keys()),list(amounts.values())):
			for k,probs in zip(list(writerProbabilities.keys()),list(writerProbabilities.values())):
				print(str(value) + " - " + str(k))
		"""
		
		pageVsClasses = {}
		for key,value in zip(list(amounts.keys()),list(amounts.values())):
			pageVsClasses[key] = self.__getNumClassesOf__(value)
		
		html_results = open(self._images.pred_path+"//Summary_Pages_Table_"+str(self._files_amount)+".html","w+")
		html_results.write(local_html_content)
		html = ""
		successful_value = 0.0
		prob_success_count = []
		for key,value in zip(pageVsClasses.keys(),pageVsClasses.values()):
			writerNums = self._images.realWritersNumInPage[str(key)]
			
			predictedWriters = []
			for writer,prob in zip(list(writerProbabilities.keys()),list(writerProbabilities.values())):
				if self._images.getiMap()[writer] in writerNums:
					predictedWriters.append(self._images.getiMap()[writer])
					
			html += """<tr id="""+str(len(prob_success_count)+1)+""">
						<td class="style text_style">
							<label class="style adjacent">"""+str(len(prob_success_count)+1)+"""</label>
						</td>
						<td class="style text_style">
							<label class="style adjacent">"""+str(key)+"""</label>
						</td>
						<td class="style text_style">
							<label class="style adjacent">"""+str(len(writerNums))+"""</label>
						</td>"""
			html += """<td class="style text_style">
							<label class="style adjacent">"""+str(len(predictedWriters))+"""</label>
						</td>"""
			
			html +=	"""<td class="style text_style">
							<label class="style adjacent">"""+str(len(predictedWriters)/len(writerNums))[:6]+"""</label>
						</td>"""
						
			html +=	"""<td class="style text_style">
							<label class="style adjacent">"""+str(predictedWriters)+"""</label>
						</td>"""
			html +=	"""<td class="style text_style">
							<label class="style adjacent">"""+str(writerNums)+"""</label>
						</td>	
						<td class="style radio_style">
							<input class="refactor_group" autocomplete="off" id="inp_radio" type="radio" name="""+str(len(prob_success_count)+1)+""" value="yes" onclick="Functions.markSelectedLine(this)"/>
						</td>	
					</tr>"""
			prob_success_count.append(str(len(predictedWriters)/len(writerNums)))
			results.write("Page: "+str(key)+" contains: "+str(len(writerNums))+" writers\nAnd classified With: "+str(len(predictedWriters))+" writers\n")
			
			predictedWriters = []
		
		if len(prob_success_count) > 0:
			proba = 0.0
			for curr_prob in prob_success_count:
				proba += float(curr_prob)
			successful_value = proba/len(prob_success_count)
		results.write("Successfull Average of: "+str(successful_value)[:6]+"\n")
		html += self.HTML_TAIL
		html_results.write(html)
		results.write("(*)NOTE: Each Page Was Selected Randomly!\n")
		results.close()
			
	
	"""
	Main purpose: output all notes classification results to html file using the name Pred_? [? = files_amount count value]
	Notification: the formula is - [1*P(n_1=1)+...+1*P(n_k=1)]/k
	@param	self	this object
	@param	predicted	predicted solutions vector from a specific classification
	@param	crossCorrMode	False for cross correlation mode otherwise True
	"""		
	def show_results_for_each(self,predicted,crossCorrMode = False):
		average = 0
		decreteResultPoints = [] # contains the correct answer for each note, to be used at the crosscorrelation time.
		local_html_content = ""
		if not crossCorrMode:
			results = open(self._images.pred_path+"//Pred_"+str(self._files_amount)+".txt","w+") 
			title = "Classified By: "
			type = ""
			if self._algorithm == 1:
				type = "SVM"
			elif self._algorithm == 2:
				type = "Random Forest"
			elif self._algorithm == 4:
				type = "KNN Using Neighbors = "+str(self._neigh)
			elif self._algorithm == 6:
				type = "Multinomial LR"
			elif self._algorithm == 8:
				type = "Combo1 Using Neighbors = "+str(self._neigh)
			elif self._algorithm == 10:
				type = "Combo2 Using Neighbors = "+str(self._neigh)
			perc = " With Training-Set Percentage Of "+str(self._perc)+"% \n"
			results.write(title+type+perc)
			local_html_content = self._HTML_NOTES_HEAD
		avg = {}
		for i in range(0,len(self._test_imgs)):
			if not crossCorrMode:
				results.write("image_id:"+str(self._identifiers[i])+" is writen by writer:"+str(self._images.getiMap()[int(self._cat_tests[i])]))
			# local vars:
			predicted_index = ""
			local_result_to_file = ""
			local_write_real = ""
			predicted_gain = ""
			if self._algorithm == 2: # random forest (into loop):
				# supervised classifier would return one value only which is the classified number:
				predicted_index = str(self._images.getiMap()[int(predicted[i])])
				if predicted_index == str(self._images.getiMap()[int(self._cat_tests[i])]):
					average += 1
				if not crossCorrMode:
					results.write(" and classfied as writer:"+predicted_index+"\n")
					local_result_to_file = predicted_index
				local_write_real = self._images.getiMap()[int(self._cat_tests[i])]
				if str(predicted_index) == str(local_write_real):
					decreteResultPoints.append([str(self._identifiers[i]),'1'])
				else:
					decreteResultPoints.append([str(self._identifiers[i]),'0'])
			else: # SVM || KNN || LogisticRegression || Combos (into loop):
				flag_exp = False
				try:
					predicted_gain = str(predicted[i][int(self._cat_tests[i])])
					local_result_to_file = predicted_gain # not use in crossCorrMode
					average += predicted[i][int(self._cat_tests[i])]
					# supervised:
					local_write_real = str(self._images.getiMap()[int(self._cat_tests[i])])
					decreteResultPoints.append([str(self._identifiers[i]),predicted_gain])
				except:
					flag_exp = True
					try:
						idx = self.__get_max_fit_index__(predicted[i])
						predicted_gain = str(predicted[i][int(self._cat_tests[idx])])
						if not crossCorrMode:
							local_result_to_file = "~"+predicted_gain # '~' means approximately fit #not use in crossCorrMode
							results.write(" and classfied as close as to writer:"+str(self._images.getiMap()[idx]))
							results.write(" with probability of:"+predicted_gain+"\n")
						#unsupervised:
						local_write_real = str(self._images.getiMap()[int(self._cat_tests[idx])])
						decreteResultPoints.append([str(self._identifiers[i]),predicted_gain])
					except Exception as ex:
						print(str(ex))
						return
				if not flag_exp:
					if not crossCorrMode:
						results.write(" and classfied as writer:"+str(self._images.getiMap()[int(self._cat_tests[i])]))
						results.write(" with probability of:"+predicted_gain+"\n")
						
			if not crossCorrMode:
				result_file = open(self._images.notes_folder+"//"+str(self._identifiers[i])+"//"+"_metadata_"+str(self._files_amount)+".txt","w+")
				image_id = str(self._identifiers[i])
				result_file.write("Img_id="+image_id+"\n")
				if self._algorithm == 2: #random forest
					result_file.write("Classified_writer="+str(local_write_real)+"\n")
				else:
					result_file.write("Classified_writer="+local_write_real+"\n") # SVM || KNN || LogisticRegression || combos:
					result_file.write("Probability_of="+local_result_to_file)
					
				realId = ""
				predictedId = ""
				intensity = ""
				if self._algorithm == 2:
					realId = str(local_write_real)
					predictedId = str(predicted_index)
					if realId == predictedId:
						intensity = "1"
					else:
						intensity = "0"
				else:
					realId = str(local_write_real)
					predictedId = str(local_write_real)
					intensity = str(predicted_gain)
				pageId = ""
				found = False
				for key,value in zip(self._IMGPreprocess.getGroupsKeys(),self._IMGPreprocess.getGroupsValues()):
					if found == True:
						break
					for val in value:
						if str(image_id) == str(val[3]):
							pageId = key
							found = True
							break
				
				local_html_content += """<tr id="""+str(i+1)+""">
						<td class="style text_style">
							<label class="style adjacent">"""+str(i+1)+"""</label>
						</td>
						<td class="style text_style">
							<label class="style adjacent">"""+str(image_id)+"""</label>
						</td>
						<td class="style text_style">
							<label class="style adjacent"><img id='note_img' alt='Classified Image' src='"""+self._images.access_img_path+"//"+image_id+"//"+image_id+".png"+"""'/></label>
						</td>
						<td class="style text_style">
							<label class="style adjacent">"""+str(pageId)+"""</label>
						</td>	
						<td class="style text_style">
							<label class="style adjacent">"""+realId+"""</label>
						</td>
						<td class="style text_style">
							<label class="style adjacent">"""+predictedId+"""</label>
						</td>	
						<td class="style text_style">
							<label class="style adjacent">"""+intensity[:6]+"""</label>
						</td>	
						<td class="style radio_style">
							<input class="refactor_group" autocomplete="off" id="inp_radio" type="radio" name="""+str(i+1)+""" value="yes" onclick="Functions.markSelectedLine(this)"/>
						</td>	
					</tr>"""
		
		if not crossCorrMode:	
			result_file.close()
			isTestImgsExist = False
			if len(self._test_imgs) <= 0:
				isTestImgsExist = True
			img_amount = len(self._test_imgs)
			if not isTestImgsExist:
				results.write("Successfull Hit Average Of:"+str(average/img_amount)+"%\n")
			else:
				results.write("Successfull Hit Average Of:0%\n")
				
			local_html_content+=self.HTML_TAIL
			
			html_results = open(self._images.pred_path+"//Summary_Table_"+str(self._files_amount)+".html","w+")
			html_results.write(local_html_content)
			html_results.close()

			results.write("(*)NOTE: Each Note Was Selected Randomly!\n")
			results.close()

		self._crossCorrelationAns.append(decreteResultPoints)
			
			
	"""
	Main purpose: make predication for preprocessed object using specific algorithm which is already picked up
	@param	self	this object
	"""		
	def predict(self):
		#event.widget.config(state = "disabled")
		predicted = []
		if self._algorithm == 1:
			predicted = self.SVM()
		elif self._algorithm == 2:
			predicted = self.RandomForest()
		elif self._algorithm == 4:
			predicted = self.KNN()
		elif self._algorithm == 6:
			predicted = self.LogisticRegression()
		elif self._algorithm == 8:
			firstPredicted = self.SVM()
			predicted = self.Combo_SecondClassify(1,firstPredicted)
		elif self._algorithm == 10:
			firstPredicted = self.LogisticRegression()
			predicted = self.Combo_SecondClassify(2,firstPredicted)
		if self._notesFocus == False:
			self.show_results_for_pages(predicted)
		else:
			self.show_results_for_each(predicted)
		#event.widget.config(state = "normal")
	
	