import stat
import shutil
import os
import time
import sys
	
def rmIMGS():
	notes_folder = "C://Users//Itay_Guy//SharePoint//Robotics//Exported Databases//Notes"
	pred_path = "C://Users//Itay_Guy//SharePoint//Robotics//Exported Databases//Predictions"
	if os.path.isdir(pred_path):
		for dirpath, dirnames, files in os.walk(pred_path,topdown = True):
			if not files:
				os.rmdir(str(pred_path))
				break
			else:
				shutil.rmtree(str(pred_path))
	if os.path.isdir(notes_folder):
		for dirpath, dirnames, files in os.walk(pred_path,topdown = True):
			if not files:
				os.rmdir(notes_folder)
				break
			else:
				shutil.rmtree(notes_folder)
		
#===========================================
# optional helper functions from ExtracIMGS:
#===========================================
	"""
	def change_metadata_to_classified_imgs(self,files_amount):
		for dir in os.listdir(self.notes_folder):
			if not dir == "Classification_Summary.html":
				os.chdir(self.notes_folder+"//"+dir)
				for file in glob(self.notes_folder+"//"+dir+"//"+"_metadata_"+files_amount+".txt"):
					with open(file) as f:
						line = f.read().split("\n")
	"""

	"""
	def clear_by_close(self):
		if os.path.isdir(self.pred_path):
			for dirpath, dirnames, files in os.walk(self.pred_path,topdown = False):
				if not files:
					shutil.rmtree(str(dirpath))
		if os.path.isdir(self.notes_folder):
			shutil.rmtree(self.notes_folder)
	"""	

	"""# These are implementations to make images more smooth:	
	def deskew(self,img):
		m = cv2.moments(img)
		if abs(m['mu02']) < 1e-2:
			return img.copy()
		skew = m['mu11']/m['mu02']
		M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
		img = cv2.warpAffine(img,M,(20, 20),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
		return img
		
	def hog(self,img):
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bins = np.int32(8*ang/(2*np.pi))
		bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), 8) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)     # hist is a 64 bit vector
		return hist
	"""
#============================================
# optional helper functions from  ImgLearner:
#============================================
	'''# optional to use for improve of this tool's development:	
	def SVD(self):
		svd = TruncatedSVD(n_components=self._notesFocus)
		svd.fit(self._feature_matrix,self._category_class_indecies)
		predicted = svd.components_
		print(self._feature_matrix)
		print()
		print(predicted)
		return predicted
	'''