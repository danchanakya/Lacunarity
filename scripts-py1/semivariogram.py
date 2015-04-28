import Image
import numpy as np
import datetime

from matplotlib import pyplot as plt


# Folders for input images and results
imgFolder1 = '/Users/danthac/Work/MSc/Lacunarity-MSMath-CSUCI/'
#imgFolder = imgFolder1 + 'imagesBrodatzTexture/textures/'
resFolder1 = '/Volumes/D16SD_TD/Study/MSc/results/'
resFolder  = resFolder1 + 'restmp/'

# Switches
histEq 		= True
histEqPic 	= True

nLags = 30

def getImageData(dataFolder, fileName, extFl):
	inim  = Image.open(dataFolder + fileName + extFl)

	imgMode = inim.mode
	print inim.format, inim.size, inim.mode
	if inim.mode != "L":
		inim = inim.convert("L")

	inar = np.asarray(inim)
	imageDim = len(inar.shape)
	print "Image " + fileName + " size: " + str(inar.shape) + ' dim (' + str(imageDim) + ')', "Image mode: ", imgMode
	# Pick image data
	if imageDim == 2:
		inar1 = inar[:,:]
	if imageDim == 3:
		inar1 = inar[:,:,0]
	return inar1

def histeq(im, imagename, nbr_bins=256):
	# get image histogram
	imhist, bins = np.histogram(im.flatten(), nbr_bins, (0,256)) #, density=True)

	if histEqPic:
		f1 = plt.figure()
		f1.clear()
		r1 = plt.hist(im.flatten(), nbr_bins)
		plt.title('Histogram Before ' + imagename)
		plt.xlabel('bin')
		plt.ylabel('Freq.')
		plt.savefig(resFolder + imagename + "_hstgrmB.png")
		plt.close(f1)
	#print len(imhist), imhist
	#print len(bins), bins
	cdf = imhist.cumsum() # cumulative distribution function
	#print 1.0 * cdf / cdf[-1]
	cdf = 255.0 * cdf / cdf[-1] #normalize

	# use linear interpolation of cdf to find new pixel values
	im2 = np.interp(im.flatten(), bins[:-1], cdf)

	if histEqPic:
		#PlotHist(imagename + '_af', idata, 18)
		#plt.imshow(idata, cmap=plt.cm.gray)
		#plt.savefig(resFolder + imagename + '_hiseq1.jpg')
		idata  = np.array(im2.reshape(im.shape), dtype='B')
		Image.fromarray(idata).save(resFolder + imagename + "_he.tif", "TIFF")

	if histEqPic:
		f1 = plt.figure()
		f1.clear()
		r1 = plt.hist(idata.flatten(), nbr_bins)
		plt.title('Histogram After ' + imagename)
		plt.xlabel('bin')
		plt.ylabel('Freq.')
		plt.savefig(resFolder + imagename + "_hstgrmA.png")
		plt.close(f1)

	return im2.reshape(im.shape), cdf

def ProcessAnImage_Variogram(imagename, ext1):
	dt1 = datetime.datetime.now()

	# Load Image
	idata1 = getImageData(imgFolder, imagename, ext1)

	if histEq:
		idata, cdf = histeq(idata1, imagename)
	else:
		idata = idata1

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1

if __name__ == '__main__':

	set = 2

	if set == 1:
		imgFolder = imgFolder1 + 'images1/'
		resFolder = resFolder1 + 'respy06_para/'

		testImages = ['honeycomb2'] #, 'Sq2_A', 'Sq2_B', 'Sq2_C', 'Sq2_D']
		for tImg in testImages:
			ProcessAnImage_Variogram(tImg, '.tif')
