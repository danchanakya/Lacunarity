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

def ProcessAnImage_SD_M_RANGE(imagename, ext1):
	dt1 = datetime.datetime.now()

	# Load Image
	idata1 = getImageData(imgFolder, imagename, ext1)

	if histEq:
		idata, cdf = histeq(idata1, imagename)
	else:
		idata = idata1

	range1 = 0
	(Xdim, Ydim) = idata.shape
	min_range = 99999

	for col in range(Xdim):
		for row in range(Ydim):
			#print "col,row", col, row
			# 0 degrees
			if (row + nLags) < Ydim:
				maxsv = 0
				for x in range(1, nLags):
					sum  = 0.0
					mult = 0.5/(nLags - x)
					for y in range(nLags - x + 1):
						diff = idata[col, row + y] - idata[col, row + y + x]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

			# 45 degrees
			if (col - nLags) > 0 and (row + nLags) < Ydim:
				maxsv = 0
				for x in range(1, nLags):
					sum = 0.0
					mult = 0.5/(nLags - x)
					for y in range(nLags - x + 1):
						diff = idata[col - y, row + y] - idata[col - y - x, row + y + x]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

			# 90 degrees
			if (col - nLags) > 0:
				maxsv = 0
				for x in range(1, nLags):
					sum = 0.0
					mult = 0.5/(nLags - x)
					for y in range(nLags - x + 1):
						diff = idata[col - y, row] - idata[col - y - x, row]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

			# 135 degrees
			if (row - nLags) > 0 and (col - nLags) > 0:
				maxsv = 0
				for x in range(1, nLags):
					sum = 0.0
					mult = 0.5/(nLags - x)
					for y in range(nLags - x + 1):
						diff = idata[col - y, row - y] - idata[col - y - x, row - y - x]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

			# 180 degrees
			if (row - nLags) > 0:
				maxsv = 0
				for x in range(1, nLags):
					sum = 0.0
					mult = 0.5/(nLags - x)
					for y in range(nLags - x + 1):
						diff = idata[col, row - y] - idata[col, row - y - x]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

			# 225 degrees
			if (col + nLags) < Xdim and (row - nLags) > 0:
				maxsv = 0
				for x in range(1, nLags):
					sum = 0.0
					mult = 0.5/(nLags - x)
					for y in range(nLags - x + 1):
						diff = idata[col + y, row - y] - idata[col + y + x, row - y - x]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

			# 270 degrees
			if (col + nLags) < Xdim:
				maxsv = 0
 				for x in range(1, nLags):
					sum = 0.0
					mult = 0.5/(nLags - x)
					for y in range(nLags - x + 1):
						diff = idata[col + y, row] - idata[col + y + x, row]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

			# 315 degrees
			if (row + nLags) < Ydim and (col + nLags) < Xdim:
				maxsv = 0
 				for x in range(1, nLags):
					sum = 0.0
					mult = 0.5/(nLags - x)
					for y in range(nLags - x + 1):
						diff = idata[col + y, row + y] - idata[col + y + x, row + y + x]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

	if range1 % 2 == 0:
		range1 -= 1
	if range1 > (nLags - 1):
		range1 = nLags - 1

	print "Range:", range1

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1


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

	if set == 2:
		imgFolder = imgFolder1 + 'images1/'
		resFolder = resFolder1 + 'respy06_para/'

		testImages = ['honeycomb2'] #, 'Sq2_A', 'Sq2_B', 'Sq2_C', 'Sq2_D']
		for tImg in testImages:
			ProcessAnImage_SD_M_RANGE(tImg, '.tif')
