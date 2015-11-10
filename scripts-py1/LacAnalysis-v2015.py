import Image
import csv
import numpy as np
import datetime

import PIL.ImageOps

from skimage.filters import threshold_otsu, threshold_adaptive, threshold_isodata

import json

import colorsys
import readColorList as colorL

from matplotlib.colors import LogNorm

from scipy.misc import imread,imsave

from matplotlib import pyplot as plt
from copy import deepcopy

from ALVLACSubRoutines  import *
from Output 			import *

# Author 		: Dantha Manikka-Baduge
# Supervisor 	: Prof. Geoff Dougherty - CSUCI
# Last updated 	: Oct-2015

# Folders for input images and results
imgFolder1 = '/Users/danthac/Work/MSc/Lacunarity-MSMath-CSUCI/'
#imgFolder = imgFolder1 + 'imagesBrodatzTexture/textures/'
resFolder1 = '/Volumes/D16SD_TD/Study/MSc/Results/'
resFolder  = resFolder1 + 'restmp/'

testList = []

# All Possible threshold methods - "fixed", "isodata", "adaptive", "otsu"
#ThresholdTypes = ["fixed", "isodata", "adaptive", "otsu"]
#ThresholdTypes = ["adaptive", "otsu"]
ThresholdTypes  = ["adaptive"]
#ThresholdTypes = ["otsu"]
#ThresholdTypes = ["fixed"]
#ThresholdTypes = ["isodata"]

imageCrop = {'MRI-slice' : (134,257,70,35), 'cows-gray' :(200,107,20,25)}


# Switches
histEq 		= True
histEqPic 	= True

CalLac 		= True
FullRg 		= True
Graphs 		= True
SaveData 	= True
BothSides  	= False

def getImageData(dataFolder, fileName, extFl):
	inim  = Image.open(dataFolder + fileName + extFl)

	imgMode = inim.mode
	print "B:", inim.format, inim.size, inim.mode
	if inim.mode != "L":
		inim = inim.convert("L")
	print "A:", inim.format, inim.size, inim.mode

	inar = np.asarray(inim)
	imageDim = len(inar.shape)
	print "Image " + fileName + " size: " + str(inar.shape) + ' dim (' + str(imageDim) + ')', "Image mode: ", imgMode
	# Pick image data
	if imageDim == 2:
		inar1 = inar[:,:]
	if imageDim == 3:
		inar1 = inar[:,:,0]
	return inar1

def histeq(im, imagename, nbr_bins=255):
	# get image histogram
	print sum(im.flatten())
	print "before eq:", np.min(im.flatten()), np.max(im.flatten())
	max1 = np.max(im.flatten())
	imhist, bins = np.histogram(im.flatten() , nbr_bins, (0.0,max1)) #, (0,256)) #, density=True)
	#print len(imhist), len(bins)
	if histEqPic:
		f1 = plt.figure()
		f1.clear()
		r1 = plt.hist(im.flatten(), nbr_bins)
		plt.title('Histogram Before ' + imagename)
		plt.xlabel('bin')
		plt.ylabel('Freq.')
		plt.savefig(resFolder + imagename + "_hstgrmB.png")
		plt.close(f1)

	cdf = imhist.cumsum() # cumulative distribution function
	#cdf = 255.0 * cdf / cdf[-1] #normalize
	cdf = max1 * cdf / cdf[-1] #normalize
	cdf1 = np.concatenate(([0.0],cdf))
	# use linear interpolation of cdf to find new pixel values
	im2 = np.interp(im.flatten(), bins, cdf1)

	#PlotHist(imagename + '_af', idata, 18)
	#plt.imshow(idata, cmap=plt.cm.gray)
	#plt.savefig(resFolder + imagename + '_hiseq1.jpg')
	#idata  = np.array(im2.reshape(im.shape), dtype='B')
	#Image.fromarray(idata).save(resFolder + imagename + "_he.tif", "TIFF")
	print "after  eq:", np.min(im2.flatten()), np.max(im2.flatten())

	if histEqPic:
		f1 = plt.figure()
		f1.clear()
		r1 = plt.hist(im2, nbr_bins)
		plt.title('Histogram After ' + imagename)
		plt.xlabel('bin')
		plt.ylabel('Freq.')
		plt.savefig(resFolder + imagename + "_hstgrmA.png")
		plt.close(f1)

	return im2.reshape(im.shape), cdf

# Still testing - varify hist eq function
def histeqTest(im, imagename, nbr_bins=256):
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
	print len(imhist), imhist
	print len(bins), bins
	cdf = imhist.cumsum() # cumulative distribution function
	print 1.0 * cdf / cdf[-1]
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

def PlotHist(imagename, data, bins):
	fileName = imagename + "_bin_" +  str(bins) + ".png"
	f = plt.figure()
	f.clear()
	r1 = plt.hist(data, bins)
	plt.title('Histogram for ' + imagename + ' bins - ' +  str(bins))
	plt.xlabel('bin')
	plt.ylabel('freq.')
	figFilename = resFolder + fileName
	plt.savefig(figFilename)
	plt.close(f)
	return fileName

def hue2rgb(p, q, t):
	if t < 0: t += 1
	if t > 1: t -= 1
	if t < 1/6: return p + (q - p) * 6 * t
	if t < 1/2: return q
	if t < 2/3: return p + (q - p) * (2/3 - t) * 6
	return p

def hslToRgb_1(h, s, l):
	"""
	 Converts an HSL color value to RGB. Conversion formula
	 adapted from http://en.wikipedia.org/wiki/HSL_color_space.
	 Assumes h, s, and l are contained in the set [0, 1] and
	 returns r, g, and b in the set [0, 255].

	 @param   Number  h       The hue
	 @param   Number  s       The saturation
	 @param   Number  l       The lightness
	 @return  Array           The RGB representation
	"""
	if s == 0:
		r = g = b = l # achromatic
	else:
		if l < 0.5:
			q = l * (1 + s)
		else:
			q = l + s - l * s
		p = 2 * l - q
		r = hue2rgb(p, q, h + 1/3)
		g = hue2rgb(p, q, h)
		b = hue2rgb(p, q, h - 1/3)
	#print r, g, b
	return [r, g, b]

def hslToRgb(h, s, l):
	"""
	 Converts an HSL color value to RGB. Conversion formula
	 adapted from http://en.wikipedia.org/wiki/HSL_color_space.
	 Assumes h, s, and l are contained in the set [0, 1] and
	 returns r, g, and b in the set [0, 255].

	 @param   Number  h       The hue
	 @param   Number  s       The saturation
	 @param   Number  l       The lightness
	 @return  Array           The RGB representation
	"""
	#h1 = h/6.0
	#c  = (1 - (2*l-1)) * s
	if s == 0:
		r = g = b = l # achromatic
	else:
		if l < 0.5:
			q = l * (1 + s)
		else:
			q = l + s - l * s
		p = 2 * l - q
		r = hue2rgb(p, q, h + 1/3)
		g = hue2rgb(p, q, h)
		b = hue2rgb(p, q, h - 1/3)
	#print r, g, b
	return [r, g, b]

def SaveLacData(fileName, datax, datay, legend):
	noH = len(legend)
	print noH
	strCSV = ','
	for i in range(len(datax)):
		strCSV = strCSV + str(datax[i]) + ','
	strCSV = strCSV	+ '\n'

	with open(fileName, 'w') as file1:
		file1.writelines(strCSV)
		for i in range(noH):
			strCSV = legend[i] + ','
			for j in range(len(datax)):
				strCSV = strCSV + str(datay[i][j]) + ','
			strCSV = strCSV	+ '\n'
			file1.writelines(strCSV)
		if BothSides:
			for i in range(noH):
				strCSV = legend[i] + ' Inv,'
				for j in range(len(datax)):
					strCSV = strCSV + str(datay[i+noH][j]) + ','
				strCSV = strCSV	+ '\n'
				file1.writelines(strCSV)

			strCSV = legend[noH] + ','
			for j in range(len(datax)):
				strCSV = strCSV + str(datay[noH * 2][j]) + ','
			strCSV = strCSV	+ '\n'
			file1.writelines(strCSV)

			strCSV = legend[noH] + ' Inv,'
			for j in range(len(datax)):
				strCSV = strCSV + str(datay[noH * 2 + 1][j]) + ','
			strCSV = strCSV	+ '\n'
			file1.writelines(strCSV)

def SaveLacDataInd(fileName, datax, datay, legend):
	with open(fileName, 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',')
		spamwriter.writerow(legend)
		spamwriter.writerow(datax)
		spamwriter.writerows(datay)

def SaveALVDataInd(fileName, datax, datay):
	with open(fileName, 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',')
		spamwriter.writerow(datax)
		spamwriter.writerows(datay)

def OtsuThreshold(imageData):
	(sx, sy)	= imageData.shape
	maxVal		= max(np.ravel(imageData))
	minVal		= min(np.ravel(imageData))
	bins		= maxVal - minVal
	total		= sx * sy

	(dataHst, nHst) = np.histogram(imageData, bins)
	dataHstProb = dataHst * 1.0/ dataHst.sum()

	threshold1 = 0.0;
	threshold2 = 0.0;
	sumT = 0
	sumB = 0
	for i in range(len(dataHst)):
		sumT += dataHst[i] * nHst[i]

	wB   = 0
	maxV = 0
	for i in range(256):
		if i >= minVal and i < maxVal:
			#print nHst[i - minVal], i - minVal
			wB += dataHst[i - minVal]

			if wB == 0:
				continue
			wF = total - wB
			if wF == 0:
				break
			sumB += i * dataHst[i - minVal]
			mB = sumB / wB
			mF = (sumT - sumB) / wF
			between = wB * wF * ((mB - mF) ** 2)
			if between >= maxV:
				threshold1 = i
				if between > maxV:
					threshold2 = i
				maxV = between
	#print dataHst, nHst, dataHst.sum(), size
	ostuT = (threshold1 + threshold2)/2
	return ostuT

def GetALV(datarray, imagename, w = 2, mjmp = 1, clipType = 0):
	"""
		Average Local Variance program for 2D data
		It uses Python PIL and select modules
		by Dantha Manikka-Baduge, V.1.0, Feb 28 2008
			Type 0 - average all
				 1 - clip out smaller boxes at edges
				 2 - fill out with zero
			mj   0 - moving
   				 1 - jumping
	"""
	ALVData 	= []
	SaveAggImgs = 0
	funcused    = np.std # var ?

	(sx, sy) = datarray.shape
	if clipType == 0:
		maxA = max(sx, sy) // w
	else:
		maxA = max(sx, sy) #-1

	if mjmp == 1: # Jumping
		d1 = JumpingImg(datarray, funcused, 2, clipType)
	else:
		d1 = SlidingImg(datarray, funcused)

	ALVData.append(np.average(d1))

	for x in [x+1 for x in range(maxA)][1:]:
		# print x
		d2 = JumpingImg(datarray, np.average, x, clipType)
		if SaveAggImgs == 1:
			d3   = array(d2, dtype='B')
			agIm = Image.fromarray(d3)
			fileName = resFolder + imagename + '_AggData__' + str(x) + '.tif'
			agIm.save(fileName, "TIFF")

		if mjmp == 1:
			d3 = JumpingImg(d2, funcused, 2, clipType)
		else:
			d3 = SlidingImg(d2, funcused)

		ALVData.append(np.average(d3))

	return ALVData

def GetLacunarity(data, mj = 0, clipType = 0):
	(sx, sy) = data.shape
	minxy = min(sx, sy);
	lac = []

	dataAve = np.average(data)
	dataStd = np.std(data)
	lac1  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
	lac.append(lac1)

	if FullRg:
		rangeW = range(2, minxy)
	else:
		rangeW = range(5, 180)

	for count in rangeW:
		if mj == 1: # Jumping
			CS = JumpingImg(data, sum1, count, clipType)
		else:
			CS = SlidingImg(data, sum1, count)
		dataAve = np.average(CS)
		dataStd = np.std(CS)
		lac1  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
		lac.append(lac1)

	#print len(lac), lac

	return lac

def CalcLacunarity(data, wd = 1, pos = []):
	(sx, sy) = data.shape
	minxy = min(sx, sy);
	if sx != 0 and sy != 0:
		if wd == 1:
			dataAve = np.average(data)
			dataStd = np.std(data)
			if dataAve > 0:
				lac     = 1 + (dataStd * dataStd)/(dataAve * dataAve)
			else:
				lac     = 1
				#print dataStd, dataAve, sx, sy

		elif minxy > wd:
			CS = SlidingImg(data, sum, wd)
			dataAve = np.average(CS)
			dataStd = np.std(CS)
			lac  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
		else:
			lac = 0
	else:
		lac = 0
		#print sx,sy,pos

	if (sx != 0 and sy == 0) or (sx == 0 and sy != 0):
		print "One dim zero - should not happend - need to check this situation"
		print sx,sy,pos
	return lac

def GetLacImageLocation(data, wdList, calPos, imagename):
	resDict = {}
	(sx, sy) = data.shape
	#minxy = min(sy, sx);

	res = []

	for wd in wdList:
		df = wd / 2
		xtop = calPos[0] - df
		ytop = calPos[1] - df
		if (ytop > 0 and xtop > 0 and xtop+wd < sx and ytop+wd <sy):
			selPicData = data[ytop:ytop+wd, xtop:xtop+wd]
			res.append(CalcLacunarity(selPicData))
			idata  = np.array(selPicData, dtype='B')
			fName = resFolder + imagename + "_" + str(calPos[0]) + "_" + str(wd) + ".tif"
			Image.fromarray(idata).save(fName, "TIFF")
		else:
			print "Size out of bound"

	return [resDict, res, str(calPos[0]) + "," + str(calPos[1])]

def GetLacImageLocationNWindow(data, wd, Pos, imagename):
	(sx, sy) = data.shape

	#print wd, Pos
	#res = []

	df = wd / 2
	xtop = Pos[0] - df
	ytop = Pos[1] - df
	if (ytop >= 0 and xtop >= 0 and xtop+wd < sx and ytop+wd <sy):
		#selPicData = data[ytop:ytop+wd, xtop:xtop+wd]
		selPicData = data[xtop:xtop+wd, ytop:ytop+wd]
		ret = CalcLacunarity(selPicData, 1, (xtop, ytop, wd))
	else:
		ret = 0

	return ret

def GetLacSegImage(iData, wd, imagename, slice = 5):
	resDict = {}
	data = maskImage(imagename, iData)
	(sx, sy) = data.shape

	#minxy = min(sy, sx);
	try:	# windows size
		wd = int(wd)
	except: # window ratio
		s  = int(wd[:-1])
		wd = min(sx/s, sy/s)
	df = wd / 2

	lacData = np.zeros([sx-df*2, sy-df*2], dtype=np.float)

	for y in range(sy-df*2):
		for x in range(sx-df*2):
			selData = data[x:x+wd, y:y+wd]
			dataAve = np.average(selData)
			dataStd = np.std(selData)
			if dataAve > 0:
				lacData [x, y]  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
			else:
				lacData [x, y]  = 1.0

	ld1 = maskImage(imagename, lacData)
	lacData = ld1
	minD = np.min(lacData)
	maxD = np.max(lacData)
	print "Lacunarity range: from ", minD, " to ", maxD, " for window = ", wd

	if (maxD - minD) > 0.0:
		lacDataN = (lacData - minD) * 255.0/(maxD-minD)
		trData2  = np.array(lacDataN, dtype='B')
		if len(imagename) > 0:
			Image.fromarray(trData2).save(resFolder + imagename + '_LacVals_w_' + str(wd) + '.tif',"TIFF")

		#trData3, cdf = histeq(trData2, imagename + '_LacVals_w_' + str(wd) )
		trData3, cdf = histeq(lacData, imagename + '_LacVals_w_' + str(wd) )
		GetSegImage(trData3, imagename, slice = 5, post= 'Window-' + str(wd))
		#GetSegImage(trData3, imagename, slice = 5, post= 'Window-' + str(wd))
	else:
		print "Image data has flat values..."

	return [resDict]

def adjustFigAspect(fig, aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

def GetSegImage(iData, imagename, slice = 5, post= '1'):
	(sx, sy) = iData.shape
	minxy = min(sy, sx);
	minD = np.min(iData.flatten())
	maxD = np.max(iData.flatten())
	lvls = np.arange(minD, maxD + 0.0001, (maxD-minD)/slice)
	x1, y1 = np.meshgrid(range(sy), range(sx))
	fig  = plt.figure()
	#ar = sy*1.25/sx
	#adjustFigAspect(fig, aspect=ar)  # '_' + str(ar) +
	ax1  = fig.add_subplot(111)
	CF   = ax1.contourf(x1, y1, iData, norm = None, levels = lvls) # , aspect="equal" norm=LogNorm()
	#CS   = ax1.contour( x1, y1, iData, norm = None, colors = 'k', levels = lvls)
	ax1.set_aspect('equal')
	cbar = plt.colorbar(CF, ticks=lvls, format='%.2f')
	fileName = imagename + '_sliced-' +  str(slice) + '_' + post + '.png'
	plt.savefig(resFolder + fileName)
	plt.close(fig)

	f = plt.figure()
	f.clear()
	r1 = plt.hist(iData.flatten(), slice)
	plt.title('Histogram ' + imagename + ' sliced-' +  str(slice) + ' ' + post)
	plt.xlabel('bin')
	plt.ylabel('freq.')
	fileName = imagename + "_histgrm+sliced-" +  str(slice) + '_' + post + ".png"
	plt.savefig(resFolder + fileName)
	plt.close(f)

	# By direct color assignments
	"""
	fileName = imagename + "_sliced-" +  str(slice) + '_' + post + "_leg.tif"
	[dataHst, nHst] = np.histogram(iData.flatten(), slice)
	[clr, clrNameLst] = colorL.colorLUT(resFolder + fileName, nHst, slice)

	# Debug
	#print dataHst, nHst
	#print np.min(iData.flatten()), np.max(iData.flatten())
	#for i in range(9):
	#	print colorNameLst[i], nHst[i+1]

	lacImage = np.zeros([sx, sy, 3], dtype=np.uint8)
	for y in range(sy):
		for x in range(sx):
			pos = 0
			while(iData[x, y] > nHst[pos+1]):
				#print iData[x, y], nHst[pos+1], pos
				pos += 1
			#print iData[x, y], nHst[pos], pos, len(nHst)
			lacImage[x, y] = clr[clrNameLst[len(clr)-1-pos]]

	fileName = imagename + "_sliced-" +  str(slice) + '_' + post + "_ext.png"
	Image.fromarray(lacImage).save(resFolder + fileName, "TIFF")
	"""

def NormalizedRoy(data):
	lacNorm = []
	Lmin = min(data)
	Lmax = max(data)
	print Lmin, Lmax
	for dt in data:
		#if (Lmax-Lmin) != 0:
		lacNorm.append((dt-Lmin)/(Lmax-Lmin))
		#else:
		#	lacNorm.append(0)
	return lacNorm

def NormalizedMalhi(data):
	lacNorm = []
	for dt in data:
		lacNorm.append(log(dt)/log(data[0]))
	return lacNorm

def NormalizedH(data, dataR):
	# Henebry
	lacNorm = []
	for n in range(len(data)):
		nl = 2 - 1/data[n] - 1/dataR[n]
		lacNorm.append(nl);
	return lacNorm

# Threshold methods
def Threshold_Fixed(idata, savePics = '', threshold = 128):
	trData2  = []
	trDataR2 = []
	trData1  = idata >= threshold
	trData2  = np.array(trData1, dtype='B')
	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_TH_fixed_' + str(threshold) + '.tif',"TIFF")

	if BothSides:
		trDataR1  = idata < threshold
		trDataR2  = np.array(trDataR1, dtype='B')
		if len(savePics) > 0:
			Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_TH_fixed_' + str(threshold) + 'R.tif',"TIFF")

	return [threshold, trData2, trDataR2]

def Threshold_ISODATA(idata, savePics = ''):
	trData2  = []
	trDataR2 = []
	threshold = threshold_isodata(idata)

	trData1  = idata >= threshold
	trData2  = np.array(trData1, dtype='B')
	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_TH_isodata_' + str(threshold) + '.tif',"TIFF")

	if BothSides:
		trDataR1  = idata < threshold
		trDataR2  = np.array(trDataR1, dtype='B')
		if len(savePics) > 0:
			Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_TH_isodata_' + str(threshold) + 'R.tif',"TIFF")

	return [int(threshold), trData2, trDataR2]

def Threshold_Adaptive(idata, savePics = ''):
	trData2  = []
	trDataR2 = []
	block_size  = 40
	threshold   = 0
	trData1     = threshold_adaptive(idata, block_size, offset = 10)

	trData2     = np.array(trData1, dtype='B')
	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_TH_adaptive_' + str(threshold) + '.tif',"TIFF")

	if BothSides:
		trDataR10  = PIL.ImageOps.mirror(Image.fromarray(idata))
		trDataR11  = inar = np.asarray(trDataR10)
		trDataR1   = threshold_adaptive(trDataR11, block_size, offset=10)

		trDataR2   = np.array(trDataR1, dtype='B')
		if len(savePics) > 0:
			Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_TH_adaptive_' + str(threshold) + 'R.tif',"TIFF")

	return [threshold, trData2, trDataR2]

def Threshold_Otsu(idata, savePics = ''):
	trData2  = []
	trDataR2 = []
	threshold = OtsuThreshold(idata)
	trData1  = idata >= threshold
	trData2  = np.array(trData1, dtype='B')
	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_TH_otsu_' + str(threshold) + '.tif',"TIFF")

	if BothSides:
		trDataR1  = idata < threshold
		trDataR2  = np.array(trDataR1, dtype='B')
		if len(savePics) > 0:
			Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_TH_otsu_' + str(threshold) + 'R.tif',"TIFF")

	return [threshold, trData2, trDataR2]

def Lac_Threshold_Image(idata, thresh_type, savePics = '', mj = 0, threshold = 128):
	dt1 = datetime.datetime.now()

	threshImageData = []
	lac1 			= []
	lacR1 			= []
	lacLeg			= ''
	lacLegR			= ''

	if thresh_type == "fixed":
		threshImageData = Threshold_Fixed(idata, savePics, threshold)
		lacLeg   = thresh_type + ' = ' + str(threshold)
		lacLegR   = 'R ' + thresh_type + ' = ' + str(threshold)

	if thresh_type == "isodata":
		threshImageData = Threshold_ISODATA(idata, savePics)
		if len(threshImageData) > 0:
			threshold = threshImageData[0]
		lacLeg   = thresh_type + ' = ' + str(threshold)
		lacLegR   = 'R ' + thresh_type + ' = ' + str(threshold)

	if thresh_type == "adaptive":
		threshImageData = Threshold_Adaptive(idata, savePics)
		threshold = 0
		lacLeg   = thresh_type
		lacLegR   = 'R ' + thresh_type

	if thresh_type == "otsu":
		threshImageData = Threshold_Otsu(idata, savePics)
		threshold = 0
		lacLeg   = thresh_type
		lacLegR   = 'R ' + thresh_type

	if thresh_type == "":
		threshImageData = [0,idata,[]]
		threshold = 0
		lacLeg    = thresh_type
		lacLegR   = 'R ' + thresh_type

	if CalLac:
		if len(threshImageData) > 1 and len(threshImageData[1]) > 0:
			lac1     = GetLacunarity(threshImageData[1], mj)
			#lacLeg   = thresh_type + ' = ' + str(threshold)

		if len(threshImageData) > 2 and len(threshImageData[2]) > 0:
			lacR1     = GetLacunarity(threshImageData[2], mj)
			#lacLegR   = 'R ' + thresh_type + ' = ' + str(threshold)

	dt2 = datetime.datetime.now()
	print 'Time Taken for Lac. with ' + thresh_type + ' Threshold = ', dt2-dt1

	return [lac1, lacR1, lacLeg, lacLegR, threshold]

def Calc_LacHistogramThreshold(idata, savePics = ''):
	dt1 = datetime.datetime.now()

	chkPnt	 = 0.1
	p2		 = 0.0
	p2acc	 = []
	lacData  = []
	lacDataR = []
	lacLeg   = []

	# Image Statistics
	maxVal		= max(np.ravel(idata))
	minVal		= min(np.ravel(idata))
	bins		= maxVal - minVal

	# Get Histogram for all possible values
	(dataHst, nHst) = np.histogram(idata, bins)
	dataHstProb = dataHst * 1.0/ dataHst.sum()

	for h in range(len(dataHstProb)):
		p2 = p2 + dataHstProb[h]
		if p2 >= chkPnt:
			print "P% = ", chkPnt, datetime.datetime.now() - dt1
			lacLeg.append("(%0.1f-%.4f)" % (chkPnt, nHst[h]))
			trData1  = idata <= nHst[h]
			trData2  = np.array(trData1, dtype='B')
			if len(savePics) > 0:
				Image.fromarray(trData2 * 255).save(resFolder + savePics + '_' + str(chkPnt) + "-" + str(nHst[h]) + '.tif',"TIFF")
			lac1     = GetLacunarity(trData2)
			lacData.append(lac1)
			if BothSides:
				trDataR1  = idata > nHst[h]
				trDataR2  = np.array(trDataR1, dtype='B')
				if len(savePics) > 0:
					Image.fromarray(trDataR2 * 255).save(resFolder + savePics + '_' + str(chkPnt) + "-" + str(nHst[h]) + 'R.tif',"TIFF")
				lacR1     = GetLacunarity(trDataR2)
				lacDataR.append(lacR1)
			chkPnt   = chkPnt + 0.1
			if chkPnt > 0.91:  # Remove 100% Threshold
				break

		p2acc.append(p2)

	dt2 = datetime.datetime.now()
	print "C LacHistogramThreshold = ", dt2-dt1

	return [lacData, lacDataR, lacLeg]

def Calc_LacSegment(imagename, idata, wd):
	dt1 = datetime.datetime.now()

	for w1 in wd:
		[retDict] = GetLacSegImage(idata, w1, imagename)

	dt2 = datetime.datetime.now()
	print "Time Taken for LacSegment = ", dt2-dt1

def Calc_LacSegSemiVar(imagename, idata, wdszs, post='-1'):
	dt1 = datetime.datetime.now()

	(dx, dy) = idata.shape
	(wx, wy) = np.array(wdszs).shape

	idata = maskImage(imagename, idata)

	if dx == wx and dy == wy:
		imageD0 = []
		for a1 in range(dx):
			irD = []
			for a2 in range(dy):
				irD.append(GetLacImageLocationNWindow(idata, wdszs[a1][a2], (a1, a2), imagename + post))
			imageD0.append(irD)

		imageD  = np.array(imageD0)
		imageD0 = maskImage(imagename, imageD)

		imageDnp  = np.array(imageD0 * 255.0/np.max(imageD0),  dtype='B')
		Image.fromarray(imageDnp).save(resFolder + imagename + post + "-Lac.tif", "TIFF")

		#npimageD, cdf = histeq(imageDnp, imagename + post + "-LacHist")

		imageD1, cdf = histeq(imageD0, imagename + post + "-LacHist")

		GetSegImage(imageD1, imagename + post + "-LacEq")

		with open(resFolder + imagename + post + "_Lac.csv", 'w') as fp:
			a = csv.writer(fp, delimiter=',')
			a.writerows(imageD0)
	else:
		print "image size and window data array size is different!"

	dt2 = datetime.datetime.now()
	print "Time Taken for LacSegSemiVar = ", dt2-dt1

def Calc_LacNormPlot(lacData):
	x  = range(1, xMax+1)
	lx = [log(x1) for x1 in x]

	if SaveData:
		# Save Data to CSV file
		saveas = resFolder + 'lac_' + imagename + ".csv"
		WriteToCSVFile(saveas, [lacData])

	if Graphs:
		# Plot
		# Lacunarity
		f0 = plt.figure()
		f0.clear()
		for dset in lacData:
			plt.plot(x, dset[1], label=dset[2])
		plt.legend()
		plt.title("All Threshold Methods")
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_allthresholds.png"
		plt.savefig(figFilename)
		plt.close(f0)

		# Log Log - Lacunarity
		f0 = plt.figure()
		f0.clear()
		for dset in lacData:
			plt.loglog(x, dset[1], label=dset[2])
		plt.legend()
		plt.title("All Threshold Methods - log/log")
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_allthresholds_loglog.png"
		plt.savefig(figFilename)
		plt.close(f0)

		# Normalize - Lacunarity
		f0 = plt.figure()
		f0.clear()
		for dset in lacData:
			y1 = NormalizedMalhi(dset[1])
			plt.plot(x, y1, label=dset[2])
		plt.legend()
		plt.title("All Threshold Methods -  Normalized (Malhi)")
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_allthresholds_NormM.png"
		plt.savefig(figFilename)
		plt.close(f0)

		# Normalize - Lacunarity
		f0 = plt.figure()
		f0.clear()
		for dset in lacData:
			y1 = NormalizedMalhi(dset[1])
			plt.plot(lx, y1, label=dset[2])
		plt.legend()
		plt.title("All Threshold Methods -  Normalized (Malhi)")
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_allthresholds_NormM-log.png"
		plt.savefig(figFilename)
		plt.close(f0)

normMethod = 'roy'
def Calc_LacBINAllThresholds(imagename, idata):
	lacData = []
	xMax = 10000
	for t in range(len(ThresholdTypes)):
		ttype = "" #ThresholdTypes[t]
		[lac1, lacR1, lacLeg1, lacLegR, thresh] = Lac_Threshold_Image(idata, ttype, imagename)
		xMax = min(xMax, len(lac1))
		lacData.append([lac1, lacLeg1, thresh])

	if not CalLac:
		return

	x  = range(1, xMax+1)
	lx = [log(x1) for x1 in x]

	if SaveData:
		# Save Data to CSV file
		saveas = resFolder + 'lac_' + imagename + ".csv"
		WriteToCSVFile(saveas, [lacData])

	if Graphs:
		# Plot
		# Lacunarity
		f0 = plt.figure()
		f0.clear()
		for dset in lacData:
			plt.plot(x, dset[0], label=dset[1])
		plt.legend()
		if len(ThresholdTypes) > 1:
			plt.title("Threshold Methods")
		else:
			plt.title("Threshold = " + ttype)
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_allthresholds.png"
		plt.savefig(figFilename)
		plt.close(f0)

		# Log Log - Lacunarity
		f0 = plt.figure()
		f0.clear()
		for dset in lacData:
			plt.loglog(x, dset[0], label=dset[1])
		plt.legend()
		if len(ThresholdTypes) > 1:
			plt.title("Threshold Methods - log/log")
		else:
			plt.title("Threshold = " + ttype + " - log/log")
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_allthresholds_loglog.png"
		plt.savefig(figFilename)
		plt.close(f0)

		if normMethod == 'malhi':
			# Normalize - Lacunarity
			f0 = plt.figure()
			f0.clear()
			for dset in lacData:
				y1 = NormalizedMalhi(dset[0])
				plt.plot(x, y1, label=dset[1])
			plt.legend()
			if len(ThresholdTypes) > 1:
				plt.title("Threshold Methods - Normalized (Malhi)")
			else:
				plt.title("Threshold = " + ttype + " - Normalized (Malhi)")
			plt.xlabel('window size')
			plt.ylabel('lacunarity')
			figFilename = resFolder + imagename + "_allthresholds_NormM.png"
			plt.savefig(figFilename)
			plt.close(f0)

			# Normalize - Lacunarity
			f0 = plt.figure()
			f0.clear()
			for dset in lacData:
				y1 = NormalizedMalhi(dset[0])
				plt.plot(lx, y1, label=dset[1])
			plt.legend()
			if len(ThresholdTypes) > 1:
				plt.title("Threshold Methods - Normalized (Malhi)")
			else:
				plt.title("Threshold = " + ttype + " - Normalized (Malhi)")
			plt.xlabel('window size (log)')
			plt.ylabel('lacunarity')
			figFilename = resFolder + imagename + "_allthresholds_NormM-log.png"
			plt.savefig(figFilename)
			plt.close(f0)

		if normMethod == 'roy':
			# Normalize - Lacunarity
			f0 = plt.figure()
			f0.clear()
			for dset in lacData:
				y1 = NormalizedRoy(dset[0])
				plt.plot(x, y1, label=dset[1])
			plt.legend()
			if len(ThresholdTypes) > 1:
				plt.title("Threshold Methods - Normalized (Roy)")
			else:
				plt.title("Threshold = " + ttype + " - Normalized (Roy)")
			plt.xlabel('window size')
			plt.ylabel('lacunarity')
			figFilename = resFolder + imagename + "_allthresholds_NormM.png"
			plt.savefig(figFilename)
			plt.close(f0)

			# Normalize - Lacunarity
			f0 = plt.figure()
			f0.clear()
			for dset in lacData:
				y1 = NormalizedRoy(dset[0])
				plt.plot(lx, y1, label=dset[1])
			plt.legend()
			if len(ThresholdTypes) > 1:
				plt.title("Threshold Methods - Normalized (Roy)")
			else:
				plt.title("Threshold = " + ttype + " - Normalized (Roy)")
			plt.xlabel('window size (log)')
			plt.ylabel('lacunarity')
			figFilename = resFolder + imagename + "_allthresholds_NormM-log.png"
			plt.savefig(figFilename)
			plt.close(f0)

def NU_Calc_LacBIN(imagename, idata, threshTy = 1, mj = 0):
	if mj == 0:
		mvjmp = " Moving Window"
	else:
		mvjmp = " Jumping Window"
	retDict = {}

	[lac1, lacR1, lacLeg1, lacLegR, thresh] = Lac_Threshold_Image(idata, ThresholdTypes[threshTy], imagename, mj)
	thresh_title_part = ThresholdTypes[threshTy] + ' Threshold = ' + str(thresh)

	retDict[ThresholdTypes[threshTy] + "_threshold"] = thresh
	thresh_type = ThresholdTypes[threshTy]

	x = range(1, len(lac1)+1)

	# Save Data to CSV file
	dataFilename = resFolder + imagename + "_BIN_Lac_threshold-" + str(thresh) + str(mvjmp) + ".csv"
	SaveLacDataInd(dataFilename, x, [lac1, lacR1], lacLeg1)
	#SaveLacData(dataFilename, x, [lac1, lacR1], [lacLeg1])
	retDict["lacBINCSV"] = imagename + "_BIN_Lac_threshold-" + str(thresh) + str(mvjmp) + ".csv"

	# Lacunarity
	f0 = plt.figure()
	f0.clear()
	plt.plot(x, lac1, label=lacLeg1)
	plt.legend()
	plt.title(thresh_title_part + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "_threshold1-" + str(thresh) + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINPlot"] = imagename + "_" + thresh_type + "_threshold1-" + str(thresh) + str(mvjmp) + ".png"
	plt.close(f0)

	if BothSides:
		# Inv Lacunarity
		f0 = plt.figure()
		f0.clear()
		plt.plot(x, lac1,  label=lacLeg1)
		plt.plot(x, lacR1, label=lacLeg1 + "_inv")
		plt.legend()
		plt.title(thresh_title_part + mvjmp)
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_" + thresh_type + "_threshold-" + str(thresh) + str(mvjmp) + ".png"
		plt.savefig(figFilename)
		retDict["lacBINPlot"] = imagename + "_" + thresh_type + "_threshold-" + str(thresh) + str(mvjmp) + ".png"
		plt.close(f0)

	# Log Log - Lacunarity
	f0 = plt.figure()
	f0.clear()
	plt.loglog(x, lac1, label=lacLeg1)
	if BothSides:
		plt.loglog(x, lacR1, label=lacLeg1 + "_Inv")
	plt.legend()
	plt.title(thresh_title_part + mvjmp)
	plt.xlabel('window size (log)')
	plt.ylabel('lacunarity (log)')
	figFilename = resFolder + imagename + "_" + thresh_type + "_loglog" + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINLogPlot"] = imagename + "_" + thresh_type + "_loglog" + str(mvjmp) + ".png"
	plt.close(f0)

	if BothSides:
		# Normalize - Lacunarity
		f0 = plt.figure()
		f0.clear()
		y3 = NormalizedH(lac1, lacR1)
		p1 = plt.plot(x, y3, label=lacLeg1 + " - H")
		plt.legend()
		plt.title(thresh_title_part + ' Normalized (Henebry)'  + mvjmp)
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_" + thresh_type + "NormH" + str(mvjmp) + ".png"
		plt.savefig(figFilename)
		retDict["lacBINNormPlot"] = imagename + "_" + thresh_type + "NormH" + str(mvjmp) + ".png"
		plt.close(f0)

	# Normalize - Lacunarity
	f0 = plt.figure()
	f0.clear()
	y1 = NormalizedMalhi(lac1)
	p1 = plt.plot(x, y1, label=lacLeg1 + " - M")
	plt.legend()
	plt.title(thresh_title_part + ' Normalized (Malhi)'  + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "NormM" + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINNormPlot"] = imagename + "_" + thresh_type + "NormM" + str(mvjmp) + ".png"
	plt.close(f0)

	"""
	# Normalize - Lacunarity
	f0 = plt.figure()
	f0.clear()
	y1 = NormalizedMalhi(lac1)
	p1 = plt.plot(x, y1, label=lacLeg1 + " - M")
	y2 = NormalizedRoy(lac1)
	p1 = plt.plot(x, y2, label=lacLeg1 + " - R")
	if BothSides:
		y3 = NormalizedH(lac1, lacR1)
		p1 = plt.plot(x, y3, label=lacLeg1 + " - H")
	plt.legend()
	if BothSides:
		plt.title(thresh_title_part + ' Normalized (Roy, Malhi & Henebry)'  + mvjmp)
	else:
		plt.title(thresh_title_part + ' Normalized (Roy & Malhi)'  + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "Norm" + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINNormPlot"] = imagename + "_" + thresh_type + "Norm" + str(mvjmp) + ".png"
	plt.close(f0)
	"""

	return retDict

plotType = 1
def Calc_LacHistThresh(imagename, idata):
	# Calculate Lacunarity using Histogram threshold levels
	[lacData, lacDataR, lacLeg] = Calc_LacHistogramThreshold(idata, imagename)

	# Calculate Lacunarity using ostsu threshold
	#[lac1, lacR1, lacLeg1, thresh] = Lac_Otsu_Threshold(idata, imagename)

	x = range(1, len(lacData[0])+1)
	lx = [log(x1) for x1 in x]

	# Save Data to CSV file
	dataFilename = resFolder + imagename + ".csv"
	# SaveLacData(dataFilename, x, lacData + lacDataR + [lac1, lacR1], lacLeg + [lacLeg1])
	SaveLacData(dataFilename, x, lacData + lacDataR, lacLeg)

	if Graphs and plotType == 1:

		# Plot Hist Threshold Lacunarity
		f1 = plt.figure()
		f1.clear()
		for n in range(len(lacData)):
			plt.plot(x, lacData[n], label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90%')
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_HistThresh.png"
		plt.savefig(figFilename)
		plt.close(f1)

		#f1 = plt.figure()
		#f1.clear()
		#for n in range(len(lacDataR)):
		#	plt.plot(x, lacDataR[n], label=lacLeg[n])
		#plt.legend()
		#plt.title('Histogram Threshold 10% ~ 90% - Inverse Images')
		#plt.xlabel('window size')
		#plt.ylabel('lacunarity')
		#figFilename = resFolder + imagename + "_HistThresh_Inverse.png"
		#plt.savefig(figFilename)
		#plt.close(f1)

		# Otsu Lacunarity
		#f1 = plt.figure()
		#f1.clear()
		#plt.plot(x, lac1, label=lacLeg1)
		#plt.plot(x, lacR1, label=lacLeg1 + "_inv")
		#plt.legend()
		#plt.title('Otsu Threshold = ' + str(thresh))
		#plt.xlabel('window size')
		#plt.ylabel('lacunarity')
		#figFilename = resFolder + imagename + "_otsu.png"
		#plt.savefig(figFilename)
		#plt.close(f1)

		# Log Log - Plot Hist Threshold Lacunarity
		f1 = plt.figure()
		f1.clear()
		for n in range(len(lacData)):
			plt.loglog(x, lacData[n], label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90%')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThresh_loglog.png"
		plt.savefig(figFilename)
		plt.close(f1)

		#f1 = plt.figure()
		#f1.clear()
		#for n in range(len(lacDataR)):
		#	plt.loglog(x, lacDataR[n], label=lacLeg[n])
		#plt.legend()
		#plt.title('Histogram Threshold 10% ~ 90% - Inverse Images')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_HistThresh_Inverse_loglog.png"
		#plt.savefig(figFilename)
		#plt.close(f1)

		# Log Log - Otsu Lacunarity
		#f1 = plt.figure()
		#f1.clear()
		#plt.loglog(x, lac1, label=lacLeg1)
		#plt.loglog(x, lacR1, label=lacLeg1 + "_Inv")
		#plt.legend()
		#plt.title('Otsu Threshold = ' + str(thresh))
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_otsu_loglog.png"
		#plt.savefig(figFilename)
		#plt.close(f1)

		if BothSides:
			# Normalize - Plot Hist Threshold Lacunarity
			f1 = plt.figure()  # Norm - H
			f1.clear()
			for n in range(len(lacData)):
				y1 = NormalizedH(lacData[n], lacDataR[n])
				p1 = plt.loglog(x, y1, label=lacLeg[n])
			plt.legend()
			plt.title('Histogram Threshold 10% ~ 90% - Normalized (Henebry)')
			plt.xlabel('window size (log)')
			plt.ylabel('lacunarity (log)')
			figFilename = resFolder + imagename + "_HistThresh_normHloglog.png"
			plt.savefig(figFilename)
			plt.close(f1)

		f1 = plt.figure()  # Norm - M
		f1.clear()
		for n in range(len(lacData)):
			y1 = NormalizedMalhi(lacData[n])
			p1 = plt.plot(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Normalized (Malhi)')
		plt.xlabel('window size')
		plt.ylabel('lacunarity (log ratio)')
		figFilename = resFolder + imagename + "_HistThresh_normMlr.png"
		plt.savefig(figFilename)
		plt.close(f1)

		#f1 = plt.figure()  # Norm - M
		#f1.clear()
		#for n in range(len(lacData)):
		#	y1 = NormalizedMalhi(lacData[n])
		#	p1 = plt.loglog(x, y1, label=lacLeg[n])
		#plt.legend()
		#plt.title('Histogram Threshold 10% ~ 90% - Normalized (Malhi)')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log logratio)')
		#figFilename = resFolder + imagename + "_HistThresh_normMloglrlog.png"
		#plt.savefig(figFilename)
		#plt.close(f1)

		#f1 = plt.figure()  # Norm - M
		#f1.clear()
		#for n in range(len(lacDataR)):
		#	y1 = NormalizedMalhi(lacDataR[n])
		#	p1 = plt.plot(lx, y1, label=lacLeg[n])
		#plt.legend()
		#plt.title('Histogram Threshold 10% ~ 90% - Inverse Images - Normalized (Malhi)')
		#plt.xlabel('window size')
		#plt.ylabel('lacunarity (log ratio)')
		#figFilename = resFolder + imagename + "_HistThresh_Inverse_normMlr.png"
		#plt.savefig(figFilename)
		#plt.close(f1)

		f1 = plt.figure() # Norm - R
		f1.clear()
		for n in range(len(lacData)):
			y1 = NormalizedRoy(lacData[n])
			p1 = plt.plot(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Normalized (Roy)')
		plt.xlabel('window size')
		plt.ylabel('lacunarity ratio')
		figFilename = resFolder + imagename + "_HistThresh_normR.png"
		plt.savefig(figFilename)
		plt.close(f1)

		f1 = plt.figure() # Norm - R
		f1.clear()
		for n in range(len(lacData)):
			y1 = NormalizedRoy(lacData[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Normalized (Roy)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThresh_normRloglog.png"
		plt.savefig(figFilename)
		plt.close(f1)

		#f1 = plt.figure()  # Norm - R
		#f1.clear()
		#for n in range(len(lacDataR)):
		#	y1 = NormalizedRoy(lacDataR[n])
		#	p1 = plt.loglog(x, y1, label=lacLeg[n])
		#plt.legend()
		#plt.title('Histogram Threshold 10% ~ 90% - Inverse Images - Normalized (Roy)')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_HistThresh_Inverse_normRloglog.png"
		#plt.savefig(figFilename)
		#plt.close(f1)

		# Normalize - Otsu Lacunarity
		#f1 = plt.figure()
		#f1.clear()
		#y1 = NormalizedMalhi(lac1)
		#p1 = plt.plot(x, y1, label=lacLeg1 + " - M")
		#y2 = NormalizedRoy(lac1)
		#p1 = plt.plot(x, y2, label=lacLeg1 + " - R")
		#y3 = NormalizedH(lac1, lacR1)
		#p1 = plt.plot(x, y3, label=lacLeg1 + " - H")
		#plt.legend()
		#plt.title('Otsu Threshold = ' + str(thresh) + ' Normalized (Roy, Malhi & Henebry)')
		#plt.xlabel('window size')
		#plt.ylabel('lacunarity')
		#figFilename = resFolder + imagename + "_otsuNorm.png"
		#plt.savefig(figFilename)
		#plt.close(f1)

		#f1 = plt.figure()
		#f1.clear()
		#y1 = NormalizedMalhi(lac1)
		#p1 = plt.loglog(x, y1, label=lacLeg1 + " - M")
		#y2 = NormalizedRoy(lac1)
		#p1 = plt.loglog(x, y2, label=lacLeg1 + " - R")
		#if BothSides:
		#	y3 = NormalizedH(lac1, lacR1)
		#	p1 = plt.loglog(x, y3, label=lacLeg1 + " - H")
		#plt.legend(loc=3)
		#plt.title('Otsu Threshold = ' + str(thresh) + ' Normalized (Roy, Malhi & Henebry)')
		#plt.title('Histogram Threshold 10% ~ 90% - Normalized (Roy, Malhi & Henebry)')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (Norm)')
		#figFilename = resFolder + imagename + "_Normloglog.png"
		#plt.savefig(figFilename)

		# 20% and %80 and reverse 20%
		#f1 = plt.figure()
		#f1.clear()
		#for n in range(len(lacData)):
		#	if n == 1 or n == 7:
		#		p1 = plt.loglog(x, lacData[n], label=lacLeg[n])
		#	if n == 1:
		#		p1 = plt.loglog(x, lacDataR[n], label=lacLeg[n] + "-Inv")
		#plt.legend()
		#plt.title('Histogram Threshold 20%, 80% and Inverse of 20%')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_HistThreshloglog_2080.png"
		#plt.savefig(figFilename)
		#plt.close(f0)

		# 20% and %80 and reverse 20%  - norm Malhi
		#f1 = plt.figure()
		#f1.clear()
		#for n in range(len(lacData)):
		#	if n == 1 or n == 7:
		#		y1 = NormalizedMalhi(lacData[n])
		#		p1 = plt.loglog(x, y1, label=lacLeg[n])
		#	if n == 1:
		#		y1 = NormalizedMalhi(lacDataR[n])
		#		p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
		#plt.legend()
		#plt.title('Histogram Threshold 20%, 80% and Inverse of 20% - Normalized (Malhi)')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_HistThreshloglog_normM_2080.png"
		#plt.savefig(figFilename)
		#plt.close(f0)

		# 20% and %80 and reverse 20%  - norm Henebry
		#f1 = plt.figure()
		#f1.clear()
		#for n in range(len(lacData)):
		#	if n == 1 or n == 7:
		#		y1 = NormalizedH(lacData[n], lacDataR[n])
		#		p1 = plt.loglog(x, y1, label=lacLeg[n])
			#if n == 1:
			#	y1 = NormalizedH(lacDataR[n], lacData[n])
			#	p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
		#plt.legend()
		#plt.title('Histogram Threshold 20%, 80% - Normalized (Henebry)')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_HistThresh_normHloglog_2080.png"
		#plt.savefig(figFilename)
		#plt.close(f0)

		# 20% and %80 and reverse 20%  - norm Roy
		#f1 = plt.figure()
		#f1.clear()
		#for n in range(len(lacData)):
		#	if n == 1 or n == 7:
		#		y1 = NormalizedRoy(lacData[n])
		#		p1 = plt.loglog(x, y1, label=lacLeg[n])
		#	if n == 1:
		#		y1 = NormalizedRoy(lacDataR[n])
		#		p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
		#plt.legend()
		#plt.title('Histogray Threshold 20%, 80% and Inverse of 20% - Normalized (Roy)')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_HistThreshloglog_normR_2080.png"
		#plt.savefig(figFilename)
		#plt.close(f0)

		# %80 - normalized
		#f1 = plt.figure()
		#f1.clear()
		#y1 = NormalizedMalhi(lacData[7])
		#p1 = plt.plot(x, y1, label=lacLeg[7] + '-M')
		#y1 = NormalizedH(lacData[7], lacDataR[7])
		#p1 = plt.plot(x, y1, label=lacLeg[7] + '-H')
		#y1 = NormalizedRoy(lacData[n])
		#p1 = plt.plot(x, y1, label=lacLeg[7] + '-R')
		#plt.legend()
		#plt.title('Histogram Threshold 80% - Normalized')
		#plt.xlabel('window size')
		#plt.ylabel('lacunarity')
		#figFilename = resFolder + imagename + "_HistThresh_norm_80.png"
		#plt.savefig(figFilename)
		#plt.close(f0)

		# %80 - normalized  log, log
		#f1 = plt.figure()
		#f1.clear()
		#y1 = NormalizedMalhi(lacData[7])
		#p1 = plt.loglog(x, y1, label=lacLeg[7] + '-M')
		#y1 = NormalizedH(lacData[7], lacDataR[7])
		#p1 = plt.loglog(x, y1, label=lacLeg[7] + '-H')
		#y1 = NormalizedRoy(lacData[n])
		#p1 = plt.loglog(x, y1, label=lacLeg[7] + '-R')
		#plt.legend()
		#plt.title('Histogram Threshold 80% - Normalized (log,log)')
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_HistThreshloglog_norm_80.png"
		#plt.savefig(figFilename)
		#plt.close(f0)

		#plt.show()

	if Graphs and plotType == 2:
		# Log Log - Plot Hist Threshold Lacunarity
		for n in range(len(lacData)):
			f10 = plt.figure()
			f10.clear()
			y1 = NormalizedMalhi(lacData[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n])
			y1 = NormalizedMalhi(lacDataR[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
# 			y1 = NormalizedRoy(lacData[n])
# 			p1 = plt.loglog(x, y1, label=lacLeg[n])
# 			y1 = NormalizedRoy(lacDataR[n])
# 			p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
			plt.legend()
			plt.title('Histogram Threshold -' + lacLeg[n])
			plt.xlabel('window size (log)')
			plt.ylabel('lacunarity (log)')
			figFilename = resFolder + imagename + "_" + str(n) + "_HistThresh_loglog-N(M).png"
			plt.savefig(figFilename)
			plt.close(f10)

def Calc_ALV(imagename, idata):
	retDict = {}

	w = 2
	# Calculate ALV
	print "ALV moving"
	alv0 = GetALV(idata, imagename, w, 0)
	print "ALV jumping"
	alv1 = GetALV(idata, imagename, w, 1)

	x = range(1, len(alv1)+1)

	# Save Data to CSV file
	dataFilename = resFolder + imagename + "_ALV.csv"
	SaveALVDataInd(dataFilename, x, [alv0, alv1])
	retDict["alvCSV"] = imagename + "_ALV.csv"

	# ALV Moving
	f1 = plt.figure()
	f1.clear()
	plt.plot(x, alv0)
	plt.title('ALV moving w = ' + str(w) + ', ' + imagename)
	plt.xlabel('Aggr. window size')
	plt.ylabel('ALV')
	figFilename = resFolder + imagename + "_alv_Moving.png"
	plt.savefig(figFilename)
	retDict["lacBINPlot"] = imagename + "_alv_Moving.png"

	# ALV Jumping
	f2 = plt.figure()
	f2.clear()
	plt.plot(x, alv1)
	plt.title('ALV jumping w = ' + str(w) + ', ' + imagename)
	plt.xlabel('Aggr. window size')
	plt.ylabel('ALV')
	figFilename = resFolder + imagename + "_alv_Jumping.png"
	plt.savefig(figFilename)
	retDict["lacBINPlot"] = imagename + "_alv_Jumping.png"

	plt.close(f1);

	return retDict

def SplitImageData(idata, levels):
	"""
	Split image data to 4 quadrants
	"""
	(sx, sy) = idata.shape

	OutIm1 = zeros((sx,sy), float)
	OutIm2 = zeros((sx,sy), float)
	OutIm3 = zeros((sx,sy), float)
	OutIm4 = zeros((sx,sy), float)
	OutIm  = []

	chkLevel = 0
	#print idata
	for x in range(sx):
			for y in range(sy):
				chkLevel = levels[0]
				if idata[x,y] >= chkLevel:
					OutIm1[x,y] = chkLevel
					idata[x,y] -= chkLevel
				else:
					OutIm1[x,y] = idata[x,y]
					idata[x,y]  = 0
				chkLevel = levels[1] - levels[0]
				if idata[x,y] >= chkLevel:
					OutIm2[x,y] = chkLevel
					idata[x,y] -= chkLevel
				else:
					OutIm2[x,y] = idata[x,y]
					idata[x,y]  = 0
				chkLevel = levels[2] - levels[1]
				if idata[x,y] >= chkLevel:
					OutIm3[x,y] = chkLevel
					idata[x,y] -= chkLevel
				else:
					OutIm3[x,y] = idata[x,y]
					idata[x,y]  = 0
				chkLevel = levels[3] - levels[2]
				if idata[x,y] >= chkLevel:
					OutIm4[x,y] = chkLevel
					idata[x,y] -= chkLevel
				else:
					OutIm4[x,y] = idata[x,y]
					idata[x,y]  = 0

	OutIm.append(sum(OutIm1))
	OutIm.append(sum(OutIm2))
	OutIm.append(sum(OutIm3))
	OutIm.append(sum(OutIm4))

	return OutIm

# Not used
def NU_SlidingImgG(idata, w = 2):
	(sx, sy) = idata.shape
	maxVal	= max(np.ravel(idata))
	minVal	= min(np.ravel(idata))
	bins	= maxVal - minVal + 1
	lev1	= [x * bins/4 for x in range(1,5)]  # Different form the paper Q = 4 instead of 5
	print "levels = ", lev1
	print "sx = %d, sy = %d, max = %d, min = %d"  % (sx, sy, maxVal, minVal)

	if min(sx-w + 1,sy - w + 1) <= 0:
		return None
	else:
		Out1 = []
		for x in range(sx - w + 1):
			#print 'x = ', x
			for y in range(sy - w + 1):
				#print 'y = ', y
				dw = deepcopy(idata[x:x+w,y:y+w])
				imgs1 =  SplitImageData(dw, lev1)
				Out1.append(imgs1)
		outArray1 = np.array(Out1)
		outArray2 = outArray1 * outArray1 / sum(Out1)
		outArray3 = outArray1 * outArray2
		ML  = sum(outArray2)
		ML2 = sum(outArray3)
		Lac = (ML2 - ML*ML)/(ML*ML)
		print outArray1.transpose()
		print outArray2.transpose()
		print outArray3.transpose()
		print Lac
		return Lac

def Cal_GrayScaleLacunarity(imagename, idata):
	lacData = GetLacunarity(idata)

	x  = range(1, len(lacData)+1)
	#lx = [log(x1) for x1 in x]

	# Save Data to CSV file
	saveas = resFolder + 'gslac_' + imagename + ".csv"
	WriteToCSVFile(saveas, [lacData])

	# Plot
	# Lacunarity Grayscale
	f0 = plt.figure()
	f0.clear()
	plt.plot(x, lacData, label='lacunarity')
	plt.title("Grayscale Lacunarity")
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_gray.png"
	plt.savefig(figFilename)
	plt.close(f0)

	# Log Log - Lacunarity
	f0 = plt.figure()
	f0.clear()
	plt.loglog(x, lacData, label='lacunarity')
	plt.title("Grayscale Lacunarity - log/log")
	plt.xlabel('window size (log)')
	plt.ylabel('lacunarity (log)')
	figFilename = resFolder + imagename + "_gray_loglog.png"
	plt.savefig(figFilename)
	plt.close(f0)

	""" Not required for gray scale - 04/17/15
	# Normalize - Lacunarity
	f0 = plt.figure()
	f0.clear()
	y1 = NormalizedMalhi(lacData)
	plt.plot(x, y1, label='lacunarity')
	plt.title("Grayscale Lacunarity -  Normalized (Malhi)")
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_gray_NormM.png"
	plt.savefig(figFilename)
	plt.close(f0)

	# Normalize - Lacunarity
	f0 = plt.figure()
	f0.clear()
	y1 = NormalizedMalhi(lacData)
	plt.plot(lx, y1, label='lacunarity')
	plt.title("Grayscale Lacunarity -  Normalized (Malhi)")
	plt.xlabel('window size (log)')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_gray_NormM-log.png"
	plt.savefig(figFilename)
	plt.close(f0)
	"""

def maskImage(imagename, data):
	try:
		mData = getImageData(imgFolder, imagename + '-mask', '.png')
	except:
		print "Does not have a mask"
		return data

	#print imagename, data.shape, mData.shape
	#print np.min(mData), np.max(mData)

	xOff = mData.shape[0] - data.shape[0]
	yOff = mData.shape[1] - data.shape[1]
	if 	data.shape[0] <= mData.shape[0] and data.shape[0] <= mData.shape[0]:
		xOff = xOff/2
		yOff = yOff/2
		mData1 = np.zeros([data.shape[0], data.shape[1]], dtype=np.float)
		for k in range(data.shape[0]):
			for j in range(data.shape[1]):
				if mData[k+xOff][j+yOff] > 0:
					mData1[k][j] = data[k][j]
				else:
					mData1[k][j] = 0.0
	else:
		print "Mask not applied!"
		mData1 = data

	"""
	mData1 = np.zeros([mData.shape[0], mData.shape[1]], dtype=np.float)
	for k in range(mData.shape[0]):
		for j in range(mData.shape[1]):
			if mData[k][j] > 0:
				mData1[k][j] = 0.0
			else:
				mData1[k][j] = 1.0
	mData2  = np.array(mData1, dtype='B')
	Image.fromarray(mData2 * 255).save(resFolder + imagename + '_ms.png',"png")
	"""

	return mData1

def getWindowSizes(imagename, idata):
	dim1 = min(idata.shape)

	#[lac1, lacR1, lacLeg1, lacLegR, thresh] = Lac_Threshold_Image(idata, ThresholdTypes[1], imagename)

	winds = [3, 5, 10, 15, 20, 25, 30, 50] #[2, 3, 5, 32, 56, 72]
	#winds = range(5,dim1/2,5)
	#print winds

	return winds

def ProcessAnImage_Seg_UsingFixedWin(imagename, ext1):
	dt1 = datetime.datetime.now()

	# Load Image
	idata2 = getImageData(imgFolder, imagename, ext1)

	if histEq:
		idata1, cdf = histeq(idata2, imagename)
		idata  = np.array(idata1, dtype='B')
		Image.fromarray(idata).save(resFolder + imagename + "_histEq.tif", "TIFF")
	else:
		idata1 = idata2

	winds = getWindowSizes(imagename, idata1)

	Calc_LacSegment(imagename, idata, winds)

	#plotMul(retR,"T",retL,0)

	dt2 = datetime.datetime.now()
	print "Time Taken for (Seg_UsingFixedWin) Image:", imagename, " = ", dt2-dt1

def ProcessAnImage_Seg_UsingSemivariantData(imagename, ext1):
	dt1 = datetime.datetime.now()

	# Load Image
	idata2 = getImageData(imgFolder, imagename, ext1)

	if histEq:
		idata1, cdf = histeq(idata2, imagename)
		idata  = np.array(idata1, dtype='B')
		Image.fromarray(idata).save(resFolder + imagename + "_histEq.tif", "TIFF")
	else:
		idata1 = idata2

	idata1 = maskImage(imagename, idata1)
	[rng, mRng] = ProcessAnImage_SD_M_RANGE(imagename, idata1)
	idata2 = idata1
	print "rng..."
	Calc_LacSegSemiVar(imagename, idata1, rng, '-rng')
	print "mRng..."
	Calc_LacSegSemiVar(imagename, idata2, mRng, '-mrng')
	print "...Done"

	dt2 = datetime.datetime.now()
	print "Time Taken for (Seg_UsingSemivariantData) Image:", imagename, " = ", dt2-dt1

def ProcessAnImage_Lacunarity_GrayScale(imagename, ext1):
	#idata = np.array([[5,4,8,7,9],[12,12,11,8,12],[11,12,9,10,5],[1,2,5,3,11],[5,9,2,7,10]])
	#print idata
	#Cal_GrayScaleLacunarity(imagename, idata, [3])

	dt1 = datetime.datetime.now()

	# Load Image
	idata1 = getImageData(imgFolder, imagename, ext1)

	if histEq:
		idata, cdf = histeq(idata1, imagename)
	else:
		idata = idata1

	Cal_GrayScaleLacunarity(imagename, idata)

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1

def ProcessAnImage_Lacunarity_HistThresholded(imagename, ext1):
	testData = {}
	dt1 = datetime.datetime.now()

	testData["imagename"] = imagename + ext1

	# Load Image
	idata1 = getImageData(imgFolder, imagename, ext1)

	if histEq:
		idata, cdf = histeq(idata1, imagename)
	else:
		idata = idata1

	Calc_LacHistThresh(imagename, idata)

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1
	return testData

def ProcessAnImage_Lacunarity(imagename, ext1, mj = 0):
	testData = {}
	dt1 = datetime.datetime.now()

	testData["imagename"] = imagename + ext1

	# Load Image
	idata1 = getImageData(imgFolder, imagename, ext1)

	if histEq:
		idata, cdf = histeq(idata1, imagename)
	else:
		idata = idata1

	Calc_LacBINAllThresholds(imagename, idata)

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1
	return testData

def ProcessAnImage_Lacunarity_Norm(testImages):
	testData = {}
	dt1 = datetime.datetime.now()

	xMax = 10000

	lacData = []
	for tImg, ext1 in testImages:
		# Load Image
		idata1 = getImageData(imgFolder, tImg, ext1)
		if histEq:
			idata, cdf = histeq(idata1, tImg)
		else:
			idata = idata1
		[lac1, lacR1, lacLeg1, lacLegR, thresh] = Lac_Threshold_Image(idata, "adaptive", tImg)
		xMax = min(xMax, len(lac1))
		lacData.append([lac1, tImg])

	if SaveData:
		# Save Data to CSV file
		saveas = resFolder + 'lac_norm.csv'
		WriteToCSVFile(saveas, [lacData])

	x  = range(1, xMax+1)
	#lx = [log(x1) for x1 in x]

	#print lacData

	if Graphs:
		# Normalize - Lacunarity
		f0 = plt.figure()
		f0.clear()
		for dset in lacData:
			#print dset[1]
			y1 = NormalizedMalhi(dset[0])
			plt.plot(x, y1, label=dset[1])
		plt.legend()
		plt.title(" Normalized (Malhi)")
		plt.xlabel('window size')
		plt.ylabel('lacunarity - Normalized')
		figFilename = resFolder + "L_NormM.png"
		plt.savefig(figFilename)
		plt.close(f0)

		# Normalize - Lacunarity
		f0 = plt.figure()
		f0.clear()
		for dset in lacData:
			y1 = NormalizedRoy(dset[0])
			plt.plot(x, y1, label=dset[1])
		plt.legend()
		plt.title(" Normalized (Roy)")
		plt.xlabel('window size')
		plt.ylabel('lacunarity - Normalized')
		figFilename = resFolder + "L_NormR.png"
		plt.savefig(figFilename)
		plt.close(f0)

	dt2 = datetime.datetime.now()
	print "Time Taken for Lacunarity - Norm = ", dt2-dt1
	return testData

def ProcessAnImage_SD_M_RANGE_Org(imagename, idata):
	dt1 = datetime.datetime.now()

	px = 200
	py = 40
	w  = 20
	h  = 25

	idataCrop = np.array(idata)[py:py+h+1,px:px+w+1]
	resData  = np.array(idataCrop,  dtype='B')
	Image.fromarray(resData).save(resFolder + imagename+"_subimg.tif", "TIFF")


	range1 = 0
	min_range = 99999

	(Xdim, Ydim) = idata.shape

	resDataRng  = []
	resDataMRng = []

	for col in range(Xdim):
		resRowRng  = []
		resRowMRng = []
		for row in range(Ydim):
			range1 = 0
			min_range = 99999
			#print "col,row", col, row
			# 0 degrees
			if (row + nLags) < Ydim:
				maxsv = 0
				for x in range(1, nLags):
					sum  = 0.0
					mult = 0.5/nLags
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
					mult = 0.5/nLags
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
					mult = 0.5/nLags
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
					mult = 0.5/nLags
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
					mult = 0.5/nLags
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
					mult = 0.5/nLags
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
					mult = 0.5/nLags
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
					mult = 0.5/nLags
					for y in range(nLags - x + 1):
						diff = idata[col + y, row + y] - idata[col + y + x, row + y + x]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x
				if range1 < min_range:
					min_range = range1

			if range1 > 0 and range1 % 2 == 0:
				range1 -= 1
			if range1 > (nLags - 1):
				range1 = nLags - 1
			if min_range > 0 and min_range % 2 == 0:
				min_range -= 1
			if min_range > (nLags - 1):
				min_range = nLags - 1

			resRowRng.append(range1)
			resRowMRng.append(min_range)
		resDataRng.append(resRowRng)
		resDataMRng.append(resRowMRng)

	print np.min(resDataRng), np.max(resDataRng),  np.average(resDataRng),  np.std(resDataRng)
	print np.min(resDataMRng),np.max(resDataMRng), np.average(resDataMRng), np.std(resDataMRng)

	resData  = np.array(np.array(resDataRng)  * 255.0/np.max(resDataRng),  dtype='B')
	Image.fromarray(resData).save(resFolder + imagename+"svtest.tif", "TIFF")

	resDataCrop = resData[py:py+h+1,px:px+w+1]
	Image.fromarray(resDataCrop).save(resFolder + imagename+"_svtest_subimg.tif", "TIFF")

	resDataM = np.array(np.array(resDataMRng) * 255.0/np.max(resDataMRng), dtype='B')
	Image.fromarray(resDataM).save(resFolder + imagename+"svtestM.tif", "TIFF")

	print np.array(resDataRng).shape

	resDataRngCrop = np.array(resDataRng)[py:py+h+1, px:px+w+1]
	with open(resFolder + imagename + "_RngCrop.csv", 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(resDataRngCrop)


	with open(resFolder + imagename + "_Rng.csv", 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(resDataRng)
	with open(resFolder + imagename + "_MRng.csv", 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(resDataMRng)

	dt2 = datetime.datetime.now()
	print "Time Taken for (SD_M_RANGE) Image:", imagename, " =", dt2-dt1

	return [resDataRng, resDataMRng]

def ProcessAnImage_SD_M_RANGE(imagename, idata):
	dt1 = datetime.datetime.now()

	(px, py, w, h) = imageCrop[imagename]
	print (px, py, w, h)

	idataCrop = np.array(idata)[py:py+h,px:px+w]
	resData  = np.array(idataCrop,  dtype='B')
	Image.fromarray(resData).save(resFolder + imagename+"_subimg.tif", "TIFF")

	nLags = 10

	#range1    = 0
	#min_range = 99999

	(Xdim, Ydim) = idata.shape

	resDataRng  = []
	resDataMRng = []

	for col in range(Xdim):
		resRowRng  = []
		resRowMRng = []
		for row in range(Ydim):

			range1 = 0
			min_range = 99999
			#print "col,row", col, row
			# 0 degrees
			if (row + 2*nLags - 1) < Ydim:
				maxsv = 0
				for x in range(nLags-1):
					sum  = 0.0
					mult = 0.5/(nLags-x-1)
					for y in range(nLags - x):
						diff = idata[col, row + 2*y] - idata[col, row + 2*y + x + 1]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x + 1
				if range1 < min_range:
					min_range = range1

			# 45 degrees
			if (col - 2*nLags + 1) > 0 and (row + 2*nLags - 1) < Ydim:
				maxsv = 0
				for x in range(nLags-1):
					sum = 0.0
					mult = 0.5/(nLags-x-1)
					for y in range(nLags - x):
						diff = idata[col - 2*y, row + 2*y] - idata[col - 2*y - x - 1, row + 2*y + x + 1]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x + 1
				if range1 < min_range:
					min_range = range1

			# 90 degrees
			if (col - 2*nLags + 1) > 0:
				maxsv = 0
				for x in range(nLags-1):
					sum = 0.0
					mult = 0.5/(nLags-x-1)
					for y in range(nLags - x):
						diff = idata[col - 2*y, row] - idata[col - 2*y - x - 1, row]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x + 1
				if range1 < min_range:
					min_range = range1

			# 135 degrees
			if (col - 2*nLags + 1) > 0 and (row - 2*nLags + 1) > 0:
				maxsv = 0
				for x in range(nLags-1):
					sum = 0.0
					mult = 0.5/(nLags-x-1)
					for y in range(nLags - x):
						diff = idata[col - 2*y, row - 2*y] - idata[col - 2*y - x - 1, row - 2*y - x - 1]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x + 1
				if range1 < min_range:
					min_range = range1

			# 180 degrees
			if (row - 2*nLags + 1) > 0:
				maxsv = 0
				for x in range(nLags-1):
					sum = 0.0
					mult = 0.5/(nLags-x-1)
					for y in range(nLags - x):
						diff = idata[col, row - 2*y] - idata[col, row - 2*y - x - 1]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x + 1
				if range1 < min_range:
					min_range = range1

			# 225 degrees
			if (col + 2*nLags - 1) < Xdim and (row - 2*nLags + 1) > 0:
				maxsv = 0
				for x in range(nLags-1):
					sum = 0.0
					mult = 0.5/(nLags-x-1)
					for y in range(nLags - x):
						diff = idata[col + 2*y, row - 2*y] - idata[col + 2*y + x + 1, row - 2*y - x - 1]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x + 1
				if range1 < min_range:
					min_range = range1

			# 270 degrees
			if (col + 2 * nLags - 1) < Xdim:
				maxsv = 0
 				for x in range(nLags-1):
					sum = 0.0
					mult = 0.5/(nLags-x-1)
					for y in range(nLags - x):
						diff = idata[col + 2*y, row] - idata[col + 2*y + x + 1, row]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x + 1
				if range1 < min_range:
					min_range = range1

			# 315 degrees
			if (row + 2 * nLags -1) < Ydim and (col + 2 * nLags - 1) < Xdim:
				maxsv = 0
 				for x in range(nLags-1):
					sum = 0.0
					mult = 0.5/(nLags-x-1)
					for y in range(nLags - x):
						diff = idata[col + 2 * y, row + 2 * y] - idata[col + 2 * y + x + 1, row + 2 * y + x + 1]
						sum += 1.0 * diff * diff
					semi = mult * sum;
					if semi > maxsv:
						maxsv = semi
						range1 = x + 1
				if range1 < min_range:
					min_range = range1

			if range1 > 0 and range1 % 2 == 0:
				range1 -= 1
			if range1 > (nLags - 1):
				range1 = nLags - 1
			if min_range > 0 and min_range % 2 == 0:
				min_range -= 1
			if min_range > (nLags - 1):
				min_range = nLags - 1

			resRowRng.append(range1)
			resRowMRng.append(min_range)
		resDataRng.append(resRowRng)
		resDataMRng.append(resRowMRng)

	print "min","max","average","std"
	print np.min(resDataRng), np.max(resDataRng),  np.average(resDataRng),  np.std(resDataRng)
	print np.min(resDataMRng),np.max(resDataMRng), np.average(resDataMRng), np.std(resDataMRng)

	resData  = np.array(np.array(resDataRng)  * 255.0/np.max(resDataRng),  dtype='B')
	Image.fromarray(resData).save(resFolder + imagename+"svtest.tif", "TIFF")

	resDataCrop = resData[py:py+h,px:px+w]
	Image.fromarray(resDataCrop).save(resFolder + imagename+"_svtest_subimg.tif", "TIFF")

	resDataM = np.array(np.array(resDataMRng) * 255.0/np.max(resDataMRng), dtype='B')
	Image.fromarray(resDataM).save(resFolder + imagename+"svtestM.tif", "TIFF")

	print np.array(resDataRng).shape

	resDataRngCrop = np.array(resDataRng)[py:py+h, px:px+w]
	with open(resFolder + imagename + "_RngCrop.csv", 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(resDataRngCrop)
	resDatamRngCrop = np.array(resDataMRng)[py:py+h, px:px+w]

	with open(resFolder + imagename + "_MRngCrop.csv", 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(resDatamRngCrop)


	with open(resFolder + imagename + "_Rng.csv", 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(resDataRng)
	with open(resFolder + imagename + "_MRng.csv", 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(resDataMRng)

	dt2 = datetime.datetime.now()
	print "Time Taken for (SD_M_RANGE) Image:", imagename, " =", dt2-dt1

	return [resDataRng, resDataMRng]

def ProcessAnImage_ALV(imagename, ext1):
	dt1 = datetime.datetime.now()

	# Load Image
	idata1 = getImageData(imgFolder, imagename, ext1)

	if histEq:
		idata, cdf = histeq(idata1, imagename)
	else:
		idata = idata1

	Calc_ALV(imagename, idata)

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1


if __name__ == '__main__':
	# Select the function below with the number
	funcSet = {"Lacunarity":0,"ALV":1,"Parameterization":2,"Seg_Fix_Win":3}
	#print funcSet
	set = 9

	# Lacunarity
	if set == 0:
		resFolder = resFolder1 + 'respy12/'
		imgFolder  = imgFolder1 + 'images2/'

		#imgFolder = imgFolder1 + 'images1/'
		#imgFolder = imgFolder1 + 'imagesCreated/'
		#imgFolder  = imgFolder1 + 'MSRC_ObjCategImageDatabase_v1/Sel1/'

		#testImages = [('honeycomb2', '.tif')]
		#testImages = [('honeycombb', '.tif')]
		#testImages = [('RAC275', '.tif'),('RAC325', '.tif')]
		#testImages = [('Sq2_A','.tif')] #, ('Sq2_B','.tif'), ('Sq2_C','.tif'), ('Sq2_D','.tif')]
		#testImages = [('peru480-blogSpan','.jpg')]
		#testImages = [('cows-gray','.bmp'),('cows-gray-smooth','.bmp')]

		testImages = [('MRI slice','.jpg')]
		for tImg, ext in testImages:
			ret = ProcessAnImage_Lacunarity(tImg, ext) #,1)

	# ALV
	if set == 1:
		imgFolder = imgFolder1 + 'images1/'
		resFolder = resFolder1 + 'respy10/'
		#testImages = ['honeycomb2'] #, 'Sq2_A', 'Sq2_B', 'Sq2_C', 'Sq2_D']
		#testImages = [('RAC275', '.tif'),('RAC325', '.tif')]
		testImages = ['RAC275', 'RAC325']
		#testImages = ['ALV_Samp_A', 'ALV_Samp_B', 'ALV_Samp_C', 'ALV_Samp_D', 'ALV_Samp_E', 'ALV_Samp_F']
		for tImg in testImages:
			ProcessAnImage_ALV(tImg, '.tif')

	# Parameterization
	if set == 2:
		resFolder  = resFolder1 + 'respy03_para/'
		#imgFolder = imgFolder1 + 'imagesBrodatzTexture/textures/'
		imgFolder  = imgFolder1 + 'images2/'
		testImages = ['lena256b-org']
		#testImages = ['1.2.03', '1.2.12']
		#testImages = ['1.1.01', '1.1.07', 'texmos3.s512', '1.5.02']
		#testImages = ['t-1.1.01', 't-1.1.07', 't-texmos3.s512', 't-1.5.02']
		for tImg in testImages:
			ProcessAnImage_Lacunarity(tImg, '.bmp')

	# Segmentation using fixed window size for lacunarity
	if set == 3:
		resFolder  = resFolder1 + 'respy17/'
		imgFolder  = imgFolder1 + 'imageSeg/'
		testImages = [('MRI-slice','.jpg'),('cows-gray','.jpg')]
		#testImages = [('MRI-slice','.jpg')]
		#testImages = [('MRI-slice-org','.jpg')]
		#testImages = [('cows-gray','.jpg')]
		for tImg, ext in testImages:
			#ProcessAnImage_ALV(tImg, ext)
			ret = ProcessAnImage_Seg_UsingFixedWin(tImg, ext)

		#imgFolder = imgFolder1 + 'MSRC_ObjCategImageDatabase_v1/Sel1/'
		#testImages = ['1_14_s', '4_7_s']
		#testImages = [('cows-gray','.bmp'),('cows-gray-smooth','.bmp')]
		#imgFolder = '../images1/'
		#testImages = ['honeycomb2A'] # 'honeycomb2'

		# Save results attributes
		#strResultsJson = json.dumps(testList)
		#with open(resFolder + "images-exp1.json", 'w') as file1:
		#	file1.write(strResultsJson)

	# Lacunarity - Grayscale
	if set == 4:
		imgFolder = imgFolder1 + 'images1/'
		resFolder = resFolder1 + 'respy09_para/'
		#testImages = ['honeycomb2A', 'honeycombb', 'RAC275', 'RAC325']
		#testImages = ['Tropical-forest-Brazil-photo-by-Edward-Mitchard-small', 'forest-20130117-full', 'back_forest', 'glight2_pho_2014', 'GoogleEarthEng_main_1203', 'peru480-blogSpan']
		testImages = [('honeycomb2','.tif')]
		for tImg, ext in testImages:
			ProcessAnImage_Lacunarity_GrayScale(tImg, ext)

	# Thresholded Lacunarity
	if set == 5:
		imgFolder = imgFolder1 + 'images1/'
		resFolder = resFolder1 + 'respy09_para/'
		testImages = [('honeycombb', '.tif')]
		for tImg, ext in testImages:
			ProcessAnImage_Lacunarity_HistThresholded(tImg, ext)

	# Space Data - Lacunarity
	if set == 6:
		imgFolder = imgFolder1 + 'imagesSpaceData/'
		resFolder = resFolder1 + 'respy10/'
		testImages = [('p1-Map_pac2 cropmid','.tif'), ('p2-Map_pac2f cropmid','.tif')]
		for tImg, ext in testImages:
			ProcessAnImage_Lacunarity(tImg, ext)

	# Compare Normalizing methods
	if set == 7:
		resFolder = resFolder1 + 'respy05_para/'
		imgFolder = imgFolder1 + 'imagesBrodatzTexture/textures/'
		testImages = [('t-1.2.12', '.tif'), ('t-1.1.01', '.tif')]
		#imgFolder = imgFolder1 + 'images1/'
		#testImages = [('honeycomb2', '.tif'), ('honeycombb', '.tif')]
		ProcessAnImage_Lacunarity_Norm(testImages)

	# Histogram Equalized
	if set == 8:
		imgFolder = imgFolder1 + 'images1/'
		idata1 = getImageData(imgFolder, 'honeycomb2', '.tif')
		histeq(idata1, 'honeycomb2')

	# Semivariogram - modified method to segment images
	if set == 9:
		resFolder  = resFolder1 + 'respy16/'
		imgFolder  = imgFolder1 + 'imageSeg/'
		testImages = [('MRI-slice','.jpg'),('cows-gray','.jpg')]
		#testImages = [('cows-gray','.jpg')]
		#testImages = [('MRI-slice','.jpg')]
		for tImg, ext in testImages:
			ret = ProcessAnImage_Seg_UsingSemivariantData(tImg, ext)
