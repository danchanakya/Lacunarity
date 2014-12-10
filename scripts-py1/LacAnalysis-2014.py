import Image
import csv
import numpy as np
import datetime

import PIL.ImageOps

from skimage.filter import threshold_otsu, threshold_adaptive, threshold_isodata

import json

import colorsys
import readColorList as colorL

from scipy.misc import imread,imsave

from matplotlib import pyplot as plt
from copy import deepcopy

from ALVLACSubRoutines  import *
from Output 			import *

imgFolder = '../images1/'

resFolder = '../results/respy6/'

testList = []

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
	h1 = h/6.0
	c  = (1 - (2*l-1)) * s
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
	noH = 9
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
	funcused    = std # var ?

	(sx, sy) = datarray.shape
	if clipType == 0:
		maxA = max(sx, sy) // w
	else:
		maxA = max(sx, sy) #-1

	if mjmp == 1: # Jumping
		d1 = JumpingImg(datarray, funcused, 2, clipType)
	else:
		d1 = SlidingImg(datarray, funcused)

	ALVData.append(average(d1))

	for x in [x+1 for x in range(maxA)][1:]:
		# print x
		d2 = JumpingImg(datarray, average, x, clipType)
		if SaveAggImgs == 1:
			d3   = array(d2, dtype='B')
			agIm = Image.fromarray(d3)
			fileName = resFolder + imagename + '_AggData__' + str(x) + '.tif'
			agIm.save(fileName, "TIFF")

		if mjmp == 1:
			d3 = JumpingImg(d2, funcused, 2, clipType)
		else:
			d3 = SlidingImg(d2, funcused)

		ALVData.append(average(d3))

	return ALVData

def GetLacunarity(data, mj = 0, clipType = 0):
	(sx, sy) = data.shape
	minxy = min(sx, sy);
	lac = []

	#print "Window = 1"
	dataAve = np.average(data)
	dataStd = np.std(data)
	lac1  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
	lac.append(lac1)

	# find lac from count = 2 to msize
	for count in range(2, minxy+1):
		#print "Window = ", count
		if mj == 1: # Jumping
			CS = JumpingImg(data, sum, count, clipType)
		else:
			CS = SlidingImg(data, sum, count)
		dataAve = np.average(CS)
		dataStd = np.std(CS)
		lac1  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
		lac.append(lac1)

	return lac

def GetLacunarityImg(data, wd, imagename, color, slice = 9):
	dict1 = {}

	(sx, sy) = data.shape
	minxy = min(sy, sx);
	try:
		wd = int(wd)
	except:
		s  = int(wd[:-1])
		wd = min(sx/s, sy/s)

	df = wd / 2
	lacData  = np.zeros([sx-df*2, sy-df*2])
	lacImage = np.zeros([sx-df*2, sy-df*2,3], dtype=np.uint8)

	for y in range(sy-df*2):
		for x in range(sx-df*2):
			selData = data[x:x+wd, y:y+wd]
			dataAve = np.average(selData)
			dataStd = np.std(selData)
			if dataAve > 0:
				v1  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
			else:
				v1  = 1
			lacData [x, y] = v1
			if not color:
				lacImage[x, y] = [v1*255, v1*255, v1*255]

	lacDataAll = []
	for x in lacData:
		for y in x:
			lacDataAll.append(y)

	# Get Histogram for all possible values
	(dataHst, nHst) = np.histogram(lacDataAll, slice)

	dict1["window"] = wd

	if color:
		#dict1["color"] = "true"
		fileName = resFolder + imagename + '_resBN_Color_' + str(wd) + '.tif'
		dict1["resImage"] = imagename + '_resBN_Color_' + str(wd) + '.tif'
		legendFileName = resFolder + imagename + '_legBN_sliced-' +  str(slice) + '_' + str(wd) + '.tif'
		clr = colorL.colorLUT(legendFileName, nHst)
		dict1["legImage"] = imagename + '_legBN_sliced-' +  str(slice) + '_' + str(wd) + '.tif'
	else:
		#dict1["color"] = "false"
		fileName = resFolder + imagename + '_resBN_' + str(wd) + '.tif'
		dict1["resImage"] = imagename + '_resBN_' + str(wd) + '.tif'

	minD = np.min(lacData)
	maxD = np.max(lacData)
	print "Lacunarity range: from ", minD, " to ", maxD, " for window = ", wd

	if color:
		for y in range(sy-df*2):
			for x in range(sx-df*2):
				pos = 1
				while(lacData[x, y] > nHst[pos]):
					pos += 1
				lacImage[x, y] = clr.items()[pos-1][1]

	if len(imagename) > 0:
		Image.fromarray(lacImage).save(fileName, "TIFF")
		#imsave("Result-" + str(op) + "-" + str(wd) + ".jpg", lacImage)

	return [lacImage, dict1]

def CalcLacunarity(data, wd):
	(sx, sy) = data.shape
	minxy = min(sx, sy);

	if minxy > wd:
		CS = SlidingImg(data, sum, wd)
		dataAve = np.average(CS)
		dataStd = np.std(CS)
		lac  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
	else:
		lac = -1

	return lac

def GetSegLacunarityGrayImage(data, wd, imagename, slice = 9):
	resDict = {}
	(sx, sy) = data.shape
	minxy = min(sy, sx);
	try:	# windows size
		wd = int(wd)
	except: # window ratio
		s  = int(wd[:-1])
		wd = min(sx/s, sy/s)

	df = wd / 2
	lacData = np.zeros([sx-df*2, sy-df*2], dtype=np.float)
	#lacData1 = np.zeros([sx-df*2, sy-df*2], dtype=np.float)

	for y in range(sy-df*2):
		for x in range(sx-df*2):
			selData = data[x:x+wd, y:y+wd]
			dataAve = np.average(selData)
			dataStd = np.std(selData)
			if dataAve > 0:
				lacData [x, y]  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
			else:
				lacData [x, y]  = 1.0

	fileName = resFolder + imagename + '_resGS_sliced-' +  str(slice) + '_wnd-' + str(wd) + '.tif'
	resDict["resImage"] = imagename + '_resGS_sliced-' +  str(slice) + '_wnd-' + str(wd) + '.tif'
	legendFileName = resFolder + imagename + '_legGS_sliced-' +  str(slice) + '_wnd-' + str(wd) + '.tif'
	resDict["legImage"] = imagename + '_legGS_sliced-' +  str(slice) + '_wnd-' + str(wd) + '.tif'

	resDict["window"] = wd

	minD = np.min(lacData)
	maxD = np.max(lacData)
	print "Lacunarity range: from ", minD, " to ", maxD, " for window = ", wd

	lacDataAll = []
	for x in lacData:
		for y in x:
			lacDataAll.append(y)

	# Get Histogram for all possible values
	(dataHst, nHst) = np.histogram(lacDataAll, slice)

	f = plt.figure()
	f.clear()
	r1 = plt.hist(lacDataAll, slice)
	plt.title('Histogram ' + imagename + ' GS sliced-' +  str(slice) + ' window-' + str(wd))
	plt.xlabel('bin')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_resGS_sliced-" +  str(slice) + '_wnd-' + str(wd) + ".png"
	resDict["hstImage"] = imagename + "_resGS_sliced-" +  str(slice) + '_wnd-' + str(wd) + ".png"
	plt.savefig(figFilename)

	#print nHst
	clr = colorL.colorLUT(legendFileName, nHst, slice)

	lacImage = np.zeros([sx-df*2, sy-df*2,3], dtype=np.uint8)
	for y in range(sy-df*2):
		for x in range(sx-df*2):
			pos = 1
			while(lacData[x, y] > nHst[pos]):
				pos += 1
			lacImage[x, y] = clr.items()[pos-1][1]


	if len(imagename) > 0:
		Image.fromarray(lacImage).save(fileName, "TIFF")

	return [lacImage, resDict]

def NormalizedRoy(data):
	lacNorm = []
	Lmin = min(data)
	Lmax = max(data)
	for dt in data:
		lacNorm.append((dt-Lmin)/(Lmax-Lmin))
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

def Lac_ISODATA_Threshold(idata, savePics = '', mj = 0):
	dt1 = datetime.datetime.now()

	block_size = 40
	threshold  = 0
	#thresh_type = "otsu"
	#threshold = OtsuThreshold(idata)
	#thresh_type = "isodata"
	threshold = threshold_isodata(idata)
	thresh_type = "fixed"
	threshold = 128
	#print threshold
	#thresh_type = "adaptive"
	#trData1 = threshold_adaptive(idata, block_size, offset=10)

	trData1  = idata <= threshold
	trData2  = np.array(trData1, dtype='B')

	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_' + thresh_type + '_' + str(threshold) + '.tif',"TIFF")
	lac1     = GetLacunarity(trData2, mj)
	lacLeg   = thresh_type + ' = ' + str(threshold)

	trDataR1  = idata > threshold
	#trDataR10  = PIL.ImageOps.mirror(Image.fromarray(idata))
	#trDataR11  = inar = np.asarray(trDataR10)
	#trDataR1 = threshold_adaptive(trDataR11, block_size, offset=10)

	trDataR2  = np.array(trDataR1, dtype='B')

	if len(savePics) > 0:
		Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_' + thresh_type + '_' + str(threshold) + 'R.tif',"TIFF")
	lacR1     = GetLacunarity(trDataR2, mj)
	lacLegR   = 'R ' + thresh_type + ' = ' + str(threshold)

	dt2 = datetime.datetime.now()
	print 'Lac_' + thresh_type + '_Threshold      = ', dt2-dt1

	return [lac1, lacR1, lacLeg, threshold]

def Lac_Otsu_Threshold(idata, savePics = '', mj = 0):
	dt1 = datetime.datetime.now()

	threshold = OtsuThreshold(idata)
	trData1  = idata <= threshold
	trData2  = np.array(trData1, dtype='B')
	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_otsu_' + str(threshold) + '.tif',"TIFF")
	lac1     = GetLacunarity(trData2, mj)
	lacLeg   = 'otsu = ' + str(threshold)

	trDataR1  = idata > threshold
	trDataR2  = np.array(trDataR1, dtype='B')
	if len(savePics) > 0:
		Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_otsu_' + str(threshold) + 'R.tif',"TIFF")
	lacR1     = GetLacunarity(trDataR2, mj)
	lacLegR   = 'R otsu = ' + str(threshold)

	dt2 = datetime.datetime.now()
	print "Lac_Otsu_Threshold      = ", dt2-dt1

	return [lac1, lacR1, lacLeg, threshold]

def Lac_Histogram_Threshold(idata, savePics = ''):
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
			lacLeg.append(str(chkPnt) + "(" + str(nHst[h]) + ")")
			trData1  = idata <= nHst[h]
			trData2  = np.array(trData1, dtype='B')
			if len(savePics) > 0:
				Image.fromarray(trData2 * 255).save(resFolder + savePics + '_' + str(chkPnt) + "-" + str(nHst[h]) + '.tif',"TIFF")
			lac1     = GetLacunarity(trData2)
			lacData.append(lac1)
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
	print "Lac_Histogram_Threshold = ", dt2-dt1

	return [lacData, lacDataR, lacLeg]

def Calc_LacImageGrayScale(imagename, idata, wd):
	retDictR = []
	dt1 = datetime.datetime.now()

	fileName = resFolder + imagename + '.tif'

	if len(imagename) > 0:
		Image.fromarray(idata).save(fileName, "TIFF")

	for w1 in wd:
		[ret, retDict] = GetSegLacunarityGrayImage(idata, w1, imagename, 18)
		#retDictR  = dict( retDictR, **retDict )
		retDictR.append(retDict)

	dt2 = datetime.datetime.now()
	print "Time Taken for Calc_LacImageGrayScale   = ", dt2-dt1

	return retDictR

def Calc_LacImageOtsu(imagename, idata, wd, color = False):
	retDict = {}
	retLst  = []
	dt1 = datetime.datetime.now()

	threshold = OtsuThreshold(idata)
	trData1  = idata <= threshold
	trData2  = np.array(trData1, dtype='B')

	retDict["otsu_threshold"] = threshold

	fileName = resFolder + imagename + '_Imgotsu_' + str(threshold) + '.tif'
	retDict["binImage"] = imagename + '_Imgotsu_' + str(threshold) + '.tif'

	if len(imagename) > 0:
		Image.fromarray(trData2 * 255).save(fileName, "TIFF")

	for w1 in wd:
		trData3  = GetLacunarityImg(trData2, w1, imagename, color)
		#retDict  = dict( retDict, **trData3[1] )
		retLst.append(trData3[1])

	retDict["res"] = retLst

	#if len(savePics) > 0:
	#	Image.fromarray(trDataR3 * 255).save(resFolder + imagename +'_Imgotsu_' + str(threshold) + 'Res.tif',"TIFF")

	dt2 = datetime.datetime.now()
	print "Time Taken for Lac_Otsu_Threshold      =", dt2-dt1

	#return [trData3, threshold]
	return retDict

def Calc_LacBIN(imagename, idata, mj = 0):
	if mj == 0:
		mvjmp = " Moving Window"
	else:
		mvjmp = " Jumping Window"
	retDict = {}

	# Calculate Lacunarity using ostsu threshold
	[lac1, lacR1, lacLeg1, thresh] = Lac_Otsu_Threshold(idata, imagename, mj)
	thresh_type = "otsu"
	thresh_title_part = 'Otsu Threshold = ' + str(thresh)

	#[lac1, lacR1, lacLeg1, thresh] = Lac_ISODATA_Threshold(idata, imagename, mj)
	#thresh_type = "isodata"
	#thresh_title_part = 'isodata Threshold ' + str(thresh)

	#thresh_type = "fixed"
	#thresh_title_part = 'fixed Threshold ' + str(thresh)

	#thresh_type = "adaptive"
	#thresh_title_part = 'Adaptive Threshold'

	x = range(1, len(lac1)+1)
	retDict[thresh_type + "_threshold"] = thresh

	# Save Data to CSV file
	dataFilename = resFolder + imagename + "_BIN_Lac_threshold-" + str(thresh) + str(mvjmp) + ".csv"
	SaveLacDataInd(dataFilename, x, [lac1, lacR1], lacLeg1)
	#SaveLacData(dataFilename, x, [lac1, lacR1], [lacLeg1])
	retDict["lacBINCSV"] = imagename + "_BIN_Lac_threshold-" + str(thresh) + str(mvjmp) + ".csv"

	# Otsu Lacunarity
	f1 = plt.figure()
	f1.clear()
	plt.plot(x, lac1, label=lacLeg1)
	plt.legend()
	plt.title(thresh_title_part + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "_threshold1-" + str(thresh) + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINPlot"] = imagename + "_" + thresh_type + "_threshold1-" + str(thresh) + str(mvjmp) + ".png"

	# Otsu Lacunarity
	f1 = plt.figure()
	f1.clear()
	plt.plot(x, lac1, label=lacLeg1)
	plt.plot(x, lacR1, label=lacLeg1 + "_inv")
	plt.legend()
	plt.title(thresh_title_part + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "_threshold-" + str(thresh) + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINPlot"] = imagename + "_" + thresh_type + "_threshold-" + str(thresh) + str(mvjmp) + ".png"

	# Log Log - Otsu Lacunarity
	f2 = plt.figure()
	f2.clear()
	plt.loglog(x, lac1, label=lacLeg1)
	plt.loglog(x, lacR1, label=lacLeg1 + "_Inv")
	plt.legend()
	plt.title(thresh_title_part + mvjmp)
	plt.xlabel('window size (log)')
	plt.ylabel('lacunarity (log)')
	figFilename = resFolder + imagename + "_" + thresh_type + "_loglog" + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINLogPlot"] = imagename + "_" + thresh_type + "_loglog" + str(mvjmp) + ".png"

	# Normalize - Otsu Lacunarity
	f3 = plt.figure()
	f3.clear()
	y3 = NormalizedH(lac1, lacR1)
	p1 = plt.plot(x, y3, label=lacLeg1 + " - H")
	plt.legend()
	plt.title(thresh_title_part + ' Normalized (Henebry)'  + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "NormH" + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINNormPlot"] = imagename + "_" + thresh_type + "NormH" + str(mvjmp) + ".png"

	# Normalize - Otsu Lacunarity
	f3 = plt.figure()
	f3.clear()
	y1 = NormalizedMalhi(lac1)
	p1 = plt.plot(x, y1, label=lacLeg1 + " - M")
	plt.legend()
	plt.title(thresh_title_part + ' Normalized (Malhi)'  + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "NormM" + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINNormPlot"] = imagename + "_" + thresh_type + "NormM" + str(mvjmp) + ".png"

	# Normalize - Otsu Lacunarity
	f4 = plt.figure()
	f4.clear()
	y1 = NormalizedMalhi(lac1)
	p1 = plt.plot(x, y1, label=lacLeg1 + " - M")
	y2 = NormalizedRoy(lac1)
	p1 = plt.plot(x, y2, label=lacLeg1 + " - R")
	y3 = NormalizedH(lac1, lacR1)
	p1 = plt.plot(x, y3, label=lacLeg1 + " - H")
	plt.legend()
	plt.title(thresh_title_part + ' Normalized (Roy, Malhi & Henebry)'  + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "Norm" + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINNormPlot"] = imagename + "_" + thresh_type + "Norm" + str(mvjmp) + ".png"

	plt.close(f1); plt.close(f2); plt.close(f3); plt.close(f4);

	return retDict

def Calc_LacBIN_1(imagename, idata):
	# Calculate Lacunarity using Histogram threshold levels
	[lacData, lacDataR, lacLeg] = Lac_Histogram_Threshold(idata, imagename)

	# Calculate Lacunarity using ostsu threshold
	#[lac1, lacR1, lacLeg1, thresh] = Lac_Otsu_Threshold(idata, imagename)

	x = range(1, len(lacData[0])+1)
	lx = [log(x1) for x1 in x]

	# Save Data to CSV file
	dataFilename = resFolder + imagename + ".csv"
	#SaveLacData(dataFilename, x, lacData + lacDataR + [lac1, lacR1], lacLeg + [lacLeg1])
	#SaveLacData(dataFilename, x, lacData + lacDataR, lacLeg)

	plotType = 2
	if plotType == 1:
		# Plot Hist Threshold Lacunarity
		#
		f0 = plt.figure()
		f0.clear()
		for n in range(len(lacData)):
			plt.plot(x, lacData[n], label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90%')
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_HistThresh.png"
		plt.savefig(figFilename)

		f1 = plt.figure()
		f1.clear()
		for n in range(len(lacDataR)):
			plt.plot(x, lacDataR[n], label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Inverse Images')
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_HistThresh_Inverse.png"
		plt.savefig(figFilename)

		# Otsu Lacunarity
		#f2 = plt.figure()
		#f2.clear()
		#plt.plot(x, lac1, label=lacLeg1)
		#plt.plot(x, lacR1, label=lacLeg1 + "_inv")
		#plt.legend()
		#plt.title('Otsu Threshold = ' + str(thresh))
		#plt.xlabel('window size')
		#plt.ylabel('lacunarity')
		#figFilename = resFolder + imagename + "_otsu.png"
		#plt.savefig(figFilename)

		# Log Log - Plot Hist Threshold Lacunarity
		f3 = plt.figure()
		f3.clear()
		for n in range(len(lacData)):
			plt.loglog(x, lacData[n], label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90%')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThresh_loglog.png"
		plt.savefig(figFilename)

		f4 = plt.figure()
		f4.clear()
		for n in range(len(lacDataR)):
			plt.loglog(x, lacDataR[n], label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Inverse Images')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThresh_Inverse_loglog.png"
		plt.savefig(figFilename)

		# Log Log - Otsu Lacunarity
		#f5 = plt.figure()
		#f5.clear()
		#plt.loglog(x, lac1, label=lacLeg1)
		#plt.loglog(x, lacR1, label=lacLeg1 + "_Inv")
		#plt.legend()
		#plt.title('Otsu Threshold = ' + str(thresh))
		#plt.xlabel('window size (log)')
		#plt.ylabel('lacunarity (log)')
		#figFilename = resFolder + imagename + "_otsu_loglog.png"
		#plt.savefig(figFilename)

		# Normalize - Plot Hist Threshold Lacunarity
		f6 = plt.figure()  # Norm - H
		f6.clear()
		for n in range(len(lacData)):
			y1 = NormalizedH(lacData[n], lacDataR[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Normalized (Henebry)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThresh_normHloglog.png"
		plt.savefig(figFilename)

		f7 = plt.figure()  # Norm - M
		f7.clear()
		for n in range(len(lacData)):
			y1 = NormalizedMalhi(lacData[n])
			p1 = plt.plot(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Normalized (Malhi)')
		plt.xlabel('window size')
		plt.ylabel('lacunarity (log ratio)')
		figFilename = resFolder + imagename + "_HistThresh_normMlr.png"
		plt.savefig(figFilename)

		f71 = plt.figure()  # Norm - M
		f71.clear()
		for n in range(len(lacData)):
			y1 = NormalizedMalhi(lacData[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Normalized (Malhi)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log logratio)')
		figFilename = resFolder + imagename + "_HistThresh_normMloglrlog.png"
		plt.savefig(figFilename)

		f8 = plt.figure()  # Norm - M
		f8.clear()
		for n in range(len(lacDataR)):
			y1 = NormalizedMalhi(lacDataR[n])
			p1 = plt.plot(lx, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Inverse Images - Normalized (Malhi)')
		plt.xlabel('window size')
		plt.ylabel('lacunarity (log ratio)')
		figFilename = resFolder + imagename + "_HistThresh_Inverse_normMlr.png"
		plt.savefig(figFilename)

		f9 = plt.figure() # Norm - R
		f9.clear()
		for n in range(len(lacData)):
			y1 = NormalizedRoy(lacData[n])
			p1 = plt.plot(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Normalized (Roy)')
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_HistThresh_normR.png"
		plt.savefig(figFilename)

		f10 = plt.figure() # Norm - R
		f10.clear()
		for n in range(len(lacData)):
			y1 = NormalizedRoy(lacData[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Normalized (Roy)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThresh_normRloglog.png"
		plt.savefig(figFilename)

		f11 = plt.figure()  # Norm - R
		f11.clear()
		for n in range(len(lacDataR)):
			y1 = NormalizedRoy(lacDataR[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n])
		plt.legend()
		plt.title('Histogram Threshold 10% ~ 90% - Inverse Images - Normalized (Roy)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThresh_Inverse_normRloglog.png"
		plt.savefig(figFilename)

		# Normalize - Otsu Lacunarity
		#f12 = plt.figure()
		#f12.clear()
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

		f13 = plt.figure()
		f13.clear()
		y1 = NormalizedMalhi(lac1)
		p1 = plt.loglog(x, y1, label=lacLeg1 + " - M")
		y2 = NormalizedRoy(lac1)
		p1 = plt.loglog(x, y2, label=lacLeg1 + " - R")
		y3 = NormalizedH(lac1, lacR1)
		p1 = plt.loglog(x, y3, label=lacLeg1 + " - H")
		plt.legend(loc=3)
		plt.title('Otsu Threshold = ' + str(thresh) + ' Normalized (Roy, Malhi & Henebry)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_otsuNormloglog.png"
		plt.savefig(figFilename)

		# 20% and %80 and reverse 20%
		f14 = plt.figure()
		f14.clear()
		for n in range(len(lacData)):
			if n == 1 or n == 7:
				p1 = plt.loglog(x, lacData[n], label=lacLeg[n])
			if n == 1:
				p1 = plt.loglog(x, lacDataR[n], label=lacLeg[n] + "-Inv")
		plt.legend()
		plt.title('Histogram Threshold 20%, 80% and Inverse of 20%')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThreshloglog_2080.png"
		plt.savefig(figFilename)

		# 20% and %80 and reverse 20%  - norm Malhi
		f15 = plt.figure()
		f15.clear()
		for n in range(len(lacData)):
			if n == 1 or n == 7:
				y1 = NormalizedMalhi(lacData[n])
				p1 = plt.loglog(x, y1, label=lacLeg[n])
			if n == 1:
				y1 = NormalizedMalhi(lacDataR[n])
				p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
		plt.legend()
		plt.title('Histogram Threshold 20%, 80% and Inverse of 20% - Normalized (Malhi)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThreshloglog_normM_2080.png"
		plt.savefig(figFilename)

		# 20% and %80 and reverse 20%  - norm Henebry
		f16 = plt.figure()
		f16.clear()
		for n in range(len(lacData)):
			if n == 1 or n == 7:
				y1 = NormalizedH(lacData[n], lacDataR[n])
				p1 = plt.loglog(x, y1, label=lacLeg[n])
			#if n == 1:
			#	y1 = NormalizedH(lacDataR[n], lacData[n])
			#	p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
		plt.legend()
		plt.title('Histogram Threshold 20%, 80% - Normalized (Henebry)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThresh_normHloglog_2080.png"
		plt.savefig(figFilename)

		# 20% and %80 and reverse 20%  - norm Roy
		f17 = plt.figure()
		f17.clear()
		for n in range(len(lacData)):
			if n == 1 or n == 7:
				y1 = NormalizedRoy(lacData[n])
				p1 = plt.loglog(x, y1, label=lacLeg[n])
			if n == 1:
				y1 = NormalizedRoy(lacDataR[n])
				p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
		plt.legend()
		plt.title('Histogray Threshold 20%, 80% and Inverse of 20% - Normalized (Roy)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThreshloglog_normR_2080.png"
		plt.savefig(figFilename)

		# %80 - normalized
		f18 = plt.figure()
		f18.clear()
		y1 = NormalizedMalhi(lacData[7])
		p1 = plt.plot(x, y1, label=lacLeg[7] + '-M')
		y1 = NormalizedH(lacData[7], lacDataR[7])
		p1 = plt.plot(x, y1, label=lacLeg[7] + '-H')
		y1 = NormalizedRoy(lacData[n])
		p1 = plt.plot(x, y1, label=lacLeg[7] + '-R')
		plt.legend()
		plt.title('Histogram Threshold 80% - Normalized')
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + imagename + "_HistThresh_norm_80.png"
		plt.savefig(figFilename)

		# %80 - normalized  log, log
		f19 = plt.figure()
		f19.clear()
		y1 = NormalizedMalhi(lacData[7])
		p1 = plt.loglog(x, y1, label=lacLeg[7] + '-M')
		y1 = NormalizedH(lacData[7], lacDataR[7])
		p1 = plt.loglog(x, y1, label=lacLeg[7] + '-H')
		y1 = NormalizedRoy(lacData[n])
		p1 = plt.loglog(x, y1, label=lacLeg[7] + '-R')
		plt.legend()
		plt.title('Histogram Threshold 80% - Normalized (log,log)')
		plt.xlabel('window size (log)')
		plt.ylabel('lacunarity (log)')
		figFilename = resFolder + imagename + "_HistThreshloglog_norm_80.png"
		plt.savefig(figFilename)

		#plt.show()
		plt.close(f0); plt.close(f1); plt.close(f3); plt.close(f4);
		plt.close(f6); plt.close(f7); plt.close(f71); plt.close(f8); plt.close(f9); plt.close(f10);
		plt.close(f11); plt.close(f13); plt.close(f14); plt.close(f15)
		plt.close(f16); plt.close(f17); plt.close(f18); plt.close(f19)
		#plt.close('all')
	if plotType == 2:
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
	alv0 = GetALV(idata, imagename, w, 0)
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

def SlidingImgG(idata, w = 2):
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

def Cal_GrayScaleLacunarity(imagename, idata, windowsSizes):
	minxy = min(idata.shape)

	for count in windowsSizes:
		print "Window = ", count, " of ", minxy
		print "Lacunarity = ", SlidingImgG(idata, count)

def ProcessAnImage(imagename, ext1):
	testData = {}
	dt1 = datetime.datetime.now()

	testData["imagename"] = imagename + ext1

	# Load Image
	idata = getImageData(imgFolder, imagename, ext1)

	#retDict = Calc_LacBIN(imagename, idata)
	#testData = dict( testData, **retDict )
	#testData["LacBIN"] = retDict
	#print testData

	winds = [3, 5, 51, '10x', '20x']
	winds = [7, 24, 33, 51]
	#winds = [12]

	#retDict = Calc_LacImageOtsu(imagename, idata, winds)
	#testData["LacBINSeg"] = retDict
	#retDict = Calc_LacImageOtsu(imagename, idata, winds, True)
	#testData["LacBINSegColor"] = retDict

	retDict = Calc_LacImageGrayScale(imagename, idata, winds)
	testData["LacGSSeg"] = retDict

	#idata = np.array([[5,4,8,7,9],[12,12,11,8,12],[11,12,9,10,5],[1,2,5,3,11],[5,9,2,7,10]])
	#print idata
	#Cal_GrayScaleLacunarity(imagename, idata, [3])
	#wins = range(1, minxy+1)
	#wins = [3, 4]
	#Cal_GrayScaleLacunarity(imagename, idata, wins)

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1
	return testData

def ProcessAnImage_Lacunarity(imagename, ext1, mj = 0):
	testData = {}
	dt1 = datetime.datetime.now()

	testData["imagename"] = imagename + ext1

	# Load Image
	idata = getImageData(imgFolder, imagename, ext1)

	retDict = Calc_LacBIN(imagename, idata, mj)
	#testData = dict( testData, **retDict )
	#testData["LacBIN"] = retDict

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1
	return testData

def ProcessAnImage_ALV(imagename, ext1):
	testData = {}
	dt1 = datetime.datetime.now()

	testData["imagename"] = imagename + ext1

	# Load Image
	idata = getImageData(imgFolder, imagename, ext1)

 	alv1 = Calc_ALV(imagename, idata)
 	print alv1

	dt2 = datetime.datetime.now()
	print "Time Taken for Image:", imagename, " =", dt2-dt1
	return testData

if __name__ == '__main__':
	testList = []
	# '../images1/'
	#ProcessAnImage('honeycomb2A', '.tif')
	#ProcessAnImage('honeycombb',  '.tif')
	#ProcessAnImage('RAC275',      '.tif')
	#ProcessAnImage('RAC325',      '.tif')

	#ProcessAnImage('Tropical-forest-Brazil-photo-by-Edward-Mitchard-small', '.jpg')
	#ProcessAnImage('forest-20130117-full',                                  '.jpg')

	#ProcessAnImage('back_forest',              '.jpg')
	#ProcessAnImage('glight2_pho_2014',         '.jpg')
	#ProcessAnImage('GoogleEarthEng_main_1203', '.jpg')
	#ProcessAnImage('peru480-blogSpan',         '.jpg')

	# '../images2/'
	#ret = ProcessAnImage('lena256b', '.bmp')
	#testList.append(ret)
	#ret = ProcessAnImage('MRI slice', '.jpg')
	#testList.append(ret)

	#strResultsJson = json.dumps(testList)
	#print
	#print strResultsJson

	#with open(resFolder + "images-exp1.json", 'w') as file1:
	#	file1.write(strResultsJson)
	set = 0
	if set == 0:
		ProcessAnImage_Lacunarity('honeycomb2', '.tif')
		ProcessAnImage_Lacunarity('Sq2_A', '.tif')
		ProcessAnImage_Lacunarity('Sq2_B', '.tif')
		ProcessAnImage_Lacunarity('Sq2_C', '.tif')
		ProcessAnImage_Lacunarity('Sq2_D', '.tif')

		#ProcessAnImage_Lacunarity('honeycomb2', '.tif', 1)
		#ProcessAnImage_Lacunarity('Sq2_A', '.tif', 1)
		#ProcessAnImage_Lacunarity('Sq2_B', '.tif', 1)
		#ProcessAnImage_Lacunarity('Sq2_C', '.tif', 1)
		#ProcessAnImage_Lacunarity('Sq2_D', '.tif', 1)

		#ProcessAnImage_ALV('honeycomb2', '.tif')
		#ProcessAnImage_ALV('Sq2_A', '.tif')
		#ProcessAnImage_ALV('Sq2_B', '.tif')
		#ProcessAnImage_ALV('Sq2_C', '.tif')
		#ProcessAnImage_ALV('Sq2_D', '.tif')

	if set == 1:
		ProcessAnImage_Lacunarity('ALV_Samp_A', '.tif')
		ProcessAnImage_Lacunarity('ALV_Samp_B', '.tif')
		ProcessAnImage_Lacunarity('ALV_Samp_C', '.tif')
		ProcessAnImage_Lacunarity('ALV_Samp_D', '.tif')
		ProcessAnImage_Lacunarity('ALV_Samp_E', '.tif')
		ProcessAnImage_Lacunarity('ALV_Samp_F', '.tif')

		ProcessAnImage_Lacunarity('ALV_Samp_A', '.tif', 1)
		ProcessAnImage_Lacunarity('ALV_Samp_B', '.tif', 1)
		ProcessAnImage_Lacunarity('ALV_Samp_C', '.tif', 1)
		ProcessAnImage_Lacunarity('ALV_Samp_D', '.tif', 1)
		ProcessAnImage_Lacunarity('ALV_Samp_E', '.tif', 1)
		ProcessAnImage_Lacunarity('ALV_Samp_F', '.tif', 1)

		ProcessAnImage_ALV('ALV_Samp_A', '.tif')
		ProcessAnImage_ALV('ALV_Samp_B', '.tif')
		ProcessAnImage_ALV('ALV_Samp_C', '.tif')
		ProcessAnImage_ALV('ALV_Samp_D', '.tif')
		ProcessAnImage_ALV('ALV_Samp_E', '.tif')
		ProcessAnImage_ALV('ALV_Samp_F', '.tif')

	if set == 2:
		#ProcessAnImage_Lacunarity('honeycomb2', '.tif')
		#ProcessAnImage_Lacunarity('brick.000', '.tif')
		ProcessAnImage_Lacunarity('1.2.03', '.tif')
		ProcessAnImage_Lacunarity('1.2.12', '.tif')



