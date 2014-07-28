import Image
import csv
import numpy as np
import datetime

from matplotlib import pyplot as plt
from copy import deepcopy

from ALVLACSubRoutines  import *
from Output 			import *

imgFolder = '../images1/'
resFolder = '../results/respy1/'

def getImageData(dataFolder, fileName, extFl):
	inim  = Image.open(dataFolder + fileName + extFl)

	if inim.mode != "L":
		inim = inim.convert("L")

	inar = np.asarray(inim)
	imageDim = len(inar.shape)
	print "Image " + fileName + " size: " + str(inar.shape) + ' dim (' + str(imageDim) + ')'
	# Pick image data
	if imageDim == 2:
		inar1 = inar[:,:]
	if imageDim == 3:
		inar1 = inar[:,:,0]
	return inar1

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

def SaveLacData_1(fileName, datax, datay, legend):
	with open(fileName, 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',')
		spamwriter.writerow(legend)
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

def GetLacunarity(data):
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
		CS = SlidingImg(data, sum, count)
		dataAve = np.average(CS)
		dataStd = np.std(CS)
		lac1  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
		lac.append(lac1)

	return lac

def NormalizedRoy(data):
	lacNorm = []
	Lmin = min(data)
	Lmax = max(data)
	for dt in data:
		lacNorm.append((dt-Lmin)/(Lmax-Lmin))
	return lacNorm

def NormalizedMahil(data):
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

def Lac_Otsu_Threshold(idata, savePics = ''):
	dt1 = datetime.datetime.now()

	threshold = OtsuThreshold(idata)
	trData1  = idata <= threshold
	trData2  = np.array(trData1, dtype='B')
	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_otsu_' + str(threshold) + '.tif',"TIFF")
	lac1     = GetLacunarity(trData2)
	lacLeg   = 'otsu = ' + str(threshold)

	trDataR1  = idata > threshold
	trDataR2  = np.array(trDataR1, dtype='B')
	if len(savePics) > 0:
		Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_otsu_' + str(threshold) + 'R.tif',"TIFF")
	lacR1     = GetLacunarity(trDataR2)
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

def Calc_BinLac(imagename, idata):
	# Calculate Lacunarity using Histogram threshold levels
	[lacData, lacDataR, lacLeg] = Lac_Histogram_Threshold(idata, imagename)

	# Calculate Lacunarity using ostsu threshold
	[lac1, lacR1, lacLeg1, thresh] = Lac_Otsu_Threshold(idata, imagename)

	x = range(1, len(lacData[0])+1)
	lx = [log(x1) for x1 in x]

	# Save Data to CSV file
	dataFilename = resFolder + imagename + ".csv"
	#SaveLacData_1(resFolder + imagename + "_1.csv", x, lacData + lacDataR + [lac1, lacR1], lacLeg + [lacLeg1])
	SaveLacData(dataFilename, x, lacData + lacDataR + [lac1, lacR1], lacLeg + [lacLeg1])

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
	f2 = plt.figure()
	f2.clear()
	plt.plot(x, lac1, label=lacLeg1)
	plt.plot(x, lacR1, label=lacLeg1 + "_inv")
	plt.legend()
	plt.title('Otsu Threshold = ' + str(thresh))
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_otsu.png"
	plt.savefig(figFilename)

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
	f5 = plt.figure()
	f5.clear()
	plt.loglog(x, lac1, label=lacLeg1)
	plt.loglog(x, lacR1, label=lacLeg1 + "_Inv")
	plt.legend()
	plt.title('Otsu Threshold = ' + str(thresh))
	plt.xlabel('window size (log)')
	plt.ylabel('lacunarity (log)')
	figFilename = resFolder + imagename + "_otsu_loglog.png"
	plt.savefig(figFilename)

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
		y1 = NormalizedMahil(lacData[n])
		p1 = plt.plot(x, y1, label=lacLeg[n])
	plt.legend()
	plt.title('Histogram Threshold 10% ~ 90% - Normalized (Mahil)')
	plt.xlabel('window size')
	plt.ylabel('lacunarity (log ratio)')
	figFilename = resFolder + imagename + "_HistThresh_normMlr.png"
	plt.savefig(figFilename)

	f71 = plt.figure()  # Norm - M
	f71.clear()
	for n in range(len(lacData)):
		y1 = NormalizedMahil(lacData[n])
		p1 = plt.loglog(x, y1, label=lacLeg[n])
	plt.legend()
	plt.title('Histogram Threshold 10% ~ 90% - Normalized (Mahil)')
	plt.xlabel('window size (log)')
	plt.ylabel('lacunarity (log logratio)')
	figFilename = resFolder + imagename + "_HistThresh_normMloglrlog.png"
	plt.savefig(figFilename)

	f8 = plt.figure()  # Norm - M
	f8.clear()
	for n in range(len(lacDataR)):
		y1 = NormalizedMahil(lacDataR[n])
		p1 = plt.plot(lx, y1, label=lacLeg[n])
	plt.legend()
	plt.title('Histogram Threshold 10% ~ 90% - Inverse Images - Normalized (Mahil)')
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
	f12 = plt.figure()
	f12.clear()
	y1 = NormalizedMahil(lac1)
	p1 = plt.plot(x, y1, label=lacLeg1 + " - M")
	y2 = NormalizedRoy(lac1)
	p1 = plt.plot(x, y2, label=lacLeg1 + " - R")
	y3 = NormalizedH(lac1, lacR1)
	p1 = plt.plot(x, y3, label=lacLeg1 + " - H")
	plt.legend()
	plt.title('Otsu Threshold = ' + str(thresh) + ' Normalized (Roy, Mahil & Henebry)')
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_otsuNorm.png"
	plt.savefig(figFilename)

	f13 = plt.figure()
	f13.clear()
	y1 = NormalizedMahil(lac1)
	p1 = plt.loglog(x, y1, label=lacLeg1 + " - M")
	y2 = NormalizedRoy(lac1)
	p1 = plt.loglog(x, y2, label=lacLeg1 + " - R")
	y3 = NormalizedH(lac1, lacR1)
	p1 = plt.loglog(x, y3, label=lacLeg1 + " - H")
	plt.legend(loc=3)
	plt.title('Otsu Threshold = ' + str(thresh) + ' Normalized (Roy, Mahil & Henebry)')
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

	# 20% and %80 and reverse 20%  - norm Mahil
	f15 = plt.figure()
	f15.clear()
	for n in range(len(lacData)):
		if n == 1 or n == 7:
			y1 = NormalizedMahil(lacData[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n])
		if n == 1:
			y1 = NormalizedMahil(lacDataR[n])
			p1 = plt.loglog(x, y1, label=lacLeg[n] + "-Inv")
	plt.legend()
	plt.title('Histogram Threshold 20%, 80% and Inverse of 20% - Normalized (Mahil)')
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
	y1 = NormalizedMahil(lacData[7])
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
	y1 = NormalizedMahil(lacData[7])
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

	plt.close(f0); plt.close(f1); plt.close(f2); plt.close(f3); plt.close(f4); plt.close(f5)
	plt.close(f6); plt.close(f7); plt.close(f71); plt.close(f8); plt.close(f9); plt.close(f10);
	plt.close(f11); plt.close(f12); plt.close(f13); plt.close(f14); plt.close(f15)
	plt.close(f16); plt.close(f17); plt.close(f18); plt.close(f19)

	#plt.close('all')

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
		#print outArray1.transpose()
		#print outArray2.transpose()
		#print outArray3.transpose()
		print Lac
		return Lac

def Cal_GrayScaleLacunarity(imagename, idata, windowsSizes):
	minxy = min(idata.shape)

	for count in windowsSizes:
		print "Window = ", count, " of ", minxy
		print "Lacunarity = ", SlidingImgG(idata, count)

def ProcessAnImage(imagename):
	dt1 = datetime.datetime.now()
	# Load Image
	idata = getImageData(imgFolder, imagename, '.tif')
	Calc_BinLac(imagename, idata)

	#idata = np.array([[5,4,8,7,9],[12,12,11,8,12],[11,12,9,10,5],[1,2,5,3,11],[5,9,2,7,10]])
	#print idata

	# wins = range(1, minxy+1)
	wins = [3, 4]

	#Cal_GrayScaleLacunarity(imagename, idata, wins)
	dt2 = datetime.datetime.now()
	print "Image:", imagename, " time taken:", dt2-dt1

if __name__ == '__main__':
	ProcessAnImage('honeycomb2A')
	ProcessAnImage('honeycombb')
	ProcessAnImage('RAC275')
	ProcessAnImage('RAC325')


