#! /usr/bin/python

"""
Lacunarity program for 2D data
It uses Python PIL numpy modules
by Dantha Manikka-Baduge, V.1.1, Feb 28 2008

Changed : Jul 22 2008
		: Jun 04 2014

Usage:  Lac2D <image file path>
			  <optional: image file name without ext. or all					default: all image files, all = all image files>
			  <optional: average = 1 or clip off smaller boxes at the end = 0.	default: 1>
			  <optional: moving  = 0, jumping = 1, both = 2.					default: 0>
			  <optional: showprogress = 1.										default: 0>
			  <optional: 1bit Lac. 					= 0,
			  			 1bit Norm. Lac 			= 1,
			  			 Grayscale Lac. 			= 2,
			  			 Gray Norm Lac 				= 3.
			  			 Gray Biplane Norm Lac 		= 4.
			  			 Gray Threshold Norm Lac 	= 5.
			  			 Gray 3D Data Lac 			= 6.
			  			 Gray 3D Data Norm Lac 		= 7.						default: 1>
parameters seperate by space

TODO:
bitplane gray   lac
threshold gray  lac ??
normalized gray lac
parameterized

thresholding methods isodata, adaptive, otsu
strech, hist equalize, normalized variance

"""

from random   import choice
from colorsys import rgb_to_yiq

# Python PIL
import Image
import ImageOps

from math  import *
from numpy import *

import time
import sys
import os.path

from UtilRoutines import *
#from PlotGnuplot import *
#from PlotGoogle import *
#from PlotCSV import *
from ALVLACSubRoutines  import *
from Output import *

def Lacunarity(datarray, norm = 1, mjmp = 0, clipType = 0, verbose = 0, threshold = 1, fileName = None):
	"""Generate the lacunarity data for the given datarray
		norm: 0 - 1 bit lacunarity
			  1 - normalized data
			  2 - gray scale
			  3 - gray scale norm
			  4 - bit plane gray scale norm
			  5 - threshold gray scale norm
			  6 - gray scale lacunarity considering 3D (8 bit/256 value)
			  7 - gray scale lacunarity considering 3D norm
		threshold : 0 - no threshold
					1 - 50% threshold
					2 - adaptive (TODO)
					3 - isodata  (TODO)
					4 - otsu     (TODO)
	"""
	w = 1
	LacData   = []
	LacDataC  = []
	LacDataN  = []
	TimeSpent = []
	(sx, sy) = datarray.shape
	TH = "none"

	#timeVal = time.clock()
	maxVal = max(ravel(datarray))
	minVal = min(ravel(datarray))
	if norm == 0 or norm == 1:
		# Converto binary use threshold method
		# threshold == 0 - no threshold.
		if threshold == 1:
			# 50
			TH = (maxVal - minVal) / 2 #82
			print "50% threshold at ", TH
			inar2 = datarray >  TH
			inar3 = datarray <= TH
			datarray  = array(inar2, dtype='B')
			datarrayN = array(inar3, dtype='B')

		if threshold == 0:
			inar4 = datarray != 1
			datarrayN = array(inar4, dtype='B')
			#ddd = datarrayN + datarray
			#print max(ravel(ddd)),min(ravel(ddd))
			#print datarrayN.sum(), datarray.sum(), ddd.sum()

		if threshold != 0 and threshold != 1:
			print "These threshold methods are not implemented yet."

		if fileName != None:
			Im1  = Image.frombuffer('L', datarray.shape,  datarray * 255,  "raw", 'L', 0, 1)
			Im1.save(fileName + "_" + str(TH) + '.tif')
		
		dataAve  = average(datarray)
		dataAveN = average(datarrayN)
		lac  = 1 + (std(datarray) * std(datarray))/(dataAve * dataAve)
		if norm == 1:
			lacC = 1 + (std(datarrayN) * std(datarrayN))/(dataAveN * dataAveN)
			lacF = 2 - 1/lac - 1/lacC;

	if norm == 2:
		lac  = 1 + (std(datarray )*std(datarray))/(average(datarray)*average(datarray ))

	if norm == 3:
		bins = maxVal - minVal + 1
		(dataHst, nHst) = histogram(datarray, bins)
		dataHstProb = dataHst * 1.0/ dataHst.sum()
		#xv = range(bins + 1)[1:]
		#plot2(xv, dataHst, "testHistD", 0, "testHD")
		#plot2(xv, dataHstProb, "testHistP", 0, "testHP")
		#sq = s2q = lac = 0.0
		lenHst = len(nHst)
		for n in [range(lenHst)]:
			sq  = dataHstProb[n] * nHst[n]
			s2q = dataHstProb[n] * nHst[n] * nHst[n]
		lac = s2q.sum()/(sq.sum()*sq.sum())
		#sqC = s2qC = lacC = 0.0
		for m in [range(lenHst-1, -1, -1)]:
			sqC  = dataHstProb[m] * nHst[n]
			s2qC = dataHstProb[m] * nHst[n] * nHst[n]
		lacC = s2qC.sum()/(sqC.sum()*sqC.sum())
		lacF = 2 - 1/lac - 1/lacC;
		#print lac, lacC, lacF

	if norm == 6 or norm == 7:
		dArray  = datarray > 0
		dArraySel = array(dArray, dtype='B')
		dArray  = datarray == 0
		dArraySelN = array(dArray, dtype='B')
		lac  = 1 + (std(dArraySel)*std(dArraySel))/(average(dArraySel)*average(dArraySel))
		if norm == 7:
			lacC = 1 + (std(dArraySelN)*std(dArraySelN))/(average(dArraySelN)*average(dArraySelN))
			
lacF = 2 - 1/lac - 1/lacC;
			#print dArraySel
			#print dArraySelN
		datarrayN = [255-x for x in datarray]
	
	LacData.append(lac);
	if norm == 1 or norm == 3 or norm == 7:
		LacDataC.append(lacC);
		LacDataN.append(lacF);

	minxy = min(sx, sy);

	# find lac from i=2 to msize
	for count in [x+1 for x in range(minxy)][1:]:
		if verbose == 1:
			print "Box size : ", count

		#timeValNew = time.clock()
		#timeSp = timeValNew - timeVal
		#TimeSpent.append(timeSp)
		# print count, timeSp
		#timeVal = timeValNew
		
	if norm == 6 or norm == 7:
				if mjmp == 1: # Jumping
					CS = JumpingImg(datarray, sum, count, clipType)
			else:
				CS = SlidingImgGray(datarray, sum, count)
			if norm == 7:
				if mjmp == 1: # Jumping
					CSC = JumpingImg(datarrayN, sum, count, clipType)
				else:
					CSC = SlidingImgGray(datarrayN, sum, count)		
		else:
			if mjmp == 1: # Jumping
				CS = JumpingImg(datarray, sum, count, clipType)
			else:
				CS = SlidingImg(datarray, sum, count)
			if norm == 1:
				if mjmp == 1: # Jumping
					CSC = JumpingImg(datarrayN, sum, count, clipType)
				else:
					CSC = SlidingImg(datarrayN, sum, count)

		if norm == 1 or norm == 0 or norm == 2 or norm == 6:
			meanscore = average(CS);
			stdscore  = std(CS);
			lac = 1 + ((stdscore * stdscore)/(meanscore * meanscore));

		if norm == 1 or norm == 7:
			meanscore = average(CSC);
			stdscore  = std(CSC);
			lacC = 1 + ((stdscore * stdscore)/(meanscore * meanscore));
			lacF = 2 - 1/lac - 1/lacC;

		if norm == 3:
			maxVal = max(ravel(CS))
			minVal = min(ravel(CS))
			bins = maxVal - minVal + 1
			# print "B M m ", bins, maxVal, minVal
			(dataHst, nHst) = histogram(CS, bins)
			dataHstProb = dataHst * 1.0/ dataHst.sum()
			#xv = range(bins + 1)[1:]
			#plot2(xv, dataHst, "testHistD", 0, "testHD")
			#plot2(xv, dataHstProb, "testHistP", 0, "testHP")
			#sq = s2q = sqC = s2qC = lac = 0.0
			lenHst = len(nHst)
			for n in [range(lenHst)]:
				sq = dataHstProb[n] * nHst[n]
				s2q = sq * nHst[n]
			lac = s2q.sum()/(sq.sum()*sq.sum())
			for m in [range(lenHst-1, -1, -1)]:
				sqC = dataHstProb[m] * nHst[n]
				s2qC = dataHstProb[m] * nHst[n] * nHst[n]
			lacC = s2qC.sum()/(sqC.sum()*sqC.sum())
			lacF = 2 - 1/lac - 1/lacC;
			# print lac, lacC, lacF

		LacData.append(lac)
		if norm == 1 or norm == 3:
			LacDataC.append(lacC)
			LacDataN.append(lacF)

	if norm == 3:
		mVal = LacDataN[0]
		for n in range(len(LacDataN)):
			LacDataN[n] /= mVal

	#timeValEnd = time.clock()
	#print minxy, timeValStat, timeValEnd, timeValEnd - timeValStat
	if norm == 1:
		xv = range(len(LacDataC)+1)[1:]
		if fileName == None:
			fileName = 'test'
		plot2(xv, LacDataC, "Lacunarity 1bit normalized", 1, fileName)
	# WriteToCSVFile("time1.csv", TimeSpent)
	return (LacData, LacDataN)

def LacunarityBitPlane(datarray, norm = 1, mjmp = 0, clipType = 0, verbose = 0, threshold = 0, thFileName = None):
	lacCalBNorm = []
	LacData     = []
	legendBP    = []
	#bitplan = range(8)
	#for h in bitplan:
	#	bitplan[h] = 2**h
	for n in range(8):
		legendBP.append(str(n+1))
		if thFileName != None:
			thFileNameBN = thFileName + str(n+1)
		else:
			thFileNameBN = None
		inar1 = (datarray & 2**n)/2**n
		# print inar1
		if verbose == 1:
			print "Bit Plane ", n
		(lacDiscard, lacCalBN) = Lacunarity(inar1, 1, mjmp, clipType, verbose, 0, thFileNameBN)
		lacCalBNorm.append(lacCalBN)
	lac1 = []
	rng = len(lacCalBNorm[0])
	for m in range(rng):
		lac1.append(0)
	for n in range(8):
		for m in range(rng):
			lac1[m] = lac1[m] + lacCalBNorm[n][m] * 2 ** n
	LacData = lac1/(255* ones(len(lac1)))

	if thFileName != None:
		WriteToCSVFile(thFileName + ".csv",  lacCalBNorm[0], lacCalBNorm[1], lacCalBNorm[2], lacCalBNorm[3],  lacCalBNorm[4], lacCalBNorm[5], lacCalBNorm[6], lacCalBNorm[7], LacData)
		plotMul(lacCalBNorm, "Lacunarity - Bitplane", legendBP, 1, thFileName)
	return (LacData, lacCalBNorm)


def LacunarityThreshold(datarray, norm = 1, mjmp = 0, clipType = 0, verbose = 0, threshold = 1, thFileName = None):
	lacCalTNorm = []
	LacData = []
	legendTH = []
	for n in range(8):
		legendTH.append(str(2**n))
		if thFileName != None:
			thFileNameTH = thFileName + str(2**n)
		else:
			thFileNameTH = None
		
		inar2 = datarray > 2**n
		inar1  = array(inar2, dtype='B')

		# print inar1
		if verbose == 1:
			print "Threshold ", 2**n
		(lacDiscard, lacCalTH) = Lacunarity(inar1, 1, mjmp, clipType, verbose, 0, thFileNameTH)
		lacCalTNorm.append(lacCalTH)
	lac1 = []
	rng = len(lacCalTNorm[0])
	for m in range(rng):
		lac1.append(0)
	for n in range(8):
		for m in range(rng):
			lac1[m] = lac1[m] + lacCalTNorm[n][m] * 2 ** n
	LacData = lac1/(255* ones(len(lac1)))

	if thFileName != None:
		WriteToCSVFile(thFileName + ".csv",  lacCalTNorm[0], lacCalTNorm[1], lacCalTNorm[2], lacCalTNorm[3],  lacCalTNorm[4], lacCalTNorm[5], lacCalTNorm[6], lacCalTNorm[7], LacData)
		plotMul(lacCalTNorm, "Lacunarity - Threshold", legendTH, 1, thFileName)
	return (LacData, lacCalTNorm)

def main(dataFolder, fileName, extFl, JumpClipping = 0, MTD = 2, verbose = 0, norm = 1):
	LACJ = 0
	LACM = 0
	if MTD == 0:
		LACM = 1
	if MTD == 1:
		LACJ = 1
	if MTD == 2:
		LACJ = 1
		LACM = 1

	# create the results folder if it does not exist
	resFolder  = dataFolder + 'resSet1/'
	if not os.path.isdir(resFolder):
		os.mkdir(resFolder)

	inim  = Image.open(dataFolder + fileName + extFl)

	if inim.mode != "L":
		inim = inim.convert("L")

	inar = asarray(inim)
	imageDim = len(inar.shape)
	print "Image size: " + str(inar.shape) + ' dim (' + str(imageDim) + ')'

	if JumpClipping == 0:
		print "Clipping off smaller boxes"
	else:
		print "Considering Weighted average of smaller boxes"

	subHeader1 = ""
	saveFNpart = ""
	if norm == 0:
		subHeader1 = "1-bit "
		saveFNpart = "1b_"
	if norm == 1:
		subHeader1 = "1-bit Normalized "
		saveFNpart = "1bn_"
	if norm == 2:
		subHeader1 = "Grayscale "
		saveFNpart = "gs_"
	if norm == 3:
		subHeader1 = "Grayscale Normalized "
		saveFNpart = "gsn_"
	if norm == 4:
		subHeader1 = "Grayscale Bitplane Normalized "
		saveFNpart = "gsbpn_"
	if norm == 5:
		subHeader1 = "Grayscale Threshold Normalized "
		saveFNpart = "gsthn_"
	if norm == 6:
		subHeader1 = "Grayscale 3D "
		saveFNpart = "gs3d_"
	if norm == 7:
		subHeader1 = "Grayscale 3D Normalized "
		saveFNpart = "gs3dn_"
	# Pick image data
	if imageDim == 2:
		inar1 = inar[:,:]
	if imageDim == 3:
		inar1 = inar[:,:,0]

	equalize = 0
	# strech - equalize
	if equalize == 1:
		scale = 255.0
		minv = min(ravel(inar1).tolist())
		inar1 = inar1 - minv * ones(inar1.shape)
		maxv = max(ravel(inar1).tolist())
		inar2 = inar1 * scale/maxv
		inar1 = array(inar2, dtype='B')

	#if LACJ == 1 or LACM == 1:
	#	lacIm1  = Image.frombuffer('L', inar1.shape,  inar1,  "raw", 'L', 0, 1)
	#	lacIm1.save(resFolder + 'LacProc_'  + fileName + extFl)

	if LACJ == 1:
		if norm == 0 or norm == 1:
			thFileName = resFolder + 'LACJ_'  + fileName + '_TH'
		else:
			thFileName = resFolder + 'LACJ_'  + fileName
		print "Calculation Lacunarity (jump)... " #+ thFileName;
		if norm == 0 or norm == 1 or norm == 2 or norm == 3 or norm == 6 or norm == 7:
			(lacd2, lacdn2) = Lacunarity(inar1, norm, 1, JumpClipping, verbose, 1, thFileName)
		if norm == 4:
			thFileName = resFolder + 'LACJ_'  + fileName + '_BP'
			(lacCal, lacBPData) = LacunarityBitPlane(inar1, norm, 1, JumpClipping, verbose, 1, thFileName)
		if norm == 5:
			thFileName = resFolder + 'LACJ_'  + fileName + '_TH'
			(lacCal, lacTHData) = LacunarityThreshold(inar1, norm, 1, JumpClipping, verbose, 1, thFileName)
		if norm == 0 or norm == 2 or norm == 6:
			lacCal = lacd2
		if norm == 1 or norm == 3 or norm == 7:
			lacCal = lacdn2

		lacDif  = DataDifference(lacCal)
		lacGrad = DataGradient(lacCal)
		saveas  = resFolder + 'LACJ_' + saveFNpart + fileName
		title   = subHeader1 + 'Lacunarity (jump) :' + fileName
		xv 		= range(len(lacCal)+1)[1:]
		plot2(xv, lacCal, title, 1, saveas)
		plot2(xv, lacCal, title, 0, saveas)
		title = subHeader1 + 'Lacunarity (jump) diff. from line:' + fileName
		#plot1(lacDif, title, saveas + 'dif')
		plot2(xv, lacDif, title, 0, saveas + 'dif')
		title = subHeader1 + 'Lacunarity (jump) grad.:' + fileName
		#plot1(lacGrad, title, saveas + 'grad')
		plot2(xv[:-1], lacGrad, title, 0, saveas + 'grad')
		WriteToCSVFile(saveas + ".csv",   xv, lacCal, lacDif, lacGrad)
		print "...Done";
	if LACM == 1:
		if norm == 0 or norm == 1:
			thFileName = resFolder + 'LACM_'  + fileName + '_TH'
		else:
			thFileName = resFolder + 'LACM_'  + fileName
		print "Calculation Lacunarity (move)..." #+ thFileName;
		if norm == 0 or norm == 1 or norm == 2 or norm == 3 or norm == 6 or norm == 7:
			(lacd1, lacdn1) = Lacunarity(inar1, norm, 0, JumpClipping, verbose, 1, thFileName)
		if norm == 4:
			thFileName = resFolder + 'LACM_'  + fileName + '_BP'
			(lacCal, lacBPData) = LacunarityBitPlane(inar1, norm, 0, JumpClipping, verbose, 1, thFileName)
		if norm == 5:
			thFileName = resFolder + 'LACM_'  + fileName + '_TH'
			(lacCal, lacTHData) = LacunarityThreshold(inar1, norm, 0, JumpClipping, verbose, 1, thFileName)

		if norm == 0 or norm == 2 or norm == 6:
			lacCal = lacd1
		if norm == 1 or norm == 3 or norm == 7:
			lacCal = lacdn1
		lacDif  = DataDifference(lacCal)
		lacGrad = DataGradient(lacCal)
		saveas  = resFolder + 'LACM_' + saveFNpart + fileName
		title   = subHeader1 + 'Lacunarity (move) :' + fileName
		xv 	    = range(len(lacCal)+1)[1:]
		plot2(xv, lacCal, title, 1, saveas)
		plot2(xv, lacCal, title, 0, saveas)
		title = subHeader1 + 'Lacunarity (move) diff. from line:' + fileName
		#plot1(lacDif, title, saveas + 'dif')
		plot2(xv, lacDif, title, 0, saveas + 'dif')
		title = subHeader1 + 'Lacunarity (move) grad.:' + fileName
		#plot1(lacGrad, title, saveas + 'grad')
		plot2(xv[:-1], lacGrad, title, 0, saveas + 'grad')
		WriteToCSVFile(saveas + ".csv",    xv, lacCal, lacDif, lacGrad)
		print "...Done";

if __name__ == '__main__':
	JumpClipping = 1			# 0 - Clipping; 1 - waited average
	method 		 = 0
	verbose 	 = 0
	norm 		 = 1

	if len(sys.argv) < 2:
		print "Usage:  Lac2D <image file path>"
		print "	  <optional: image file name without ext. all = all image files"
		print "	   default: all image files>"
		print "	  <optional: average = 1 or clip off smaller boxes at the end = 0."
		print "	   default: 1>"
		print "	  <optional: moving = 0, jumping = 1, both = 2."
		print "	   default: 0>"
		print "	  <optional: showprogress = 1."
		print "	   default: 0>"
		print "	  <optional: 1bit Lac. = 0, 1bit Norm. Lac = 1,"
		print "	  <          Grayscale Lac. = 2, Gray Norm Lac = 3,"
		print "	  <          Gray Biplane Norm Lac = 4, Gray Threshold Norm Lac = 5."
		print "	   default: 1>"
		print ""
		print "parameters seperate by space"
	else:
		extFl = ".tif"
		#print sys.argv
		dataFolder = sys.argv[1]

		if len(sys.argv) > 2:
			fileName = sys.argv[2]
			if fileName == "all":
				fileName = ''
		else:
			fileName = ''

		if len(sys.argv) > 3:
			JumpClipping = int(sys.argv[3])

		if len(sys.argv) > 4:
			method = int(sys.argv[4])

		if len(sys.argv) > 5:
			verbose = int(sys.argv[5])

		if len(sys.argv) > 6:
			norm = int(sys.argv[6])

		if len(fileName) > 0:
			print fileName
			main(dataFolder, fileName, extFl, JumpClipping, method, verbose, norm)
		else:
			for fileName in lstDir(dataFolder, extFl):
				print fileName
				main(dataFolder, fileName, extFl, JumpClipping, method, verbose, norm)


