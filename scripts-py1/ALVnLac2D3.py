#! /usr/bin/python

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

from random   import choice
from colorsys import rgb_to_yiq

 # Python PIL
import Image
import ImageOps

# Give proper credits here and website
# from select import median
# import psyco; psyco.full()

from math  import *
from numpy import *
import sys

from PlotCSV import *

def JumpingImg(datarray, func, w = 2, clipType = 0):
	(sx, sy) = datarray.shape
	(ma, na) = {
		0: lambda : (sx//w, sy//w),
		1: lambda : (sx//w + (sx % w >= 1 and 1 or 0), sy//w + (sy % w >= 1 and 1 or 0))
				}[clipType]()
	OutIm = zeros((ma, na), float)
	(mab, nab) = (sx % w, sy % w)
	# print ((sx, sy), (ma, na))
	# print w, clipType
	if clipType == 1:
		nal = na - (sx % w >= 1 and 1 or 0)
		mal = ma - (sy % w >= 1 and 1 or 0)
	else:
		nal = na
		mal = ma
	#print ((mab, nab), (mal,nal))
	for y in range(nal):
		for x in range(mal):
			OutIm[x,y] = func(datarray[x*w:x*w+w,y*w:y*w+w])
	if clipType == 1:
		if mal != ma:
			for x in range(mal):
				OutIm[x,nal] = func(datarray[x*w:x*w+w,nal*w:nal*w+nab]) # * w / nab
		if nal != na:
			for y in range(nal):	
				OutIm[mal,y] = func(datarray[mal*w:mal*w+mab,y*w:y*w+w]) # * w / mab
		if mal != ma and nal != na:
			OutIm[mal,nal] = func(datarray[mal*w:mal*w+mab,nal*w:nal*w+nab]) # * w * w / (mab*nab)
	return OutIm
    
def SlidingImg(datarray, func, w = 2):
	(sx, sy) = datarray.shape
	#print "Slide in :"
	#print (sx, sy)
	if min(sx-w + 1,sy - w + 1) == 0:
		OutIm = zeros((1,1), float)
		OutIm[0,0] = datarray[0,0]
	else:
		OutIm = zeros((sx-w + 1,sy - w + 1), float) 
		for y in range(sy - w + 1): 
			for x in range(sx - w + 1):
				OutIm[x,y] = func(datarray[x:x+w,y:y+w])
	#print "Slide out :"
	#print OutIm.shape
	return OutIm        
    
def ALV(datarray, w = 2, mjmp = 1, clipType = 0):
	SaveAggImgs = 1
	funcused = std # var ?
	ALVData = []
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
		#if x == 19:
		#	plot3Dgrid(d2, 6)
		# print d2.shape
		if SaveAggImgs == 1:
			d3 = array(d2, dtype='B')
			agIm = Image.fromarray(d3)
			agIm.save("aggImgs/AggData_" + str(x) + ".tif","TIFF")
		if mjmp == 1:
			d3 = JumpingImg(d2, funcused, 2, clipType)
		else:
			d3 = SlidingImg(d2, funcused)
		# print d3.shape
		#if d3.shape[0] > 0:
		ALVData.append(average(d3))	
	return ALVData
	
def Lacunarity(datarray, mjmp = 1, datarrayN = None, clipType = 0):
	"""Generate the lacunarity data for the give datarray
	   norm = 1 normalized data
	"""
	w = 1
	norm = 0
	LacData = []
	LacDataC = []
	LacDataN = []
	(sx, sy) = datarray.shape
	if datarrayN != None:
		(sxc, syc) = datarrayN.shape
	else:
		(sxc, syc) = (0, 0)
	norm = sx == sxc and sy == syc
	print norm

	# find LacData(1)
	lac  = 1 + (std(datarray )*std(datarray ))/(average(datarray )*average(datarray ));
	lacC = 1 + (std(datarrayN)*std(datarrayN))/(average(datarrayN)*average(datarrayN));
	lacF = 2 - 1/lac - 1/lacC;
	LacData.append(lac);
	LacDataC.append(lacC);
	LacDataN.append(lacF);
	minxy = min(sx, sy);

	# find lac from i=2 to msize
	for count in [x+1 for x in range(minxy)][1:]:
		# print count
		if mjmp == 1: # Jumping
			CS = JumpingImg(datarray, sum, count, clipType)
		else:
			CS = SlidingImg(datarray, sum, count)
		if mjmp == 1: # Jumping
			CSC = JumpingImg(datarrayN, sum, count, clipType)
		else:
			CSC = SlidingImg(datarrayN, sum, count)

		meanscore = average(CS);
		stdscore  = std(CS);
		lac = 1 + ((stdscore*stdscore)/(meanscore*meanscore));

		meanscore = average(CSC);
		stdscore  = std(CSC);
		lacC = 1 + ((stdscore*stdscore)/(meanscore*meanscore));
		lacF = 2 - 1/lac - 1/lacC;
		LacData.append(lac)
		LacDataC.append(lacC)
		LacDataN.append(lacF)
	return (LacData, LacDataC, LacDataN)	

def DataGrad(datarray):
	lenAr = len(datarray)
	DataGradArray = []
	for pos in range(lenAr)[1:]:
		DataGradArray.append(datarray[pos] - datarray[pos-1])
	return DataGradArray

def DataDiff(datarray):
	lenAr = len(datarray)
	DataDiffArray = []
	Grad = (datarray[lenAr-1] - datarray[0])/lenAr
	Val = datarray[0]
	for pos in range(lenAr):
		DataDiffArray.append(Val + pos * Grad - datarray[pos])
	return DataDiffArray
	
def main():
	ALVJ = 1
	ALVM = 1
	LACJ = 0
	LACM = 0
	
	JumpClipping = 0 # 0 - Clipping; 1 - waited average
	
	if len(sys.argv) < 2:
		print "usage ALVnLac3D.py <name of the image(.tif) file>"
		return
		
	dataFoler = 'data4//'
	resFoler  = 'res_set3//'
	fileName  = sys.argv[1]
	if len(sys.argv) == 3:
		JumpClipping = int(sys.argv[2])
	
	inim = Image.open(dataFoler + fileName + '.tif')

	inimC = ImageOps.invert(inim)
	
	inar = asarray(inim)
	imageDim = len(inar.shape)
	print "Image size: " + str(inar.shape) + ' dim (' + str(imageDim) + ')'
	if JumpClipping == 0:
		print "Clipping smaller boxes"
	else:
		print "Using    smaller boxes"
	if imageDim == 2:
		inar1 = inar[:,:]
	if imageDim == 3:
		inar1 = inar[:,:,0]
	
	# strech
	scale = 255.0
	minv = min(ravel(inar1).tolist())
	inar1 = inar1 - minv * ones(inar1.shape)
	maxv = max(ravel(inar1).tolist())
	inar2 = inar1 * scale/maxv
	inar1 = array(inar2, dtype='B')
	
	# plot3Dgrid(inar)
	#plot1pdf(None, None, "test")
	
	agIm1 = Image.frombuffer('L', inar1.shape, inar1, "raw", 'L', 0, 1)
	agIm1.save("aggImgs/AggData_1.tif","TIFF")
	
	if ALVJ == 1:
		print "Calculation AVL (jump)...";
		ald0 = ALV(inar1, 2, 1, JumpClipping)
		saveas = dataFoler + resFoler + 'ALVJ_' + fileName
		title = 'ALV Jumping (2x2) for ' + fileName
		#print len(ald0)
		#pdataj = ald0
		#pdatam.insert(0,0)
		#print len(ald0)
		if ALVM != 1:
			plot1(ald0, title, 0, saveas)
		WriteToCSVFile(saveas + ".csv", ald0)
	if ALVM == 1:
		print "Calculation AVL (move)...";
		ald1 = ALV(inar1, 2, 0, JumpClipping)
		saveas = dataFoler + resFoler + 'ALVM_' + fileName
		title = 'ALV Moving (2x2) for ' + fileName
		#pdataj = ald1
		#pdataj.insert(0,0)
		if ALVJ != 1:
			plot1(ald1, title, 0, saveas)
		WriteToCSVFile(saveas + ".csv", ald1)

	if ALVJ == 1 and ALVM == 1:
		title = 'ALV Jumping and Moving (2x2) for ' + fileName
		saveas = dataFoler + resFoler + 'ALV_' + fileName
		#plotMul(pdatam, pdataj, title, saveas)
		dx = range(len(ald0)+1)[1:]
		# print len(dx), len(ald0), len(ald1)
		plotMulXY(dx, ald0, ald1, title, saveas)
	inarC = asarray(inimC)
	#print inarC.shape
	
	# Pick image data
	if imageDim == 2:
		inar1 = inar[:,:]
	if imageDim == 3:
		inar1 = inar[:,:,0]
		
	# strech
	scale = 255.0
	minv = min(ravel(inar1).tolist())
	inar1 = inar1 - minv * ones(inar1.shape)
	maxv = max(ravel(inar1).tolist())
	inar2 = inar1 * scale/maxv
	inar1 = array(inar2, dtype='B')

	# Converto binary
	TH = 128 # 114, 128
	inar2 = inar1 > TH
	inar1 = inar2 * 255
	
	# Pick image data
	if imageDim == 2:
		inarC1 = inarC[:,:]
	if imageDim == 3:
		inarC1 = inarC[:,:,0]
	
	# strech
	scale = 255.0
	minv = min(ravel(inarC1).tolist())
	inarC1 = inarC1 - minv * ones(inarC1.shape)
	maxv = max(ravel(inarC1).tolist())
	inarC2 = inarC1 * scale/maxv
	inarC1 = array(inarC2, dtype='B')
	
	# Converto binary
	TH = 128 # 114, 128
	inarC2 = inarC1 > TH
	inarC1 = inarC2 * 255

	lacIm1  = Image.frombuffer('L', inar1.shape,  inar1,  "raw", 'L', 0, 1)
	lacImC1 = Image.frombuffer('L', inarC1.shape, inarC1, "raw", 'L', 0, 1)	
	lacIm1.save(dataFoler + resFoler + 'LacT128_'  + fileName + '.tif',"TIFF")
	lacImC1.save(dataFoler + resFoler + 'LacT128C_' + fileName + '.tif',"TIFF")
		
	if LACM == 1:
		print "Calculation Lacunarity (move)...";
		(lacd1, lacdc1, lacdn1) = Lacunarity(inar1, 0, inarC1) # JumpClipping irrelavant
		lacDif  = DataDiff(lacdn1)
		lacGrad = DataGrad(lacdn1)
		saveas  = dataFoler + resFoler + 'LACM1_' + fileName
		title   = '1-bit Normalized Lacunarity (move) :' + fileName
		xv 	    = range(len(lacd1)+1)[1:]

		plot2(xv, lacdn1, title, 1, saveas)
		plot2(xv, lacdn1, title, 0, saveas)
		title = '1-bit Normalized Lacunarity (move) diff. from line:' + fileName
		#plot1(lacDif, title, saveas + 'dif')
		plot2(xv, lacDif, title, 0, saveas + 'dif')
		title = '1-bit Normalized Lacunarity (move) grad.:' + fileName
		#plot1(lacGrad, title, saveas + 'grad')
		plot2(xv[:-1], lacGrad, title, 0, saveas + 'grad')
		title = '1-bit Lacunarity (move) p n:' + fileName
		# plotMulLg(lacd1, lacdc1, xv, title, saveas + '_pn')
		# WriteToCSVFile(saveas + "log.csv", lx, llac, llacc, llacn)
		WriteToCSVFile(saveas + ".csv",    xv, lacd1, lacdc1, lacdn1, lacDif, lacGrad)
	if LACJ == 1:
		print "Calculation Lacunarity (jump)...";
		(lacd2, lacdc2, lacdn2) = Lacunarity(inar1, 1, inarC1, JumpClipping)
		#lacDif  = DataDiff(lacdn2)
		#lacGrad = DataGrad(lacdn2)
		saveas   = dataFoler + resFoler + 'LACJ1_' + fileName
		title    = '1-bit Normalized Lacunarity (jump) :' + fileName
		xv       = range(len(lacd2)+1)[1:]

		plot2(xv, lacdn2, title, 1, saveas)
		plot2(xv, lacdn2, title, 0, saveas)	

		title    = '1-bit Lacunarity (jump) :' + fileName
		plot2(xv, lacd2, title, 0, saveas)	

		#title = '1-bit Normalized Lacunarity (jump) diff. from line:' + fileName
		#plot1(lacDif, title, saveas + 'dif')
		#plot2(xv, lacDif, title, 0, saveas + 'dif')
		#title = '1-bit Normalized Lacunarity (jump) grad.:' + fileName
		#plot1(lacGrad, title, saveas + 'grad')
		#plot2(xv[:-1], lacGrad, title, 0, saveas + 'grad')
		#title = '1-bit Lacunarity (jump) p n:' + fileName
		#plotMulLg(lacd2, lacdc2, xv, title, saveas + '_pn')
		# WriteToCSVFile(saveas + "log.csv", lx, llac, llacc, llacn)
		# WriteToCSVFile(saveas + ".csv",   xv, lacd2, lacdc2, lacdn2, lacDif, lacGrad)
		WriteToCSVFile(saveas + ".csv",   xv, lacd2, lacdc2, lacdn2)
	
if __name__ == '__main__':
	main()


# TODO as of 04/01/08

# bitplane gray lac
# threshold gray lac
# normalized gray lac

# thresholding methods isodata, adaptive, o
# strech, equalize
