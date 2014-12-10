#! /usr/bin/python

"""
Average Local Variance program for 2D data
It uses Python PIL, Psyco and select modules
by Dantha Manikka-Baduge, V.1.0, Feb 28 2008

   Type 0 - average all
        1 - clip out smaller boxes at edges
        2 - fill out with zero 
        
   mj   0 - moving
   		1 - jumping     
"""

from random   import choice
from colorsys import rgb_to_yiq
import Image 	# Python PIL
import ImageOps
# Give proper credits here and website
# from select import median
# import psyco; psyco.full()
from math  import *
from numpy import *
import sys
from PlotCSV import *

# If the package has been installed correctly, this should work:
import Gnuplot, Gnuplot.funcutils, Gnuplot.PlotItems

def WriteToCSVFile(datax, datay, saveas):
	"""Write to a file."""
	file = open(saveas, 'w')
	try:
		dd = ""
		for dx in datax:
			dd = dd + str(dx) + ", "
		file.write(dd)
		file.write("\n")
		if datay != None:
			dd = ""
			for dy in datay:
				dd = dd + str(dy) + ", "
			file.write(dd)
			file.write("\n")
	finally:
		file.close()

def plot3D(data):
	g = Gnuplot.Gnuplot(debug=1)
	g('set parametric')
	#g('set data style lines')
	#g('set hidden')
	#g('set contour base')
	g.title('3D Plot')
	g.xlabel('x')
	g.ylabel('y')
	g.splot(Gnuplot.Data(data, with='linesp', inline=1))
	del g
	
def plot3Dgrid(data, sz=120):
	g = Gnuplot.Gnuplot(debug=1)
	g('set parametric')
	#g('set data style lines')
	#g('set hidden')
	#g('set contour base')
	g.title('3D Plot')
	g.xlabel('x')
	g.ylabel('y')
	x = range(sz)
	y = range(sz)
	g.splot(Gnuplot.GridData(data,x,y, binary=0, inline=1))
	del g

def plot2(datax, datay, title, saveas = None):
	"""Plot the result using Gnuplot package."""

	g = Gnuplot.Gnuplot(debug=1)

	d = Gnuplot.Data(datax, datay)
	g.title(title)
	#g('set data style lines')
	g.xlabel('log(r)')
	g.ylabel('Lacanurity')
	g.plot(d)

	# Save what we just plotted as a color postscript file.
	if saveas != None:
		g.hardcopy(saveas + '.ps', enhanced=1, color=1)

def plotMul(data1, data2, title, saveas = None):
	"""Plot the result using Gnuplot package."""

    # A straightforward use of gnuplot.  The `debug=1' switch is used
    # in these examples so that the commands that are sent to gnuplot
    # are also output on stderr.
	g = Gnuplot.Gnuplot(debug=1)

	g.title(title)
	g('set data style lines') # give gnuplot an arbitrary command
	#g('set key spacing 1.5')
	#g('set key box')
	g.xlabel('aggregation level')
	g.ylabel('ALV')
	g.plot(Gnuplot.Data(data1, title='Jump'), Gnuplot.Data(data2, title='Move'))
	if saveas != None:
		g.hardcopy(saveas + '.ps', enhanced=1, color=1)
		
def plot(data, title, saveas = None):
	"""Plot the result using Gnuplot package."""

    # A straightforward use of gnuplot.  The `debug=1' switch is used
    # in these examples so that the commands that are sent to gnuplot
    # are also output on stderr.
	g = Gnuplot.Gnuplot(debug=1)

	g.title(title)
	g('set data style linespoints') # give gnuplot an arbitrary command
	g.plot(data)
	if saveas != None:
		g.hardcopy(saveas + '.ps', enhanced=1, color=1)

def JumpingImg(datarray, func, w = 2, clipType = 2):
	(sx, sy) = datarray.shape
	(ma, na) = {
		0: lambda : (sx, sy),
		1: lambda : ((sx//w)*w, (sy//w)*w),
		2: lambda : (ceil(sx/w)*w, ceil(sy/w)*w)
				}[clipType]()
	(newm, newn) = (int(ceil(ma/w)), int(ceil(na/w))) # check w non float issue
	OutIm = zeros((newm, newn), float)
	for y in range(newn):
		for x in range(newm):
			OutIm[x,y] = func(datarray[x*w:x*w+w,y*w:y*w+w])
	return OutIm
    
def SlidingImg(datarray, func, w = 2):
	(sx, sy) = datarray.shape
	OutIm = zeros((sx-w + 1,sy - w + 1), float) 
	for y in range(sy - w + 1): 
		for x in range(sx - w + 1):
			OutIm[x,y] = func(datarray[x:x+w,y:y+w])
	return OutIm        
    
def ALV(datarray, w = 2, mj = 1, clipType = 2):
	SaveAggImgs = 1
	funcused = std # var
	ALVData = []
	(sx, sy) = datarray.shape
	maxA = max(sx, sy) // w
	# print maxA
	if mj == 1:
		d1 = JumpingImg(datarray, funcused)
	else:
		d1 = SlidingImg(datarray, funcused)
	ALVData.append(average(d1))
	for x in [x+1 for x in range(maxA)][1:]:
		d2 = JumpingImg(datarray, average, x)
		#if x == 19:
		#	plot3Dgrid(d2, 6)
		if SaveAggImgs == 1:
			d3 = array(d2, dtype='B')
			agIm = Image.fromarray(d3)
			agIm.save("aggImgs/AggData_" + str(x) + ".tif","TIFF")
		#print x
		#print d2.shape
		if mj == 1:
			d3 = JumpingImg(d2, funcused)
		else:
			d3 = SlidingImg(d2, funcused)
		#print d3.shape
		ALVData.append(average(d3))	
	return ALVData
	
def Lacunarity(datarray, norm = 0):
	"""Generate the lacunarity data for the give datarray
	   norm = 1 normalized data
	"""
	w = 1
	LacData = []
	LacDataN = []
	(sx, sy) = datarray.shape
#	dataNeg = [not x for x in datarray]
#	print dataNeg
	# find LacData(1)
	lac = 1 + (std(datarray)*std(datarray))/(average(datarray)*average(datarray));
#	lacC = 1 + (std(dataNeg)*std(dataNeg))/(average(dataNeg)*average(dataNeg));
#	lacF = 2 - 1/lac - 1/lacC;
	LacData.append(lac);
#	LacDataN.append(lacF);
	minxy = min(sx, sy);

	# find lac from i=2 to msize
	for count in [x+1 for x in range(minxy)][1:]:
		CS = SlidingImg(datarray, sum, count)
		meanscore = average(CS);
		stdscore = std(CS);
		lac = 1 + ((stdscore*stdscore)/(meanscore*meanscore));
#		CSC = SlidingImg(dataNeg, sum, count)
#		meanscore = average(CSC);
#		stdscore = std(CSC);
#		lacC = 1 + ((stdscore*stdscore)/(meanscore*meanscore));
#		lacF = 2 - 1/lacN - 1/lacC;
		LacData.append(lac)
#		LacDataN.append(LacF)
	return LacData	

def main():
	ALVJ = 1
	ALVM = 1
	LAC  = 0
	
	if len(sys.argv) < 2:
		print "usage ALVnLac3D.py <name of the image(.tif) file>"
		return
	
	dataFoler = 'data1//'	 # set2
	fileName = sys.argv[1] # 'ALV_Samp_A'
	
#	inim = Image.open("ALV_Samp_A.tif")
#	inim = Image.open("honeycombb.tif")
#	inim = Image.open("dla.gif")

	inim = Image.open(dataFoler + fileName + '.tif') # convert("L")

	inar1 = asarray(inim)
	#print inar1.shape
	inar = inar1[:,:] #,0]
	# print inar
	
	# strech
	scale = 255.0
	minv = min(ravel(inar).tolist())
	inar = inar - minv * ones(inar.shape)
	maxv = max(ravel(inar).tolist()) #+ 0.0000001
	inar2 = inar * scale/maxv
	inar = array(inar2, dtype='B')
	# plot3Dgrid(inar)
	#agIm1 = Image.fromarray(inar, mode='L')
	agIm1 = Image.frombuffer('L', (120,120), inar, "raw", 'L', 0, 1)
	agIm1.save("aggImgs/AggData_1.tif","TIFF")
	if ALVJ == 1:
		ald0 = ALV(inar, mj = 0)
		# print ald0
		saveas = dataFoler + 'ALVJ_' + fileName
		title = 'ALV Jumping (2x2) for ' + fileName
		if ALVM != 1:
			plot(ald0, title, saveas)
		WriteToCSVFile(ald0, None, saveas + ".csv")
	if ALVM == 1:	
		ald1 = ALV(inar, mj = 1)
		# print ald1
		saveas = dataFoler + 'ALVM_' + fileName
		title = 'ALV Moving (2x2) for ' + fileName
		if ALVJ != 1:
			plot(ald1, title, saveas)
		WriteToCSVFile(ald1, None, saveas + ".csv")

	if ALVJ == 1 and ALVM == 1:
		title = 'ALV Jumping & Moving (2x2) for ' + fileName
		plotMul(ald0, ald1, title)

#	inim = Image.open("X.tif")
#	inar = asarray(inim)
	if LAC == 1:
		print inar.shape
		# Pick image data
		inar1 = inar[:,:] #,0]
		# Converto binary
		TH = 128 # 114, 128
		inar2 = inar1 # > TH
#		print inar2
		lacd = Lacunarity(inar2)
#		print lacd
		llac = log(lacd)
		lx = log(range(len(llac)+1)[1:])
		saveas = dataFoler + 'LAC1_' + fileName
		title = '1-bit Lacunarity'
		# plot2(lx, llac, title, saveas)
		WriteToCSVFile(lx, llac, saveas + ".csv")
	
if __name__ == '__main__':
	main()

# bitplane gray lac
# threshold gray lac
# normalized gray lac


# ALV  - filename, size of window, moving or jumping, 