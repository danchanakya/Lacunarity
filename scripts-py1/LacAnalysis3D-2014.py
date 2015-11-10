import numpy as np
#import math

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ALVLACSubRoutines  import *
from Output 			import *

# Folders for input images and results
imgFolder1 = '/Users/danthac/Work/MSc/Lacunarity-MSMath-CSUCI/'
imgFolder = imgFolder1 + 'imagesSpaceData/'
resFolder1 = '/Volumes/D16SD_TD/Study/MSc/Results/'
resFolder  = resFolder1 + 't/'

scaleIt = True
lacCalc = False

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

def Lacunarity3D(data, mj = 0, clipType = 0):
	(sx, sy, sz) = data.shape
	minxyz = min(sx, sy, sz);
	lac = []

	dataAve = np.average(data)
	dataStd = np.std(data)
	print sx, np.sum(data), dataAve, dataStd
	lac1  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
	lac.append(lac1)

	LacData  = []

	# find lac from i=2 to msize
	for count in range(2, minxyz+1):
		if mj == 1:  # Jumping
			CS = [0]
			#CS = JumpingImg(data, sum1, count, clipType)
		else:
			CS = SlidingImg3D(data, sum1, count)
		dataAve = np.average(CS)
		dataStd = np.std(CS)
		lac1  = 1 + (dataStd * dataStd)/(dataAve * dataAve)
		lac.append(lac1)

	return lac

class Space3DData(object):
	dataset  = []
	datasetN = []
	dataNx   = []
	min1	 = [1000,1000,1000]
	max1	 = [-1000,-1000,-1000]
	def __init__(self, filename):
		firstline = 0
		self.cur  = 0
		self.file = open(filename, 'r')
		try:
			for line in self.file:
				if firstline == 1:
					ldata = line.split()
					self.dataset[self.cur,0] = float(ldata[0])
					if self.max1[0] < self.dataset[self.cur,0]: self.max1[0] = self.dataset[self.cur,0]
					if self.min1[0] > self.dataset[self.cur,0]: self.min1[0] = self.dataset[self.cur,0]
					self.dataset[self.cur,1] = float(ldata[1])
					if self.max1[1] < self.dataset[self.cur,1]: self.max1[1] = self.dataset[self.cur,1]
					if self.min1[1] > self.dataset[self.cur,1]: self.min1[1] = self.dataset[self.cur,1]
					self.dataset[self.cur,2] = float(ldata[2])
					if self.max1[2] < self.dataset[self.cur,2]: self.max1[2] = self.dataset[self.cur,2]
					if self.min1[2] > self.dataset[self.cur,2]: self.min1[2] = self.dataset[self.cur,2]
					self.cur = self.cur + 1
				else:
					self.dataset = np.zeros((int(line), 3), float)
					firstline = 1
		finally:
			self.file.close()
		#print self.min1, self.max1

	def __getitem__(self, (x, y)):
		return self.dataset[x % self.cur, y % 3]

	def NormalizeAndScale(self, scale):
		minv = min(np.ravel(self.dataset).tolist())
		#print min(self.dataset.tolist()),max(self.dataset.tolist())
		#print "minv = %f" % minv
		#print self.dataset
		self.dataset  = self.dataset - minv # * np.ones(self.dataset.shape)
		#print self.dataset
		#minv = min(np.ravel(self.dataset).tolist())
		maxv = max(np.ravel(self.dataset).tolist()) + 0.0000001
		print "minv = %f, maxv = %f" % (minv, maxv+minv)
		self.dataset  = self.dataset * scale/maxv
		self.datasetN = np.zeros((scale, scale, scale), float)
		self.dataNx   = []
		for v in self.dataset:
			x = np.floor(v[0])
			y = np.floor(v[1])
			z = np.floor(v[2])
			self.datasetN[x, y, z] = self.datasetN[x, y, z] + 1
			self.dataNx.append([x, y, z])
		#print min(np.ravel(self.dataset).tolist()), max(np.ravel(self.dataset).tolist())

	def GetArray(self):
		return [self.datasetN, np.array(self.dataNx)]

def main():
	filename = 'pac2fXYZ_3'
	fullfileName = imgFolder + filename + '.txt'

	sp = Space3DData(fullfileName)

	[dlen, a] = sp.dataset.shape
	print "File:", filename, "Shape:", [dlen, a]
	datasetNt = np.transpose(sp.dataset)
	xd = datasetNt[0]
	yd = datasetNt[1]
	zd = datasetNt[2]
	#xd = [sp.dataset[i][0] for i in range(dlen)]
	#yd = [sp.dataset[i][1] for i in range(dlen)]
	#zd = [sp.dataset[i][2] for i in range(dlen)]

	f0 = plt.figure()
	f0.clear()
	ax = f0.add_subplot(111, projection='3d')
	ax.scatter(xd, yd, zd, c='b', marker='.')
	f0.add_axes(ax)
	plt.title("3D data")
	figFilename = resFolder + filename + "_data_3d_full.png"
	plt.savefig(figFilename)
	#plt.show()
	plt.close(f0)

	if scaleIt:
		scale1 = 10.0
		sp.NormalizeAndScale(scale1)
		r0 = sp.GetArray()
		d1 = r0[0]
		d2 = r0[1]
		print "Final min, max, shape, sum", min(np.ravel(d1).tolist()), max(np.ravel(d1).tolist()), d1.shape, sum(np.ravel(d1))
		print "Final min, max, shape, count", min(np.ravel(d2).tolist()), max(np.ravel(d2).tolist()), d2.shape, len(np.ravel(d2).tolist())

		datasetNt = np.transpose(sp.datasetN)
		xd = datasetNt[0]
		yd = datasetNt[1]
		zd = datasetNt[2]

		f0 = plt.figure()
		f0.clear()
		ax = f0.add_subplot(111, projection='3d')
		ax.scatter(xd, yd, zd, c='b', marker='.')
		f0.add_axes(ax)
		plt.title("3D data")
		figFilename = resFolder + filename + "_data_3d_" + str(scale1) + ".png"
		plt.savefig(figFilename)
		#plt.show()
		plt.close(f0)

		datat = np.array(sp.dataNx)
		dataNt = np.transpose(datat)
		xd = dataNt[0]
		yd = dataNt[1]
		zd = dataNt[2]

		f0 = plt.figure()
		f0.clear()
		ax = f0.add_subplot(111, projection='3d')
		ax.scatter(xd, yd, zd, c='b', marker='.')
		f0.add_axes(ax)
		plt.title("3D data")
		figFilename = resFolder + filename + "_data_3d_Nx" + str(scale1) + ".png"
		plt.savefig(figFilename)
		#plt.show()
		plt.close(f0)

	if lacCalc:
		lac3d = Lacunarity3D(d1)
		saveas = resFolder + 'Lac_' + filename + "_" + str(scale1) + ".csv"
		WriteToCSVFile(saveas, [lac3d])
		x = range(1, len(lac3d)+1)
		lx = [log(x1) for x1 in x]
		lac3dLog = [log(x1) for x1 in lac3d]

		# Plot
		# Lacunarity
		f0 = plt.figure()
		f0.clear()
		plt.plot(x, lac3d, label='lacunarity - 3D')
		plt.legend()
		plt.title('Lacunarity 3D data')
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + filename + '_3d_' + str(scale1) + '.png'
		plt.savefig(figFilename)
		plt.close(f0)

		f0 = plt.figure()
		f0.clear()
		plt.plot(lx, lac3dLog, label='log lacunarity - 3D')
		plt.legend()
		plt.title('Lacunarity 3D data')
		plt.xlabel('window size(log)')
		plt.ylabel('lacunarity(log)')
		figFilename = resFolder + filename + '_log_3d_' + str(scale1) + '.png'
		plt.savefig(figFilename)
		plt.close(f0)

		# Normalize - Lacunarity
		f0 = plt.figure()
		f0.clear()
		lac3dN = NormalizedRoy(lac3d)
		plt.plot(x, lac3dN, label='Normalized lacunarity - 3D')
		plt.legend()
		plt.title('Normalized (Roy)')
		plt.xlabel('window size')
		plt.ylabel('lacunarity')
		figFilename = resFolder + filename + '_3d_NormM_' + str(scale1) + '.png'
		plt.savefig(figFilename)
		plt.close(f0)

if __name__ == '__main__':
	main()
