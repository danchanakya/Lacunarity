import numpy as np
import math

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ALVLACSubRoutines  import *
from Output 			import *

# Folders for input images and results
imgFolder = '../images1/'
resFolder = '../results/respy13/'
datFolder = '../spacedata/'

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
	def __init__(self, filename):
		firstline = 0
		self.cur  = 0
		self.file = open(filename, 'r')
		try:
			for line in self.file:
				if firstline == 1:
					ldata = line.split()
					self.dataset[self.cur,0] = float(ldata[0])
					self.dataset[self.cur,1] = float(ldata[1])
					self.dataset[self.cur,2] = float(ldata[2])
					self.cur = self.cur + 1
				else:
					self.dataset = np.zeros((int(line), 3), float)
					firstline = 1
		finally:
			self.file.close()

	def __getitem__(self, (x, y)):
		return self.dataset[x % self.cur, y % 3]

	def NormalizeAndScale(self, scale):
		minv = min(np.ravel(self.dataset).tolist())
		self.dataset  = self.dataset - minv # * np.ones(self.dataset.shape)
		maxv = max(np.ravel(self.dataset).tolist()) + 0.0000001
		self.dataset  = self.dataset * scale/maxv
		self.datasetN = np.zeros((scale, scale, scale), float)
		for v in self.dataset:
			x = math.floor(v[0])
			y = math.floor(v[1])
			z = math.floor(v[2])
			#print x, y, z
			self.datasetN[x,y,z] = self.datasetN[x,y,z] + 1

	def GetArray(self):
		return self.datasetN

def main():
	filename = 'pac2fXYZ_3'
	fullfileName = datFolder + filename + '.txt'

	sp = Space3DData(fullfileName)

	[dlen, a] = sp.dataset.shape
	print [dlen, a]
	xd = [sp.dataset[i][0] for i in range(dlen)]
	yd = [sp.dataset[i][1] for i in range(dlen)]
	zd = [sp.dataset[i][2] for i in range(dlen)]

	scale1 = 25.0
	sp.NormalizeAndScale(scale1)
	d1 = sp.GetArray()
	print d1.shape, sum(np.ravel(d1))
	print min(np.ravel(d1).tolist()), max(np.ravel(d1).tolist())

	r1 = sp.datasetN.shape
	print r1
	dlen = r1[0]
	xd = [sp.datasetN[i][0] for i in range(dlen)]
	yd = [sp.datasetN[i][1] for i in range(dlen)]
	zd = [sp.datasetN[i][2] for i in range(dlen)]

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
	plt.title("Lacunarity 3D data")
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + filename + "_3d_" + str(scale1) + ".png"
	plt.savefig(figFilename)
	plt.close(f0)

	f0 = plt.figure()
	f0.clear()
	plt.plot(lx, lac3dLog, label='log lacunarity - 3D')
	plt.legend()
	plt.title("Lacunarity 3D data")
	plt.xlabel('window size(log)')
	plt.ylabel('lacunarity(log)')
	figFilename = resFolder + filename + "_log_3d_" + str(scale1) + ".png"
	plt.savefig(figFilename)
	plt.close(f0)

	# Normalize - Lacunarity
	f0 = plt.figure()
	f0.clear()
	lac3dN = NormalizedMalhi(lac3d)
	plt.plot(x, lac3dN, label='Normalized lacunarity - 3D')
	plt.legend()
	plt.title("Normalized (Malhi)")
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + filename + "_3d_NormM_" + str(scale1) + ".png"
	plt.savefig(figFilename)
	plt.close(f0)

if __name__ == '__main__':
	main()
