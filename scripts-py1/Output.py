"""
Supporting sub routines for CSV files and ploting graphs
It uses following Python modules modules
by Dantha Manikka-Baduge, V.1.1, Feb 28 2008

Changed : Jul 22 2008
		: Jul 06 2011 changed file name from PlotCSV to Output
		              added new ploting library

TODO:
"""

import matplotlib.pyplot as plt


#
# Writing to files
#
def WriteToCSVFile(saveas, *set):
	"""Write to a file."""
	file = open(saveas, 'w')
	try:
		for datax in set:
			dd = ""
			for dx in datax:
				dd = dd + str(dx) + ", "
			file.write(dd)
			file.write("\n")
	finally:
		file.close()

#
# Plot a graph
#
def plot2(datax, datay, title, log = 1, saveas = None):
	"""Plot the result using matplotlib package."""
	extname = ''
	f1 = plt.figure(1)
	f1.clear()
	#lines = plt.scatter(datax, datay, linestyle = 'solid')
	plt.plot(datax, datay)
	#print lines
	ca = plt.gca()
	if log == 1:
		ca.set_xscale("log")
		plt.xlabel('log(r)')
		extname = '_log.png'
	else:
		plt.xlabel('r')
		extname = '.png'
	plt.ylabel('Lacanurity')
	figFilename = saveas + extname
	# Save what we just plotted as a color postscript file.
	if saveas != None:
		#plt.savefig(figFilename,format='png')
		plt.savefig(figFilename)
	#plt.show()

def plotMul(data, title, legend, log = 1, saveas = None):
	"""Plot the result using matplotlib package."""
	xv = range(len(data[0])+1)[1:]
	extname = ''
	f1 = plt.figure(1)
	f1.clear()
	#lines = plt.scatter(datax, datay, linestyle = 'solid')
	for l in range(len(data)):
		plt.plot(xv, data[l], label = legend[l])
		#plt.plot(xv, data[l], "bo", xv, data[l], "k", label = str(l), linewidth=1.0)
	plt.legend()
	#print lines
	ca = plt.gca()
	if log == 1:
		ca.set_xscale("log")
		plt.xlabel('log(r)')
		extname = '_log.png'
	else:
		plt.xlabel('r')
		extname = '.png'
	plt.ylabel('y')  # 'Lacanurity'

	# Save what we just plotted as a color postscript file.
	if saveas != None:
		figFilename = saveas + extname
		#plt.savefig(figFilename,format='png')
		plt.savefig(figFilename)
	plt.show()
