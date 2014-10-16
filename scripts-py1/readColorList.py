# color list
import Image
import numpy as np

def colorLUT(legendFilename):
	colorLst = {}
	colorNameLst = ['Red', 'Blue', 'Aquamarine', 'Lawn Green', 'Dark Orange', 'Beige', 'Indian Red', 'Lime Green', 'Gold']

	clrTotalList = ReadColorList()
	for clrSet in clrTotalList:
		for v in clrTotalList[clrSet]:
			if v in colorNameLst:
				colorLst[v] = clrTotalList[clrSet][v]
				#print v, clrTotalList[clrSet][v]

	colorLUT = np.zeros([len(colorNameLst) * 10, 100, 3], dtype=np.uint8)
	for u in range(len(colorNameLst)):
		for v2 in range(10):
			for x in range(100):
				colorLUT[u * 10 + v2, x] = colorLst[colorNameLst[u]]

	if len(legendFilename) > 0:
		Image.fromarray(colorLUT).save(legendFilename, "TIFF")

	return colorLst

def ReadColorList():
	colorGrpName = "None"
	colorList = {}
	with open('colorList1.txt', 'r') as file1:
		rl = file1.readlines()
		colorGrp = {}
		for rl1 in rl:
			lcomp = rl1[:-1].split('\t')
			if len(lcomp) == 1 and len(lcomp[0]) > 0:
				if len(colorGrp) > 0:
					colorList[colorGrpName] = colorGrp
				colorGrp = {}
				colorGrpName = lcomp[0]
			if len(lcomp) == 3:
				rgb = lcomp[1].split('-')
				if len(rgb) == 3:
					colorGrp[lcomp[0]] = [int(x) for x in rgb]

	return colorList

if __name__ == '__main__':
	print colorLUT()