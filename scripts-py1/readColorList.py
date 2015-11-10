# color list
import Image
import ImageDraw
import numpy as np

clrMap9 = {
		'Red'			: [189,10,19],
		'Orange'		: [252,72,31],
		'Gold'			: [253,177,43],
		'Yellow'		: [216,253,62],
		'Lime Green'	: [130,253,128],
		'Cyan'			: [55,254,215],
		'Deep Sky Blue'	: [28,159,252],
		'Light Slate Blue'	: [15,56,251],
		'Blue'			: [8,25,188]
		}
clrMap = {   # 5
		'Red'			: [238,18,25],
		'Gold'			: [254,195,45],
		'Lime Green'	: [130,253,128],
		'Cyan'			: [30,178,252],
		'Blue'			: [10,33,237]
		}


def colorLUT(legendFilename, HistRange, sliced = 9):
	colorRowHeight = 12
	colorLst = {}
	# colorNameLst = ['Lime Green', 'Spring Green', 'Goldenrod', 'Gold', 'Yellow', 'Khaki', 'Beige',
	#				'Red', 'Sienna','Chocolate', 'Dark Orange',
	#				'Blue', 'Turquoise', 'Aquamarine',
	 #               'Orchid', 'Violet', 'Pink', 'Gray', 'Indian Red', 'Lawn Green']
	colorNameLst9 = ['Red', 'Orange', 'Gold', 'Yellow', 'Lime Green', 'Cyan', 'Deep Sky Blue', 'Light Slate Blue', 'Blue']
	colorNameLst = ['Red', 'Gold', 'Lime Green', 'Cyan', 'Blue']

	clrTotalList = ReadColorList()
	for clrSet in clrTotalList:
		#print clrTotalList[clrSet]
		#print "====="
		for v in clrTotalList[clrSet]:
			if v in colorNameLst:
				colorLst[v] = clrTotalList[clrSet][v]
				#print v, clrTotalList[clrSet][v]

	len1 = sliced # len(colorNameLst)
	#print colorLst, len(colorLst)
	colorLUT = np.zeros([len1 * colorRowHeight, 120, 3], dtype=np.uint8)
	for u in range(len1):
		#clr1 = colorLst[colorNameLst[8-u]]
		clr1 = clrMap[colorNameLst[len(colorNameLst)-1-u]]
		for v2 in range(colorRowHeight):
			for x in range(120):
				colorLUT[u * colorRowHeight + v2, x] = clr1

	if len(legendFilename) > 0:
		im1 = Image.fromarray(colorLUT)
		draw = ImageDraw.Draw(im1)
		for u in range(1,len(HistRange)):
			draw.text((8, u * colorRowHeight - colorRowHeight),'{0:.2f}'.format(HistRange[u-1]) + ' ~ ' + '{0:.2f}'.format(HistRange[u]))
		del draw
		im1.save(legendFilename, "TIFF")

	return [clrMap, colorNameLst] # colorLst

def allColors(filenamePath):
	colorRowHeight 	= 12
	colorRowWidth   = 100
	maxNoColors		= 0

	clrTotalList = ReadColorList()
	NoColorGrps	= len(clrTotalList)

	for clrSet in clrTotalList:
		if len(clrTotalList[clrSet]) > maxNoColors:
			maxNoColors = len(clrTotalList[clrSet])

	colorLUT = np.zeros([(maxNoColors+2) * colorRowHeight, colorRowWidth * NoColorGrps, 3], dtype=np.uint8)
	print (maxNoColors+2) * colorRowHeight, colorRowWidth * NoColorGrps
	g = -1
	for clrSet in clrTotalList:
		g += 1
		u = -1
		for clr in clrTotalList[clrSet]:
			u += 1
			clr1 = clrTotalList[clrSet][clr]
			for v in range(colorRowHeight):
				for x in range(colorRowWidth):
					colorLUT[u * colorRowHeight + v, g * colorRowWidth + x] = clr1

	if len(filenamePath) > 0:
		fileName = filenamePath + "allColors.tif"
		im1 = Image.fromarray(colorLUT)
		#draw = ImageDraw.Draw(im1)
		#for u in range(1,len(HistRange)):
		#	draw.text((8, u * colorRowHeight - colorRowHeight),'{0:.2f}'.format(HistRange[u-1]) + ' ~ ' + '{0:.2f}'.format(HistRange[u]))
		#del draw
		im1.save(fileName, "TIFF")

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
		if len(colorGrp) > 0:
			colorList[colorGrpName] = colorGrp
	return colorList

if __name__ == '__main__':
	#print colorLUT()
	allColors('./')