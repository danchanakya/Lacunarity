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

	#[lac1, lacR1, lacLeg1, thresh] = Lac_Fixed_Threshold(idata, imagename, mj)
	#thresh_type = "fixed"
	#thresh_title_part = 'fixed Threshold ' + str(thresh)

	#[lac1, lacR1, lacLeg1, thresh] = Lac_Adaptive_Threshold(idata, imagename, mj)
	#thresh_type = "adaptive"
	#thresh_title_part = 'Adaptive Threshold'

	x = range(1, len(lac1)+1)
	retDict[thresh_type + "_threshold"] = thresh

	# Save Data to CSV file
	dataFilename = resFolder + imagename + "_BIN_Lac_threshold-" + str(thresh) + str(mvjmp) + ".csv"
	SaveLacDataInd(dataFilename, x, [lac1, lacR1], lacLeg1)
	#SaveLacData(dataFilename, x, [lac1, lacR1], [lacLeg1])
	retDict["lacBINCSV"] = imagename + "_BIN_Lac_threshold-" + str(thresh) + str(mvjmp) + ".csv"

	# Lacunarity
	f0 = plt.figure()
	f0.clear()
	plt.plot(x, lac1, label=lacLeg1)
	plt.legend()
	plt.title(thresh_title_part + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "_threshold1-" + str(thresh) + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINPlot"] = imagename + "_" + thresh_type + "_threshold1-" + str(thresh) + str(mvjmp) + ".png"

	# Inv Lacunarity
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

	# Log Log - Lacunarity
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

	# Normalize - Lacunarity
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

	# Normalize - Lacunarity
	f4 = plt.figure()
	f4.clear()
	y1 = NormalizedMalhi(lac1)
	p1 = plt.plot(x, y1, label=lacLeg1 + " - M")
	plt.legend()
	plt.title(thresh_title_part + ' Normalized (Malhi)'  + mvjmp)
	plt.xlabel('window size')
	plt.ylabel('lacunarity')
	figFilename = resFolder + imagename + "_" + thresh_type + "NormM" + str(mvjmp) + ".png"
	plt.savefig(figFilename)
	retDict["lacBINNormPlot"] = imagename + "_" + thresh_type + "NormM" + str(mvjmp) + ".png"

	# Normalize - Lacunarity
	f5 = plt.figure()
	f5.clear()
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

	plt.close(f0); plt.close(f1); plt.close(f2); plt.close(f3); plt.close(f4); plt.close(f5)

	return retDict


def Lac_Adaptive_Threshold(idata, savePics = '', mj = 0):
	dt1 = datetime.datetime.now()

	block_size  = 40
	threshold   = 0
	thresh_type = "adaptive"
	trData1     = threshold_adaptive(idata, block_size, offset = 10)

	trData2  = np.array(trData1, dtype='B')

	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_' + thresh_type + '_' + str(threshold) + '.tif',"TIFF")
	lac1     = GetLacunarity(trData2, mj)
	lacLeg   = thresh_type + ' = ' + str(threshold)

	trDataR10  = PIL.ImageOps.mirror(Image.fromarray(idata))
	trDataR11  = inar = np.asarray(trDataR10)
	trDataR1   = threshold_adaptive(trDataR11, block_size, offset=10)

	trDataR2   = np.array(trDataR1, dtype='B')

	if len(savePics) > 0:
		Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_' + thresh_type + '_' + str(threshold) + 'R.tif',"TIFF")
	lacR1     = GetLacunarity(trDataR2, mj)
	lacLegR   = 'R ' + thresh_type + ' = ' + str(threshold)

	dt2 = datetime.datetime.now()
	print 'Lac_' + thresh_type + '_Threshold      = ', dt2-dt1

	return [lac1, lacR1, lacLeg, threshold]

def Lac_Fixed_Threshold(idata, savePics = '', mj = 0, threshold = 128):
	dt1 = datetime.datetime.now()

	thresh_type = "fixed"
	#print threshold

	trData1  = idata <= threshold
	trData2  = np.array(trData1, dtype='B')

	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_' + thresh_type + '_' + str(threshold) + '.tif',"TIFF")
	lac1     = GetLacunarity(trData2, mj)
	lacLeg   = thresh_type + ' = ' + str(threshold)

	trDataR1  = idata > threshold
	trDataR2  = np.array(trDataR1, dtype='B')

	if len(savePics) > 0:
		Image.fromarray(trDataR2 * 255).save(resFolder + savePics +'_' + thresh_type + '_' + str(threshold) + 'R.tif',"TIFF")
	lacR1     = GetLacunarity(trDataR2, mj)
	lacLegR   = 'R ' + thresh_type + ' = ' + str(threshold)

	dt2 = datetime.datetime.now()
	print 'Lac_' + thresh_type + '_Threshold      = ', dt2-dt1

	return [lac1, lacR1, lacLeg, threshold]

def Lac_ISODATA_Threshold(idata, savePics = '', mj = 0):
	dt1 = datetime.datetime.now()

	thresh_type = "isodata"
	threshold = threshold_isodata(idata)
	#thresh_type = "fixed"
	#threshold = 128
	#print threshold

	trData1  = idata <= threshold
	trData2  = np.array(trData1, dtype='B')

	if len(savePics) > 0:
		Image.fromarray(trData2 * 255).save(resFolder + savePics + '_' + thresh_type + '_' + str(threshold) + '.tif',"TIFF")
	lac1     = GetLacunarity(trData2, mj)
	lacLeg   = thresh_type + ' = ' + str(threshold)

	trDataR1  = idata > threshold
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

# Obsolete
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


# Obsolete
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

......


	#retDict = Calc_LacImageOtsu(imagename, idata, winds)
	#testData["LacBINSeg"] = retDict
	#retDict = Calc_LacImageOtsu(imagename, idata, winds, True)
	#testData["LacBINSegColor"] = retDict
