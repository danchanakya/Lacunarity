#! /usr/bin/python

"""
Supporting sub routines for ALV and Lacunarity calculations
It uses Python PIL and select modules
by Dantha Manikka-Baduge, V.1.1, Feb 28 2008

Changed : Jul 22 2008

"""

#from random   import choice
#from colorsys import rgb_to_yiq

# Python PIL
#import Image
#import ImageOps

import numpy as np

from math  import *
#from numpy import *

#import time
#import sys
#import os.path

def JumpingImg(datarray, func, w = 2, clipType = 0):
	(sx, sy) = datarray.shape
	(ma, na) = {
		0: lambda : (sx//w, sy//w),
		1: lambda : (sx//w + (sx % w >= 1 and 1 or 0), sy//w + (sy % w >= 1 and 1 or 0))
				}[clipType]()
	OutIm = np.zeros((ma, na), float)
	(mab, nab) = (sx % w, sy % w)
	if clipType == 1:
		nal = na - (sx % w >= 1 and 1 or 0)
		mal = ma - (sy % w >= 1 and 1 or 0)
	else:
		nal = na
		mal = ma
	for y in range(nal):
		for x in range(mal):
			OutIm[x,y] = func(datarray[x*w:x*w+w,y*w:y*w+w])
	# print sx, sy, ma, na, mab, nab, mal, nal
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

# Moving / Sliding window
def SlidingImg(datarray, func, w = 2):
	(sx, sy) = datarray.shape
	if min(sx - w + 1, sy - w + 1) > 0:
		OutIm = np.zeros((sx - w + 1, sy - w + 1), float)
		for y in range(sy - w + 1):
			for x in range(sx - w + 1):
				OutIm[x, y] = func(datarray[x:x + w, y:y + w])
	else:
		OutIm = np.zeros((1, 1), float)
		OutIm[0, 0] = datarray[0, 0]

	return OutIm


# Moving / Sliding window
def SlidingImgGray(datarray, func, w = 2):
	(sx, sy) = datarray.shape
	if min(sx-w + 1,sy - w + 1) == 0:
		OutIm = np.zeros((1,1,1), float)
		OutIm[0,0,0] = datarray[0,0]
	else:
		OutIm = zeros((sx-w + 1,sy - w + 1,255), float)
		for y in range(sy - w + 1):
			for x in range(sx - w + 1):
				for z in range(255 - w + 1):

					dA1 = reshape(datarray[x:x+w,y:y+w], w*w)
					dArray  = [z < val and val <=(z+w) for val in dA1]
					dArrayN = array(dArray, dtype='B')
					#print dA1

					#print dArrayN

					OutIm[x,y,z] = func(dArrayN)
	return OutIm

# this function needs to be updated ...
def JumpingImg3D(datarray, func, w = 2, clipType = 2):
	(sx, sy, sz) = datarray.shape
	(ma, na, pa) = {
		0: lambda : (sx, sy, sz),
		1: lambda : ((sx//w)*w, (sy//w)*w, (sz//w)*w),
		2: lambda : (ceil(sx/w)*w, ceil(sy/w)*w, ceil(sz/w)*w)
				}[clipType]()
	(newm, newn, newp) = (int(ceil(ma/w)), int(ceil(na/w)), int(ceil(pa/w))) # check w non float issue
	OutIm = zeros((newm, newn, newp), float)
	for z in range(newp):
		for y in range(newn):
			for x in range(newm):
				OutIm[x,y,z] = func(datarray[x*w:x*w+w,y*w:y*w+w,z*w:z*w+w])
	return OutIm

def sum1(data):
	return sum(np.ravel(data))

def SlidingImg3D(datarray, func, w = 2):
	(sx, sy, sz) = datarray.shape
	if min(sx - w + 1, sy - w + 1, sz - w + 1) > 0:
		OutIm = np.zeros((sx - w + 1,sy - w + 1, sz - w + 1), float)
		for z in range(sz - w + 1):
			for y in range(sy - w + 1):
				for x in range(sx - w + 1):
					d0 = datarray[x:x+w,y:y+w,z:z+w]
					OutIm[x,y,z] = func(d0)
	else:
		OutIm = zeros((1, 1, 1), float)
		OutIm[0, 0, 0] = datarray[0, 0, 0]

	return OutIm

# piecewise gradient
def DataGradient(datarray):
	lenAr = len(datarray)
	DataGradArray = []
	for pos in range(lenAr)[1:]:
		DataGradArray.append(datarray[pos] - datarray[pos-1])
	return DataGradArray

# distance to the straight line drawn from 1st point to last point
def DataDifference(datarray):
	lenAr = len(datarray)
	DataDiffArray = []
	Grad = (datarray[lenAr-1] - datarray[0])/lenAr
	Val = datarray[0]
	for pos in range(lenAr):
		DataDiffArray.append(Val + pos * Grad - datarray[pos])
	return DataDiffArray