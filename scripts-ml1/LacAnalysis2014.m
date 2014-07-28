function LacAnalysis2014()
    imagename = 'honeycomb2A';

    imgFolder = ['../images1/'];
	resFolder = [imgFolder, 'resml1/'];
	%% Load Image
	fileName = strcat(imgFolder, imagename,'.tif');
	idata = imread(fileName);
    
  Cal_BinLac(imagename, idata, resFolder);
    
	%imagename = "test"
	%idata = np.array([[5,4,8,7,9],[12,12,11,8,12],[11,12,9,10,5],[1,2,5,3,11],[5,9,2,7,10]])
	%Cal_GrayScaleLacunarity(imagename, idata)

%% ---------------------------------------------------------------------
function Cal_BinLac(imagename, idata, resFolder)
	% Image Statistics
	[sx sy]	= size(idata);
	minxy	= min(sx, sy);
	maxVal	= max(max(idata));
	minVal	= min(min(idata));
	bins	= maxVal - minVal;
	sizeT	= sx * sy;

	% Max occupancy
	P = sum(sum(idata)) * 1.0/sizeT;

	idata1D = reshape(idata,1,[]);
	% Get Histogram for all possible values
	[dataHst, nHst] = hist(double(idata1D), double(bins));

	d1 = ['P = ', num2str(P), ' sx = ', num2str(sx), ' sy = ', num2str(sy), ' max = ', num2str(maxVal), ' min = ', num2str(minVal)];
	disp(d1);

   [lacLeg, lacData, lacDataR] = Lac_Histogram_Threshold(resFolder, imagename, idata, dataHst, nHst);

   [lacLeg1, lac1, lacR1, thresh] = Lac_Otsu_Threshold(resFolder, imagename, idata);

	dataFilename = strcat(resFolder, imagename);
	SaveLacData(dataFilename, [lacData; lacDataR; lac1; lacR1], [lacLeg; lacLeg1])

    %% Plot data
    
    x = 1:length(lacData(1,:));
	f1 = figure;
	plot(x, lacData);
	legend(lacLeg);
	%set(gcf,'position',[10 300 512 384]);
	legend('boxoff');
	legend('show');

	figFilename = strcat(resFolder, imagename, '_1.png');
	%savefig(figFilename);
    	saveas(f1, figFilename,'png');
    
	f1 = figure;
	plot(log(x),log(lacData));
	legend(lacLeg);
	%set(gcf,'position',[10 300 512 384]);
	legend('boxoff');
	legend('show');

	figFilename = strcat(resFolder, imagename, '_log1.png');
	%savefig(figFilename);
    	saveas(f1, figFilename,'png');
    
	f1 = figure;
	lx = log(x);
	for n = 1:length(lacData)
		y1 = NormalizedMahil(lacData(n)); % [log(y)/log(lacData[n][0]) for y in lacData[n]]
		plot(lx, y1); %, label=lacLeg(n));
	end
	figFilename = strcat(resFolder, imagename, '_log1norm.png');
	%savefig(figFilename);
    	saveas(f1, figFilename,'png');
    
 	figure;
	plot(x, lac1);
	msg = ['Threshold : Otsu'];
	title (strcat(msg));
	%label='otsu=' + str(threshold)
	figFilename = strcat(resFolder, imagename, '_otsu1.png');
	saveas(gcf, figFilename, 'png');

	figure;
	plot(log(x),log(lac1));
	% label='otsu=' + str(threshold))
	figFilename = strcat(resFolder, imagename, '_otsulog1.png');
	saveas(gcf, figFilename, 'png');

	figure;
	lx = log(x);
	y1 = NormalizedMahil(lac1);   %  log-log
	plot(lx, y1); %, label=lacLeg + "-M")
	y2 = NormalizedRoy(lac1);     %  x y
	plot(lx, y2); %, label=lacLeg + "-R")
	y3 = NormalizedHenebry(lac1, lacR1, 1);  % log log
	plot(lx, y3); % , label=lacLeg + "-P")

	figFilename = strcat(resFolder, imagename, '_otsulog1norm.png');
	saveas(gcf, figFilename, 'png');
        
	% 20% and %80 and reverse 20%
	f1 = figure;
	for n = 1:length(lacData)
		if (n == 1 || n == 7)
			y1 = NormalizedMahil(lacData(n)); % [log(y)/log(lacData[n][0]) for y in lacData[n]]
			plot(lx, y1); %, label=lacLeg[n])
		end
		if (n == 1)
			y1 = NormalizedMahil(lacDataR(n)); % [log(y)/log(lacDataR[n][0]) for y in lacDataR[n]]
			plot(lx, y1); %, label=lacLeg(n));
		end
	end
	figFilename = strcat(resFolder, imagename, '_log1norm_2080.png');
    saveas(f1, figFilename,'png');
    %savefig(figFilename);

%------------------------------------------------------------------------------
function SaveLacData(fileName, datax, datay, legend)
	%f1 = strcat(fileName, '_leg_1.csv');
	%csvwrite(f1, legend);
	%f2 = strcat(fileName, '_dat_x.csv');
	%csvwrite(f2, datax);
	%f3 = strcat(fileName, '_dat_y.csv');
	%csvwrite(f3, datay);
	%dlmwrite(fileName, lac, ','); 	% .txt
    %save(fileName, 'lac');   		% .mat
    
    save(fileName, 'datax', 'datay', 'legend');   		% .mat

%------------------------------------------------------------------------------
function lacNorm = NormalizedRoy(idata)
	lacNorm = [];
	maxVal	= max(max(idata));
	minVal	= min(min(idata));
	for i = 1:length(idata)
		lacNorm = [lacNorm ((idata(i)-minVal)/(maxVal-minVal))];
	end

%------------------------------------------------------------------------------
function lacNorm = NormalizedMahil(idata)
	lacNorm = [];
	for i = 1:length(idata)
		lacNorm = [lacNorm log(idata(i))/log(idata(1))];
	end

%------------------------------------------------------------------------------
function lacNorm = NormalizedHenebry(data, dataR)
	lacNorm = [];
	for i = 1:length(data)
		nl = 2 - 1/data(i) - 1/dataR(i);
		lacNorm = [lacNorm nl];
	end

%------------------------------------------------------------------------------
function lac = GetLacunarity(trData)
	[sx sy]	= size(trData);
	minxy	= min(sx, sy);
	% find lac(1)
	meanscore = mean(trData(:));
	stdscore  = std(trData(:));
	lac(1) = 1 + ((stdscore*stdscore)/(meanscore*meanscore));

	% use NLFILTER with (new) SUM2: find lac from i=2 to msize
	for count=2:minxy
		CS = slidingBox(trData,[count count],'sum2');
		meanscore = mean(CS(:));
		stdscore  = std(CS(:));
		lac(count)= 1 + ((stdscore*stdscore)/(meanscore*meanscore));
	end

%------------------------------------------------------------------------------
function [lacLeg, lacData, lacDataR] = Lac_Histogram_Threshold(resFolder, imagename, idata, dataHst, nHst)
	[sx sy]	= size(idata);
	%minxy	= min(sx, sy);

	chkPnt	 = 0.1;
	p2		 = 0.0;
	lacData  = [];
	lacDataR = [];
	lacLeg   = [];
	hpLen 	 = length(dataHst);
	p2acc	 = zeros(1,hpLen);

	dataHstProb = dataHst / sum(dataHst);
	ndx = 1;
	for h = 0:hpLen-1;
		p2 = p2 + dataHstProb(h+1);
		if (p2 >= chkPnt)
			disp (['P% = ', num2str(chkPnt)]);
			lacLeg = [lacLeg num2str(chkPnt)];
			chkPnt   = chkPnt + 0.1;
			trData1  = idata <= nHst(h+1);
			lac1     = GetLacunarity(trData1);
			lacData = [lacData;lac1];

			trDataR1  = idata > nHst(h+1);
			lacR1     = GetLacunarity(trDataR1);
			lacDataR = [lacDataR;lacR1];
			ndx = ndx + 1;
		end
		if p2 > 0.91 break; end;
		p2acc(h+1) = p2;
	end

%------------------------------------------------------------------------------
function SlidingImgG(idata, count)
	[sx sy]	= size(idata);

	maxVal	= max(np.ravel(idata))
	minVal	= min(np.ravel(idata))
	bins	= maxVal - minVal + 1
	lev1	= [0.25:0.25:1] * bins;  % Different form the paper Q = 4 instead of 5
	disp('levels = ', lev1);
    disp('sx = ', sx, ' sy = ', sy, ' max = ', maxVal, ' min = ', minVal);

	if (min(sx-w + 1,sy - w + 1) > 0)
		Out1 = []
		for x = 1:(sx - w + 1)
			%print 'x = ', x
			for y = 1:(sy - w + 1)
				%print 'y = ', y
				%dw = deepcopy(idata[x:x+w,y:y+w]);
				imgs1 =  SplitImageData(dw, lev1);
				Out1.append(imgs1);
            end
        end
		outArray1 = np.array(Out1)
		outArray2 = outArray1 * outArray1 / sum(Out1)
		outArray3 = outArray1 * outArray2
		ML  = sum(outArray2);
		ML2 = sum(outArray3);
		Lac = (ML2 - ML*ML)/(ML*ML);
        SlidingImgG = Lac;
    end

%------------------------------------------------------------------------------
function Cal_GrayScaleLacunarity(resFolder, imagename, idata)
	[sx sy]	= size(idata);
	minxy	= min(sx, sy);

	lac(1) = 0;

	% find lac from count = 2 to msize
	%for count 1:minxy
	%	disp(["Window = ", num2str(count), " of ", num2str(minxy)]);
	%	lac1  = SlidingImgG(idata, count)
	%	lac(count) = lac1
	%end
	Cal_GrayScaleLacunarity = lac;

%------------------------------------------------------------------------------
function [lacLegend, lac1, lacR1, threshold] = Lac_Otsu_Threshold(resFolder, imagename, idata)
	threshold = 109; %graythresh(idata);
	trData1  = idata <= threshold;
	lac1     = GetLacunarity(trData1);
	lacLeg   = strcat('otsu=', num2str(threshold));
	trDataR1 = idata > threshold;
	lacR1    = GetLacunarity(trDataR1);
	lacLegR  = strcat('Inv otsu=', num2str(threshold));

    lacLegend = [lacLeg, lacLegR];


%plot(log(x),log(lac))
%title (strcat(['Gray scale lacunarity:' 13 10 'Normalized 1bit lacunarity of bit planes']));
%axis([0 ceil(log(max(x))) 0 1]);
%set(gcf,'position',[10 300 512 384]);
