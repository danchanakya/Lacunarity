function b = slidingBox(varargin)
% slidingBox Perform general sliding-box operations.
%
% This function is a modified version of nlfilter.m of images toolbox
%
%   B = slidingBox(A,[M N],FUN) applies the function FUN to each M-by-N
%   sliding block of A.  FUN is a function that accepts an M-by-N matrix as
%   input and returns a scalar:
%
%       C = FUN(X)
%
%   FUN must be a FUNCTION_HANDLE.
%
%   C is the output value for the center pixel in the M-by-N block
%   X. NLFILTER calls FUN for each pixel in A. NLFILTER zero pads the M-by-N
%   block at the edges, if necessary.
%
%   Class Support
%   -------------
%   The input image A can be of any class supported by FUN. The class of B
%   depends on the class of the output from FUN.
%
%   Example
%   -------
%   This example produces the same result as calling MEDFILT2 with a 3-by-3
%   neighborhood:
%
%       A = imread('cameraman.tif');
%       fun = @(x) median(x(:));
%       B = slidingBox(A,[3 3],fun);


[a, nhood, fun] = parse_inputs(varargin{:});

% Expand A
[ma,na] = size(a);

% Find out what output type to make.
rows = 0:(nhood(1)-1);
cols = 0:(nhood(2)-1);

b = mkconstarray(class(feval(fun,a(1,1))), 0, [ma-nhood(1) + 1, na-nhood(2) + 1]);

% Apply m-file to each neighborhood of a
% f = waitbar(0,'Applying neighborhood operation...');
for i=1:ma-nhood(1) + 1,
 for j=1:na-nhood(2) + 1,
    x = a(i+rows,j+cols);
    b(i,j) = feval(fun,x);
  end
  % waitbar(i/na)
end
% close(f)

%%%
%%% Function parse_inputs
%%%
function [a, nhood, fun] = parse_inputs(varargin)

switch nargin
case 3
        % slidingBox(A, [M N], 'fun')
        a = varargin{1};
        nhood = varargin{2};
        fun = varargin{3};
otherwise
    eid = sprintf('Images:%s:tooFewInputs',mfilename);
    msg = 'Too few inputs to NLFILTER';
    error(eid,'%s',msg);
end

%fun = fcnchk(fun,0);
