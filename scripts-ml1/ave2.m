function y = ave2(x)
%SUM2 Compute sum of matrix elements.
%   B = SUM2(A) computes the sum of the values in A.
%
%   Class Support
%   -------------
%   A can be numeric or logical. B is a scalar of class double. 
%
%   Example
%   -------
%       I = imread('liftingbody.png');
%       val = sum2(I)
%  
%   See also MEAN,MEAN2, STD, STD2.


y = mean(x(:)) ;
