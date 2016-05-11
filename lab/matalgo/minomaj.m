function [ minoX, majX, minolab ] = minomaj( X,y )
%MINOMAJ Summary of this function goes here
%   Detailed explanation goes here
    tabu=tabulate(y);
    [val, ind]=min(tabu(:, 2));
    minolab=tabu(ind,1);
    minoX = X(y==minolab, :);
    majX=X(y~=minolab, :);
end

