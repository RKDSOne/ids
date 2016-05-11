function [ ret ] = Overlap( X,y,eps,MinPts )
%OVERLAP Summary of this function goes here
%   Detailed explanation goes here
    X=X-repmat(mean(X), [size(X,1),1]);
    X=X./repmat(sqrt(var(X)), [size(X,1),1]);
    addpath('./DBSCAN Clustering');
    IDX=myDBSCAN(X,eps,MinPts);
    ret=tabulate(IDX);
end