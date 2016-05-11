function [r] = WBNR(X, y, k)
%% WBNR: With-in Bwtween Neighbors Ratio

    %% here select all rows with corresponding `y` the minority
    [minoX, majX, minolab]=minomaj(X,y)
    [IDX, D]=knnsearch(X, Xmino, 'k', k+1);
    D=transpose(D);
    sumdis=sqrt(sum(D(2:end, :).^2));
    [IDX, D]=knnsearch(Xmaj, Xmino, 'k', k+1);
    D=transpose(D);
    sumdis_maj=sqrt(sum(D(2:end, :).^2));
    r=sumdis./sumdis_maj;

end