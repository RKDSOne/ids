function [X, y] = DataLoader(fpath)
%% DataLoader: load format-formulated data in X, y (features and labels)
%% fpath to be
%% DataLoader('\\urbcomp03\d$\Users\v-tianhe\idsdata\dataset\pima\pima.data');
    ret = csvread(fpath,1);
    % Note that array form is required in matlab mulvalue return
    X=ret(:, 1:end-1);
    y=ret(:, end);
end