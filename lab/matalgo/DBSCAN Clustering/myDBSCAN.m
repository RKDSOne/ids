function IDX = myDBSCAN(X, eps, MinPts)
%% myDBSCAN: DBSCAN for personal use

    N=size(X, 1);
    Dis = pdist2(X,X);
    IDX = zeros(N,1);
    vst=false(N,1);
    C = 0;
    for i = 1:N
        if vst(i)
            continue;
        end
        vst(i)=true;
        neighs=Neigh(i);
        if numel(neighs)>=MinPts
            C=C+1;
            IDX(i)=C;
            BFS(neighs, C);
        end
    end

    %% BFS: given a expand-able point of a cluster
    %% search all the related points
    function k = BFS(neighs, C)
        k=1;
        while true
            cur=neighs(k);
            IDX(cur)=C;
            if ~vst(cur)
                vst(cur)=true;
                news=Neigh(cur);
                if numel(news)>=MinPts
                    %% en-queue new DenseNeighbors
                    neighs=[neighs, news];
                end
            end
            k=k+1;
            if k>numel(neighs)
                break;
            end
        end
    end

    %% Neigh: return neighbors within `eps` distance
    function rnei = Neigh(p)
        rnei = find(Dis(p, :)<eps);
        rnei = rnei(rnei~=p);
    end
end