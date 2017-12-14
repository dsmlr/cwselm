function [ vIdx , centroid ] = SimpleKMeans( X, k, varargin )
%SIMPLEKMEANS Summary of this function goes here
%   Detailed explanation goes here

    if nargin < 2
        error('At least two input arguments required.');
    end
    
    randSeed  = find(strcmp('randSeed', varargin));
    if ~isempty(randSeed)
        randSeed = varargin{randSeed+1};
    else
        randSeed = 1;
    end
    
    init  = find(strcmp('start', varargin));
    if ~isempty(init)
        init = varargin{init+1};
        if size(init,1) ~= k
            error('number of initial centroid is not equal to k.')
        end
    else
        rng(randSeed)
        init = X(randperm(size(X,1),k),:);
    end
    
    maxIter  = find(strcmp('MaxIter', varargin));
    if ~isempty(maxIter)
        maxIter = varargin{maxIter+1};
    else
        maxIter = 100;
    end
    
    measure  = find(strcmp('measure', varargin));
    if ~isempty(measure)
        measure = varargin{measure+1};
    else
        measure = 'euclidean';
    end
    
    vIdx = [];
    centroidOld = [];
    centroid = init;
    
    round = 1;
    while ~isequal(centroid,centroidOld) && round <= maxIter
        centroidOld = centroid;
        ctdist = calDist(X,centroid,measure);
        [~, vIdx] = min(ctdist,[],2);
        
%         % Check any k still exist
        [~, noExistK] = hist(vIdx,unique(vIdx));
        if numel(noExistK) ~= k
            ctdistTemp = ctdist;
            while numel(noExistK) ~= k
                kDisappear = sortrows(setdiff([1:k],unique(vIdx)),1);
                [tv, tIdx] = min(ctdistTemp(:,kDisappear),[],1);
                minTvIdx = find(tv == min(tv));
                minTvIdx = minTvIdx(1);
                vIdx(tIdx(minTvIdx),1) = kDisappear(minTvIdx);
                ctdistTemp(tIdx(minTvIdx),:) = Inf;
                
                [~, noExistK] = hist(vIdx,unique(vIdx));
            end
        end
        
        centroid = [];
        for i = 1 : k
            XvIdx = X(find(vIdx==i),:);
            if size(XvIdx,1) == 1
                centroid(i,:) = XvIdx;
            else
                centroid(i,:) = mean(XvIdx);
            end
        end
        
        round = round + 1;
    end
    
    if round > maxIter
        warning(['Failed to converge in ' num2str(maxIter) ' iterations.'])
    end
end

function distM = calDist(data, ct, ms)
    [ n , a , b , c , d ] = FindComponent( ct , data );
    distM = 1-Measures( n , a , b , c , d , ms );
end

