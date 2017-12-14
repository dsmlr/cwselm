function [ W , beta ] = TrainWELM( X, T, NumberHiddenNodes, regularizationC, balance, measureName, randSeed, method )
%TRAINWELM Summary of this function goes here
%   Detailed explanation goes here

    [ W ] = initHidden(NumberHiddenNodes,X,randSeed,method,measureName);
    [ n , a , b , c , d ] = FindComponent( W , X );
    H = Measures( n , a , b , c , d , measureName);
    
    % For imbalance dataset
    if balance == 1
        [S,SI] = hist(T,unique(T));
        maxS = max(S);
        S = sqrt(S.\maxS);
    else
        SI = unique(T);
        S = [1 1];
    end
    
    B = ones(size(T,1),1)*S(1);
    B(find(T == SI(2))) = S(2);
    H = repmat(B,1,size(H,2)).*H;
    beta = ( (H'*H) + ( (1/regularizationC) * eye(size(H,2)) ) ) \ (H'*(T.*B));

end

function [ W ] = initHidden( h , X , randSeed , method , measureName)
    % set max iterations
    maxIter = 100;
    h = min(h,size(X,1));
    
    if strcmp(method{1},'kmeans')
        [ idx , C ] = SimpleKMeans1(X,h,'randSeed',randSeed,'MaxIter',maxIter,'measure',measureName);
        
        W = [];
        for i = 1 : h
            tempCluster = X(find(idx==i),:);
            [ n , a , b , c , d ] = FindComponent( C(i,:) , tempCluster );
            tempScore = Measures( n , a , b , c , d , measureName );
            tempMax = find(tempScore == max(tempScore));
            W = [W; tempCluster(tempMax(1),:)];
        end
    elseif strcmp(method{1},'svc')
        svcData.X = X;
        model = svc(svcData,'method','CG','ker','sim','C',method{2},'simf',measureName);
        W = X(model.sv_ind,:);
    else
        rng(randSeed)
        W = X(randperm(size(X,1),h),:);
    end
    
end
