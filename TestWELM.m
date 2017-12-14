function [ accuracy, recall, GMean, BAC, AUC, score, sortScore ] = ...
    TestWELM( X , T , W , beta , measureName )
%TESTWELM Summary of this function goes here
%   Detailed explanation goes here

    [ n , a , b , c , d ] = FindComponent( W , X );
    H = Measures( n , a , b , c , d , measureName );
    Hbeta = H*beta;
    predictT = sign(Hbeta);
    
    [T,predictT] = prepareLabel(T,predictT);
    cp = classperf(T,predictT,'Positive', 1, 'Negative', 0);
    accuracy = cp.CorrectRate;
    specificity = cp.Specificity;
    recall = cp.Sensitivity;

    GMean =  abs(recall*specificity);
    BAC = (1/2) * (recall+specificity);
    
    [~,~,~,AUC,~] = perfcurve(T,Hbeta,1);
    
    score = [Hbeta predictT T];
    tempSortrows = sortrows(score,-1);
    sortScore = dataset(tempSortrows(:,1),tempSortrows(:,2),tempSortrows(:,3),'VarNames',{'score','predictLabel','trueLabel'});
    score = dataset(score(:,1),score(:,2),score(:,3),'VarNames',{'score','predictLabel','trueLabel'});

end

function [T,predictT] = prepareLabel(T,predictT)
    T(T==-1) = 0;
    predictT(predictT==-1) = 0;
end
