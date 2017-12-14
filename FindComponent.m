function [ n , a , b , c , d ] = FindComponent( refStructure , dbStructure )
%FINDCOMPONENT Summary of this function goes here
%   Detailed explanation goes here
    n = size(dbStructure,2);
    sumRefStructure = sum(refStructure,2);
    sumDBStructure = sum(dbStructure,2);
    a = dbStructure*refStructure';       % dbStructure intersec refStructure
    b = repmat(sumRefStructure',size(a,1),1) - a;
    c = repmat(sumDBStructure,1,size(a,2)) - a;
    d = n - (a + b + c);
end

