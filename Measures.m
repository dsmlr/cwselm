function [ score ] = Measures( n , a , b , c , d , measureName )
%MEASURES Summary of this function goes here
%   Detailed explanation goes here

    switch measureName
        case 'tanimoto'     %Jaccard
            [ score ] = Tanimoto( a , b , c );
        case 'dice'
            [ score ] = Dice( a , b , c );
        case 'russell'
            [ score ] = Russell( a , n );
        case 'sokal1'
            [ score ] = Sokal1( a , b , c );
        case 'kulczynski1'
            [ score ] = Kulczynski1( a , b , c );
        case 'simpleMatching'
            [ score ] = SimpleMatching( a , d , n );
        case 'hamann'
            [ score ] = Hamann( a , b , c , d , n );
        case 'sokal2'
            [ score ] = Sokal2( a , b , c , d );
        case 'rogers'
            [ score ] = Rogers( a , b , c , d );
        case 'sokal3'
            [ score ] = Sokal3( a , b , c , d );
        case 'buser'
            [ score ] = Buser( a , b , c , d );
        case 'ochiai'
            [ score ] = Ochiai( a , b , c );
        case 'kulczynski2'
            [ score ] = Kulczynski2( a , b , c );
        case 'forbes'
            [ score ] = Forbes( a , b , c , n );
        case 'fossum'
            [ score ] = Fossum( a , b , c , n );
        case 'simpson'
            [ score ] = Simpson( a , b , c );
        case 'pearson'
            [ score ] = Pearson( a , b , c , d );
        case 'yule'
            [ score ] = Yule( a , b , c , d );
        case 'mcConnaughey'
            [ score ] = McConnaughey( a , b , c );
        case 'stiles'
            [ score ] = Stiles( a , b , c , d , n );
        case 'dennis'
            [ score ] = Dennis( a , b , c , d , n );
        case 'meanManhattan'
            [ score ] = MeanManhattan( b , c , n );
        case 'euclidean'
            [ score ] = Euclidean( b , c , n );
        otherwise
            warning('Unexpected Measure Name.')
            return
    end
end

function [ SS ] = Tanimoto( a , b , c )     %Jaccard
    SS = a./ (a+b+c);
end

function [ SS ] = Dice( a , b , c )
    SS = (2*a)./ ((2*a)+b+c);
end

function [ SS ] = Russell( a , n )
    SS = (a./n);  % Cannot norm
end

function [ SS ] = Sokal1( a , b , c ) 
    SS = a./(a+(2*(b+c)));
end

function [ SS ] = Kulczynski1( a , b , c ) 
    SS = a./(b+c);
end

function [ SS ] = SimpleMatching( a , d , n ) 
    SS = (a+d)./n;
end

function [ SS ] = Hamann( a , b , c , d , n ) 
    SS = (((a+d-b-c)./n)+1)./2;
end

function [ SS ] = Sokal2( a , b , c , d ) 
    SS = 0.25*( (a./(a+b)) + (a./(a+c)) + (d./(b+d)) + (d./(c+d)) );
end

function [ SS ] = Rogers( a , b , c , d ) 
    SS = (a+d)./(a+(2*(b+c))+d);
end

function [ SS ] = Sokal3( a , b , c , d ) 
    SS = (a.*d)./sqrt((a+b).*(a+c).*(d+b).*(d+c));
end

function [ SS ] = Buser( a , b , c , d ) 
    SS = (a+sqrt(a.*d))./(a+b+c+sqrt(a.*d));
end

function [ SS ] = Ochiai( a , b , c ) 
    SS = a./sqrt((a+b).*(a+c));
end

function [ SS ] = Kulczynski2( a , b , c ) 
    SS = 0.5*( (a./(a+b)) + (a./(a+c)) );
end

function [ SS ] = Forbes( a , b , c , n ) 
    SS = ((n*a)./((a+b).*(a+c)))/2; % Normalize /2 % Cannot Norm
end

function [ SS ] = Fossum( a , b , c , n ) 
    SS = (n*(a-0.5).*(a-0.5))./((a+b).*(a+c));
end

function [ SS ] = Simpson( a , b , c ) 
    SS = a./min((a+b),(a+c));
end

function [ SS ] = Pearson( a , b , c , d ) 
    SS = ((((a.*d)-(b.*c))./sqrt((a+b).*(a+c).*(b+d).*(c+d)) )+1)./2;
end

function [ SS ] = Yule( a , b , c , d ) 
    SS = (((a.*d)-(b.*c))./((a.*d)+(b.*c))+1)./2;
end

function [ SS ] = McConnaughey( a , b , c ) 
    SS = ((((a.*a)-(b.*c))./((a+b).*(a+c)))+1)./2;
end

function [ SS ] = Stiles( a , b , c , d , n ) 
    SS = log10( (n*power((abs((a.*d)-(b.*c))-(n/2)), 2))./ ((a+b).*(a+c).*(b+d).*(c+d)) );
end

function [ SS ] = Dennis( a , b , c , d , n ) 
    SS = (((((a.*d)-(b.*c))./sqrt(n*(a+b).*(a+c)))./(sqrt(n)/2))+1)./2;
end

function [ SS ] = MeanManhattan( b , c , n )
    SS = 1 - ((b+c)./n);
end

function [ SS ] = Euclidean( b , c , n )
    SS = 1 - sqrt((b+c)./n);
end