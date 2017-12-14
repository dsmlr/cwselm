% You must install this
% https://sites.google.com/site/daewonlee/research/svctoolbox for using SVC

% X is samples
% Y is labels
% seed is a random seed 
% regularizationC (can be one of these regularizationC = power(10,[ -6 : 6 ]))
% balance for balancing imbalance dataset i.e. 1 or 0
% method{1} for selection weights in ELM i.e. random kmeans and svc
% method{1} for svcRegC is a parameter of SVC (can be one of these svcRegC = [ 0.1 : 0.1 : 1 ])

load([pwd '/MUV_Sample_Dataset.mat']);
data = ECFP4.aid846;
X = data(1:100,:);
T = X(:,end);
X = X(:,2:end-1);

numberNodeUsed = 50;
regularizationC = power(10,1);
balance = 1;
svcRegC = 0.1;
method = {'svc',svcRegC};
measureName = 'tanimoto';
seed = 1;

[ W , beta ] = TrainWELM( X, T, numberNodeUsed, regularizationC, balance, measureName, seed, method );
[ accuracy, recall, GMean, BAC, AUC, score, sortScore ] = TestWELM( X , T , W , beta , measureName );
