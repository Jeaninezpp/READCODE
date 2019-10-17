clear all;
clc;
% dataName=[dataPath datasetName{dataIndex} '.mat'];%自行设置
dataName = ['/home/zpp/Documents/dataset/MultiView/' 'handwritten' '.mat'];%自行设置
load(dataName);% 从dataName中加载数据,如果是MAT文件，会把变量加载到工作区．如果是ASCII，创建双精度数组．
kmeansK = max(Y)-min(Y)+1; %% Clusters of samples
ViewN = length(feas);  % Number of views
TotalSampleNo = length(Y); %% Labels of samples
N = 130;% select N features
%% LPP Paras
options = [];
options.NeighborMode = 'KNN';%%%%%?哪里用到的K近邻呢?　在constructW中用到
options.k = 5;
options.WeightMode = 'HeatKernel';%%%%%热核特征HKS??　在constructW中用到
options.t = 10;
%%
tic;
[U,V,Ustar,alpha,beta,gamma,ConX,obj]=ConsensusNMF(X,TotalSampleNo,kmeansK,ViewN,options);
%输入：数据，样本数，聚类个数，视图个数，options
Vcon=[];
for vConIndex=1:ViewN
    Vcon=[Vcon;V{vConIndex}];
end
wc=sum(Vcon.*Vcon,2);
[dumb idx] = sort(wc,'descend');
feature_idx = idx(1:N); %% select N features
selecteddata=[];
selecteddata=ConX(:,feature_idx);
res = kmeans(selecteddata, kmeansK, 'emptyaction','singleton');
toc;
newres = bestMap(Y,res);
%             =============  evaluate AC: accuracy ==============
AC = length(find(Y == newres))/length(Y);
MIhat=MutualInfo(Y,res);
%%
