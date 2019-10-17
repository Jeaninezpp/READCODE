function [A] = Incomplete_graph(M,Index,k)
% M:data matrices, cell, M{1}\in\mathcal{R}^{n\times d}
% Index: {0,1}\in\{n\times V}
% k:number of neighbors k=10 in our experiments
[num,V] = size(Index);
for v = 1:V
    aa = Index(:,v);
   [B, ~] = selftuning(M{v}(find(aa==1),:),k(v));
%     B = constructW_PKN(M{v}(find(aa==1),:)',k(v),0);
%% issymmetric    
     A{v} = (B+B')/2;
end