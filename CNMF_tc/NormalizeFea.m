function fea = NormalizeFea(fea,row)
% if row == 1, normalize each row of fea to have unit norm;
% if row == 0, normalize each column of fea to have unit norm;
%
%   version 3.0 --Jan/2012 
%   version 2.0 --Jan/2012 
%   version 1.0 --Oct/2003 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%

if ~exist('row','var')
%如果row中无变量则为真  ///exist('name','kind'),name可以是变量名，函数名，文件名，目录．kind可以是java类class，目录dir，文件file，变量var
    row = 1;%缺省值为１
end

if row%row为１
    nSmp = size(fea,1);%fea的行数
    feaNorm = max(1e-14,full(sum(fea.^2,2)));%fea:n*d,feaNorm:n*1
    %fea.^2对fea矩阵逐元素平方
    %sum(fea.^2,2)i.e.sum(X,2)求行和
    %full稀疏矩阵转化为完整矩阵
    %max(1e-14,...)截断在1e-14,小于它的都用它代替
    fea = spdiags(feaNorm.^-.5,0,nSmp,nSmp)*fea;%其实实现的就是每个元素除以每行的平方和的平方根
else%row为０
    nSmp = size(fea,2);%fea的列数
    feaNorm = max(1e-14,full(sum(fea.^2,1))');
    fea = fea*spdiags(feaNorm.^-.5,0,nSmp,nSmp);%每个元素除以每列的平方和的平方根
end
            
return;




if row
    [nSmp, mFea] = size(fea);%获取行,列
    if issparse(fea)%如果是稀疏矩阵
        fea2 = fea';
        feaNorm = mynorm(fea2,1);%%%%%%%%%%%%%%%mynorm????
        for i = 1:nSmp
            fea2(:,i) = fea2(:,i) ./ max(1e-10,feaNorm(i));
        end
        fea = fea2';
    else
        feaNorm = sum(fea.^2,2).^.5;
        fea = fea./feaNorm(:,ones(1,mFea));
    end
% if not row
else
    [mFea, nSmp] = size(fea);
    if issparse(fea)
        feaNorm = mynorm(fea,1);
        for i = 1:nSmp
            fea(:,i) = fea(:,i) ./ max(1e-10,feaNorm(i));
        end
    else
        feaNorm = sum(fea.^2,1).^.5;
        fea = fea./feaNorm(ones(1,mFea),:);
    end
end
            

