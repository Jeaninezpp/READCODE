clc;clear;
f=load('dataset/handwritten.mat');
data=f.X';% data is a view*1 dim cell
label=f.Y;

para1=[10 20 30];
para2=[.001 .01 .1];
para3=[.000001 .00001 .0001];

for i=1:size(data,1)%view num
    dist = max(max(data{i})) - min(min(data{i}));
    m01 = (data{i} - min(min(data{i})))/dist;
    data{i} = 2 * m01 - 1;
end
[v1,v2]=size(data);% v1 is the num of view
FV=cell(v1,v2);
SV=cell(v1,v2);
LV=cell(v1,v2);
for i=1:v1  % 初始化W和F
    Sv= constructW_PKN(data{v1}', 5, 1);
    D = diag(sum(Sv));
    Lv = D-Sv;
    [Fv, ~, ev]=eig1(Lv,length(unique(label)),0);% 取前k小个特征值和特征向量
    FV{i}=Fv;
    SV{i}=Sv;
    LV{i}=Lv;
end

B=cell(size(data));
for i=1:length(para1)
    for ii=1:size(data,1)% view
        B{ii}= inv(data{ii}*data{ii}'+para1(i)*eye(size(data{1},1)));% compute B=inv(alpha*I+XX')
    end
    for j=1:length(para2)
        for k=1:length(para3)
            [result,FVret,obj,Wv]=UMVSC(data,label,B,para1(i),para2(j),para3(k),FV,SV,LV);
            dlmwrite('result_handwriten.txt',[para1(i) para2(j) para3(k) result(1) result(2) result(3) result(4) result(5)],'-append','delimiter','\t','newline','pc');
        end
    end
end
      %result = [Fscore Precision Recall nmi AR]  
