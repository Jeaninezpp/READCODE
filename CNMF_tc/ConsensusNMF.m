%ConsensusNMF
%InPut    :X,N,k,vN,options
%OutPut  :U,V,Ustar,alpha,beta,gamma,ConsenX,obj
%Function:更新u,v,u*,alpha,beta,gamma,ConsenX,obj
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% function格式：function [返回值１,返回值2,...]=fun_name(输入值1,输入值2,....)
function [U,V,Ustar,alpha,beta,gamma,ConsenX,obj]=ConsensusNMF(X,N,k,vN,options)
%%initial U,V,alpha,beta,gamma
r1=2;r2=2;r3=2;%r1,r2,r3是alpha,beta,gamma的指数
tau=1e8;%tau是优化U的时候的参数
alpha=cell(1,vN);beta=cell(1,vN);gamma=cell(1,vN);%alpha,beta,gamma是权衡参数
gv=cell(1,vN);hv=cell(1,vN);fv=cell(1,vN);% cell_array=cell(m,n)创建m*n的元胞数组
U=cell(1,vN);V=cell(1,vN);L=cell(1,vN);Xnor=cell(1,vN);%%%%%%%%%%%%%%%%%%  Xnor是啥？
Ustar=rand(N,k);%%%%%%%一致矩阵是随机初始化的？？？
MaxIter=50;
ConsenX=[];
for vIndex=1:vN%得到每个视图的数据，正则化数据，相似度矩阵，拉普拉斯矩阵，以及所有视图拼在一起的一直矩阵
    alpha{vIndex}=1/vN;beta{vIndex}=1/vN;gamma{vIndex}=1/vN; %对每个视图的alpha,beta,gamma进行初始化，初始值为1/vN
    U{vIndex}=rand(N,k);        %U是　N*k　的
    TempvData=[];TempvData=X{vIndex};%TempvData存放的是当前索引的单视图
    TempD=size(TempvData,2);%size(X,2)返回X数组的列数，如果是size(X,1)则返回X数组的行数．所以TempD是当前视图的列数，di
    V{vIndex}=rand(TempD,k);% V是　di * k　的
    TempNorData=[];
    TempNorData=NormalizeFea(TempvData); %NormalizeFea input n*d,输入単视图数据，进行正则化．输出正则化后的数据．
    Xnor{vIndex}=TempNorData;%赋值给Xnor的第vIndex个元素
    ConsenX=[ConsenX,TempNorData];%将TempNorData拼接在ConsenX后
    Sv = constructW(TempNorData,options);%构建相似度矩阵
    D_valuev = sum(Sv,2);%对Sv求行和就是D的对角线元素
    Dv = spdiags(D_valuev,0,N,N);%N是TotalSampleNumber,D是N*N的矩阵，把D_valuev的值放在对角线位置
    Lv = Dv-Sv;%拉普拉斯矩阵
    L{vIndex}=((full(Dv))^(-0.5))*Lv*((full(Dv))^(-0.5));%正则化拉普拉斯
    t=3;
end
obj=zeros(1,MaxIter);
for iterIndex=1:MaxIter
    disp(['Iter: ',num2str(iterIndex)]);
    %% Update Uv Vv
    for vIndexIter=1:vN
        % Update Uv
        Utop=Xnor{vIndexIter}*V{vIndexIter}+alpha{vIndexIter}.^r1*Ustar+2*tau*U{vIndexIter};
        Udown=U{vIndexIter}*V{vIndexIter}'*V{vIndexIter}+alpha{vIndexIter}.^r1*U{vIndexIter}+gamma{vIndexIter}.^r3*L{vIndexIter}*U{vIndexIter}+2*tau*U{vIndexIter}*U{vIndexIter}'*U{vIndexIter}+eps;
        UFrac=Utop./Udown;
        U{vIndexIter}=U{vIndexIter}.*UFrac;
        % Update Vv
        %         AvalueTemp = sqrt(sum(V{vIndexIter}.*V{vIndexIter},2)+eps);
        %         Al = 0.5./AvalueTemp;
        %         A = spdiags(Al,0,size(Xnor{vIndexIter},2),size(Xnor{vIndexIter},2));
        %         Vtop=Xnor{vIndexIter}'*U{vIndexIter};
        %         Vdown=V{vIndexIter}*U{vIndexIter}'*U{vIndexIter}+beta{vIndexIter}.^r2*A*V{vIndexIter}+eps;
        %         VFrac=Vtop./Vdown;
        %         V{vIndexIter}=V{vIndexIter}.*VFrac;
        %         t=3;
    end
    
    % Update Vv
    alphaTemp=cell(1,vN);
    for UpdatevIndexIter=1:vN
        Ktemp=Xnor{UpdatevIndexIter}'*U{UpdatevIndexIter};
        [Kr,Kl]=size(Ktemp);
        Vtemp=zeros(size(X{UpdatevIndexIter},2),k);
        alphaTemp{UpdatevIndexIter}=beta{UpdatevIndexIter}.^r2;
        for KrIndex=1:Kr
            if norm(Ktemp(KrIndex,:),2)>0.5*beta{UpdatevIndexIter}.^r2;
                Vtemp(KrIndex,:)=(1-(0.5*beta{UpdatevIndexIter}.^r2)/norm(Ktemp(KrIndex,:),2));
            else
                Vtemp(KrIndex,:)=zeros(1,k);
            end
        end
        V{UpdatevIndexIter}=Vtemp;
    end
    %% Update Ustar
    sum1=zeros(N,k);sum2=zeros(N,k);
    for sumIndex=1:vN
        sum1=sum1+alpha{sumIndex}.^r1*U{sumIndex};
        sum2=sum2+alpha{sumIndex}.^r1*Ustar;
    end
    Ustartop=sum1+2*tau*Ustar;Ustardown=sum2+2*tau*Ustar*Ustar'*Ustar;
    UstarFrac=Ustartop./Ustardown;
    Ustar=Ustar.*UstarFrac;
    
    %% Update gv sum hv sum fv sum
    gvsum=0;hvsum=0;fvsum=0;
    for ghfvIndex=1:vN
        gv{ghfvIndex}=norm(U{ghfvIndex}-Ustar,'fro').^2;
        gvsum=gvsum+gv{ghfvIndex}.^(1/(1-r1));
        
        hv{ghfvIndex}=sum(sqrt(sum(V{ghfvIndex}.*V{ghfvIndex},2)));
        hvsum=hvsum+hv{ghfvIndex}.^(1/(1-r2));
        
        fv{ghfvIndex}=trace(U{ghfvIndex}'*L{ghfvIndex}*U{ghfvIndex});
        fvsum=fvsum+fv{ghfvIndex}.^(1/(1-r3));
    end
    %% Update alpha beta gamma
    for abgIndex=1:vN
        gtemp=norm(U{abgIndex}-Ustar,'fro').^2;gtemp=gtemp.^(1/(1-r1));
        alpha{abgIndex}=gtemp/gvsum;
        
        htemp=sum(sqrt(sum(V{abgIndex}.*V{abgIndex},2)));htemp=htemp.^(1/(1-r2));
        beta{abgIndex}=htemp/hvsum;
        
        ftemp=trace(U{abgIndex}'*L{abgIndex}*U{abgIndex});ftemp=ftemp.^(1/(1-r3));
        gamma{abgIndex}=ftemp/fvsum;
    end
    
    %% Calculate obj
    tempobj=0;
    for objIndex=1:vN
        Term1=norm(Xnor{objIndex}-U{objIndex}*V{objIndex}','fro').^2;
        Term2=alpha{objIndex}.^r1*norm(U{objIndex}-Ustar,'fro').^2;
        Term3=beta{objIndex}.^r2*sum(sqrt(sum(V{objIndex}.*V{objIndex},2)));
        Term4=gamma{objIndex}.^r3*trace(U{objIndex}'*L{objIndex}*U{objIndex});
        tempobj=tempobj+Term1+Term2+Term3+Term4;
    end
    obj(iterIndex)=tempobj;
    disp(['Obj: ',num2str(tempobj)]);
    
    
end
