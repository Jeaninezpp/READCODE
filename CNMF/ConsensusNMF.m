function [U,V,Ustar,alpha,beta,gamma,ConsenX,obj]=ConsensusNMF(X,N,k,vN,options)
%%initial U,V,alpha,beta,gamma
r1=2;r2=2;r3=2;tau=1e8;
alpha=cell(1,vN);beta=cell(1,vN);gamma=cell(1,vN);
gv=cell(1,vN);hv=cell(1,vN);fv=cell(1,vN);
U=cell(1,vN);V=cell(1,vN);L=cell(1,vN);Xnor=cell(1,vN);
Ustar=rand(N,k);
MaxIter=10;
ConsenX=[];
for vIndex=1:vN
    alpha{vIndex}=1/vN;beta{vIndex}=1/vN;gamma{vIndex}=1/vN;
    U{vIndex}=rand(N,k);
    TempvData=[];TempvData=X{vIndex};TempD=size(TempvData,2);
    V{vIndex}=rand(TempD,k);
    TempNorData=[];TempNorData=NormalizeFea(TempvData); %NormalizeFea input n*d
    Xnor{vIndex}=TempNorData;
    ConsenX=[ConsenX,TempNorData];
    Sv = constructW(TempNorData,options);
    D_valuev = sum(Sv,2);
    Dv = spdiags(D_valuev,0,N,N);
    Lv = Dv-Sv;
    L{vIndex}=((full(Dv))^(-0.5))*Lv*((full(Dv))^(-0.5));
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