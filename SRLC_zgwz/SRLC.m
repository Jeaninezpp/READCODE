function [YII,obj] = SRLC(A, Index, gamma, lambda)
%% A:cell, partial grpah A{1}\in\mathcal{R}^{n^{(v)}\times n^{(v)}}
%% Index: {0,1}\in\{n\times V}
%% G0: initial Y
maxIter = 30;
[num, V] = size(Index);
U = diag(1./sqrt(sum(Index)));
for i=1:num
    D_ind{i} = find(Index(i,:)==1);% 找到每一行中为1的元素的列号
end
for v=1:V
    V_ind{v} = find(Index(:,v)==1);% 每一列中为１的元素的行号
end
C = size(G0,2); 
T = eye(C);
H = zeros(num,C);

%% Initialize F{v},R{v}
[F,L,L_ti] = initialF(A,C,Index,V_ind);
PY = initialG(C,num);

for iter=1:maxIter
    fprintf('SRLC iteration %d...\n', iter);
   %% Update matrix R^(v)
    for v = 1:V
        % %Լ��λ��
        M = PY(V_ind{v},:)'*F{v}(V_ind{v},:);
        [U_R,~,V_R] = svd(M);
        R{v} = V_R*U_R';
    end   
   %% update F{v}
   F_old = F;
   [F] = updateF(F_old, L, L_ti,R,PY,V_ind,lambda);        
   %% Update label matrix Y
    Y = zeros(num,C);
    H = zeros(num,C);
    for i = 1:num
        for v = 1:max(size(D_ind{i}))
            hh = F{D_ind{i}(v)}(i,:)*R{D_ind{i}(v)};
            for c=1:C
                qq(i,D_ind{i}(v),c) = norm(hh-T(c,:),2)^2;
                H(i,c) = H(i,c)+qq(i,D_ind{i}(v),c);
            end
        end
    end
    if gamma==1
        for i = 1:num
            [~,Y_i] = min(H(i,:));
            Y(i,Y_i) = 1;
        end
    else
        Y = (gamma.*H).^(1/(1-gamma));
        s0 = sum(Y,2);%列向量，求行和.得到N*1的列向量
        Y = Y./repmat(s0,1,C);% N*C
    end
    PY = Y.^(gamma);
    %% Calculate obj
    if iter>0
        obj(iter) = sum(sum(PY.*H));
        [~,YII] = max(Y');
        YII = YII';
        for v=1:V
            obj(iter) = obj(iter)+lambda*trace(F{v}(V_ind{v},:)'*L{v}*F{v}(V_ind{v},:));
        end
        if iter>2
            if (obj(iter-1)-obj(iter))/abs(obj(iter-1))<1e-6;
              break;
            end
        end
    end    
end
[~,YII] = max(Y');
YII = YII';
end

function [F,L,L_ti] = initialF(A,C,Index,V_ind)
[num,V] = size(Index);
for v=1:V
    F{v} = zeros(num,C);
    T = full(A{v});
    S = (T+T')/2;
    L{v} = diag(sum(S,2))-S;
    ww=rand(size(S,1),1);
    for i=1:20
        m1=S*ww;
        q=m1./norm(m1,2);
        ww=q;
    end
    alpha(v)=10*abs(ww'*S*ww);
    L_ti{v} = alpha(v).*eye(size(L{v},1))-L{v};
    F{v}(V_ind{v},:) = orth(rand(size(S,1),C));
end
end

function [F] = updateF(F_old, L, L_ti,R,PY,V_ind,lambda)
[num, C] = size(PY);
V =  max(size(L_ti));
D = sum(PY,2);
for v=1:V
    F{v} = zeros(num,C);
    A_til=lambda*L_ti{v};
    A2 = lambda*L{v};
    T = PY(V_ind{v},:)*R{v}';
    Q_old = F_old{v}(V_ind{v},:);
    for tt = 1:20
        M=A_til*Q_old+T;
        [U,~,V]=svd(M,'econ');
        Q=U*V';
        obj(tt)=trace(Q'*(A2*Q-T));
        if tt<2
            Q_old = Q;
        else
            if (obj(tt-1)-obj(tt))/abs(obj(tt-1))<1e-5
                break;
            else
                Q_old = Q;
            end
        end
    end
    F{v}(V_ind{v},:) = Q;
end
end
function[G0] = initialG(C,num)
I = eye(C);   
R = randperm(size(I,1)); 
g = ceil(num/C);
inG0 = repmat(I(R,:),g,1); 
G0 = inG0(1:num,:);
end
function [eig_vector,eig_value] = mineig(M)
[vec,val] = eig(M);
if issymmetric(M)
    eig_vector = vec;
    eig_value = val;
else
    temp = diag(val);
    [B,I] = sort(temp,'ascend');
    eig_vector = vec(:,I);
    eig_value = diag(B);
end
end