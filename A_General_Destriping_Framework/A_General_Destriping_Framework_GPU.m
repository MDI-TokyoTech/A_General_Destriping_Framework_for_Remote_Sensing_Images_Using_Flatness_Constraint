%% A General Destriping Framework for Remote Sensing Images Using Flatness Constraint

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Kazuki Naganuma (naganuma.k.aa@m.titech.ac.jp)
% Last version: Feb 26, 2022
% Article: K. Naganuma, ``A General Destriping Framework for Remote Sensing Images Using Flatness Constraint,''
% IEEE Transactions on Geoscience and Remote Sensing, 2022.
% Test images: https://pixabay.com/en/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DATA_est = A_General_Destriping_Framework_GPU(DATA_noisy, para)
%% 
[n1, n2, n3] = size(DATA_noisy);

%% Settings and initializing
V = gpuArray(DATA_noisy); % observed data

lambda_S = para.lambda_S; % balancing parameter
max_iteration = para.max_iteration; % maximum number of iterations
stopping_criterion = para.stopping_criterion; % stopping criterion
epsilon = para.epsilon; 
is_tinv = para.is_tinv;
regularization = para.regularization;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating preconditioners
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

precon1_D_v = cat(1, ones(1, n2, n3), 2*ones(n1-2, n2, n3), ones(1, n2, n3));
precon1_D_h = cat(2, ones(n1, 1, n3), 2*ones(n1, n2-2, n3), ones(n1, 1, n3));
precon1_D_b = cat(3, ones(n1, n2, 1), 2*ones(n1, n2, n3-2), ones(n1, n2, 1));

Gamma2_Y2   = gpuArray(1/2);
Gamma2_Y4   = gpuArray(1/2);

if is_tinv == 1
    Gamma2_Y3 = gpuArray(1/2);
    Gamma1_S  = 1./(precon1_D_v + precon1_D_b + 1);
elseif is_tinv == 0
    Gamma1_S    = gpuArray(1./(precon1_D_v + 1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializing primal variables
% U is a estimated image component
% S is a stripe noise component
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U = gpuArray(zeros(n1, n2, n3));
S = gpuArray(zeros(n1, n2, n3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializing dual variables
% Y1はSSTVの項に用いる変数
% Y2はzero-Gradient constraintの項に用いる変数
% Y3はF-norm制約の項に用いる変数
% Y1   = D12(U)
% Y2   = D1(S)
% Y3   = U + S
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y2   = gpuArray(zeros(n1, n2, n3));
Y4   = V;

if is_tinv == 1
    Y3 = gpuArray(zeros(n1, n2, n3));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setting linear operators, proximity operators, ....
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(regularization, 'HTV')
    K = 1;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) D(X);
    Lt{1} = @(Y) Dt(Y);
    
    proxL{1} = @(Y, Gamma) prox_ml12(Y, Gamma);
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    
    Gamma1_U    = gpuArray(1./(precon1_D_v + precon1_D_h + 1));
    
    Gamma2_Y1{1}   = gpuArray(1/2);
    
elseif strcmp(regularization, 'SSTV')
    K = 2;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) Dv(Db(X));
    L{2} = @(X) Dh(Db(X));
    Lt{1} = @(Y) Dbt(Dvt(Y));
    Lt{2} = @(Y) Dbt(Dht(Y));
    
    proxL{1} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{2} = @(Y, Gamma) prox_l1(Y, Gamma);
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    Y1{2} = gpuArray(zeros(size(L{2}(U))));
    
    Gamma1_U = gpuArray(1./(precon1_D_v.*precon1_D_b + precon1_D_h.*precon1_D_b + 1));
    
    Gamma2_Y1{1} = gpuArray(1/4);
    Gamma2_Y1{2} = gpuArray(1/4);
    
elseif strcmp(regularization, 'ASSTV')
    K = 3;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) Dv(X);
    L{2} = @(X) Dh(X);
    L{3} = @(X) Db(X);
    Lt{1} = @(Y) Dvt(Y);
    Lt{2} = @(Y) Dht(Y);
    Lt{3} = @(Y) Dbt(Y);
    
    proxL{1} = @(Y, Gamma) Dv(V) + prox_l1(Y - Dv(V), Gamma);
    proxL{2} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{3} = @(Y, Gamma) prox_l1(Y, Gamma);
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    Y1{2} = gpuArray(zeros(size(L{2}(U))));
    Y1{3} = gpuArray(zeros(size(L{3}(U))));
    
    Gamma1_U    = gpuArray(1./(precon1_D_v + precon1_D_h + precon1_D_b + 1));
    
    Gamma2_Y1{1} = gpuArray(1/2);
    Gamma2_Y1{2} = gpuArray(1/2);
    Gamma2_Y1{3} = gpuArray(1/2);
    
elseif strcmp(regularization, 'TNN')
    K = 1;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) X;
    Lt{1} = @(Y) Y;
    
    proxL{1} = @(Y, Gamma) prox_TNN(Y, Gamma);
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    
    Gamma1_U    = gpuArray(1/2);
    
    Gamma2_Y1{1}   = gpuArray(1);
    
elseif strcmp(regularization, 'SSTV+TNN')
    K = 3;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) Dv(Db(X));
    L{2} = @(X) Dh(Db(X));
    L{3} = @(X) permute(X, [1, 3, 2]);
    Lt{1} = @(Y) Dbt(Dvt(Y));
    Lt{2} = @(Y) Dbt(Dht(Y));
    Lt{3} = @(Y) permute(Y, [1, 3, 2]);
    
    proxL{1} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{2} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{3} = @(Y, Gamma) gpuArray(prox_TNN(gather(Y), 100*Gamma));
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    Y1{2} = gpuArray(zeros(size(L{2}(U))));
    Y1{3} = gpuArray(zeros(size(L{3}(U))));
    
    Gamma1_U    = gpuArray(1./(precon1_D_v.*precon1_D_b + precon1_D_h.*precon1_D_b + 2));
    
    Gamma2_Y1{1} = gpuArray(1/4);
    Gamma2_Y1{2} = gpuArray(1/4);
    Gamma2_Y1{3} = gpuArray(1);
    
elseif strcmp(regularization, 'l0l1HTV')
    K = 3;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) Dv(Db(X));
    L{2} = @(X) Dh(Db(X));
    L{3} = @(X) D(X);
    Lt{1} = @(Y) Dbt(Dvt(Y));
    Lt{2} = @(Y) Dbt(Dht(Y));
    Lt{3} = @(Y) Dt(Y);
    
    eps_L10ball = floor(0.3*n1*n2);
    
    proxL{1} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{2} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{3} = @(Y, Gamma) ProjL10ball(Y, eps_L10ball);
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    Y1{2} = gpuArray(zeros(size(L{2}(U))));
    Y1{3} = gpuArray(zeros(size(L{3}(U))));
    
    Gamma1_U    = gpuArray(1./(precon1_D_v.*precon1_D_b + precon1_D_h.*precon1_D_b + precon1_D_v + precon1_D_h + 1)); 
    
    Gamma2_Y1{1} = gpuArray(1/4);
    Gamma2_Y1{2} = gpuArray(1/4);
    Gamma2_Y1{3} = gpuArray(1/2);
    
elseif strcmp(regularization, 'ATV')
    K = 3;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) Dv(X);
    L{2} = @(X) Dh(X);
    L{3} = @(X) Db(X);
    Lt{1} = @(Y) Dvt(Y);
    Lt{2} = @(Y) Dht(Y);
    Lt{3} = @(Y) Dbt(Y);
    
    proxL{1} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{2} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{3} = @(Y, Gamma) prox_l1(Y, Gamma);
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    Y1{2} = gpuArray(zeros(size(L{2}(U))));
    Y1{3} = gpuArray(zeros(size(L{3}(U))));
    
    Gamma1_U    = gpuArray(1./(precon1_D_v + precon1_D_h + precon1_D_b + 1)); 
    
    Gamma2_Y1{1} = gpuArray(1/2);
    Gamma2_Y1{2} = gpuArray(1/2);
    Gamma2_Y1{3} = gpuArray(1/2);
    
elseif strcmp(regularization, 'ITV')
    K = 1;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) Dall(X);
    Lt{1} = @(Y) Dallt(Y);
    
    proxL{1} = @(Y, Gamma) prox_ml12_2(Y, Gamma);
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    
    Gamma1_U    = gpuArray(1./(precon1_D_v + precon1_D_h + precon1_D_b + 1));
    
    Gamma2_Y1{1}   = gpuArray(1/2);
    
elseif strcmp(regularization, 'ATV+NN')
    K = 3;
    
    L = cell(K, 1);
    Lt = cell(K, 1);
    Y1 = cell(K, 1);
    Gamma2_Y1 = cell(K, 1);
    proxL = cell(K, 1);
    
    L{1} = @(X) Dv(X);
    L{2} = @(X) Dh(X);
    L{3} = @(X) Db(X);
    L{4} = @(X) reshape(X, n1*n2, n3);
    Lt{1} = @(Y) Dvt(Y);
    Lt{2} = @(Y) Dht(Y);
    Lt{3} = @(Y) Dbt(Y);
    Lt{4} = @(Y) reshape(Y, n1, n2, n3);
    
    proxL{1} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{2} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{3} = @(Y, Gamma) prox_l1(Y, Gamma);
    proxL{4} = @(Y, Gamma) prox_NN(Y, Gamma);
    
    Y1{1} = gpuArray(zeros(size(L{1}(U))));
    Y1{2} = gpuArray(zeros(size(L{2}(U))));
    Y1{3} = gpuArray(zeros(size(L{3}(U))));
    Y1{4} = gpuArray(zeros(size(L{4}(U))));
    
    Gamma1_U    = gpuArray(1./(precon1_D_v + precon1_D_h + precon1_D_b + 2)); 
    
    Gamma2_Y1{1} = gpuArray(1/2);
    Gamma2_Y1{2} = gpuArray(1/2);
    Gamma2_Y1{3} = gpuArray(1/2);
    Gamma2_Y1{4} = gpuArray(1);
    
end


Y1_next = cell(K, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setting other variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

is_converged = 0;
iteration = 0;

%% Optimization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('********* Optimization start *********')

while is_converged == 0

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % regularization term = SSTV
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating U (line 4 of Alg. 1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    LY = gpuArray(zeros(size(U)));
    for k = 1:K
        LY = LY + Lt{k}(Y1{k});
    end
    U_next = U - Gamma1_U.*(LY + Y4);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating S (lines 5 and 6 of Alg. 1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if is_tinv == 0 % without temporal FC
        S_next = prox_l1(S - Gamma1_S.*(Dvt(Y2) + Y4), Gamma1_S.*lambda_S);
    elseif is_tinv == 1 % with temporal FC
        S_next = prox_l1(S - Gamma1_S.*(Dvt(Y2) + Dbt(Y3) + Y4), Gamma1_S.*lambda_S);
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y11, ..., Y1K (Lines 7, 8, and 9 of Alg. 1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k = 1:K
        Y1_tmp  = Y1{k} + Gamma2_Y1{k}.*L{k}(2*U_next - U);     
        Y1_next{k} = Y1_tmp - Gamma2_Y1{k}.*proxL{k}(Y1_tmp./Gamma2_Y1{k}, 1./Gamma2_Y1{k});
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y2 (Line 11 of Alg. 1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y2_next = Y2 + Gamma2_Y2.*Dv(2*S_next - S);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y3 (Line 12 of Alg. 1) for temporal invariant stripe noise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if is_tinv == 1
        Y3_next = Y3 + Gamma2_Y3.* Db(2*S_next - S);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y4 (Lines 13 and 14 of Alg. 1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y4_tmp  = Y4 + Gamma2_Y4.*(2*(U_next + S_next) - (U + S));
    Y4_next = Y4_tmp - Gamma2_Y4.*proj_Fball(Y4_tmp./Gamma2_Y4, V, epsilon);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculating error
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 収束レートの計算
    error_U = gather(sqrt(sum((U - U_next).^2, "all")/sum(U.^2, "all")));

    % 収束しているか否か
    if (error_U <= stopping_criterion || iteration >= max_iteration)
        is_converged = 1;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating all variables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U  = U_next;
    S  = S_next;
    for k = 1:K
        Y1{k} = Y1_next{k};
    end
    Y2 = Y2_next;
    Y4 = Y4_next;
    if is_tinv == 1
        Y3 = Y3_next;
    end

    
    
    iteration = iteration + 1;
    if (mod(iteration, 100) == 0)
        disp("parameter : " + num2str(lambda_S) + ", iteration : " + num2str(iteration) + ", error : " + num2str(error_U))
    end
end

disp('********* Optimization end ***********')

DATA_est = gather(U);



end
