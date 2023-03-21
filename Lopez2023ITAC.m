clc,clear;

% Lopez V G, Alsalti M, Müller M A. 
% Efficient off-policy Q-learning for data-based discrete-time LQR problems[J].
% IEEE Transactions on Automatic Control, 2023.

% Luenberger, D. (1967).Canonical forms for linear multivariable systems. 
% IEEE Transactions on Automatic Control, 12(3), 290–293.doi:10.1109/tac.1967.1098584

% random A and B
n = 20;
m = 2;

disp(['Generate random systems (A, B) ...'])

while 1
    A = random('uniform', -1, 1, [n, n]);
    B = random('uniform', -1, 1, [n, m]);
    % rescale the eigenvalues of A     
    A = A / abs(eigs(A,1,'LM'));
    B = B / abs(eigs(A, 1, 'LM'));
    % controllability
    Co = ctrb(A, B);  % size (n, n x m)
    if rank(Co) == n
        break;
    end
end

Q = eye(n);
R = eye(m);
Q_bar = blkdiag(Q, R);

% direct solution
[P_true, K_true, L_true] = idare(A, B, Q, R, [], []);

% norm(K_true - (R + B' * P_true * B) \ B' * P_true * A)
% 
% norm(sort(eig(A - B * K_true)) - sort(L_true))

%% generate exciting data (u,x)
N = 2 * (n + 1) * (m + 1);   % N >= (n+1)*(m+1)-1, number of samples

disp(['Generate exciting control of PE order n + 1 ...'])

%  control input U_ is of PE order n+1

L = n + 1;
U_ = random('uniform', -0.5, 0.5, [m, N - L + 1]);
Hk_U = U_;
while 1
    u_i = random('uniform', -0.5, 0.5, [m, 1]);
    if rank([Hk_U; U_(:,end-N+L+1:end), u_i(:)]) - rank(Hk_U) == m
        U_ = [U_, u_i(:)];
        Hk_U = [Hk_U; U_(:,end-N+L:end)];
    end
    if rank(Hk_U) == m * L
        break;
    end
end

disp(['Generate data samples (u, x) ...'])

X_ = zeros(n, N+1);
X_(:,1) = random('uniform', -0.5, 0.5, [n, 1]);
for i = 1 : N
    X_(:,i+1) = A * X_(:,i) + B * U_(:,i);
end
X_0 = X_(:, 1 : N);
X_1 = X_(:, 2 : N+1);

% rank(Hankel(U_, m, N, n+1)) == m*L
% rank([Hankel(X_0, n, N, 1); Hankel(U_, m, N, 1)]) == m + n


%% Q-Learning

disp(['Off-Policy Q-Learning...'])

% Step 1 : Construct initial stablizing controller

disp(['--- Construct initial stablizing controller...'])

F = pinv(X_0);   % size (N, n)
G = null(X_0);   % size (N, N-n)
A_bar = X_1 * F;  % size (n, n)
B_bar = X_1 * G;  % size (n, N-n)

r = rank(B_bar);  % it is very likely r = m

B_F = [];
c_i = [];
i = 1;
while 1
    if rank([B_F, B_bar(:,i)]) > rank(B_F)
        c_i = [c_i, i];
        B_F = [B_F, B_bar(:,i)];
        i = i + 1;
    end
    if rank(B_F) == r  % B_F of size (n, r)
        break;
    end
end

% feedback gain K for system (A_bar, B_F), K is of size (r, n)
[T, A_tilde, B_tilde, A_cf, B_cf, Kc, K] = MIMO_Canonical(n, m, A_bar, B_F);

H = zeros(N - n, n);
H(c_i,:) = K;

% norm(B_F * K - B_bar * H)

% deadbeat controller gain for system (A, B), K_db is of size (m, n)
K_db = - U_ * (F - G * H);

% eig(A - B * K_db)

% Step 2 : Collect data samples
idx = 1 : n + m;
Z = [X_0(:, idx); U_(:, idx)];
% for i = 2 : N-n-m
%     if min(abs(eig([X_0(:,i:i+n+m-1); U_(:,i:i+n+m-1)]))) > min(abs(eig(Z)))
%         i
%         Z = [X_0(:,i:i+n+m-1); U_(:,i:i+n+m-1)];
%     end
% end

iter = 1;
err = 1;
K_old = K_db;

disp(['--- Policy Iterations ...'])

while err > 1e-8
% Step 3 : Solve the generalized discrete-time Lyapunov equation
    Y = [X_(:, idx + 1); - K_old * X_(:, idx + 1)];
    Z_inv = inv(Z);
    Theta_new = dlyap(Z_inv' * Y', Q_bar);
% Step 4 : Update feedback gain matrix
    K_new = Theta_new(n+1:n+m,n+1:n+m) \ Theta_new(n+1:n+m,1:n);
    err = norm(K_new - K_old);
    disp(['       Iteration  ' num2str(iter),': ', num2str(err)])
    K_old = K_new;
    iter = iter + 1;
end

disp(['Error of Controller Gain K : ', num2str(norm(K_true - K_new))])


