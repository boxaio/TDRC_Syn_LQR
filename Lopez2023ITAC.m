clc,clear;

% Lopez V G, Alsalti M, Müller M A. 
% Efficient off-policy Q-learning for data-based discrete-time LQR problems[J].
% IEEE Transactions on Automatic Control, 2023.

% Luenberger, D. (1967).Canonical forms for linear multivariable systems. 
% IEEE Transactions on Automatic Control, 12(3), 290–293.doi:10.1109/tac.1967.1098584

% random A and B
n = 10;
m = 2;
while 1
    A = random('uniform', -1, 1, [n, n]);
    B = random('uniform', -1, 1, [n, m]);
    % controllability
    Co = ctrb(A, B);  % size (n, n*m)
    if rank(Co) == n
        % rescale the eigenvalues of A     
        A = A / abs(eigs(A,1,'LM'));
        break;
    end
end

rank(ctrb(A, B)) == n

Q = eye(n);
R = eye(m);

% direct solution
[P_true, K_true, L_true] = idare(A, B, Q, R, [], []);

norm(K_true - (R + B' * P_true * B) \ B' * P_true * A)

norm(sort(eig(A - B * K_true)) - sort(L_true))

%% generate exciting data (u,x)
N = 2 * (n + 1) * (m + 1);   % N >= (n+1)*(m+1)-1, number of samples

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
        fprintf('\n exciting input found. \n')
        break;
    end
end

X_ = zeros(n, N+1);
X_(:,1) = random('uniform', -0.5, 0.5, [n, 1]);
for i = 1 : N
    X_(:,i+1) = A * X_(:,i) + B * U_(:,i);
end
X_0 = X_(:, 1 : N);
X_1 = X_(:, 2 : N+1);

rank(Hankel(U_, m, N, n+1)) == m*L
rank([Hankel(X_0, n, N, 1); Hankel(U_, m, N, 1)]) == m + n

%% construct initial stablizing controller

P_temp = B;
P_temp2 = B;
v = ones(1, m);
while rank(P_temp) < n
    for i = 1 : m
        P_temp2 = [P_temp, A^v(i) * B(:,i)];
        if rank(P_temp2) == size(P_temp2, 2) % linearly independent columns
            v(i) = v(i) + 1;
            P_temp = P_temp2;
        end
    end
end
v
rank(P_temp)
v_cum = cumsum(v);

% % construct matrix P
% P_temp = B;
% P_temp2 = B;
% v = zeros(1, m);
% count = 1;
% for i = 1 : m
%     while size(P_temp, 2) < n
%         P_temp2 = [P_temp, A^count * B(:,i)];
%         if rank(P_temp2) == size(P_temp2, 2)  % linearly independent columns
%             count = count + 1;
%             v(i) = v(i) + 1;
%             P_temp = P_temp2;
%         else
%             count = 1;
%             break;
%         end
%     end
% end
% v = v + ones(1, m);

P = [];
for i = 1 : m
    for j = 0 : v(i)-1
        P = [P, A^j * B(:,i)];
    end
end

P_inv = inv(P);
ee = P_inv(cumsum(v), :);
T = [];
for i = 1 : m
    for j = 1 : v(i)
        T = [T; ee(i,:) * A^(j-1)];
    end
end
det(T)

A_tilde = T * A / T;
B_tilde = T * B;

% sort(eig(A)) - sort(eig(A_tilde))

% Brunovsky Canonical Form
A_cf = zeros(n, n);
B_cf = zeros(n, m);
for i = 1 : n-1
    A_cf(i, i + 1) = 1;
end
for j = 1 : m
    A_cf(v_cum(j), v_cum(j)-v(j)+1 : v_cum(j)) = A_tilde(v_cum(j), v_cum(j)-v(j)+1 : v_cum(j));
    B_cf(v_cum(j), j) = 1;
end

J = pinv(B_tilde) * B_cf;

K1 = pinv(B_cf) * (A_cf - A_tilde);

% pole placement for the m decoupled systems
Kc = [];
for j = 1 : m
    Kj = acker(A_cf(v_cum(j)-v(j)+1 : v_cum(j), v_cum(j)-v(j)+1 : v_cum(j)), ...
               B_cf(v_cum(j)-v(j)+1 : v_cum(j), j), zeros(1,v(j)));
    Kc = blkdiag(Kc, Kj);
end

K = J * (K1 + Kc) * T;

% eig(A_cf - B_cf * Kc)'

%% Q-Learning


% solve the generalized discrete-time Lyapunov equation















