function [T, A_tilde, B_tilde, A_cf, B_cf, Kc, K] = MIMO_Canonical(n, m, A, B)

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
v_cum = cumsum(v);


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
% det(T)

A_tilde = T * A / T;
B_tilde = T * B;

% Brunovsky Canonical Form
A_cf = zeros(n, n);
B_cf = zeros(n, m);
for j = 1 : m
    for i = 1 : v(j) - 1
        A_cf(v_cum(j)-v(j)+i, v_cum(j)-v(j)+i+1) = 1;
    end
    A_cf(v_cum(j), v_cum(j)-v(j)+1 : v_cum(j)) = A_tilde(v_cum(j), v_cum(j)-v(j)+1 : v_cum(j));
    B_cf(v_cum(j), j) = 1;
end

B_tilde_inv = pinv(B_tilde, 1e-30);

K1 = B_tilde_inv * (A_cf - A_tilde);
K2 = B_tilde_inv * B_cf;

% K1 = lsqminnorm(B_tilde, A_cf - A_tilde, 1e-16);
% K2 = lsqminnorm(B_tilde, B_cf, 1e-16);

% pole placement for the m decoupled systems
Kc = [];
for j = 1 : m
    Kj = acker(A_cf(v_cum(j)-v(j)+1 : v_cum(j), v_cum(j)-v(j)+1 : v_cum(j)), ...
               B_cf(v_cum(j)-v(j)+1 : v_cum(j), j), zeros(1,v(j)));
    Kc = blkdiag(Kc, Kj);
end

% eig(A_tilde + B_tilde *  (K1 - K2 * Kc))'

% feedback gain for system (A, B)
K = -(K1 - K2 * Kc) * T;