function H = Hankel(x, n, N, L)
% definition of Hankel matrix

H = zeros(L * n, N - L + 1);
for i = 1 : L
    H((i - 1) * n + 1 : i * n, :) = x(:, i : i + N - L);
end
