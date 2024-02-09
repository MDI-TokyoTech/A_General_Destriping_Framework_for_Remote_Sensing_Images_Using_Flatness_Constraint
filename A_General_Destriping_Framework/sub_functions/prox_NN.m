function result = prox_NN(X, lambda)
    [U, S, V] = svd(X, "econ");
    S_ST = max(S - lambda, 0);
    result = U*S_ST*ctranspose(V);
end