% this is the proximal operator of TNN
% Y is a 3-way tensor 
% tau is a real value and a parameter
% Y has the size of n_1*n_2*n_3
% the result is a 3-way tensor
% the result has the size of n_1*n_2*n_3
function result = prox_TNN_GPU(Y, tau)
    n1 = size(Y, 1);
    n2 = size(Y, 2);
    n3 = size(Y, 3);
    FFT_of_Y = gather(fft(Y, [], 3));
    FFT_of_result = gpuArray(complex(zeros(n1, n2, n3)));
    for i = 1:n3
        if ceil((n3 + 1)/2) >= i
            [U, S, V] = svd(FFT_of_Y(:, :, i), "econ");
            FFT_of_result(:, :, i) = gpuArray(U)*(max(gpuArray(S) - tau, 0))*ctranspose(gpuArray(V));
        elseif ceil((n3 + 1)/2) < i
            FFT_of_result(:, :, i) = conj(FFT_of_result(:, :, n3 - i + 2));
        end
    end
    result = abs(ifft(FFT_of_result, [], 3));
end