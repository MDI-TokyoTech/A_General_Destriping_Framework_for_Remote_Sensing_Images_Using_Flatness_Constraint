
function result = prox_TNN(Y, tau)
    n1 = size(Y, 1);
    n2 = size(Y, 2);
    n3 = size(Y, 3);
    FFT_of_Y = fft(Y, [], 3);
    FFT_of_result = complex(zeros(n1, n2, n3));
    for i = 1:n3
        if ceil((n3 + 1)/2) >= i
            [U, S, V] = svd(FFT_of_Y(:, :, i), "econ");
            FFT_of_result(:, :, i) = U*(max(S - tau, 0))*ctranspose(V);
        elseif ceil((n3 + 1)/2) < i
            FFT_of_result(:, :, i) = conj(FFT_of_result(:, :, n3 - i + 2));
        end
    end
    result = ifft(FFT_of_result, [], 3);
end