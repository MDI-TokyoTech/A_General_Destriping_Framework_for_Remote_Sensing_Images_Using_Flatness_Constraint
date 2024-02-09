
function DATA = add_stripe_noise(DATA_clean, para_stripe, para_gaussian)
    [n1, n2, n3] = size(DATA_clean);
    
    rate_stripe = para_stripe.rate_stripe;
    sigma_stripe = para_stripe.sigma_stripe;
    intensity_stripe = para_stripe.intensity_stripes;
    is_tinv = para_stripe.is_tinv;
    
    sigma_gaussian = para_gaussian.sigma_gaussian;
    is_gaussian = para_gaussian.is_gaussian;
    
    if is_tinv == 0
        sparse_stripe = 2*(imnoise(0.5*ones(1, n2, n3), "salt & pepper", rate_stripe) - 0.5).*rand(1, n2, n3).*ones(n1, n2, n3);
        warm_stripe = sigma_stripe*randn(1, n2, n3).*ones(n1, n2, n3);
        true_stripe = warm_stripe + sparse_stripe;
        true_stripe = intensity_stripe.*true_stripe./max(abs(true_stripe), [], "all");
        DATA_noisy = DATA_clean + true_stripe;
    elseif is_tinv == 1
        sparse_stripe = 2*(imnoise(0.5*ones(1, n2, 1), "salt & pepper", rate_stripe) - 0.5).*rand(1, n2, 1).*ones(n1, n2, n3);
        warm_stripe = sigma_stripe*randn(1, n2, 1).*ones(n1, n2, n3);
        true_stripe = warm_stripe + sparse_stripe;
        true_stripe = intensity_stripe.*true_stripe./max(abs(true_stripe), [], "all");
        DATA_noisy = DATA_clean + true_stripe;
    else
        disp('invalid value for is_tinv');
    end

    % gaussian_noiseの作成
    % variance_of_gaussianの中にgaussian noiseの分散を入れること
    if is_gaussian == 1
        true_random_noise = sigma_gaussian*randn(n1, n2, n3);
        DATA_noisy = DATA_noisy + true_random_noise;
    elseif is_gaussian == 0
        true_random_noise = zeros(n1, n2, n3);
    else
        disp('invalid value for is_gaussian');
    end
    
    DATA = cell(3, 1);
    DATA{1} = DATA_noisy;
    DATA{2} = true_stripe;
    DATA{3} = true_random_noise;

end
