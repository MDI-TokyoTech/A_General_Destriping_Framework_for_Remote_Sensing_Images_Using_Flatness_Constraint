%% A General Destriping Framework for Remote Sensing Images Using Flatness Constraint

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Kazuki Naganuma (naganuma.k.aa@m.titech.ac.jp)
% Last version: Feb 26, 2022
% Article: K. Naganuma, S. Ono, ``A General Destriping Framework for Remote Sensing Images Using Flatness Constraint,''
% IEEE Transactions on Geoscience and Remote Sensing, 2022.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
addpath('./sub_functions/')

%%%%%%%%%%%%%%%% Choose target image %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
target_image = 'HSI';
% target_image = 'IR_video';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Preparing image

if strcmp(target_image, 'HSI')
    % HSI
    load('./images/Moffett_field.mat');

    para_stripe.is_tinv = 0;  % (stripe noise is variant in spectral direction)
    para_stripe.rate_stripe = 0.3;
    para_stripe.sigma_stripe = 0.05;
    para_stripe.intensity_stripes = 0.3;

    para_gaussian.is_gaussian = 1; % 0 or 1
    para_gaussian.sigma_gaussian = 0.05;

    DATA = add_stripe_noise(DATA_clean, para_stripe, para_gaussian);
    DATA_noisy = DATA{1};
    true_random_noise = DATA{3};
elseif strcmp(target_image, 'IR_video')
    % IR video
    load('./images/Bats1.mat');

    para_stripe.is_tinv = 1;  % (stripe noise is invariant in temporal direction)
    para_stripe.rate_stripe = 0.3;
    para_stripe.sigma_stripe = 0.05;
    para_stripe.intensity_stripes = 0.3;

    para_gaussian.is_gaussian = 0;
    para_gaussian.sigma_gaussian = 0;

    DATA = add_stripe_noise(DATA_clean, para_stripe, para_gaussian);
    DATA_noisy = DATA{1};
    true_random_noise = DATA{3};
else
end

%% Setting parameters
%%%%%%%%%%%%%%%%%%%%% User Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%
para.lambda_S = 0.01; % balancing parameter
para.is_tinv = para_stripe.is_tinv;

use_GPU = 1; % if using GPU, set use_GPU = 1. Otherwise, you set use_GPU = 0

% Regularizations for HSIs
% para.regularization = 'HTV';
para.regularization = 'SSTV';
% para.regularization = 'ASSTV';
% para.regularization = 'TNN';
% para.regularization = 'SSTV+TNN';
% para.regularization = 'l0l1HTV';

% Regularizations for IR videos
% para.regularization = 'ATV';
% para.regularization = 'ITV';
% para.regularization = 'ATV+NN';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

para.epsilon = norm(true_random_noise(:), 2);
para.max_iteration = 10000; % maximum number of iterations
para.stopping_criterion = 1e-4; % stopping criterion

%% Destriping
if use_GPU == 1
    DATA_est = A_General_Destriping_Framework_GPU(DATA_noisy, para); % for GPU user
elseif use_GPU == 0
    DATA_est = A_General_Destriping_Framework_CPU(DATA_noisy, para); % for CPU user
else
end

%% Plotting results
[n1, n2, n3] = size(DATA_clean);

% Calculating MPSNR
DIFF_cle2est = DATA_clean - DATA_est;
v_psnrs = 20*log10(sqrt(n1*n2)./reshape(sqrt(sum(sum(DIFF_cle2est.*DIFF_cle2est, 1), 2)), [1, n3]));
v_mpsnr = mean(v_psnrs);

% Calculating MSSIM
v_ssims = zeros(1, n3);
for j = 1:n3
    v_ssims(j) = ssim(DATA_clean(:, :, j), DATA_est(:, :, j));
end
v_mssim = mean(v_ssims);

disp('*********** destriping results **************')
disp(append('MPSNR : ', num2str(v_mpsnr)))
disp(append('MSSIM : ', num2str(v_mssim)))
disp('*********************************************')

method_name = cell(2, 1);
method_name{1} = 'Noisy';
method_name{2} = append('FC-', para.regularization);
DATAs = cell(2, 1);
DATAs{1} = DATA_noisy;
DATAs{2} = DATA_est;
en_list = 1:2;

% plotting results
figure;
show_results(DATAs, DATA_clean, 0, 1, method_name, en_list, 1, n3)

