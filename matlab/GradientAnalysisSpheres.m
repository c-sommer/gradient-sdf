% BSD 3-Clause License
%
% This file is part of the code accompanying the paper
% Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction
% by Christiane Sommer*, Lu Sang*, David Schubert, and Daniel Cremers (* denotes equal contribution).
%
% Copyright (c) 2021, Christiane Sommer and Lu Sang.
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
%
% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
%
% * Neither the name of the copyright holder nor the names of its
%   contributors may be used to endorse or promote products derived from
%   this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%%

ml_blue = [0 0.4470 0.7410];
ml_yellow = [0.9290 0.6940 0.1250];
ml_green = [0.4660 0.6740 0.1880];
ml_red = [0.6350 0.0780 0.1840];

%% read in per-voxel SDF and stored Gradient-SDF gradient vectors

path = './'; % modify accordingly
file_prefix = 'gradient_sdf_sdf_';

 % adapt the following values to those from grid_info.txt
sz = [162 176 123]; % voxel dim
dmin = [-77 -92 -66]; % voxel min
dmax = [84 83 56]; % voxel max
vs = 0.01;
vs_inv = 1/vs;
T = 10;

D = vs * T * ones(sz); % init SDF volume
tmp = load([path file_prefix 'd.txt']);
D(tmp(:,1)+1) = tmp(:,2);

Gx1 = zeros(sz);
tmp = load([path file_prefix 'n0.txt']);
Gx1(tmp(:,1)+1) = tmp(:,2);

Gy1 = zeros(sz);
tmp = load([path file_prefix 'n1.txt']);
Gy1(tmp(:,1)+1) = tmp(:,2);

Gz1 = zeros(sz);
tmp = load([path file_prefix 'n2.txt']);
Gz1(tmp(:,1)+1) = tmp(:,2);

% normalize stored Gradient-SDF vectors
inv_normG1 = 1 ./ sqrt(Gx1.^2 + Gy1.^2 + Gz1.^2);
Gx1 = Gx1 .* inv_normG1;
Gy1 = Gy1 .* inv_normG1;
Gz1 = Gz1 .* inv_normG1;

clear tmp inv_normG1 path file_prefix

%% compute gradients with finite differences

[Gy2, Gx2, Gz2] = gradient(D, vs);

% normalize computed gradients
inv_normG2 = 1 ./ sqrt(Gx2.^2 + Gy2.^2 + Gz2.^2);
Gx2 = Gx2 .* inv_normG2;
Gy2 = Gy2 .* inv_normG2;
Gz2 = Gz2 .* inv_normG2;

clear inv_normG2

%% compute GT gradients

% load spheres in format [cx cy cz R]
load('spheres.mat')

[y,x,z] = meshgrid((dmin(2):dmax(2))*vs, (dmin(1):dmax(1))*vs, (dmin(3):dmax(3))*vs) ;
xyz = [x(:), y(:), z(:)];
% clear x y z
D_GT = spheres(:,4)' - pdist2(xyz, spheres(:,1:3)); % signed distances
[D_GT, idx] = max(D_GT, [], 2);
D_GT = reshape(D_GT, sz);

gx = reshape(xyz(:,1) - spheres(idx, 1), sz);
gy = reshape(xyz(:,2) - spheres(idx, 2), sz);
gz = reshape(xyz(:,3) - spheres(idx, 3), sz);

% normalize GT gradients
inv_norm_g = 1 ./ sqrt(gx.^2 + gy.^2 + gz.^2);
gx = gx .* inv_norm_g;
gy = gy .* inv_norm_g;
gz = gz .* inv_norm_g;

clear inv_norm_g xyz idx

%% compute gradients with forward finite differences

Gx3 = vs_inv * (D(2:end, :, :) - D(1:end-1, :, :));
Gx3 = cat(1, Gx3, zeros(1, sz(2), sz(3)));

Gy3 = vs_inv * (D(:, 2:end, :) - D(:, 1:end-1, :));
Gy3 = cat(2, Gy3, zeros(sz(1), 1, sz(3)));

Gz3 = vs_inv * (D(:, :, 2:end) - D(:, :, 1:end-1));
Gz3 = cat(3, Gz3, zeros(sz(1), sz(2), 1));

% normalize computed gradients
inv_normG3 = 1 ./ sqrt(Gx3.^2 + Gy3.^2 + Gz3.^2);
Gx3 = Gx3 .* inv_normG3;
Gy3 = Gy3 .* inv_normG3;
Gz3 = Gz3 .* inv_normG3;

clear inv_normG3

%% compute gradients with backward finite differences

Gx4 = vs_inv * (D(2:end, :, :) - D(1:end-1, :, :));
Gx4 = cat(1, zeros(1, sz(2), sz(3)), Gx4);

Gy4 = vs_inv * (D(:, 2:end, :) - D(:, 1:end-1, :));
Gy4 = cat(2, zeros(sz(1), 1, sz(3)), Gy4);

Gz4 = vs_inv * (D(:, :, 2:end) - D(:, :, 1:end-1));
Gz4 = cat(3, zeros(sz(1), sz(2), 1), Gz4);

% normalize computed gradients
inv_normG4 = 1 ./ sqrt(Gx4.^2 + Gy4.^2 + Gz4.^2);
Gx4 = Gx4 .* inv_normG4;
Gy4 = Gy4 .* inv_normG4;
Gz4 = Gz4 .* inv_normG4;

clear inv_normG4

%% general preparation

d = 0.001:.001:(vs*T); % x-axis of plot

ax_limits = [0 T 0 10];

%% analyze and visualize our gradients and finite differences vs GT

% our Gradient-SDF gradients
cos_phi = Gx1.*gx + Gy1.*gy + Gz1.*gz;
phi = acosd(min(abs(cos_phi),1));
% statistical measures: mean, median, RMSE, 95th percentile
stats1 = phi_statistics(d, D, phi);

% central finite differences
cos_phi = Gx2.*gx + Gy2.*gy + Gz2.*gz;
phi = acosd(min(abs(cos_phi),1));
% statistical measures: mean, median, RMSE, 95th percentile
stats2 = phi_statistics(d, D, phi);

% forward finite differences
cos_phi = Gx3.*gx + Gy3.*gy + Gz3.*gz;
phi = acosd(min(abs(cos_phi),1));
% statistical measures: mean, median, RMSE, 95th percentile
stats3 = phi_statistics(d, D, phi);

% backward finite differences
cos_phi = Gx4.*gx + Gy4.*gy + Gz4.*gz;
phi = acosd(min(abs(cos_phi),1));
% statistical measures: mean, median, RMSE, 95th percentile
stats4 = phi_statistics(d, D, phi);

clear phi cos_phi

%% actually visualize

figure

hold on
% plot actual evaluation data
plot(NaN, 'w-'); % for legend entry
plot(d / vs, stats1(:,1), '-', 'LineWidth', 1, 'Color', ml_red) % mean
plot(d / vs, stats1(:,2), '-', 'LineWidth', 1, 'Color', ml_blue) % median
plot(d / vs, stats1(:,4), '-', 'LineWidth', 1,  'Color', ml_green) % 95th percentile
plot(NaN, 'w-'); % for legend entry
plot(d / vs, stats2(:,1), '--', 'LineWidth', 1, 'Color', ml_red) % mean
plot(d / vs, stats2(:,2), '--', 'LineWidth', 1, 'Color', ml_blue) % median
plot(d / vs, stats2(:,4), '--', 'LineWidth', 1,  'Color', ml_green) % 95th percentile
% comment following four lines to exclude forward differences as in paper
plot(NaN, 'w-'); % for legend entry
plot(d / vs, stats3(:,1), ':', 'LineWidth', 1, 'Color', ml_red) % mean
plot(d / vs, stats3(:,2), ':', 'LineWidth', 1, 'Color', ml_blue) % median
plot(d / vs, stats3(:,4), ':', 'LineWidth', 1,  'Color', ml_green) % 95th percentile
% comment following four lines to exclude backward differences as in paper
plot(NaN, 'w-'); % for legend entry
plot(d / vs, stats4(:,1), '-.', 'LineWidth', 1, 'Color', ml_red) % mean
plot(d / vs, stats4(:,2), '-.', 'LineWidth', 1, 'Color', ml_blue) % median
plot(d / vs, stats4(:,4), '-.', 'LineWidth', 1,  'Color', ml_green) % 95th percentile
hold off
grid on

axis(ax_limits)

legend('Gradient-SDF', '   mean', '   median', '   95%', ...
    'Central differences', '   mean', '   median', '   95%', ...
    'Forward differences', '   mean', '   median', '   95%', ... % comment this line to exclude forward differences
    'Backward differences', '   mean', '   median', '   95%', ... % comment this line to exclude backward differences
    'Location', 'eastoutside')

%% cleanup

clear d ax_limits