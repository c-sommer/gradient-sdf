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


%% setup

% intrinsics matrix - mimic Kinect
K = [525 0 319.5; 0 525 239.5; 0 0 1];

% spheres in format [cx cy cz R]

% create five random non-intersecting spheres
% comment following code block to load spheres from file instead (if one exists already)
% parameters are chosen to make sure depth values are in Kinect range
spheres = [rand(1, 3)-0.5 0.0625+0.4375*rand(1)];
while size(spheres,1)<5
    c = rand(1, 3) - 0.5;
    r = 0.0625 + 0.4375*rand(1);
    if all(sqrt(sum((spheres(:,1:3)-c).^2,2)) > (spheres(:,4)+r))
        spheres = [spheres; c r];
    end
end
save('spheres.mat', 'spheres')

% uncomment next line to load spheres from file
% load('spheres.mat')

% load poses from file (spiral shape) in TUM RGB-D format
poses = load('poses.txt');
t_vec = poses(:,2:4);
q_vec = poses(:,[8 5:7]);
clear poses

% pixel meshgrid
[u,v] = meshgrid(0:639, 0:479);
coeff_u = (u - K(1,3)) / K(1,1);
coeff_v = (v - K(2,3)) / K(2,2);
A = coeff_u.^2 + coeff_v.^2 + 1;

% output path, should contain subfolders depth/ and rgb/
out_path = './'; % modify accordingly
% create subfolders if they do not exist
if ~exist([out_path 'depth/'], 'dir')
       mkdir([out_path 'depth/']);
end
if ~exist([out_path 'rgb/'], 'dir')
       mkdir([out_path 'rgb/']);
end

color = [0 0.4470 0.7410; 
        0.8500 0.3250 0.0980;
        0.9290 0.6940 0.1250;
        0.4940 0.1840 0.5560;
       0.4660 0.6740 0.1880];

%% rendering

for k=1:length(t_vec)

    I = inf(480, 640); % depth image, 1 channel
    I_color = zeros(480, 640, 3); % color image, 3 channels

    t = t_vec(k,:);
    Q = q_vec(k,:);

    R = quat2rotm(Q);

    c = (spheres(:,1:3) - t) * R;
    c_sq_r = c(:,1).^2 + c(:,2).^2 + c(:,3).^2 - spheres(:,4).^2;

    for s=1:size(spheres,1)
        I_tmp = zeros(480, 640);
        % A, B, and C are coefficients of A*z^2 + B*z + C = 0
        B = coeff_u * c(s,1) + coeff_v * c(s,2) + c(s,3);
        B = -2 * B;
        C = c_sq_r(s);
        mask = B.^2 < 4 * A * C;
        I_tmp(mask) = inf;
        I_tmp(~mask) = .5 * (- B(~mask) - sqrt(B(~mask).^2 - 4 * A(~mask) * C)) ./ A(~mask);
        
        color_mask = (I >= I_tmp)& ~mask;

        for ch=1:3
            color_tmp = I_color(:,:,ch);
            color_tmp(color_mask) = color(s,ch);
            I_color(:,:,ch) = color_tmp;
        end

        I = min(I, I_tmp);
    end

    I(isinf(I)) = 0;
    I = add_kinect_noise(I);

    % visualize current renderings
    subplot(121)
    imagesc(I)
    axis image
    subplot(122)
    image(I_color)
    axis image

    % write to file
     imwrite(uint16(1000 * I), [out_path 'depth/' num2str(k,'%03d') '.png'])
     imwrite(I_color, [out_path 'rgb/' num2str(k,'%03d') '.png'])

    pause(.2)
end