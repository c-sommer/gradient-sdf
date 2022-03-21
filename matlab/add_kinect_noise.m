function z = add_kinect_noise(z, mask)
% ADD_KINECT_NOISE adds a Kinect-like noise pattern to a synthetic depth image
%
% z = ADD_KINECT_NOISE(z)
% z = ADD_KINECT_NOISE(z, mask)
%
% Input:
% - z: synthetic depth image
% - mask (optional): values to which noise shall be added, defaul: all >0
%
% Output:
% - z: noisy depth image, same size as original one
%
% Noise model follows Khoshelham et al. 2012

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

if nargin<2
    mask = z>0;
end

sz = size(z);

% Compute normalized disparity d
% Eq. (5) and experiments:
% in cm: z^-1 = -2.85e-5 * d + 0.03
% in m:  z^-1 = -2.85e-3 * d + 3
%           d = (3 - z^-1) / 2.85e-3
d = zeros(sz);
d(mask) = (3 - 1./z(mask)) / 2.85e-3;

% Add noise to normalized disparity
% sigma_d = 0.5 pixels (p. 1450)
tmp = 0.5 * randn(sz);
d(mask) = d(mask) + tmp(mask);

% Add quantization to normalized disparity
d = round(d);

% Go back to z^-1 and z
z_inv = -2.85e-3 * d + 3;
z(mask) = 1 ./ z_inv(mask);