function [stats, hist2d] = phi_statistics(d, D, phi, max_phi)
% PHI_STATISTICS computes statistical measures on angular deviation
%
% stats = PHI_STATISTICS(d, D, phi)
% stats = PHI_STATISTICS(d, D, phi, max_phi)
% [stats, hist2d] = PHI_STATISTICS(d, D, phi)
% [stats, hist2d] = PHI_STATISTICS(d, D, phi, max_phi)
%
% Input:
% - d: vector of distances for which the statistics shall be computed
% - D: 3D SDF volume
% - phi: 3D volume of deviation angles (in degrees)
% - max_phi (optional): cut-off value for phi in histogram (default: 2°)
%
% Output:
% - stats: array of size length(d) x 4, with the four columns
%    1: mean of all phi with abs(D)<d
%    2: median of all phi with abs(D)<d
%    3: root mean square of all phi with abs(D)<d
%    4: 95th percentile of all phi with abs(D)<d
% - hist (optional):  histogram of all phi with abs(D)<d, per entry in d

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

if nargin<4
    max_phi = 25;
end

% statistical measures: mean, median, RMSE, 95th percentile
stats = zeros(size(d,1), 4);

bin_edges = 0:.2:max_phi;
if nargout > 1
    hist2d = zeros(length(bin_edges)-1, size(d,1));
end

for k=1:length(d)
    idx = abs(D(:))<d(k) & ~isnan(phi(:));
    stats(k,1) = mean(phi(idx));
    stats(k,3) = sqrt(mean(phi(idx).^2));
    stats(k,[2 4]) = prctile(phi(idx),[50 95]);
    if nargout > 1
        hist2d(:,k) = histcounts(phi(idx), bin_edges, 'Normalization', 'probability');
    end
end