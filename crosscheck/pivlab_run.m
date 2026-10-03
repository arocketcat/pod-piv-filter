% pivlab_run.m  -- PIVlab 3.09 piv_FFTmulti on a folder of pair_XXXX_A/B.tif
% usage (Octave or MATLAB):
%   pivlab_run(pivlab_dir, image_dir, out_mat, k0, k1, prep)
%   prep: 0 none, 1 CLAHE (PIVlab default; needs MATLAB, Octave lacks adapthisteq),
%         2 PIVlab high-pass filter (size 15)
function pivlab_run(pivlab_dir, image_dir, out_mat, k0, k1, prep)
  addpath(pivlab_dir); cd(pivlab_dir);   % smoothn checks for +misc/dctn relative to the working folder
  if exist('OCTAVE_VERSION', 'builtin'), pkg load image; end
  n = k1 - k0 + 1;
  X = []; Y = []; U = []; V = []; T = [];
  for j = 1:n
    k = k0 + j - 1;
    a = imread(fullfile(image_dir, sprintf('pair_%04d_A.tif', k)));
    b = imread(fullfile(image_dir, sprintf('pair_%04d_B.tif', k)));
    if prep == 1      % PIVlab default preprocessing: CLAHE, 64 px tiles
      a = preproc.PIVlab_preproc(a, [], 1, 64, 0, 15, 0, 0, 3, 0, 1);
      b = preproc.PIVlab_preproc(b, [], 1, 64, 0, 15, 0, 0, 3, 0, 1);
    elseif prep == 2  % PIVlab high-pass filter, kernel size 15
      a = preproc.PIVlab_preproc(a, [], 0, 64, 1, 15, 0, 0, 3, 0, 1);
      b = preproc.PIVlab_preproc(b, [], 0, 64, 1, 15, 0, 0, 3, 0, 1);
    end
    % 64 px first pass (step 32), then three 32 px passes (step 16), linear deformation (PIVlab default; Octave interp2 spline needs a regular grid)
    [x, y, u, v, tv] = piv.piv_FFTmulti(a, b, 64, 32, 1, [], [], 4, 32, 32, 32, '*linear', 0, 0, 0, 0, 0, 0.025);
    % PIVlab's own post processing as in its command line example: stdev + local median, then inpaint
    [uf, vf] = postproc.PIVlab_postproc(u, v, [], [], [], 1, 7, 1, 3);
    uf = misc.inpaint_nans(double(uf), 4); vf = misc.inpaint_nans(double(vf), 4);
    if j == 1
      X = double(x); Y = double(y);
      U = zeros([size(x) n]); V = U; T = U;
    end
    U(:,:,j) = uf; V(:,:,j) = vf; T(:,:,j) = double(isnan(u) | isnan(v));
  end
  save('-v7', out_mat, 'X', 'Y', 'U', 'V', 'T', 'k0', 'k1');
end
