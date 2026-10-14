% %% [markdown]
% # Octave · Kriging — cross-binding consistency: the Octave side
%
% Same protocol as `python_Kriging_cross-binding-reference.ipynb` (see there), on the
% shared datasets of `data/`; writes `results/xbind_octave.csv`, compared with
% the Python values by `tools/report.py`. mLibKriging must be on the path
% (`OCTAVE_PATH` or `addpath`).

% %%
addpath(fullfile(pwd, "tools", "octave"));
s = lkb_suite("octave_Kriging_cross-binding");
out = {};
function out = put(out, dataset, quantity, values)
  v = double(values(:));
  for i = 1:numel(v)
    out(end + 1, :) = {"octave", dataset, quantity, i, sprintf("%.17g", v(i))};
  end
end

% %%
for name = {"branin", "hartmann3", "borehole"}
  name = name{1};
  X = read_data(name, "X"); y = read_data(name, "y"); Xt = read_data(name, "Xt")(1:50, :);
  d = columns(X);
  theta0 = 0.3 * ones(1, d);
  kf = Kriging(y, X, "matern5_2", "constant", false, "none", "LL", Params("theta", theta0, "is_theta_estim", false));
  v = kf.sigma2(); out = put(out, name, "fixed_sigma2", v);
  v = kf.beta(); out = put(out, name, "fixed_beta", v);
  ll = kf.logLikelihood();
  out = put(out, name, "fixed_ll", ll);
  loo = kf.leaveOneOut();
  out = put(out, name, "fixed_loo", loo);
  [m, sd] = kf.predict(Xt, true, false, false);
  out = put(out, name, "fixed_mean", m);
  out = put(out, name, "fixed_stdev", sd);
  [llf, grad] = kf.logLikelihoodFun(theta0', true, false);
  out = put(out, name, "llfun", llf);
  out = put(out, name, "llfun_grad", grad);
  s = lkb_close(s, [name ": logLikelihood() == logLikelihoodFun(theta0)"], ll, llf, 1e-10);
  k = Kriging(y, X, "matern5_2");
  v = k.theta(); out = put(out, name, "fit_theta", v);
  llk = k.logLikelihood();
  out = put(out, name, "fit_ll", llk);
  [mk, ~] = k.predict(Xt, false, false, false);
  out = put(out, name, "fit_mean", mk);
  S = kf.simulate(int32(3), int32(123), Xt(1:5, :), false);
  out = put(out, name, "sim", S);
end
d = getenv("LKBENCH_RESULTS"); if isempty(d), d = "results"; end
if ~exist(d, "dir"), mkdir(d); end
fid = fopen(fullfile(d, "xbind_octave.csv"), "w");
fprintf(fid, "lang,dataset,quantity,index,value\n");
for i = 1:size(out, 1)
  fprintf(fid, "%s,%s,%s,%d,%s\n", out{i, :});
end
fclose(fid);
printf("%d values written\n", size(out, 1));

% %%
lkb_finalize(s);
