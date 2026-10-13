% %% [markdown]
% # Octave 02 — mLibKriging vs STK
%
% [STK](https://github.com/stk-kriging/stk) (Small Toolbox for Kriging) is the
% reference kriging toolbox for Octave/MATLAB.
%
% 1. **Same hyperparameters → same numbers** (error-level). STK's anisotropic
%    Gaussian covariance is σ² exp(−‖(x−x')/ρ‖²) and libKriging's `gauss` is
%    σ² exp(−½ Σ ((xⱼ−x'ⱼ)/θⱼ)²): the same kernel with ρ = θ √2. With a constant
%    mean (ordinary/universal kriging, β estimated by GLS in both), the
%    predicted mean and variance at fixed (θ, σ²) must coincide. (STK's Matérn
%    covariances are radial, libKriging's are product-form: not comparable
%    hyperparameter by hyperparameter, hence the Gaussian kernel here.)
% 2. **Default fits** on the shared datasets of `data/`: Matérn 5/2, each
%    toolbox's own estimation (warn-level comparison of Q²).
%
% STK's location: the `STK_PATH` environment variable (skipped if unset).

% %%
s = lkb_suite("oct02_vs_stk");
stk_path = getenv("STK_PATH");
have_stk = ~isempty(stk_path) && exist(fullfile(stk_path, "stk_init.m"), "file");
if have_stk
  run(fullfile(stk_path, "stk_init.m"));
  printf("STK %s\n", stk_version());
else
  s = lkb_check(s, "STK available", false, "absent", "STK_PATH", "STK not found: comparison skipped", "warn");
end
q2 = @(y, m) 1 - sum((y - m).^2) / sum((y - mean(y)).^2);
datasets = {"branin", "hartmann3", "borehole"};

% %% [markdown]
% ## 1. Same hyperparameters, same numbers

% %%
if have_stk
  for i = 1:numel(datasets)
    name = datasets{i};
    X = read_data(name, "X"); y = read_data(name, "y"); Xt = read_data(name, "Xt");
    d = columns(X);
    theta = linspace(0.2, 0.6, d);
    k = Kriging(y, X, "gauss", "constant", false, "none", "LL", Params("theta", theta, "is_theta_estim", false));
    sigma2 = k.sigma2();
    [m_lk, s_lk] = k.predict(Xt, true, false, false);
    model = stk_model(@stk_gausscov_aniso, d);
    model.param = [log(sigma2); -log(theta(:) * sqrt(2))];
    model.lm = stk_lm_constant();
    zp = stk_predict(model, X, y, Xt);
    s = lkb_close(s, [name ": gauss UK mean == STK"], m_lk, zp.mean, 1e-6, 1e-8 * std(y));
    s = lkb_close(s, [name ": gauss UK variance == STK"], s_lk.^2, zp.var, 1e-4, 1e-8 * sigma2);
  end
end

% %% [markdown]
% ## 2. Default fits

% %%
for i = 1:numel(datasets)
  name = datasets{i};
  X = read_data(name, "X"); y = read_data(name, "y"); Xt = read_data(name, "Xt"); yt = read_data(name, "yt");
  d = columns(X);
  tic; k = Kriging(y, X, "matern5_2"); t = toc;
  [m, ~] = k.predict(Xt, false, false, false);
  q_lk = q2(yt, m);
  s = lkb_metric(s, name, "mLibKriging", "q2", q_lk); s = lkb_metric(s, name, "mLibKriging", "fit_s", t);
  tic; k10 = Kriging(y, X, "matern5_2", "constant", false, "BFGS10", "LL"); t = toc;
  [m, ~] = k10.predict(Xt, false, false, false);
  q_lk10 = q2(yt, m);
  s = lkb_metric(s, name, "mLibKriging BFGS10", "q2", q_lk10); s = lkb_metric(s, name, "mLibKriging BFGS10", "fit_s", t);
  s = lkb_check(s, [name ": Q2(mLibKriging BFGS10) > 0.9"], q_lk10 > 0.9, q_lk10, "> 0.9");
  if have_stk
    tic;
    model = stk_model(@stk_materncov52_aniso, d);
    model = stk_param_estim(model, X, y);  % (STK >= 2.8 returns the model)
    zp = stk_predict(model, X, y, Xt);
    t = toc;
    q_stk = q2(yt, zp.mean);
    s = lkb_metric(s, name, "STK", "q2", q_stk); s = lkb_metric(s, name, "STK", "fit_s", t);
    s = lkb_check(s, [name ": Q2(mLibKriging) - Q2(STK)"], q_lk - q_stk > -0.02, q_lk - q_stk, "> -0.02", "", "warn");
  end
end
disp(s.metrics(:, 3:6))

% %%
lkb_finalize(s);
