# %% [markdown]
# # 10 — Comparison with Python GP packages
#
# Two kinds of comparisons, which should not be confused:
#
# 1. **Same model, same hyperparameters → same numbers.** With θ, σ², β fixed,
#    libKriging's predictor and likelihood are textbook formulas: another
#    implementation must give the same values to rounding error. Done against
#    scikit-learn (`RBF` = libKriging's `gauss`, both product-form) and
#    OpenTURNS (product of 1D Matérn 5/2 = libKriging's `matern5_2`). A gap
#    here is a bug in one of the packages (error-level).
# 2. **Same problem, each package's default fit → accuracy and cost.** Each
#    package estimates its hyperparameters its own way (optimizer, restarts,
#    bounds, mean model), so differences are expected; the table is a
#    benchmark, and the checks are warn-level: libKriging should not be
#    markedly less accurate than the best package.
#
# Packages (each optional: skipped if not installed): scikit-learn, SMT,
# OpenTURNS, GPy.

# %%
import time
import warnings

import numpy as np
import pandas as pd
import pylibkriging as lk

import lkbench
from lkbench import TESTS, design, metrics, compat

warnings.filterwarnings("ignore")
suite = lkbench.Suite("py10_compare_python_packages")
sk = compat.optional("sklearn.gaussian_process")
smt = compat.optional("smt.surrogate_models")
ot = compat.optional("openturns")
GPy = compat.optional("GPy")
print("pylibkriging", compat.version(), "| sklearn", bool(sk), "| SMT", bool(smt), "| OpenTURNS", bool(ot),
      "| GPy", bool(GPy))


def lk_fixed(y, X, kernel, theta, sigma2, beta):
    return lk.Kriging(y, X, kernel, optim="none",
                      parameters={"theta": np.reshape(theta, (1, -1)), "is_theta_estim": False,
                                  "sigma2": float(sigma2), "is_sigma2_estim": False,
                                  "beta": np.reshape(beta, (-1, 1)), "is_beta_estim": False})

# %% [markdown]
# ## 1. Same hyperparameters, same numbers

# %%
tf = TESTS["hartmann3"]
X, y, Xt, yt = design.dataset(tf, 40, n_test=300, seed=81)
kg = lk.Kriging(y, X, "gauss")
theta, sigma2, beta = np.ravel(kg.theta()), float(kg.sigma2()), float(np.ravel(kg.beta())[0])
k_simple = lk_fixed(y, X, "gauss", theta, sigma2, beta)  # simple kriging, known beta
m_lk, s_lk, *_ = (np.ravel(a) for a in k_simple.predict(Xt, return_stdev=True)[:2])

if sk is not None:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    kern = ConstantKernel(sigma2, "fixed") * RBF(theta, "fixed")
    gp = GaussianProcessRegressor(kern, alpha=1e-12, optimizer=None, normalize_y=False).fit(X, y - beta)
    m_sk, s_sk = gp.predict(Xt, return_std=True)
    suite.close("sklearn RBF == libKriging gauss: simple kriging mean", m_sk + beta, m_lk, rtol=1e-6,
                atol=1e-7 * np.std(y))
    suite.close("sklearn RBF == libKriging gauss: simple kriging stdev (beta fixed)", s_sk, s_lk, rtol=1e-4,
                atol=1e-6 * np.sqrt(sigma2))
    # with beta estimated, libKriging's stdev is the universal kriging one: the simple kriging variance
    # plus the trend-estimation term (f - F'C^-1 c)^2 / (F'C^-1 F), computed here by hand
    C = kern(X); c = kern(X, Xt)
    w = np.linalg.solve(C, c)
    one = np.ones((len(y), 1))
    v_uk = s_sk ** 2 + ((1 - one.T @ w) ** 2).ravel() / (one.T @ np.linalg.solve(C, one)).item()
    s_uk = np.ravel(kg.predict(Xt, return_stdev=True)[1])
    suite.close("libKriging gauss (beta estimated) stdev == universal kriging formula", s_uk, np.sqrt(v_uk),
                rtol=1e-4, atol=1e-6 * np.sqrt(sigma2))
    # the (concentrated) log-likelihood at the MLE equals the full Gaussian log-density at (theta, sigma2, beta)
    lml = gp.log_marginal_likelihood_value_
    suite.close("sklearn log-marginal-likelihood == libKriging logLikelihood()", lml, kg.logLikelihood(),
                rtol=1e-6)
else:
    suite.skip("sklearn exact comparison", "scikit-learn not installed")

# %%
if ot is not None:
    km = lk.Kriging(y, X, "matern5_2")
    th, s2 = np.ravel(km.theta()), float(km.sigma2())
    cov = ot.ProductCovarianceModel([ot.MaternModel([t], [1.0], 2.5) for t in th])
    cov.setAmplitude([np.sqrt(s2)])
    basis = ot.ConstantBasisFactory(3).build()
    algo = ot.KrigingAlgorithm(ot.Sample(X), ot.Sample(y.reshape(-1, 1)), cov, basis)
    algo.setOptimizeParameters(False)
    ot.ResourceMap.SetAsScalar("GeneralLinearModelAlgorithm-DefaultOptimizationLowerBound", 1e-6)
    algo.run()
    res = algo.getResult()
    m_ot = np.ravel(res.getConditionalMean(ot.Sample(Xt)))
    v_ot = np.ravel(res.getConditionalMarginalVariance(ot.Sample(Xt)))
    # OpenTURNS re-estimates beta by GLS: compare with libKriging's universal kriging at the same theta, sigma2
    ku = lk.Kriging(y, X, "matern5_2", optim="none",
                    parameters={"theta": th.reshape(1, -1), "is_theta_estim": False,
                                "sigma2": s2, "is_sigma2_estim": False})
    m_u, s_u = (np.ravel(a) for a in ku.predict(Xt, return_stdev=True)[:2])
    suite.close("OpenTURNS product Matern 5/2 == libKriging matern5_2: mean", m_ot, m_u, rtol=1e-6,
                atol=1e-6 * np.std(y))
    suite.close("OpenTURNS product Matern 5/2 == libKriging matern5_2: variance (universal kriging)", v_ot, s_u ** 2,
                rtol=1e-4, atol=1e-8 * s2)
else:
    suite.skip("OpenTURNS exact comparison", "openturns not installed")

# %% [markdown]
# ## 2. Default fits on a set of test functions
#
# Matérn 5/2 (or the closest the package offers), constant mean, maximum
# likelihood with each package's default optimizer, 3 designs per case.

# %%
def fit_libkriging(X, y):
    k = lk.Kriging(y, X, "matern5_2")
    return lambda Z: tuple(np.ravel(a) for a in k.predict(Z, return_stdev=True)[:2])


def fit_libkriging_ms(X, y):
    k = lk.Kriging(y, X, "matern5_2", optim="BFGS10")
    return lambda Z: tuple(np.ravel(a) for a in k.predict(Z, return_stdev=True)[:2])


PACKAGES = {"pylibkriging": fit_libkriging, "pylibkriging BFGS10": fit_libkriging_ms}

if sk is not None:
    from sklearn.gaussian_process.kernels import Matern

    def fit_sklearn(X, y):
        d = X.shape[1]
        kern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.full(d, 0.5), (1e-3, 1e3), nu=2.5)
        gp = GaussianProcessRegressor(kern, alpha=1e-10, normalize_y=True, n_restarts_optimizer=5,
                                      random_state=0).fit(X, y)
        return lambda Z: gp.predict(Z, return_std=True)
    PACKAGES["scikit-learn"] = fit_sklearn

if smt is not None:
    def fit_smt(X, y):
        m = smt.KRG(corr="matern52", poly="constant", print_global=False)
        m.set_training_values(X, y)
        m.train()
        return lambda Z: (np.ravel(m.predict_values(Z)), np.sqrt(np.maximum(np.ravel(m.predict_variances(Z)), 0)))
    PACKAGES["SMT"] = fit_smt

if ot is not None:
    def fit_openturns(X, y):
        d = X.shape[1]
        cov = ot.MaternModel([0.5] * d, [float(np.std(y))], 2.5)
        algo = ot.KrigingAlgorithm(ot.Sample(X), ot.Sample(y.reshape(-1, 1)), cov, ot.ConstantBasisFactory(d).build())
        algo.run()
        r = algo.getResult()
        return lambda Z: (np.ravel(r.getConditionalMean(ot.Sample(Z))),
                          np.sqrt(np.maximum(np.ravel(r.getConditionalMarginalVariance(ot.Sample(Z))), 0)))
    PACKAGES["OpenTURNS"] = fit_openturns

if GPy is not None:
    def fit_gpy(X, y):
        kern = GPy.kern.Matern52(X.shape[1], ARD=True)
        m = GPy.models.GPRegression(X, y.reshape(-1, 1), kern, mean_function=None)
        m.Gaussian_noise.variance.fix(1e-8)
        m.optimize_restarts(3, verbose=False)
        def p(Z):
            mu, var = m.predict(Z)
            return np.ravel(mu), np.sqrt(np.maximum(np.ravel(var) - 1e-8, 0))
        return p
    PACKAGES["GPy"] = fit_gpy

CASES = {"branin": 30, "six_hump_camel": 40, "hartmann3": 50, "ishigami": 100, "friedman": 80,
         "otl_circuit": 80, "piston": 100, "borehole": 80, "wing_weight": 100, "hartmann6": 120}
rows = []
for fname, n in CASES.items():
    tf = TESTS[fname]
    for rep in range(3):
        X, y, Xt, yt = design.dataset(tf, n, n_test=1000, seed=100 + rep)
        for pkg, fit in PACKAGES.items():
            try:
                t0 = time.time()
                predictor = fit(X, y)
                t_fit = time.time() - t0
                t0 = time.time()
                m, s = predictor(Xt)
                t_pred = time.time() - t0
                rows.append(dict(function=fname, n=n, rep=rep, package=pkg, fit_s=t_fit, predict_s=t_pred,
                                 **metrics.all_metrics(yt, m, s)))
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(function=fname, n=n, rep=rep, package=pkg, error=str(exc)[:100]))
                if pkg.startswith("pylibkriging"):
                    suite.true(f"{fname}/rep{rep}/{pkg}: fit", False, detail=str(exc)[:200])
res = pd.DataFrame(rows)
summary = res.groupby(["function", "package"], sort=False)[["q2", "nlpd", "coverage95", "fit_s"]].median()
for (fname, pkg), r in summary.iterrows():
    suite.metric(fname, pkg, **r.to_dict())
summary.unstack("package")["q2"].round(4)

# %%
summary.unstack("package")[["fit_s"]].round(3)

# %%
for fname in CASES:
    s = summary.loc[fname]["q2"]
    others = s.drop(index=[p for p in s.index if p.startswith("pylibkriging")])
    if len(others):
        suite.greater(f"{fname}: Q2(pylibkriging) - best other package", s["pylibkriging"] - others.max(), -0.02,
                      severity="warn", detail=f"best: {others.idxmax()}")
    suite.greater(f"{fname}: Q2(pylibkriging BFGS10) - best other package", s["pylibkriging BFGS10"] - others.max(),
                  -0.05, detail=f"best: {others.idxmax()}") if len(others) else None
    suite.greater(f"{fname}: Q2(pylibkriging BFGS10) > 0.5", s["pylibkriging BFGS10"], 0.5)

# %%
suite.finalize()
