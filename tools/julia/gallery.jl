# Helpers of the libKriging gallery notebooks (Julia), in one file: include("tools/julia/gallery.jl").
#
# The same functions, under the same names, as tools/python/gallery.py (where they are
# methods of Suite: suite.check_close(...) is check_close!(s, ...) here), and the same CSV
# outputs (results/<suite>.checks.csv, .metrics.csv), aggregated with the other
# languages by tools/report.py.
#
#   s = Suite(name)                                    a suite of checks, named after the notebook
#   check!(s, check, ok; value, expected, detail, severity)
#   check_true!, check_less!, check_greater!, check_between!, check_close!, check_finite!,
#   check_runs!(f, s, check) (do-block), check_skip!   the usual checks
#   metric!(s, case, package; name=value, ...)         record metrics
#   suite_summary(s)                                   end-of-notebook summary
#   suite_finalize!(s)                                 write the CSV files, error if an error-level check failed
#   read_data(name, stem), rmse, q2
#
# Paths are relative to the repository root ($LKGALLERY_ROOT, else the working
# directory, where the notebooks are executed).
using DelimitedFiles, Printf

const GALLERY_ROOT = get(ENV, "LKGALLERY_ROOT", ".")
const RESULTS_DIR = get(ENV, "LKGALLERY_RESULTS", joinpath(GALLERY_ROOT, "results"))
mkpath(RESULTS_DIR)

# ---- suite and checks ------------------------------------------------------------------

function read_known()
    f = joinpath(GALLERY_ROOT, "tools", "known_issues.csv")
    isfile(f) || error("run the notebooks from the repository root ($f not found)")
    rows = readdlm(f, ',', String; header=true, quotes=true)[1]   # quoted fields may hold commas
    Dict{Tuple{String,String},String}((rows[i, 1], rows[i, 2]) => rows[i, 3] for i in 1:size(rows, 1))
end

mutable struct Suite
    name::String
    checks::Vector{Vector{String}}
    metrics::Vector{Vector{String}}
    t0::Float64
    known::Dict{Tuple{String,String},String}
end
Suite(name) = Suite(name, [], [], time(), read_known())

fmt(v::Real) = @sprintf("%.6g", v)
fmt(v) = string(v)

function check!(s::Suite, check, ok; value="", expected="", detail="", severity="error")
    status = ok ? "PASS" : (severity == "error" ? "FAIL" : "WARN")
    if !ok && haskey(s.known, (s.name, check))
        status = "KNOWN"; detail = s.known[(s.name, check)] * " | " * detail
    end
    ok || println("[$status] $check: value=$(fmt(value)) expected $expected $detail")
    push!(s.checks, [s.name, "julia", check, severity, status, fmt(value), expected, detail])
    ok
end

check_true!(s::Suite, check, cond; detail="", severity="error") =
    check!(s, check, cond === true; value=string(cond), expected="true", detail, severity)
check_less!(s::Suite, check, value, bound; detail="", severity="error") =
    check!(s, check, isfinite(value) && value < bound; value, expected="< $bound", detail, severity)
check_greater!(s::Suite, check, value, bound; detail="", severity="error") =
    check!(s, check, isfinite(value) && value > bound; value, expected="> $bound", detail, severity)
check_between!(s::Suite, check, value, lo, hi; detail="", severity="error") =
    check!(s, check, isfinite(value) && lo <= value <= hi; value, expected="in [$lo, $hi]", detail, severity)

function check_close!(s::Suite, check, a, b; rtol=1e-6, atol=0.0, detail="", severity="error")
    a = Float64.(vec(collect(a))); b = Float64.(vec(collect(b)))   # vec first: Float64.() of a 0-dim array is a scalar
    length(a) == length(b) || return check!(s, check, false; value="length $(length(a))",
                                            expected="length $(length(b))", detail, severity)
    gap = abs.(a .- b)
    ok = all(isfinite, a) && all(gap .<= atol .+ rtol .* abs.(b))
    rel = maximum(gap ./ max.(abs.(b), 1e-300))
    check!(s, check, ok; value=rel, expected="rtol $rtol, atol $atol", detail, severity)
end

check_finite!(s::Suite, check, a; detail="", severity="error") =
    check!(s, check, !isempty(a) && all(isfinite, a); value="$(count(!isfinite, a)) non-finite",
           expected="all finite", detail, severity)

function check_runs!(f, s::Suite, check; severity="error")
    # check_runs!(s, "fit") do ... end: an error becomes a failed check instead of stopping the notebook
    try
        f()
        check!(s, check, true; value="ok", expected="no error", severity)
    catch e
        check!(s, check, false; value=string(typeof(e)), expected="no error",
               detail=first(sprint(showerror, e), 200), severity)
    end
end

function check_skip!(s::Suite, check, reason)
    push!(s.checks, [s.name, "julia", check, "info", "SKIP", "", "", reason])
    println("[SKIP] $check: $reason")
end

# ---- metrics and results ---------------------------------------------------------------

metric!(s::Suite, case, package; kw...) =
    for (k, v) in kw; push!(s.metrics, [s.name, "julia", case, package, string(k), fmt(v)]); end

function suite_summary(s::Suite)
    for r in s.metrics
        println(rpad(r[3], 28), rpad(r[4], 28), rpad(r[5], 12), r[6])
    end
    st = [c[5] for c in s.checks]
    println(length(st), " checks: ", join(["$k $(count(==(k), st))" for k in ("PASS", "FAIL", "WARN", "KNOWN", "SKIP")], ", "))
    for c in s.checks
        c[5] == "PASS" || println(rpad(c[5], 7), rpad(c[4], 7), c[3], ": value ", c[6], ", expected ", c[7], " ", c[8])
    end
end

csvq(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""

function write_rows(f, header, rows)
    open(f, "w") do io
        println(io, join(header, ","))
        for r in rows; println(io, join(csvq.(r), ",")); end
    end
end

function suite_finalize!(s::Suite)
    metric!(s, "_suite", "_"; elapsed_s=time() - s.t0)
    write_rows(joinpath(RESULTS_DIR, s.name * ".checks.csv"),
               ["suite", "lang", "check", "severity", "status", "value", "expected", "detail"], s.checks)
    write_rows(joinpath(RESULTS_DIR, s.name * ".metrics.csv"),
               ["suite", "lang", "case", "package", "metric", "value"], s.metrics)
    st = [c[5] for c in s.checks]
    println(s.name, ": ", join(["$k $(count(==(k), st))" for k in ("PASS", "FAIL", "WARN", "KNOWN", "SKIP")], ", "))
    failed = [c[3] for c in s.checks if c[5] == "FAIL"]
    isempty(failed) || error("$(length(failed)) check(s) failed in $(s.name): " * join(first(failed, 20), "; "))
end

# ---- data and metrics ------------------------------------------------------------------

read_data(name, stem) = readdlm(joinpath(GALLERY_ROOT, "data", "$(name)_$(stem).csv"), ',', Float64)
rmse(y, m) = sqrt(sum((y .- m) .^ 2) / length(y))
q2(y, m) = 1 - sum((y .- m) .^ 2) / sum((y .- sum(y) / length(y)) .^ 2)
