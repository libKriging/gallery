# Same CSV outputs as lkbench/checks.py (see there), for the Julia notebooks.
# Paths are relative to the repository root, where the notebooks are executed.
using DelimitedFiles, Printf

const RESULTS_DIR = get(ENV, "LKBENCH_RESULTS", "results")
mkpath(RESULTS_DIR)

function read_known()
    f = joinpath("tools", "known_issues.csv")
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

function add_check!(s::Suite, check, ok; value="", expected="", detail="", severity="error")
    status = ok ? "PASS" : (severity == "error" ? "FAIL" : "WARN")
    if !ok && haskey(s.known, (s.name, check))
        status = "KNOWN"; detail = s.known[(s.name, check)] * " | " * detail
    end
    ok || println("[$status] $check: value=$(fmt(value)) expected $expected $detail")
    push!(s.checks, [s.name, "julia", check, severity, status, fmt(value), expected, detail])
    ok
end

function check_close!(s::Suite, check, a, b; rtol=1e-6, atol=0.0, detail="", severity="error")
    a = vec(Float64.(collect(a))); b = vec(Float64.(collect(b)))
    length(a) == length(b) || return add_check!(s, check, false; value="length $(length(a))",
                                                expected="length $(length(b))", detail, severity)
    gap = abs.(a .- b)
    ok = all(isfinite, a) && all(gap .<= atol .+ rtol .* abs.(b))
    rel = maximum(gap ./ max.(abs.(b), 1e-300))
    add_check!(s, check, ok; value=rel, expected="rtol $rtol, atol $atol", detail, severity)
end

add_metric!(s::Suite, case, package; kw...) =
    for (k, v) in kw; push!(s.metrics, [s.name, "julia", case, package, string(k), fmt(v)]); end

csvq(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""

function write_rows(f, header, rows)
    open(f, "w") do io
        println(io, join(header, ","))
        for r in rows; println(io, join(csvq.(r), ",")); end
    end
end

function finalize!(s::Suite)
    add_metric!(s, "_suite", "_"; elapsed_s=time() - s.t0)
    write_rows(joinpath(RESULTS_DIR, s.name * ".checks.csv"),
               ["suite", "lang", "check", "severity", "status", "value", "expected", "detail"], s.checks)
    write_rows(joinpath(RESULTS_DIR, s.name * ".metrics.csv"),
               ["suite", "lang", "case", "package", "metric", "value"], s.metrics)
    st = [c[5] for c in s.checks]
    println(s.name, ": ", join(["$k $(count(==(k), st))" for k in ("PASS", "FAIL", "WARN", "KNOWN", "SKIP")], ", "))
    failed = [c[3] for c in s.checks if c[5] == "FAIL"]
    isempty(failed) || error("$(length(failed)) check(s) failed in $(s.name): " * join(first(failed, 20), "; "))
end

read_data(name, stem) = readdlm(joinpath("data", "$(name)_$(stem).csv"), ',', Float64)
q2(y, m) = 1 - sum((y .- m) .^ 2) / sum((y .- sum(y) / length(y)) .^ 2)
