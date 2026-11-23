"""L6: the job interface. A logical job in; compiled circuits, run records, a manifest and a report out; counts
mapped back to logical bits after the run.

A logical job (JSON):

    {"kind": "echo" | "sh6" | "qca", "n": 40, "D": 12, "hz": 0.0, "h": 0.6, "shots": 1000,
     "inputs": ["zero", "one", "neel", "antineel"]}            # the four inputs by default

Each kind is a ring circuit with a known answer (echo: the input flipped; sh6, qca: the input). Every input comes
with its readout reference on the same physical qubits. The folder layout matches TOQB's (runs/<cell>.json +
<cell>.qpy, job/manifest.json, job/circuits.qpy, then job/raw_counts.json and job/submission.json), so TOQB's readers
and psf-os's read the same files.

Forward scoring (site_signal_forward, full_match) is ported from TOQB toqb/qca_test.py at 8d44f05.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

from psf_os import circuits as C
from psf_os.compile import VERSION, RuleCompiler, measured_qubits
from psf_os.registry import sha256_file
from psf_os.scan import INPUTS, input_bits

KINDS = ("echo", "sh6", "qca", "echo_clifford", "clifford_random", "clifford_stab", "kising_forward", "cluster_parity")
PROBLEMS = ("clifford_random", "clifford_stab", "kising_forward", "cluster_parity")  # practice problems with a recorded exact answer per site


def cell_name(job):
    if job["kind"] == "echo":
        tag = f"_lf{''.join(map(str, job['flip']))}" if job.get("flip") else ""
        return f"kecho_n{job['n']}_D{job['D']}_hz{job.get('hz', 0.0):g}_h{job.get('h', 0.6):g}-fold_rule{tag}"
    if job["kind"] == "echo_clifford":
        return f"kecho_n{job['n']}_D{job['D']}_hz0_h0.785398-fold_rule"
    if job["kind"] == "clifford_random":
        return f"pclif_n{job['n']}_k{job['k']}_s{job['seed']}-fold_rule"
    if job["kind"] == "clifford_stab":
        return f"pstab_n{job['n']}_k{job['k']}_s{job['seed']}_o{job.get('offset', 0)}-fold_rule"
    if job["kind"] == "cluster_parity":
        return f"ppar_n{job['n']}-fold_rule"
    if job["kind"] == "kising_forward":
        return f"pkis_n{job['n']}_L{job['L']}_h{job.get('h', 0.6):g}-fold_rule"
    if job["kind"] == "sh6":
        return f"kqsh_n{job['n']}_D{job['D']}-fold_rule"
    return f"kqca_n{job['n']}_D{job['n']}-fold_rule"


def logical_circuit(job, bits):
    n = job["n"]
    if job["kind"] == "echo":
        if job["D"] % 2:
            raise ValueError("the echo's D is 2L")
        qc = C.echo_circuit(bits, n, job["D"] // 2, job.get("h", 0.6), job.get("hz", 0.0))
        if job.get("flip"):
            qc = C.frame_flip(qc, n, job["flip"])
        return qc
    if job["kind"] == "echo_clifford":
        return C.echo_clifford_circuit(bits, n, job["D"] // 2)
    if job["kind"] == "clifford_random":
        return C.clifford_random_circuit(bits, n, job["k"], job["seed"])
    if job["kind"] == "clifford_stab":
        return C.clifford_stab_circuit(bits, n, job["k"], job["seed"], stab_observables(job))
    if job["kind"] == "cluster_parity":
        return C.cluster_parity_circuit(bits, n)
    if job["kind"] == "kising_forward":
        return C.kising_forward_circuit(bits, n, job["L"], job.get("h", 0.6), job.get("hz", 0.0))
    if job["kind"] == "sh6":
        return C.sh6_circuit(bits, n, job["D"])
    if job["kind"] == "qca":
        return C.qca_circuit(bits, n)
    raise ValueError(f"unknown kind {job['kind']!r}")


def build(dev, job, runs, compiler=None, role="user", refs=True, name=None):
    """Compile one logical job on the device record into a run record (TOQB layout). Returns the record.
    Acceptance: every compiled circuit has exactly D x (one ring layer's two-qubit gates).
    role: "user", "scan" or "sentinel" (written into the record and the manifest). refs=False leaves out the readout
    references (a sentinel reads per-site signals, which need no reference; the user's references are in the job)."""
    from qiskit import qpy
    if job["n"] % 2:
        raise ValueError("n must be even (Neel inputs)")
    rc = compiler or RuleCompiler(dev)
    inputs = job.get("inputs", list(INPUTS))
    D = {"qca": job["n"], "clifford_random": job.get("k"), "clifford_stab": job.get("k"), "kising_forward": job.get("L"), "cluster_parity": 1}.get(job["kind"], job.get("D"))
    cs, points = [], []
    for i in inputs:
        cs.append(rc.compile(logical_circuit(job, input_bits(i, job["n"]))))
        points.append(dict(input=i, setting="echo"))
    meas0 = measured_qubits(cs[0])
    if any(measured_qubits(c) != meas0 for c in cs):
        raise RuntimeError("the compiled circuits of one record measure on different qubits")
    for i in (inputs if refs else []):
        rcirc, _ = C.reference_circuit(cs[0], input_bits(i, job["n"]))
        cs.append(rcirc)
        points.append(dict(input=i, setting="ref"))
    per = rc.per_layer_q2(job["n"])
    q2s = [C.q2_count(c) for c in cs[:len(inputs)]]
    accepted = all(q == D * per for q in q2s)
    name = name or cell_name(job)
    for k, c in enumerate(cs):  # fixed names: the same build gives byte-identical QPY (Qiskit's default names count up)
        c.name = f"{name}-{k}"
    os.makedirs(runs, exist_ok=True)
    jf, qf = os.path.join(runs, name + ".json"), os.path.join(runs, name + ".qpy")
    for f in (jf, qf):
        if os.path.exists(f):
            raise FileExistsError(f"{f} exists: a run never overwrites another")
    cs = delays_to_dt(cs, getattr(dev, "dt", None))
    with open(qf, "wb") as f:
        qpy.dump(cs, f)
    rec = dict(kind="psfos", psfos_role=role, psfos_job=dict(job), arm="fold:rule", version=VERSION, device=dev.name,
               masked=dev.masked, task=dict(n=job["n"], layers=D),
               echo=dict(D=D, kind=job["kind"], inputs=inputs, measured_qubits={str(k): v for k, v in sorted(meas0.items())},
                         q2_per_layer=per, q2_circuits=q2s, accepted=bool(accepted), placement=dict(rc.last)),
               points=points, inputs=[input_bits(p["input"], job["n"]) for p in points],
               qca_test=job["kind"] in ("sh6", "qca"), valid=bool(accepted), shots=job.get("shots", 1000),
               export=dict(qpy=os.path.basename(qf), circuits=len(cs), sha256=sha256_file(qf)))
    if job["kind"] in PROBLEMS:
        rec["answers"] = exact_answers(job, inputs)
    json.dump(rec, open(jf, "w"), indent=1)
    return rec


ENV_PACKAGES = ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy")


def build_env():
    """The versions of the build environment (v0.2 prototype, 2026-11-18): python and the packages that shape a
    compiled circuit's bytes (the ring sentinel's one-qubit synthesis differed between environments on 2026-11-18)."""
    import platform
    from importlib import metadata
    env = dict(python=platform.python_version())
    for p in ENV_PACKAGES:
        try:
            env[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            env[p] = None
    return env


def verify_env(manifest, quiet=False):
    """Compare the build environment recorded in a job's manifest (or manifest path / job folder) with this one. Never
    stops anything: returns dict(same, recorded, here, differ, note) and prints the note when they differ (unless
    quiet). A job built before 2026-11-18's prototype has no record (same None)."""
    if isinstance(manifest, str):
        p = manifest if manifest.endswith(".json") else os.path.join(manifest, "manifest.json")
        manifest = json.load(open(p))
    here = build_env()
    rec = manifest.get("env")
    if rec is None:
        return dict(same=None, recorded=None, here=here, differ={}, note="no build environment recorded")
    differ = {k: (rec.get(k), here.get(k)) for k in set(rec) | set(here) if rec.get(k) != here.get(k)}
    note = ("same build environment" if not differ else
            "the build environment differs (" + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in sorted(differ.items()))
            + "): a rebuild may not match the ring sentinel's circuit bytes (the sentinel is never judged); the user, "
              "off and floor circuits are compared as before")
    if differ and not quiet:
        print("verify_env: " + note)
    return dict(same=not differ, recorded=rec, here=here, differ=differ, note=note)


DELAY_LIMIT_US = 100.0     # [OWNER] the longest delay one circuit may hold (0.3.2 candidate); allow_long_delay=True passes
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}


def delays_to_dt(circuits, dt):
    """The circuits with every delay written in a unit of time (s, ms, us, ns, ps) rewritten in dt, the device's clock
    step: qiskit's QPY keeps a delay's duration but not its unit (2.5.2: a delay in us comes back as the same number of
    dt, one in s is refused), so a delay reaches the job file in dt or not at all. Without a dt in the device record a
    delay in a unit of time is refused (ValueError); delays already in dt are kept. A duration that is not a whole
    number of dt (within 1e-6 of one) is refused. Circuits without such a delay are returned as they are."""
    from qiskit.circuit import Delay
    out = []
    for c in circuits:
        if not any(x.operation.name == "delay" and x.operation.unit in TIME_UNITS for x in c.data):
            out.append(c)
            continue
        if not dt:
            raise ValueError(f"{c.name}: a delay in a unit of time but the device record has no dt (fetch the device "
                             "with psf-os 0.3.2 or later, or write the delay in dt)")
        n = c.copy_empty_like()
        for x in c.data:
            op = x.operation
            if op.name == "delay" and op.unit in TIME_UNITS:
                k = float(op.duration) * TIME_UNITS[op.unit] / dt
                if abs(k - round(k)) > 1e-6:
                    raise ValueError(f"{c.name}: a delay of {op.duration} {op.unit} is not a whole number of dt "
                                     f"({dt:g} s)")
                op = Delay(int(round(k)), unit="dt")
            n.append(op, x.qubits, x.clbits)
        out.append(n)
    return out


def delay_grid_problems(circuits, timing):
    """0.3.3rc1: the delays (in dt, top level and inside conditional blocks) that do not fit the device's time grid
    by its timing constraints (Device.timing): a length that is not a whole multiple of `granularity` or of
    `pulse_alignment`, or a non-zero length below `min_length`. [] without constraints (every record before 0.3.3rc1:
    nothing is checked, as before)."""
    con = (timing or {}).get("constraints") or {}
    if not con:
        return []
    g = int(con.get("granularity") or 1)
    a = int(con.get("pulse_alignment") or 1)
    m = int(con.get("min_length") or 0)
    bad = []

    def rec(c, k):
        for x in c.data:
            op = x.operation
            if op.name == "if_else":
                for b in op.blocks:
                    if b is not None:
                        rec(b, k)
            elif op.name == "delay" and op.unit == "dt":
                d = int(op.duration)
                if d % g or d % a or (0 < d < m):
                    bad.append(dict(pub=k, dt=d))
    for k, c in enumerate(circuits):
        rec(c, k)
    return bad


def delay_record(circuits, dt):
    """Per circuit holding a delay: its delay in dt (the longest sum over one qubit; every delay is in dt by
    delays_to_dt) and, with a dt, in microseconds. Returns None when no circuit holds a delay."""
    per = []
    for k, c in enumerate(circuits):
        tot = {}
        for x in c.data:
            if x.operation.name == "delay":
                if x.operation.unit != "dt":
                    raise ValueError(f"circuit {k}: a delay in {x.operation.unit}, not in dt (delays_to_dt first)")
                q = c.find_bit(x.qubits[0]).index
                tot[q] = tot.get(q, 0) + int(x.operation.duration)
        if tot:
            m = max(tot.values())
            per.append(dict(pub=k, dt=m, us=round(m * dt * 1e6, 6) if dt else None))
    return per or None


def assemble(dev, runs, out, max_seconds=60, extra=None, build=None, allow_long_delay=False, compiled=None):
    """One job folder from every valid record in `runs`: circuits.qpy (in record order, references after the echo
    points) and manifest.json (cells, pubs, shots, the device record). Refuses an over-long job by the time model.
    v0.2 prototype (2026-11-18): the manifest also records the build environment (env, build_env()) and any `extra`
    fields the builder passes; circuits.qpy is unchanged.
    0.3.2 (candidate): a job holding delays records them in dt and microseconds (manifest "delays"); a circuit whose
    delay is longer than DELAY_LIMIT_US is refused before anything is written unless allow_long_delay; the delays are
    priced in the time model (qpu_seconds, delay_seconds). A job without a delay gets the manifest and the seconds it
    had before. Without a dt in the device record a delay in dt is recorded in dt only, neither limited nor priced
    (the records before 0.3.2 rebuild unchanged).
    0.3.3rc1 (candidate): with timing constraints in the device record, a delay off the device's time grid is refused
    before anything is written (delay_grid_problems). Every manifest gets "compiled": per circuit a summary of the
    circuit as sent (psf_os.compiled.summary: CZ, depth, the SHA-256 of its gate sequence) and, from the builders that
    pass `compiled`, the per-record checks (waits, parking checked, the fences' cost). loop.same_job leaves the key out
    when a recorded manifest has none, so every night recorded before verifies as before."""
    from qiskit import qpy
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists: a job folder is never overwritten")
    recs = sorted(f for f in os.listdir(runs) if f.endswith(".json"))
    cells, circuits, pubs, shots = [], [], [], None
    for f in recs:
        r = json.load(open(os.path.join(runs, f)))
        if not r.get("valid"):
            raise RuntimeError(f"{f}: not a valid record (acceptance failed)")
        qf = os.path.join(runs, r["export"]["qpy"])
        if sha256_file(qf) != r["export"]["sha256"]:
            raise RuntimeError(f"{f}: its QPY file differs from the record")
        if shots is None:
            shots = r["shots"]
        elif shots != r["shots"]:
            raise RuntimeError(f"{f}: {r['shots']} shots, an earlier record {shots}")
        cs = qpy.load(open(qf, "rb"))
        cid = len(cells)
        cells.append(dict(cell=cid, name=f[:-5], run_sha256=sha256_file(os.path.join(runs, f)),
                          qpy_sha256=r["export"]["sha256"], points=len(cs), role=r.get("psfos_role", "user")))
        for k, c in enumerate(cs):
            pubs.append(dict(cell=cid, point=k))
            circuits.append(c)
    step = getattr(dev, "dt", None)
    dly = delay_record(circuits, step)
    delay_s = sum(d["dt"] for d in dly) * step if (dly and step) else 0.0
    if dly and step:
        long_ = [d for d in dly if d["us"] > DELAY_LIMIT_US]
        if long_ and not allow_long_delay:
            w = max(long_, key=lambda d: d["us"])
            raise RuntimeError(f"{len(long_)} circuit(s) hold a delay over {DELAY_LIMIT_US:g} us (the longest "
                               f"{w['us']:g} us = {w['dt']} dt at dt {step:g} s, circuit {w['pub']}); shorten the "
                               "delays or pass allow_long_delay=True")
    grid = delay_grid_problems(circuits, getattr(dev, "timing", None))
    if grid:
        w = grid[0]
        raise RuntimeError(f"{len(grid)} delay(s) off the device's time grid {dev.timing['constraints']} (the first: "
                           f"{w['dt']} dt in circuit {w['pub']}): write the delay as a multiple of the granularity")
    seconds = qpu_seconds(len(circuits), shots, delay_s)
    if seconds > max_seconds:
        import math
        per = qpu_seconds(1, shots) - 3.0
        fit = int(max(0, (max_seconds - 3.0) // per)) if per > 0 else 0
        raise RuntimeError(f"the job is {seconds:.1f} s by the time model, over {max_seconds} s: {len(circuits)} "
                           f"circuits at {shots} shots ({per:.2f} s each + 3 s per job); {fit} fit in {max_seconds} s. "
                           f"Use --max-seconds {math.ceil(seconds)} or fewer circuits (run2d: each user circuit adds "
                           f"three, itself, its floor and its mirror; the ring sentinel adds four)")
    os.makedirs(out)
    with open(os.path.join(out, "circuits.qpy"), "wb") as f:
        qpy.dump(circuits, f)
    from zoneinfo import ZoneInfo
    man = dict(written=dt.datetime.now(ZoneInfo("Europe/Zurich")).isoformat(timespec="seconds"), device=dev.to_json(),
               fetch=dict(backend=dev.name, fake=dev.fake), cells=cells, pubs=pubs, shots=shots,
               circuits_sha256=sha256_file(os.path.join(out, "circuits.qpy")),
               qpy_sha256=sha256_file(os.path.join(out, "circuits.qpy")),  # the key TOQB s2rc_device submit checks
               qpu_seconds_model=round(seconds, 1),
               stay=True,  # placed for this device's registry: TOQB's submit never moves it (TOQB job_stays)
               roles={c["name"]: c["role"] for c in cells},  # which cells are the user's, which are sentinels
               user_pubs=[k for k, p in enumerate(pubs) if cells[p["cell"]]["role"] == "user"])
    if dly:
        # 0.3.2 (candidate): only a job holding a delay gets this key
        man["delays"] = dict(dt_seconds=step, limit_us=DELAY_LIMIT_US, allow_long_delay=bool(allow_long_delay),
                             per_circuit=dly, longest_us=max(d["us"] for d in dly) if step else None,
                             delay_seconds_per_shot=round(delay_s, 9) if step else None,
                             qpu_seconds_delays=round(delay_s * shots, 1) if step else None,
                             note=("priced in qpu_seconds_model: shots x the sum of the circuits' delays" if step else
                                   "the device record has no dt: recorded in dt only, neither limited nor priced"))
    if any(c.has_control_flow_op() for c in circuits):
        # 2026-11-18: the job's broadcasts (conditionals and the bits their tests read), recorded, never enforced;
        # jobs without conditionals keep the manifest they had
        from psf_os.feedforward import broadcast_record
        man["broadcast"] = broadcast_record(circuits)
    man["env"] = build_env()
    # release-0.3.0: every manifest names the psf-os version that built it and, for the builders that park, the park
    # setting and the declared psf-os (beside the job: loop.same_job leaves it out, as the build environment)
    from psf_os import __version__
    man["psf_os_build"] = dict(version=__version__, **(build or {}))
    # 0.3.3rc1: the circuits as sent, summarised (CZ, depth, gate-sequence hash), and the builder's checks
    from psf_os.compiled import summary
    man["compiled"] = dict(summary=[summary(c) for c in circuits], **(compiled or {}))
    for k, v in (extra or {}).items():
        if k in man:
            raise RuntimeError(f"assemble: extra field {k} would overwrite the manifest's own")
        man[k] = v
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    return man


def qpu_seconds(n_circuits, shots, delay_seconds=0.0):
    """The project's time model: 0.036 s per circuit + 0.00027 s per shot + 3 s per job, and (0.3.2 candidate) the
    delays, which the device charges: shots x delay_seconds, the sum over the job's circuits of each circuit's delay in
    seconds. Without delays it is the model it was."""
    return n_circuits * 0.036 + n_circuits * shots * 0.00027 + 3.0 + shots * delay_seconds


# ------------------------------------------------------------------------------------------------ after the run

def logical_counts(counts, n):
    """Counts keyed by the logical bit string, bit j = ring site j (clbit j is ring site j by construction)."""
    out = {}
    for key, c in counts.items():
        b = key.replace(" ", "")[-n:]
        lg = "".join(b[n - 1 - j] for j in range(n))
        out[lg] = out.get(lg, 0) + c
    return out


def site_signal_forward(counts_by_input, n):
    """a_j = mean over the inputs of <z_in,j Z_j> (1: every shot returns the input bit; 0: random; < 0: flipped),
    and its shot-noise SE."""
    a, v = np.zeros(n), np.zeros(n)
    for i in INPUTS:
        zin = 1 - 2 * np.array(input_bits(i, n))
        c = counts_by_input[i]
        N = float(sum(c.values()))
        m = np.zeros(n)
        for key, k in c.items():
            b = key.replace(" ", "")
            m += k * np.array([1 - 2 * int(b[n - 1 - j]) for j in range(n)])
        m /= N
        a += zin * m / len(INPUTS)
        v += (1 - m ** 2) / N / len(INPUTS) ** 2
    return a, np.sqrt(v)


def full_match(counts_by_input, n):
    """Fraction of shots, over the inputs, that return the whole input string."""
    hit = tot = 0
    for i in INPUTS:
        want = "".join(str(b) for b in reversed(input_bits(i, n)))
        c = counts_by_input[i]
        hit += sum(k for key, k in c.items() if key.replace(" ", "") == want)
        tot += sum(c.values())
    return hit / tot


def read(job_dir, runs):
    """{cell: (record, {(input, setting): counts})} of a finished job."""
    from psf_os.scan import load_counts
    man = json.load(open(os.path.join(job_dir, "manifest.json")))
    rows = load_counts(job_dir)
    # TOQB's submit may move a job to the next device in its fixed order when the first has a failed element under it.
    # A psf-os job was placed with one device's registry, so a result from another device is refused.
    sub = os.path.join(job_dir, "submission.json")
    ran_on = json.load(open(sub)).get("backend") if os.path.exists(sub) else None
    if ran_on and ran_on != man["fetch"]["backend"]:
        raise RuntimeError(f"the job was placed for {man['fetch']['backend']} but ran on {ran_on}: its registry "
                           "entries do not apply there; not scored")
    out = {}
    for cid, c in enumerate(man["cells"]):
        rf = os.path.join(runs, c["name"] + ".json")
        if sha256_file(rf) != c["run_sha256"]:
            raise RuntimeError(f"{rf} differs from the manifest")
        rec = json.load(open(rf))
        cc = {}
        for k, p in enumerate(man["pubs"]):
            if p["cell"] == cid:
                pt = rec["points"][p["point"]]
                cc[(pt["input"], pt["setting"])] = rows[k]["counts"]
        out[c["name"]] = (rec, cc)
    return out


def score(job_dir, runs):
    """Per cell: the echo ratio (echo) or the forward site signal (sh6, qca), with the ring and its dead runs."""
    from psf_os import scan as S
    res = {}
    for name, (rec, cc) in read(job_dir, runs).items():
        if rec.get("psfos_role") == "sentinel":
            continue  # judged by psf_os.sentinel.verdict
        if rec["psfos_job"]["kind"] in PROBLEMS:
            res[name] = problem_score(rec, cc)
            continue
        n = rec["task"]["n"]
        mq = rec["echo"]["measured_qubits"]
        ring = [int(mq[str(j)]) for j in range(n)]
        if rec.get("qca_test"):
            a, se = site_signal_forward({i: cc[(i, "echo")] for i in INPUTS}, n)
            res[name] = dict(kind=rec["echo"]["kind"], n=n, ring=ring, a=[float(x) for x in a],
                             mean=float(a.mean()), per_bit=float((1 + a.mean()) / 2),
                             whole=full_match({i: cc[(i, "echo")] for i in INPUTS}, n),
                             usable=len(S.usable(a)), dead=[[ring[(s + t) % n] for t in range(L)]
                                                            for s, L in S.dead_runs(a)])
        else:
            est = S.estimate(cc, n)
            a = S.site_signal({i: cc[(i, "echo")] for i in INPUTS}, n)
            res[name] = dict(kind="echo", n=n, ring=ring, a=[float(x) for x in a], rho4=est["rho4"], se4=est["se4"],
                             usable=len(S.usable(a)), dead=[[ring[(s + t) % n] for t in range(L)]
                                                            for s, L in S.dead_runs(a)])
    return res


# ------------------------------------------------------------------------------------------------ practice problems

def stab_observables(job):
    """The fixed set of read stabilizers of a clifford_stab job (P1)."""
    return C.clifford_stab_observables(job["n"], job["k"], job["seed"], job.get("offset", 0), job.get("max_weight"))


def exact_answers(job, inputs, method=None):
    """{input: exact <Z_j> per ring site} of a practice problem, from its logical circuit.
    clifford_random: stabilizer simulation (every value is +1, -1 or 0); kising_forward: statevector (n <= 22).
    `method` forces "stabilizer", "statevector" or "density" (the independent checks of test Q1)."""
    from qiskit.quantum_info import DensityMatrix, Pauli, StabilizerState, Statevector
    n = job["n"]
    m = method or ("stabilizer" if job["kind"] in ("clifford_random", "clifford_stab", "cluster_parity") else "statevector")
    out = {}
    if job["kind"] == "clifford_stab":
        # answer per read stabilizer: the parity of the bits on its sites after the basis change, exactly +1 or -1
        obs = stab_observables(job)
        for i in inputs:
            b = input_bits(i, n)
            if m == "analytic":
                out[i] = [float(sign * (-1) ** b[j]) for j, sign, _s, _l in obs]
                continue
            qc = logical_circuit(job, b).remove_final_measurements(inplace=False)
            vals = []
            for _j, _sign, sites, _l in obs:
                z = "".join("Z" if q in sites else "I" for q in reversed(range(n)))
                if m == "stabilizer":
                    vals.append(float(StabilizerState(qc).expectation_value(Pauli(z))))
                else:
                    if n > 22:
                        raise ValueError("the statevector answer is limited to n <= 22")
                    vals.append(float(Statevector(qc).expectation_value(Pauli(z)).real))
            out[i] = vals
        return out
    for i in inputs:
        qc = logical_circuit(job, input_bits(i, n)).remove_final_measurements(inplace=False)
        if job["kind"] == "cluster_parity":  # the answer is the parity of all bits, not a per-site value
            if m == "analytic":
                out[i] = [float((-1) ** sum(input_bits(i, n)))]
            elif m == "stabilizer":
                out[i] = [float(StabilizerState(qc).expectation_value(Pauli("Z" * n)))]
            else:
                ps = Statevector(qc).probabilities()
                par = sum(p * (-1) ** bin(x).count("1") for x, p in enumerate(ps))
                out[i] = [float(par)]
            continue
        if m == "stabilizer":
            st = StabilizerState(qc)
            z = [float(st.expectation_value(Pauli("I" * (n - 1 - j) + "Z" + "I" * j))) for j in range(n)]
        else:
            if n > 22:
                raise ValueError("the statevector answer is limited to n <= 22")
            st = Statevector(qc) if m == "statevector" else DensityMatrix(qc)
            z = [float(1 - 2 * st.probabilities([j])[1]) for j in range(n)]
        out[i] = z
    return out


def problem_score(rec, cc):
    """Score of a practice problem against its recorded answers, per input and per site.
    clifford_stab: per read stabilizer, answer x the parity of the bits on its sites ("signal", 1 = right every shot)
    and the fraction of right parities (1 + signal) / 2.
    clifford_random: on sites with answer +-1 the fraction of right bits (1 + answer x <Z>) / 2; on sites with answer 0
    the size |<Z>| (0 is right). kising_forward: the mean squared difference of <Z_j> from the answer."""
    n = rec["task"]["n"]
    kind = rec["psfos_job"]["kind"]
    if kind == "cluster_parity":
        par = {}
        for i, (ans,) in rec["answers"].items():
            c = cc[(i, "echo")]
            N = float(sum(c.values()))
            par[i] = sum(k * (-1) ** key.replace(" ", "").count("1") for key, k in c.items()) / N
        right = [(1 + rec["answers"][i][0] * par[i]) / 2 for i in par]
        return dict(kind=kind, n=n, parity=par, answers={i: a[0] for i, a in rec["answers"].items()},
                    right=float(np.mean(right)))
    if kind == "clifford_stab":
        obs = [dict(j=j, sign=sg, sites=st, letters=lt) for j, sg, st, lt in stab_observables(rec["psfos_job"])]
        per, signal = {}, []
        for i, ans in rec["answers"].items():
            c = cc[(i, "echo")]
            N = float(sum(c.values()))
            vals = []
            for o, a in zip(obs, ans):
                par = sum(k * (-1) ** sum(int(key.replace(" ", "")[n - 1 - q]) for q in o["sites"]) for key, k in c.items()) / N
                vals.append(float(par))
                signal.append(a * par)
            per[i] = vals
        return dict(kind=kind, n=n, observables=obs, measured=per, answers=rec["answers"],
                    signal=float(np.mean(signal)), right=float(np.mean([(1 + x) / 2 for x in signal])),
                    weights=[len(o["sites"]) for o in obs])
    per_site, det, undet, sq = {}, [], [], []
    for i, ans in rec["answers"].items():
        c = cc[(i, "echo")]
        N = float(sum(c.values()))
        m = np.zeros(n)
        for key, k in c.items():
            b = key.replace(" ", "")
            m += k * np.array([1 - 2 * int(b[n - 1 - j]) for j in range(n)])
        m /= N
        per_site[i] = [float(x) for x in m]
        for j in range(n):
            a = ans[j]
            if kind == "clifford_random":
                if abs(abs(a) - 1) < 1e-9:
                    det.append((1 + a * m[j]) / 2)
                else:
                    undet.append(abs(m[j]))
            sq.append((m[j] - a) ** 2)
    out = dict(kind=kind, n=n, measured=per_site, mse=float(np.mean(sq)))
    if kind == "clifford_random":
        out.update(right_on_determined=float(np.mean(det)) if det else None, n_determined=len(det),
                   size_on_undetermined=float(np.mean(undet)) if undet else None, n_undetermined=len(undet))
    return out
