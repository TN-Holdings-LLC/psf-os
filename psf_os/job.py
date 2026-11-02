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


def assemble(dev, runs, out, max_seconds=60):
    """One job folder from every valid record in `runs`: circuits.qpy (in record order, references after the echo
    points) and manifest.json (cells, pubs, shots, the device record). Refuses an over-long job by the time model."""
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
    seconds = qpu_seconds(len(circuits), shots)
    if seconds > max_seconds:
        raise RuntimeError(f"the job is {seconds:.1f} s by the time model, over {max_seconds} s")
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
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    return man


def qpu_seconds(n_circuits, shots):
    """The project's time model: 0.036 s per circuit + 0.00027 s per shot + 3 s per job."""
    return n_circuits * 0.036 + n_circuits * shots * 0.00027 + 3.0


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
