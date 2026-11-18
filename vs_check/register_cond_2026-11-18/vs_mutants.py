"""Mutation test of the branch's own tests (tests/test_register_cond.py): psf_os is copied, one line is damaged, and
the test file is run on the copy. Each mutant should make at least one test fail. The checkout is never changed.

  python vs_mutants.py PSFOS_CHECKOUT WORKDIR [OUT.json]
"""
import json
import os
import shutil
import subprocess
import sys

MUTANTS = [
    ("rep3: the test that corrects d0 compares with 2 instead of 1 (a value shifted by one)",
     "for value, k in ((1, 0), (3, 1), (2, 2)):", "for value, k in ((2, 0), (3, 1), (2, 2)):"),
    ("rep3: the syndrome bits exchanged (register bit order)",
     "    qc.measure(a[0], s[0])\n    qc.measure(a[1], s[1])", "    qc.measure(a[0], s[1])\n    qc.measure(a[1], s[0])"),
    ("restrict: the rebuilt register's bits in reverse order",
     "ClassicalRegister(bits=[out.clbits[k] for k in key]", "ClassicalRegister(bits=[out.clbits[k] for k in key[::-1]]"),
    ("condition_of: the value 2^width accepted (range off by one)",
     "if not 0 <= val < 2 ** w:", "if not 0 <= val <= 2 ** w:"),
    ("condition_of: 33 bits accepted", "REGISTER_MAX_BITS = 32 ", "REGISTER_MAX_BITS = 33 "),
    ("analyse: a register test before its bits are measured accepted",
     "late = [b for b in bits if b not in written]", "late = []"),
    ("_check_registers: overlapping registers accepted",
     "    if len(set(order)) != len(order):\n        raise Refused(\"classical registers overlap",
     "    if False:\n        raise Refused(\"classical registers overlap"),
    ("walk_bits: a register test lists only its first bit (bundle_acceptance would miss the others)",
     "cs = cs + [cmap[c.find_bit(x).index] for x in cb]", "cs = cs + [cmap[c.find_bit(x).index] for x in list(cb)[:1]]"),
    ("wire_signature: a register's bits in reverse order",
     "cond = (\"register\", tuple(c.find_bit(x).index for x in cb), int(op.condition[1]))",
     "cond = (\"register\", tuple(c.find_bit(x).index for x in cb)[::-1], int(op.condition[1]))"),
    ("broadcast_record: a register test counted as one bit",
     "bits += len(cb) if type(cb).__name__ == \"ClassicalRegister\" else 1", "bits += 1"),
]


def main():
    src, work = sys.argv[1:3]
    rows = []
    for k, (what, old, new) in enumerate(MUTANTS):
        dst = os.path.join(work, f"m{k}")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(os.path.join(src, "psf_os"), os.path.join(dst, "psf_os"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(os.path.join(src, "tests"), os.path.join(dst, "tests"), ignore=shutil.ignore_patterns("__pycache__"))
        p = os.path.join(dst, "psf_os", "feedforward.py")
        s = open(p, encoding="utf-8").read()
        if s.count(old) != 1:
            rows.append(dict(mutant=what, applied=False))
            print(f"m{k} NOT APPLIED ({s.count(old)} matches): {what}")
            continue
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
        env = dict(os.environ, PYTHONPATH=dst)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_register_cond.py"],
                           cwd=dst, env=env, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        caught = r.returncode != 0
        rows.append(dict(mutant=what, applied=True, caught=caught, summary=last))
        print(f"m{k} {'CAUGHT' if caught else 'NOT CAUGHT'}  {what}  [{last}]", flush=True)
    print(f"{sum(r.get('caught', False) for r in rows)} of {len(rows)} mutants caught")
    if len(sys.argv) > 3:
        json.dump(rows, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
