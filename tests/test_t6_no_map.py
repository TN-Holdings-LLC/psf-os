"""T6: psf-os does not read the map. The fake device's Target is wrapped so that an error value can only be compared
with the failed mark (>= 1) and any other use of it, any gate duration and any qubit property (T1, T2, frequency)
raises. The whole chain (fetch, plan with and without a registry, compile, assemble) runs through it."""
import json

import pytest

pytest.importorskip("qiskit_ibm_runtime")

from psf_os import device as Dv  # noqa: E402
from psf_os import plan as P  # noqa: E402
from psf_os import registry as R  # noqa: E402


class MapRead(AssertionError):
    pass


class FailMark:
    """An error value that may only be tested against the failed mark."""
    def __init__(self, v):
        self._v = v

    def __ge__(self, other):
        if other != Dv.FAILED_ERROR:
            raise MapRead(f"error compared with {other}")
        return self._v >= other

    def __getattr__(self, k):
        raise MapRead(f"error value used ({k})")

    def __float__(self):
        raise MapRead("error value used as a number")

    def __lt__(self, o):
        raise MapRead("error value compared")

    __gt__ = __le__ = __lt__

    def __add__(self, o):
        raise MapRead("arithmetic on an error value")

    __radd__ = __mul__ = __rmul__ = __sub__ = __rsub__ = __truediv__ = __add__


class Props:
    def __init__(self, p):
        self._p = p

    @property
    def error(self):
        return None if self._p.error is None else FailMark(self._p.error)

    def __getattr__(self, k):
        raise MapRead(f"instruction property {k} read")


class Guarded:
    def __init__(self, t):
        self._t = t
        self.num_qubits = t.num_qubits
        self.operation_names = t.operation_names

    def __getitem__(self, name):
        inner = self._t[name]

        class M(dict):
            def get(self_, qa, default=None):
                p = inner.get(qa, default)
                return None if p is None else Props(p)

            def __iter__(self_):
                return iter(inner)

            def items(self_):
                return [(qa, None if p is None else Props(p)) for qa, p in inner.items()]
        return M()

    def __getattr__(self, k):
        raise MapRead(f"Target.{k} read")


def test_the_guard_trips():
    from qiskit_ibm_runtime import fake_provider
    g = Guarded(fake_provider.FakeFez().target)
    p = g["cz"].get(next(iter(g["cz"])))
    with pytest.raises(MapRead):
        float(p.error)
    with pytest.raises(MapRead):
        p.duration
    with pytest.raises(MapRead):
        g.qubit_properties


def test_chain_reads_no_map(tmp_path):
    from qiskit_ibm_runtime import fake_provider
    b = fake_provider.FakeFez()
    dev = Dv.from_target(Guarded(b.target), b.name, fake=True)
    plain = Dv.fetch_fake("FakeFez")
    assert (dev.couplers, dev.failed_couplers, dev.failed_qubits) == (plain.couplers, plain.failed_couplers,
                                                                     plain.failed_qubits)
    text = json.dumps(dev.to_json())
    for word in ("error", "duration", "t1", "t2", "frequency", "readout"):
        assert word not in text.lower()
    t = Dv.map_free_target(dev)
    assert t.qubit_properties is None
    assert all(p is None for name in t.operation_names for p in t[name].values())
    now = P.now_cet()
    rep = P.plan(dev, dict(kind="sh6", n=40, D=12, shots=100), R.empty(), str(tmp_path / "a"), now, standard=True)
    assert rep["accepted"]
    reg = R.empty()
    reg["gray"] = [dict(device=dev.name, day=now.date().isoformat(), job="x", n=40, line=0.2,
                        path=rep["ring"][4:8], qubits=rep["ring"][3:9], mean=0.1)]
    reg["observations"] = [dict(device=dev.name, job="x", label="x", submitted=now.isoformat(sep=" "), n=40,
                                ring=rep["ring"], median=0.4, dead=[])]
    rep2 = P.plan(dev, dict(kind="sh6", n=40, D=12, shots=100), reg, str(tmp_path / "b"), now)
    assert rep2["category"] == "self-checked" and rep2["masked_couplers"] and rep2["ring"] != rep["ring"]
