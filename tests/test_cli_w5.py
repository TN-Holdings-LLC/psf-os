"""run2d's time model is checked before anything is written (W5): an over-long batch is refused with the numbers and
a usable --max-seconds, and --out is not created."""
from psf_os import cli
from psf_os import device as Dv


def test_overlong_batch_is_refused_before_writing(tmp_path, capsys):
    from qiskit import QuantumCircuit, qpy
    dev = Dv.fetch_fake("FakeKingston")
    dev.save(str(tmp_path / "dev"))
    files = []
    for k in range(22):
        qc = QuantumCircuit(5)
        qc.h(3)
        qc.cz(3, 4)
        f = tmp_path / f"c{k:02d}.qpy"
        with open(f, "wb") as fh:
            qpy.dump(qc, fh)
        files.append(str(f))
    out = tmp_path / "run"
    rc = cli.main(["run2d", "--device", str(tmp_path / "dev"), "--circuit", *files, "--out", str(out)])
    err = capsys.readouterr().err
    assert rc == 2 and not out.exists()
    assert "70 circuits" in err and "--max-seconds 82" in err and "Nothing was written" in err
