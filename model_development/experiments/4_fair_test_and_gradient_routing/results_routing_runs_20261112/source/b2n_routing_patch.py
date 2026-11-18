"""Board2Notes gradient-routing patch for PaddleOCR tools/program.py (applied only to this package's own copy of the
checkout, so older experiments keep their repository identity).

With Global.b2n_ctc_trains unset the training step is byte-for-byte the original one (avg_loss.backward()).
With Global.b2n_ctc_trains = "stem,1,2" (for example):
  1. the CTC loss is back-propagated (graph kept) -> every part the CTC branch reaches gets its CTC gradient;
  2. the CTC gradient is zeroed on the backbone parts NOT listed (here stages 3 and 4); it has already flowed through
     them to the listed earlier parts, so those still get the full CTC signal;
  3. the NRTR loss is back-propagated on the same graph; Paddle adds it to what is there.
Result per parameter: NRTR head = NRTR gradient; CTC neck and head = CTC gradient; listed backbone parts = CTC + NRTR;
unlisted backbone parts = NRTR only. The losses are MultiLoss's own weighted terms (weights 1 and 1).
Requires guidance off (use_guide: False), otherwise the CTC gradient never reaches the backbone.
"""
from pathlib import Path

MARK = "# board2notes routing patch"
FUNC = '''

def _b2n_ctc_reaches(name, parts):  # board2notes routing patch
    import re as _re
    if not name.startswith("backbone."):
        return True
    if name.startswith("backbone.stem."):
        return "stem" in parts
    m = _re.match(r"backbone\\.stages\\.(\\d)\\.", name)
    return bool(m) and str(int(m.group(1)) + 1) in parts


def _b2n_routed_backward(model, loss, spec):  # board2notes routing patch
    parts = {p.strip() for p in str(spec).split(",") if p.strip()}
    loss["CTCLoss"].backward(retain_graph=True)
    for n, p in model.named_parameters():
        if p.grad is not None and not _b2n_ctc_reaches(n, parts):
            p.clear_gradient(set_to_zero=True)
    loss["NRTRLoss"].backward()
'''
OLD = '''                loss = loss_class(preds, batch)
                avg_loss = loss["loss"]
                avg_loss.backward()
                optimizer.step()
'''
NEW = '''                loss = loss_class(preds, batch)
                avg_loss = loss["loss"]
                if config["Global"].get("b2n_ctc_trains"):  # board2notes routing patch
                    _b2n_routed_backward(model, loss, config["Global"]["b2n_ctc_trains"])
                else:
                    avg_loss.backward()
                optimizer.step()
'''
ANCHOR = "\ndef train(\n"


def apply(repo):
    f = Path(repo) / "tools" / "program.py"
    s = f.read_text(encoding="utf-8")
    if MARK in s:
        assert s.count(NEW) == 1 and s.count("def _b2n_routed_backward") == 1, "routing patch present but altered"
        return False
    assert s.count(OLD) == 1, "tools/program.py: training step not found exactly once; PaddleOCR version differs"
    assert s.count(ANCHOR) == 1, "tools/program.py: train() not found exactly once"
    s = s.replace(OLD, NEW).replace(ANCHOR, FUNC + ANCHOR)
    f.write_text(s, encoding="utf-8")
    return True
