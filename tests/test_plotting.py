import re

from atml4.common import plotting


def test_style_and_type42(tmp_path):
    font = plotting.apply_style()
    assert font in plotting.SERIF_CHAIN or font
    fig, ax = plotting.figure(plotting.HALF_W)
    ax.plot([0, 1, 2], [1, 3, 2], label=r"$\alpha$ curve")
    ax.set_xlabel("sparsity")
    ax.legend()
    path = plotting.savefig(fig, "dummy", outdir=tmp_path)
    raw = path.read_bytes()
    assert raw.startswith(b"%PDF")
    # Type 42 fonts are embedded as TrueType (/FontFile2); Type 3 must not appear
    assert b"/FontFile2" in raw or b"/TrueType" in raw or b"/CIDFontType2" in raw
    assert not re.search(rb"/Subtype\s*/Type3", raw)
