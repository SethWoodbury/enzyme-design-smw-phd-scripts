from utils.pdbio import parse_remark_666


def test_parse_remark_666_basic():
    lines = [
        "REMARK 666 MATCH TEMPLATE A XDW 0 MATCH MOTIF A HIS 13 1 1",
        "REMARK 666 MATCH TEMPLATE B LIG 0 MATCH MOTIF A ASP 53 5 1",
    ]
    entries = parse_remark_666(lines)
    assert len(entries) == 2
    assert entries[0]["motif_chain"] == "A"
    assert entries[0]["motif_resname"] == "HIS"
    assert entries[0]["motif_resno"] == 13
    assert entries[0]["block_index"] == 1
    assert entries[0]["template_chain"] == "A"
