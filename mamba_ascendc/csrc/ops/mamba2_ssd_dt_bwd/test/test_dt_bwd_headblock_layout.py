from pathlib import Path


OP_ROOT = Path(__file__).resolve().parents[1]
KERNEL = OP_ROOT / "op_kernel" / "mamba2_ssd_dt_bwd.cpp"
HOST = OP_ROOT / "op_host" / "mamba2_ssd_dt_bwd.cpp"


def _load_public_x(batch, seqlen, nheads, headdim):
    return [
        (((b * seqlen + token) * nheads + head) * headdim + p)
        for b in range(batch)
        for token in range(seqlen)
        for head in range(nheads)
        for p in range(headdim)
    ]


def _stage_head_block(public_x, batch, first_token, seqlen, nheads,
                      headdim, head_base, valid_heads, chunk_size):
    staged = []
    for token in range(first_token, first_token + chunk_size):
        base = ((batch * seqlen + token) * nheads + head_base) * headdim
        staged.extend(public_x[base:base + valid_heads * headdim])
    return staged


def _unpack_head(staged, local_head, valid_heads, headdim, chunk_size):
    unpacked = []
    row_size = valid_heads * headdim
    for token in range(chunk_size):
        start = token * row_size + local_head * headdim
        unpacked.extend(staged[start:start + headdim])
    return unpacked


def _pack_heads(per_head, valid_heads, headdim, chunk_size):
    packed = []
    for token in range(chunk_size):
        for local_head in range(valid_heads):
            start = token * headdim
            packed.extend(per_head[local_head][start:start + headdim])
    return packed


def test_two_head_stage_unpack_pack_preserves_public_layout():
    batch, seqlen, nheads, headdim, chunk_size = 2, 8, 6, 4, 4
    public_x = _load_public_x(batch, seqlen, nheads, headdim)
    staged = _stage_head_block(
        public_x, 1, 4, seqlen, nheads, headdim, 2, 2, chunk_size)
    heads = [
        _unpack_head(staged, local_head, 2, headdim, chunk_size)
        for local_head in range(2)
    ]
    assert _pack_heads(heads, 2, headdim, chunk_size) == staged


def test_odd_head_tail_does_not_touch_virtual_head():
    batch, seqlen, nheads, headdim, chunk_size = 1, 4, 5, 4, 4
    public_x = _load_public_x(batch, seqlen, nheads, headdim)
    staged = _stage_head_block(
        public_x, 0, 0, seqlen, nheads, headdim, 4, 1, chunk_size)
    tail = _unpack_head(staged, 0, 1, headdim, chunk_size)
    expected = []
    for token in range(chunk_size):
        base = (token * nheads + 4) * headdim
        expected.extend(public_x[base:base + headdim])
    assert tail == expected
    assert len(_pack_heads([tail], 1, headdim, chunk_size)) == len(staged)


def test_source_contract_limits_headblock_to_t64_d_path():
    host = HOST.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    assert "(chunkSize == 64 && nheads >= 2) ? 2 : 1" in host
    assert "batch * headBlockCount" in host
    assert "hasD_ != 0 && headBlock_ == 2" in kernel
    assert "for (int64_t chunk = 0; chunk < nchunks_; ++chunk)" in kernel
    assert "localHead < validHeads" in kernel
