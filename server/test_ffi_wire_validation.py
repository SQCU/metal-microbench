import pytest

import gemma_ffi as g


def test_encode_request_maps_negative_seed_sentinel_to_default():
    spec = g.StreamSpec(
        stream_id=1,
        action=0,
        tokens=[1, 2, 3],
        sampling=g.SamplingParams(seed=-1, max_new_tokens=2),
    )

    encoded = g._encode_request([spec])

    assert isinstance(encoded, bytes)


def test_encode_request_rejects_impossible_unsigned_wire_values():
    spec = g.StreamSpec(
        stream_id=-1,
        action=0,
        tokens=[1, 2, 3],
        sampling=g.SamplingParams(max_new_tokens=2),
    )

    with pytest.raises(ValueError, match="stream_id=-1 outside uint64 range"):
        g._encode_request([spec])
