"""Bounded metadata-only preflight; Rust remains authoritative for runtime tensors."""

import json
import os
import stat
import struct


FILE_LIMIT = 256 * 1024 * 1024
HEADER_LIMIT = 64 * 1024
CURRENT_METADATA = {
    "action_schema_hash": "10658390830565586343",
    "feature_schema_hash": "1861607613534772372",
    "model_schema_hash": "13592057279889489276",
    "ppo_schema_version": "30",
    "ppo_schema_hash": "16275284022255703821",
    "ppo_rules_audit_version": "25",
    "map2_reward_schema_version": "1",
    "map2_reward_schema_hash": "798798703797057220",
}
REWARD_DESCRIPTOR_KEY = "map2_reward_schema_descriptor"
assert len(CURRENT_METADATA) == 8
assert 8 < HEADER_LIMIT < FILE_LIMIT


def read_runtime_metadata(directory):
    """Check the current nine-key identity without loading or trusting tensor data."""
    path = directory / "drysua.weights.safetensors"
    try:
        header = read_header(path)
        if not header.startswith(b"{"):
            raise ValueError("Safetensors header must start with a JSON object")
        data = json.loads(header.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)
        metadata = data.get("__metadata__")
        expected_keys = CURRENT_METADATA.keys() | {REWARD_DESCRIPTOR_KEY}
        if (not isinstance(metadata, dict) or metadata.keys() != expected_keys
                or any(metadata.get(key) != value for key, value in CURRENT_METADATA.items())
                or not isinstance(metadata.get(REWARD_DESCRIPTOR_KEY), str)):
            raise ValueError("incompatible runtime weights metadata: expected exact nine-key "
                             "F15/M17, A5, PPO30/rules25, Map2 reward v1 identity")
        descriptor = metadata[REWARD_DESCRIPTOR_KEY].encode("utf-8")
        if fnv1a(descriptor) != int(CURRENT_METADATA["map2_reward_schema_hash"]):
            raise ValueError("incompatible runtime weights metadata: F15/M17 Map2 reward descriptor mismatch")
        assert len(metadata) == 9
        assert len(header) <= HEADER_LIMIT
        return metadata
    except (OSError, ValueError, RecursionError) as error:
        raise RuntimeError(f"runtime weights preflight failed: {path}: {error}; provide --weights-directory "
                           "with compatible F15/M17 runtime weights; no migration or Teacher fallback") from error


def read_header(path):
    # O_NONBLOCK prevents a substituted FIFO from blocking before fstat rejects it.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or not 8 < status.st_size <= FILE_LIMIT:
            raise ValueError(f"expected regular non-symlink file in 9..{FILE_LIMIT} bytes")
        prefix = os.read(descriptor, 8)
        if len(prefix) != 8:
            raise ValueError("truncated Safetensors length prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 1 <= length <= HEADER_LIMIT:
            raise ValueError(f"Safetensors header length must be in 1..{HEADER_LIMIT} bytes")
        if length > status.st_size - 8:
            raise ValueError("truncated Safetensors header")
        header = os.read(descriptor, length)
        if len(header) != length or os.fstat(descriptor).st_size != status.st_size:
            raise ValueError("runtime weights changed while reading header")
        assert len(header) <= HEADER_LIMIT
        assert 8 + len(header) <= status.st_size
        return header
    finally:
        os.close(descriptor)


def unique_object(pairs):
    assert len(pairs) <= HEADER_LIMIT
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Safetensors JSON key: {key}")
        result[key] = value
    assert len(result) == len(pairs)
    return result


def invalid_constant(value):
    raise ValueError(f"invalid Safetensors JSON constant: {value}")


def fnv1a(data):
    assert isinstance(data, bytes)
    assert len(data) <= HEADER_LIMIT
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & (2**64 - 1)
    return value
