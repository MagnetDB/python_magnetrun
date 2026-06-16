"""Tests for _BinaryFileReaderBase and ChannelVariable."""

import struct
from unittest.mock import patch

import pandas as pd
import pytest

from python_magnetrun.hybrid._binary_reader_base import (
    DIGITAL_SIZE,
    FLOAT32_SIZE,
    ChannelVariable,
)
from python_magnetrun.hybrid.rms.rms_reader import RMSFileReader, RMSVariable
from python_magnetrun.hybrid.vprocess.vprocess_reader import (
    VProcessFileReader,
    VProcessVariable,
)

# ── ChannelVariable ───────────────────────────────────────────────────────────

class TestChannelVariable:
    def test_float32_is_analog(self):
        v = ChannelVariable("I", "float32")
        assert v.is_analog is True
        assert v.byte_size == FLOAT32_SIZE

    def test_bit_is_digital(self):
        v = ChannelVariable("D", "bit")
        assert v.is_analog is False
        assert v.byte_size == DIGITAL_SIZE

    def test_dig_is_digital(self):
        # VProcess uses "dig" for digital channels
        v = ChannelVariable("D", "dig")
        assert v.is_analog is False
        assert v.byte_size == DIGITAL_SIZE

    def test_float_prefix_variants(self):
        # Any var_type containing "float" is analog
        for vt in ("float32", "float64", "FLOAT32"):
            assert ChannelVariable("x", vt).is_analog is True

    def test_optional_fields_default_none(self):
        v = ChannelVariable("x", "float32")
        assert v.unit is None
        assert v.min_val is None
        assert v.max_val is None
        assert v.display_format is None

    def test_repr(self):
        v = ChannelVariable("Amps", "float32", unit="A")
        assert "ChannelVariable" in repr(v)
        assert "Amps" in repr(v)

    def test_rms_variable_alias(self):
        assert RMSVariable is ChannelVariable

    def test_vprocess_variable_alias(self):
        assert VProcessVariable is ChannelVariable


# ── _make_timestamps round-trip ───────────────────────────────────────────────

class TestMakeTimestamps:
    UNIX_TS = [1_730_000_000.0, 1_730_000_001.0, 1_730_000_002.0]

    def test_rms_make_timestamps_utc_aware(self):
        reader = RMSFileReader.__new__(RMSFileReader)
        idx = reader._make_timestamps(self.UNIX_TS)
        assert isinstance(idx, pd.DatetimeIndex)
        assert idx.tz is not None  # UTC-aware
        assert len(idx) == 3
        # Round-trip: first timestamp matches
        assert idx[0].timestamp() == pytest.approx(self.UNIX_TS[0])

    def test_vprocess_make_timestamps_utc_aware(self):
        reader = VProcessFileReader.__new__(VProcessFileReader)
        idx = reader._make_timestamps(self.UNIX_TS)
        assert isinstance(idx, pd.DatetimeIndex)
        assert len(idx) == 3
        assert idx[0].timestamp() == pytest.approx(self.UNIX_TS[0])

    def test_both_subclasses_agree(self):
        rms = RMSFileReader.__new__(RMSFileReader)
        vp = VProcessFileReader.__new__(VProcessFileReader)
        rms_idx = rms._make_timestamps(self.UNIX_TS)
        vp_idx = vp._make_timestamps(self.UNIX_TS)
        # Both should produce the same UTC datetimes
        for r, v in zip(rms_idx, vp_idx, strict=False):
            assert r.timestamp() == pytest.approx(v.timestamp())


# ── parse_header via minimal synthetic files ──────────────────────────────────

def _make_rms_header() -> bytes:
    # XXXXXXXX is an 8-char placeholder replaced with the actual header length.
    # The replacement keeps the total byte count identical so the offset is self-consistent.
    template = (
        b"# variables = CH1 [type:float32|unit:A|min:0.00|max:100.00|df:%.1f]; D1 [type:bit|min:0|max:1]\n"
        b"# windows = [UTC] 05/11/2025-00:00:00.000 -> 05/11/2025-01:00:00.000\n"
        b"# frequency = 10.0 Hz\n"
        b"# data-helper [offset:0xXXXXXXXX - time:8(B),absolute - width:13(B)]\n"
        b"# format = binary,test-1.0\n"
        b"# processed on testhost [1.2.3.4] (test) offline at 05/11/2025-01:00:00\n"
    )
    return template.replace(b"XXXXXXXX", f"{len(template):08x}".encode())


def _make_vprocess_header() -> bytes:
    template = (
        b"# variables = TT1 [type:float32|unit:K|min:_|max:_|df:%.2f]; DIG1 [type:dig|min:0|max:1]\n"
        b"# windows = [UTC] 05/11/2025-00:00:00.000 -> 05/11/2025-01:00:00.000\n"
        b"# frequency = 1.0 Hz\n"
        b"# data-helper [offset:0xXXXXXXXX - time:8(B) - width:13(B)]\n"
        b"# vprocess data file - test-vp-1.0\n"
    )
    return template.replace(b"XXXXXXXX", f"{len(template):08x}".encode())


def _make_binary_sample(endian: str, ts: float, analog: float, digital: int) -> bytes:
    e = ">" if endian == "big" else "<"
    return (
        struct.pack(f"{e}d", ts)
        + struct.pack(f"{e}f", analog)
        + struct.pack("B", digital)
    )


class TestRMSFileReaderHeader:
    def test_parse_header_variables(self, tmp_path):
        data = _make_rms_header() + b"\x00" * 5  # minimal binary stub
        f = tmp_path / "test.rms"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.rms.rms_reader.validate_rms_format"):
            reader = RMSFileReader(str(f))
            reader.parse_header()
        assert len(reader.variables) == 2
        assert reader.variables[0].name == "CH1"
        assert reader.variables[0].is_analog is True
        assert reader.variables[0].unit == "A"
        assert reader.variables[1].name == "D1"
        assert reader.variables[1].is_analog is False

    def test_parse_header_metadata(self, tmp_path):
        data = _make_rms_header() + b"\x00" * 5
        f = tmp_path / "test.rms"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.rms.rms_reader.validate_rms_format"):
            reader = RMSFileReader(str(f))
            reader.parse_header()
        assert reader.metadata["frequency"] == pytest.approx(10.0)
        assert reader.metadata["format"] == "binary,test-1.0"
        assert "processed_info" in reader.metadata
        assert reader.metadata["sample_width"] == 13
        assert reader.metadata["timestamp_bytes"] == 8

    def test_get_variable_info_has_byte_size(self, tmp_path):
        data = _make_rms_header() + b"\x00" * 5
        f = tmp_path / "test.rms"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.rms.rms_reader.validate_rms_format"):
            reader = RMSFileReader(str(f))
            reader.parse_header()
        info = reader.get_variable_info()
        assert "byte_size" in info.columns
        assert info.loc[info["name"] == "CH1", "byte_size"].iloc[0] == FLOAT32_SIZE
        assert info.loc[info["name"] == "D1", "byte_size"].iloc[0] == DIGITAL_SIZE

    def test_read_binary_data(self, tmp_path):
        ts = 1_730_000_000.0
        sample = _make_binary_sample("big", ts, 42.5, 1)
        data = _make_rms_header() + sample
        f = tmp_path / "test.rms"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.rms.rms_reader.validate_rms_format"):
            reader = RMSFileReader(str(f))
            df = reader.read()
        assert len(df) == 1
        assert "CH1" in df.columns
        assert "D1" in df.columns
        assert df["CH1"].iloc[0] == pytest.approx(42.5, rel=1e-5)
        assert df["D1"].iloc[0] == 1


class TestVProcessFileReaderHeader:
    def test_parse_header_variables(self, tmp_path):
        data = _make_vprocess_header() + b"\x00" * 5
        f = tmp_path / "test.vprocess"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.vprocess.vprocess_reader.validate_vprocess_format"):
            reader = VProcessFileReader(str(f))
            reader.parse_header()
        assert len(reader.variables) == 2
        assert reader.variables[0].name == "TT1"
        assert reader.variables[0].is_analog is True
        assert reader.variables[0].unit == "K"
        assert reader.variables[0].min_val is None  # "_" sentinel → None
        assert reader.variables[1].name == "DIG1"
        assert reader.variables[1].is_analog is False

    def test_parse_header_metadata(self, tmp_path):
        data = _make_vprocess_header() + b"\x00" * 5
        f = tmp_path / "test.vprocess"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.vprocess.vprocess_reader.validate_vprocess_format"):
            reader = VProcessFileReader(str(f))
            reader.parse_header()
        assert reader.metadata["frequency"] == pytest.approx(1.0)
        assert reader.metadata["format"] == "test-vp-1.0"
        assert reader.metadata["sample_width"] == 13

    def test_get_variable_info_no_byte_size(self, tmp_path):
        data = _make_vprocess_header() + b"\x00" * 5
        f = tmp_path / "test.vprocess"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.vprocess.vprocess_reader.validate_vprocess_format"):
            reader = VProcessFileReader(str(f))
            reader.parse_header()
        info = reader.get_variable_info()
        assert "byte_size" not in info.columns

    def test_read_binary_data(self, tmp_path):
        ts = 1_730_000_000.0
        sample = _make_binary_sample("big", ts, 273.15, 0)
        data = _make_vprocess_header() + sample
        f = tmp_path / "test.vprocess"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.vprocess.vprocess_reader.validate_vprocess_format"):
            reader = VProcessFileReader(str(f))
            df = reader.read()
        assert len(df) == 1
        assert df["TT1"].iloc[0] == pytest.approx(273.15, rel=1e-5)
        assert df["DIG1"].iloc[0] == 0


class TestGetMetadataCopy:
    """get_metadata() must return a copy, not a reference."""

    def test_rms_returns_copy(self, tmp_path):
        data = _make_rms_header() + b"\x00" * 5
        f = tmp_path / "test.rms"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.rms.rms_reader.validate_rms_format"):
            reader = RMSFileReader(str(f))
            reader.parse_header()
        meta = reader.get_metadata()
        meta["injected"] = True
        assert "injected" not in reader.metadata

    def test_vprocess_returns_copy(self, tmp_path):
        data = _make_vprocess_header() + b"\x00" * 5
        f = tmp_path / "test.vprocess"
        f.write_bytes(data)
        with patch("python_magnetrun.hybrid.vprocess.vprocess_reader.validate_vprocess_format"):
            reader = VProcessFileReader(str(f))
            reader.parse_header()
        meta = reader.get_metadata()
        meta["injected"] = True
        assert "injected" not in reader.metadata
