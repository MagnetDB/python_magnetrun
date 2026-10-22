"""Tests for python_magnetrun.utils.validation.

Covers each validator function independently, plus integration tests that
exercise the validation calls inside MagnetData factory methods.
"""

from pathlib import Path

import pytest

from python_magnetrun.magnetdata import load_magnetdata
from python_magnetrun.utils.validation import (
    FileFormatError,
    classify_pupitre_header,
    count_valid_pupitre_rows,
    validate_csv_format,
    validate_fepc_binary_format,
    validate_file_exists,
    validate_rms_format,
    validate_tdms_format,
    validate_txt_format,
    validate_vprocess_format,
)

SAMPLE_TXT = Path(__file__).parent / "data" / "sample_pupitre.txt"


# ---------------------------------------------------------------------------
# validate_file_exists
# ---------------------------------------------------------------------------


class TestValidateFileExists:
    def test_existing_file_passes(self, tmp_path):
        f = tmp_path / "ok.txt"
        f.write_text("hello")
        validate_file_exists(str(f))  # no exception

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="file not found"):
            validate_file_exists(str(tmp_path / "no_such_file.txt"))


# ---------------------------------------------------------------------------
# validate_txt_format
# ---------------------------------------------------------------------------


class TestValidateTxtFormat:
    def test_valid_pupitre_file_passes(self):
        validate_txt_format(str(SAMPLE_TXT))  # no exception

    def test_wrong_extension_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("metadata\nDate\tTime\tField\n1\t2\t3\n")
        with pytest.raises(FileFormatError, match="expected .txt extension"):
            validate_txt_format(str(f))

    def test_missing_date_time_header_raises(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("metadata row\nField\tIcoil1\n1.0\t2.0\n")
        with pytest.raises(FileFormatError, match="missing required header columns"):
            validate_txt_format(str(f))

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        with pytest.raises(FileFormatError, match="empty"):
            validate_txt_format(str(f))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_txt_format(str(tmp_path / "ghost.txt"))


# ---------------------------------------------------------------------------
# classify_pupitre_header (line 1 of a pupitre file)
# ---------------------------------------------------------------------------


def _write_line1(tmp_path, line1: bytes):
    f = tmp_path / "2024.12.05 - 11:16:02.txt"
    f.write_bytes(line1 + b"\nDate\tTime\tField\n2024.12.05\t11:16:04\t0.0\n")
    return str(f)


_COEFS = b"\t0.681129\t4.96613e-06\t6.87325e-11\t0.0035"


class TestClassifyPupitreHeader:
    def test_full(self, tmp_path):
        line1 = b"2\tM19071101\t14200\t13900" + _COEFS + b"\t0\t0\t0\t0"
        assert classify_pupitre_header(_write_line1(tmp_path, line1)) == "full"

    def test_full_with_empty_magnet_code(self, tmp_path):
        """M3 and M5 write an empty MagnetCode: two tabs in a row."""
        line1 = b"1\t\t14200\t13900" + _COEFS
        assert classify_pupitre_header(_write_line1(tmp_path, line1)) == "full"

    def test_nbcoil_only_is_short(self, tmp_path):
        assert classify_pupitre_header(_write_line1(tmp_path, b"2")) == "short"

    def test_nbcoil_and_label_is_short(self, tmp_path):
        line1 = b"16\tHL-28_1_BI04_1_BE02_1_13/05/2014"
        assert classify_pupitre_header(_write_line1(tmp_path, line1)) == "short"

    def test_wrong_field_count_is_short(self, tmp_path):
        line1 = b"2\tM19071101\t14200\t13900" + _COEFS
        assert classify_pupitre_header(_write_line1(tmp_path, line1)) == "short"

    def test_date_first_token_is_data(self, tmp_path):
        line1 = b"2011.02.09\t09:06:50\t0.0000\t8.6"
        assert classify_pupitre_header(_write_line1(tmp_path, line1)) == "data"

    def test_empty_line_is_empty(self, tmp_path):
        assert classify_pupitre_header(_write_line1(tmp_path, b"")) == "empty"

    def test_nul_bytes_is_empty(self, tmp_path):
        assert classify_pupitre_header(_write_line1(tmp_path, b"\x00" * 32)) == "empty"


# ---------------------------------------------------------------------------
# count_valid_pupitre_rows
# ---------------------------------------------------------------------------


class TestCountValidPupitreRows:
    def test_all_valid(self):
        dates = ["2024.12.05"] * 3
        times = ["12:15:30", "12:15:31", "12:15:32"]
        assert count_valid_pupitre_rows(dates, times) == 3

    def test_stops_at_unparseable(self):
        dates = ["2024.12.05", "2024.12.05", None, "2014.07.27"]
        times = ["12:15:31", "12:15:32", "20897.5", "22:38:31"]
        assert count_valid_pupitre_rows(dates, times) == 2

    def test_stops_at_backward_jump_over_one_hour(self):
        dates = ["2024.12.05", "2024.12.05", "2024.12.05"]
        times = ["12:15:31", "12:15:32", "11:15:31"]
        assert count_valid_pupitre_rows(dates, times) == 2

    def test_keeps_backward_step_of_one_hour(self):
        dates = ["2018.10.28"] * 3
        times = ["02:59:59", "02:00:00", "02:00:01"]
        assert count_valid_pupitre_rows(dates, times) == 3

    def test_unparseable_first_row(self):
        assert count_valid_pupitre_rows(["junk"], ["12:00:00"]) == 0


# ---------------------------------------------------------------------------
# validate_tdms_format
# ---------------------------------------------------------------------------


class TestValidateTdmsFormat:
    def test_valid_magic_passes(self, tmp_path):
        f = tmp_path / "real.tdms"
        f.write_bytes(b"TDSm" + b"\x00" * 100)
        validate_tdms_format(str(f))  # no exception

    def test_wrong_magic_raises(self, tmp_path):
        f = tmp_path / "fake.tdms"
        f.write_bytes(b"\x89PNG" + b"\x00" * 100)
        with pytest.raises(FileFormatError, match="expected TDMS magic"):
            validate_tdms_format(str(f))

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.tdms"
        f.write_bytes(b"")
        with pytest.raises(FileFormatError, match="expected TDMS magic"):
            validate_tdms_format(str(f))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_tdms_format(str(tmp_path / "ghost.tdms"))


# ---------------------------------------------------------------------------
# validate_csv_format
# ---------------------------------------------------------------------------


class TestValidateCsvFormat:
    def test_valid_csv_passes(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2,col3\n1,2,3\n")
        validate_csv_format(str(f))  # no exception

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_bytes(b"")
        with pytest.raises(FileFormatError, match="empty"):
            validate_csv_format(str(f))

    def test_required_columns_present_passes(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("Date,Time,Field\n2022.01.01,00:00:00,1.0\n")
        validate_csv_format(str(f), required_columns=["Date", "Time"])  # no exception

    def test_required_columns_missing_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("Field,Icoil1\n1.0,2.0\n")
        with pytest.raises(FileFormatError, match="missing required columns"):
            validate_csv_format(str(f), required_columns=["Date", "Time"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_csv_format(str(tmp_path / "ghost.csv"))


# ---------------------------------------------------------------------------
# validate_rms_format
# ---------------------------------------------------------------------------


class TestValidateRmsFormat:
    def test_valid_rms_passes(self, tmp_path):
        f = tmp_path / "data.rms"
        f.write_bytes(b"# variables = foo\n" + b"\x00" * 64)
        validate_rms_format(str(f))  # no exception

    def test_wrong_first_byte_raises(self, tmp_path):
        f = tmp_path / "bad.rms"
        f.write_bytes(b"TDSm" + b"\x00" * 64)
        with pytest.raises(FileFormatError, match="expected ASCII header marker '#'"):
            validate_rms_format(str(f))

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.rms"
        f.write_bytes(b"")
        with pytest.raises(FileFormatError, match="expected ASCII header marker '#'"):
            validate_rms_format(str(f))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_rms_format(str(tmp_path / "ghost.rms"))


# ---------------------------------------------------------------------------
# validate_vprocess_format
# ---------------------------------------------------------------------------


class TestValidateVprocessFormat:
    def test_valid_vprocess_passes(self, tmp_path):
        f = tmp_path / "data.vprocess"
        f.write_bytes(b"# variables = foo\n" + b"\x00" * 64)
        validate_vprocess_format(str(f))  # no exception

    def test_wrong_first_byte_raises(self, tmp_path):
        f = tmp_path / "bad.vprocess"
        f.write_bytes(b"\xff\x00\x00\x00")
        with pytest.raises(FileFormatError, match="expected ASCII header marker '#'"):
            validate_vprocess_format(str(f))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_vprocess_format(str(tmp_path / "ghost.vprocess"))


# ---------------------------------------------------------------------------
# validate_fepc_binary_format
# ---------------------------------------------------------------------------


class TestValidateFEPCBinaryFormat:
    def test_valid_analog_file_passes(self, tmp_path):
        from python_magnetrun.hybrid.kHz.fepc_reader import ANALOG_BLOCK_SIZE

        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00" * (ANALOG_BLOCK_SIZE * 3))
        validate_fepc_binary_format(str(f), "ANA")  # no exception

    def test_valid_digital_file_passes(self, tmp_path):
        from python_magnetrun.hybrid.kHz.fepc_reader import DIGITAL_BLOCK_SIZE

        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00" * (DIGITAL_BLOCK_SIZE * 5))
        validate_fepc_binary_format(str(f), "DIG")  # no exception

    def test_misaligned_analog_file_raises(self, tmp_path):
        from python_magnetrun.hybrid.kHz.fepc_reader import ANALOG_BLOCK_SIZE

        f = tmp_path / "bad.bin"
        f.write_bytes(b"\x00" * (ANALOG_BLOCK_SIZE * 2 + 7))
        with pytest.raises(FileFormatError, match="not a multiple of block size"):
            validate_fepc_binary_format(str(f), "ANA")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_fepc_binary_format(str(tmp_path / "ghost.bin"), "ANA")


# ---------------------------------------------------------------------------
# Integration: MagnetData factory methods
# ---------------------------------------------------------------------------


class TestMagnetDataFromtxtValidation:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_magnetdata(str(tmp_path / "no_such.txt"))

    def test_wrong_headers_raises_file_format_error(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("metadata\nField\tIcoil1\n1.0\t2.0\n")
        with pytest.raises(FileFormatError, match="missing required header columns"):
            load_magnetdata(str(f))

    def test_valid_file_loads(self):
        md = load_magnetdata(str(SAMPLE_TXT))
        assert "Date" in md.getKeys()
        assert "Time" in md.getKeys()


class TestMagnetDataFromtdmsValidation:
    def test_wrong_magic_raises_file_format_error(self, tmp_path):
        f = tmp_path / "fake.tdms"
        f.write_bytes(b"\x89PNG" + b"\x00" * 100)
        with pytest.raises(FileFormatError, match="expected TDMS magic"):
            load_magnetdata(str(f))

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_magnetdata(str(tmp_path / "no_such.tdms"))


class TestMagnetDataFromcsvValidation:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_magnetdata(str(tmp_path / "no_such.csv"))

    def test_empty_file_raises_file_format_error(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_bytes(b"")
        with pytest.raises(FileFormatError, match="empty"):
            load_magnetdata(str(f))
