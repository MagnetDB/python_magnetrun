"""
Tests for the configuration module.

These tests verify that the refactored configuration dataclasses
produce identical results to the original setup() function.
"""

import pytest

from python_magnetrun.analysis.config import (
    DEFAULT_BINS,
    DEFAULT_GROUP,
    DEFAULT_LEVELS,
    DEFAULT_WINDOW_SIZE,
    # Pre-defined configurations
    HOUSING_CONFIGS,
    LAG_THRESHOLD_RATIO,
    SAMPLING_RATE_ARCHIVE,
    SAMPLING_RATE_INCIDENTS,
    # Constants
    SAMPLING_RATE_OVERVIEW,
    TIME_OFFSET_ARCHIVE,
    TIME_OFFSET_INCIDENTS,
    TIME_OFFSET_OVERVIEW,
    AnalysisConfig,
    ChannelMapping,
    # Dataclasses
    ColorConfig,
    ThresholdConfig,
    VoltageChannelMapping,
)


class TestConstants:
    """Test module-level constants."""

    def test_sampling_rates(self):
        """Sampling rates should match expected values."""
        assert SAMPLING_RATE_OVERVIEW == 1.0
        assert SAMPLING_RATE_ARCHIVE == 120.0
        assert SAMPLING_RATE_INCIDENTS == 4800.0

    def test_time_offsets(self):
        """Time offsets should be half the sampling period."""
        assert TIME_OFFSET_OVERVIEW == 0.5
        assert abs(TIME_OFFSET_ARCHIVE - (1 / 120.0) / 2.0) < 1e-10
        assert abs(TIME_OFFSET_INCIDENTS - (1 / 4800.0) / 2.0) < 1e-10

    def test_analysis_defaults(self):
        """Default analysis parameters should match original."""
        assert LAG_THRESHOLD_RATIO == 0.2
        assert DEFAULT_WINDOW_SIZE == 50
        assert DEFAULT_BINS == 10
        assert DEFAULT_LEVELS == 4

    def test_default_group(self):
        """Default TDMS group should be Courants_Alimentations."""
        assert DEFAULT_GROUP == "Courants_Alimentations"


class TestColorConfig:
    """Test ColorConfig dataclass."""

    def test_default_colors(self):
        """Default colors should match original setup()."""
        colors = ColorConfig()
        assert colors.up_color == "red"
        assert colors.plateau_color == "green"
        assert colors.down_color == "blue"

    def test_get_method(self):
        """get() should return correct colors."""
        colors = ColorConfig()
        assert colors.get("U") == "red"
        assert colors.get("P") == "green"
        assert colors.get("D") == "blue"
        assert colors.get("X") == "black"  # Unknown defaults to black

    def test_to_dict(self):
        """to_dict() should match original color_dict."""
        colors = ColorConfig()
        expected = {"U": "red", "P": "green", "D": "blue"}
        assert colors.to_dict() == expected


class TestChannelMapping:
    """Test ChannelMapping dataclass."""

    def test_default_mapping(self):
        """Default mapping should match original channels_dict."""
        mapping = ChannelMapping()
        assert mapping.reference_gr1 == "Courant_GR1"
        assert mapping.reference_gr2 == "Courant_GR2"

    def test_get_current_channel(self):
        """get_current_channel() should return correct channel."""
        mapping = ChannelMapping()
        assert mapping.get_current_channel("Référence_GR1") == "Courant_GR1"
        assert mapping.get_current_channel("Référence_GR2") == "Courant_GR2"

    def test_get_current_channel_invalid(self):
        """get_current_channel() should raise KeyError for invalid key."""
        mapping = ChannelMapping()
        with pytest.raises(KeyError):
            mapping.get_current_channel("Invalid_Key")

    def test_to_dict(self):
        """to_dict() should match original channels_dict."""
        mapping = ChannelMapping()
        expected = {
            "Référence_GR1": "Courant_GR1",
            "Référence_GR2": "Courant_GR2",
        }
        assert mapping.to_dict() == expected


class TestVoltageChannelMapping:
    """Test VoltageChannelMapping dataclass."""

    def test_default_gr1_channels(self):
        """GR1 should have internal probe channels."""
        mapping = VoltageChannelMapping()
        expected = (
            "ALL_internes",
            "Interne1",
            "Interne2",
            "Interne3",
            "Interne4",
            "Interne5",
            "Interne6",
            "Interne7",
        )
        assert mapping.reference_gr1_channels == expected

    def test_default_gr2_channels(self):
        """GR2 should have external probe channels."""
        mapping = VoltageChannelMapping()
        expected = ("ALL_externes", "Externe1", "Externe2")
        assert mapping.reference_gr2_channels == expected

    def test_get_channels(self):
        """get_channels() should return correct channels."""
        mapping = VoltageChannelMapping()
        gr1 = mapping.get_channels("Référence_GR1")
        assert "Interne1" in gr1
        assert "ALL_internes" in gr1

        gr2 = mapping.get_channels("Référence_GR2")
        assert "Externe1" in gr2
        assert "ALL_externes" in gr2

    def test_to_dict(self):
        """to_dict() should match original uchannels_dict."""
        mapping = VoltageChannelMapping()
        result = mapping.to_dict()
        assert "Référence_GR1" in result
        assert "Référence_GR2" in result
        assert isinstance(result["Référence_GR1"], list)


class TestHousingConfig:
    """Test HousingConfig dataclass."""

    def test_available_sites(self):
        """All expected sites should be available."""
        assert "M8" in HOUSING_CONFIGS
        assert "M9" in HOUSING_CONFIGS
        assert "M10" in HOUSING_CONFIGS

    def test_m9_configuration(self):
        """M9 should have correct channel mappings."""
        config = HOUSING_CONFIGS["M9"]
        assert config.name == "M9"
        assert config.reference_gr1_current == "IH"
        assert config.reference_gr2_current == "IB"
        assert config.reference_gr1_flow == "FlowH"
        assert config.reference_gr2_flow == "FlowB"
        assert config.reference_gr1_rpm == "RpmH"
        assert config.reference_gr2_rpm == "RpmB"
        assert config.reference_gr1_pin == "HPH"
        assert config.reference_gr2_pin == "HPB"

    def test_m8_configuration(self):
        """M8 should have swapped channel mappings vs M9."""
        config = HOUSING_CONFIGS["M8"]
        assert config.name == "M8"
        # M8 swaps GR1/GR2 compared to M9
        assert config.reference_gr1_current == "IB"
        assert config.reference_gr2_current == "IH"

    def test_m10_configuration(self):
        """M10 should have same mapping as M8."""
        m8 = HOUSING_CONFIGS["M8"]
        m10 = HOUSING_CONFIGS["M10"]
        assert m10.reference_gr1_current == m8.reference_gr1_current
        assert m10.reference_gr2_current == m8.reference_gr2_current

    def test_get_pupitre_channel(self):
        """get_pupitre_channel() should return correct channel."""
        m9 = HOUSING_CONFIGS["M9"]
        assert m9.get_pupitre_channel("Référence_GR1") == "IH"
        assert m9.get_pupitre_channel("Référence_GR2") == "IB"

        m8 = HOUSING_CONFIGS["M8"]
        assert m8.get_pupitre_channel("Référence_GR1") == "IB"
        assert m8.get_pupitre_channel("Référence_GR2") == "IH"

    def test_voltage_channels_m9(self):
        """M9 voltage channels should be correct."""
        m9 = HOUSING_CONFIGS["M9"]
        assert m9.voltage_channels_gr1 == (
            "Ucoil1",
            "Ucoil2",
            "Ucoil3",
            "Ucoil4",
            "Ucoil5",
            "Ucoil6",
            "Ucoil7",
            "Ucoil8",
            "Ucoil9",
            "Ucoil10",
            "Ucoil11",
            "Ucoil12",
            "Ucoil13",
            "Ucoil14",
        )
        assert m9.voltage_channels_gr2 == ("Ucoil15", "Ucoil16")

    def test_voltage_channels_m8_m10(self):
        """M8/M10 voltage channels should be swapped vs M9."""
        m8 = HOUSING_CONFIGS["M8"]
        assert m8.voltage_channels_gr1 == ("Ucoil15", "Ucoil16")
        assert m8.voltage_channels_gr2 == (
            "Ucoil1",
            "Ucoil2",
            "Ucoil3",
            "Ucoil4",
            "Ucoil5",
            "Ucoil6",
            "Ucoil7",
            "Ucoil8",
            "Ucoil9",
            "Ucoil10",
            "Ucoil11",
            "Ucoil12",
            "Ucoil13",
            "Ucoil14",
        )

    def test_to_pupitre_dict(self):
        """to_pupitre_dict() should match original format."""
        m9 = HOUSING_CONFIGS["M9"]
        result = m9.to_pupitre_dict()
        assert result["Référence_GR1"] == "IH"
        assert result["Référence_GR2"] == "IB"
        assert result["Référence_GR1_Q"] == "FlowH"
        assert result["Référence_GR1_Rpm"] == "RpmH"
        assert result["Référence_GR1_Pin"] == "HPH"


class TestThresholdConfig:
    """Test ThresholdConfig dataclass."""

    def test_default_thresholds(self):
        """Default thresholds should match original threshold_dict."""
        config = ThresholdConfig.default()

        # Reference/current channels
        assert config.get("Référence_GR1") == 0.5
        assert config.get("Courant_GR1") == 0.5
        assert config.get("Référence_GR2") == 0.5
        assert config.get("Courant_GR2") == 0.5

        # Pupitre channels
        assert config.get("IH") == 1.0
        assert config.get("IB") == 1.0
        assert config.get("UH") == 0.1
        assert config.get("UB") == 0.1

        # Internal probes
        for i in range(1, 8):
            assert config.get(f"Interne{i}") == 1.0e-2

        # Coil voltages
        for i in range(1, 15):
            assert config.get(f"Ucoil{i}") == 1.0e-2
        assert config.get("Ucoil15") == 0.1
        assert config.get("Ucoil16") == 0.1

        # Flow and pressure
        assert config.get("debitbrut") == 25.0
        assert config.get("Pmagnet") == 0.1

    def test_get_with_default(self):
        """get() should return default for unknown keys."""
        config = ThresholdConfig.default()
        assert config.get("unknown_key", 999.0) == 999.0
        assert config.get("another_unknown") == 0.1  # Default default

    def test_dict_access(self):
        """Should support dictionary-style access."""
        config = ThresholdConfig.default()
        assert config["IH"] == 1.0
        assert "IH" in config
        assert "nonexistent" not in config

    def test_to_dict(self):
        """to_dict() should return plain dictionary."""
        config = ThresholdConfig.default()
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["IH"] == 1.0


class TestAnalysisConfig:
    """Test AnalysisConfig dataclass."""

    def test_for_housing_valid(self):
        """for_housing() should create config for valid housings."""
        for housing in ["M8", "M9", "M10"]:
            config = AnalysisConfig.for_housing(housing)
            assert config.housing.name == housing

    def test_for_housing_invalid(self):
        """for_housing() should raise ValueError for unknown housing."""
        with pytest.raises(ValueError, match="Unknown housing"):
            AnalysisConfig.for_housing("INVALID")

    def test_default_components(self):
        """Config should have all default components."""
        config = AnalysisConfig.for_housing("M9")
        assert isinstance(config.channels, ChannelMapping)
        assert isinstance(config.voltage_channels, VoltageChannelMapping)
        assert isinstance(config.thresholds, ThresholdConfig)
        assert isinstance(config.colors, ColorConfig)

    def test_convenience_methods(self):
        """Convenience methods should delegate correctly."""
        config = AnalysisConfig.for_housing("M9")

        assert config.get_pupitre_channel("Référence_GR1") == "IH"
        assert config.get_current_channel("Référence_GR1") == "Courant_GR1"


class TestIntegration:
    """Integration tests verifying complete workflows."""

    def test_housing_specific_workflow(self):
        """Test complete workflow for each housing."""
        for housing_name in ["M8", "M9", "M10"]:
            config = AnalysisConfig.for_housing(housing_name)

            # Should be able to get all channel mappings
            for ref_key in ["Référence_GR1", "Référence_GR2"]:
                current = config.get_current_channel(ref_key)
                pupitre = config.get_pupitre_channel(ref_key)
                threshold = config.thresholds.get(current)

                assert current is not None
                assert pupitre is not None
                assert threshold > 0

    def test_immutability(self):
        """Frozen dataclasses should be immutable."""
        config = AnalysisConfig.for_housing("M9")

        with pytest.raises(Exception):  # noqa: B017 # FrozenInstanceError
            config.housing = HOUSING_CONFIGS["M8"]
