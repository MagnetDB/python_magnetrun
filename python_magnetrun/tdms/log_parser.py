#!/usr/bin/env python3
"""
Log parser for LabVIEW/DAQmx acquisition system logs.

Log file format:

- Records start with: ``<day_fr> <date> <time>: <message>``
  e.g. ``"jeu. 19-09-2019 16:14:33: Test de présence des boitiers ENET"``
- Day abbreviations are in French: lun., mar., mer., jeu., ven., sam., dim.
- Messages are a mix of French and English
- Multi-line records: Error and ENET messages include description lines
  that continue until the next timestamped record

Parses log entries including:
- ENET box presence tests (with OK/KO results per device)
- Acquisition start/stop events (Groupe, aimant)
- File creation events (Archive, Overview, Stats, Spike, Default, ManuelTrig)
- DAQmx and TDMS errors (with error codes and descriptions)
- Manual trigger events
- Fault detection events (SpikeAimant, DefautNums, Courants50Hz)

Output dictionaries:
- defaut_files: Maps fault TDMS files to their type and description
- files_with_errors: Maps Archive/Overview files to associated DAQ errors
"""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from natsort import natsorted

logger = logging.getLogger(__name__)

# French day abbreviations mapping
FRENCH_DAYS = {
    "lun.": "Mon",
    "mar.": "Tue",
    "mer.": "Wed",
    "jeu.": "Thu",
    "ven.": "Fri",
    "sam.": "Sat",
    "dim.": "Sun",
}

# Timestamp pattern: "jeu. 19-09-2019 16:14:33" - also handles encoding issues
TIMESTAMP_PATTERN = re.compile(
    r"^(lun\.|mar\.|mer\.|jeu\.|ven\.|sam\.|dim\.)\s+"
    r"(\d{2}-\d{2}-\d{4})\s+"
    r"(\d{2}:\d{2}:\d{2}):\s*(.*)$"
)


def normalize_text(text: str) -> str:
    """Normalize text to handle encoding issues with French characters."""
    # Common replacements for broken French characters
    replacements = {
        "é": "é",
        "è": "è",
        "ê": "ê",
        "ë": "ë",
        "à": "à",
        "â": "â",
        "ä": "ä",
        "ù": "ù",
        "û": "û",
        "ü": "ü",
        "ô": "ô",
        "ö": "ö",
        "î": "î",
        "ï": "ï",
        "ç": "ç",
        "\x82": "é",
        "\x8a": "è",
        "\x88": "ê",
        "\x85": "à",
        "\x83": "â",
        "\x97": "ù",
        "\x96": "û",
        "\x93": "ô",
        "\x8c": "î",
        "\x87": "ç",
        # Handle common mojibake patterns
        "Ã©": "é",
        "Ã¨": "è",
        "Ãª": "ê",
        "Ã ": "à",
        "Ã¢": "â",
        "Ã¹": "ù",
        "Ã»": "û",
        "Ã´": "ô",
        "Ã®": "î",
        "Ã§": "ç",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


class EventType(Enum):
    ENET_TEST_START = auto()
    ENET_TEST_RESULT = auto()
    ACQUISITION_START = auto()
    ACQUISITION_STOP = auto()
    FILE_CREATED = auto()
    MANUAL_TRIGGER = auto()
    ERROR = auto()
    DEFAUT_SPIKE = auto()
    DEFAUT_NUMS = auto()
    DEFAUT_50HZ = auto()
    DEFAUT_RECORDING = auto()  # "Enregistrement du défaut..."
    UNKNOWN = auto()


class DefautType(Enum):
    """Types of detected faults/defects."""

    SPIKE_AIMANT = "SpikeAimant"
    DEFAUT_NUMS = "DefautNums"
    COURANTS_50HZ = "Courants50Hz"
    UNKNOWN = "Unknown"

    @property
    def description(self) -> str:
        """Human-readable description of the fault type."""
        descriptions = {
            DefautType.SPIKE_AIMANT: "Spike de courant anormal détecté sur capteurs aimant (internes ou externes)",
            DefautType.DEFAUT_NUMS: "Défaut généré suite à un trigger matériel de l'installation (relais I_MAX)",
            DefautType.COURANTS_50HZ: "Courant 50 Hz anormalement élevé (perturbation, bruit réseau, surtension)",
            DefautType.UNKNOWN: "Type de défaut non identifié",
        }
        return descriptions.get(self, "Description non disponible")

    @property
    def file_folder(self) -> str:
        """Expected folder for this fault type."""
        folders = {
            DefautType.SPIKE_AIMANT: "Fichiers_Spike",
            DefautType.DEFAUT_NUMS: "Fichiers_Default",
            DefautType.COURANTS_50HZ: "Fichiers_Default",
        }
        return folders.get(self, "Unknown")


# DAQ Error descriptions from documentation
DAQ_ERROR_DESCRIPTIONS = {
    -200279: {
        "short": "Buffer overflow",
        "description": "L'application ne lit pas les données assez vite → débordement du buffer",
        "solutions": "Augmenter la taille du buffer, lire plus souvent, lire un nombre fixe d'échantillons",
    },
    -200284: {
        "short": "Samples not acquired",
        "description": "Certains échantillons demandés n'ont pas encore été acquis",
        "solutions": "Augmenter le timeout de lecture, effectuer la lecture plus tard, ou fournir une horloge externe",
    },
    -200019: {
        "short": "ADC timing error",
        "description": "Conversion ADC commencée avant la fin de la précédente",
        "solutions": "Augmenter la période entre conversions, vérifier l'horloge externe, vérifier les boîtiers ENET",
    },
    -200220: {
        "short": "Invalid device",
        "description": "Le périphérique spécifié n'est pas valide",
        "solutions": "Vérifier le nom du périphérique dans la configuration DAQmx",
    },
    -50103: {
        "short": "Resource reserved",
        "description": "La ressource spécifiée est réservée, l'opération n'a pas pu se faire",
        "solutions": "Libérer la ressource ou attendre qu'elle soit disponible",
    },
    -2532: {
        "short": "Invalid TDMS group/channel",
        "description": "Nom de groupe et/ou nom de voie TDMS non valide",
        "solutions": "Vérifier la configuration des groupes et voies dans le fichier TDMS",
    },
}


def get_error_info(error_code: int) -> dict:
    """Get error description and solutions for a DAQ error code."""
    if error_code in DAQ_ERROR_DESCRIPTIONS:
        return DAQ_ERROR_DESCRIPTIONS[error_code]
    return {
        "short": f"Error {error_code}",
        "description": "Erreur DAQ non documentée",
        "solutions": "Consulter la documentation NI-DAQmx",
    }


@dataclass
class LogEntry:
    timestamp: datetime | None
    event_type: EventType
    message: str
    raw_line: str


@dataclass
class ENETTestResult:
    timestamp: datetime
    results: dict[str, bool] = field(default_factory=dict)

    @property
    def ok_count(self) -> int:
        return sum(1 for v in self.results.values() if v)

    @property
    def ko_count(self) -> int:
        return sum(1 for v in self.results.values() if not v)

    @property
    def failed_devices(self) -> list[str]:
        return [k for k, v in self.results.items() if not v]


@dataclass
class AcquisitionEvent:
    timestamp: datetime
    action: str  # 'start' or 'stop'
    group: str
    magnet: str


@dataclass
class FileCreatedEvent:
    timestamp: datetime
    filepath: str

    @property
    def magnet(self) -> str | None:
        # Extract magnet from path like F:\...\M9\...
        match = re.search(r"\\(M\d+|HT\d+|A\d+)\\", self.filepath)
        return match.group(1) if match else None

    @property
    def file_type(self) -> str | None:
        # Extract type from filename like M9_Archive_191002-1445.tdms
        if "Archive" in self.filepath:
            return "Archive"
        elif "Overview" in self.filepath:
            return "Overview"
        elif "ManuelTrig" in self.filepath:
            return "ManuelTrig"
        return None


@dataclass
class ErrorEvent:
    timestamp: datetime
    error_code: int
    location: str
    description: str
    raw_message: str


@dataclass
class DefautEvent:
    """Represents a detected fault/defect."""

    timestamp: datetime
    defaut_type: DefautType
    message: str
    details: dict = field(default_factory=dict)
    associated_file: str | None = None

    @property
    def description(self) -> str:
        return self.defaut_type.description


def parse_timestamp(day_fr: str, date_str: str, time_str: str) -> datetime:
    """Parse French-formatted timestamp."""
    day, month, year = date_str.split("-")
    return datetime.strptime(f"{year}-{month}-{day} {time_str}", "%Y-%m-%d %H:%M:%S")


def parse_log_lines(lines: list[str]) -> Iterator[LogEntry]:
    """
    Parse log lines into LogEntry objects.

    Records are organized as:
    - <day_fr> <date> <time>: <message>
    - For Error or ENET messages, description lines follow until the next timestamped record
    """
    records = []
    current_record = None
    current_extra_lines: list[str] = []

    for line in lines:
        line_stripped = line.rstrip()
        if not line_stripped:
            # Empty line - could be part of multi-line record or separator
            if current_record is not None:
                current_extra_lines.append("")
            continue

        match = TIMESTAMP_PATTERN.match(line_stripped)
        if match:
            # New record starts - save previous record if exists
            if current_record is not None:
                records.append((current_record, current_extra_lines))

            day_fr, date_str, time_str, message = match.groups()
            timestamp = parse_timestamp(day_fr, date_str, time_str)
            event_type = classify_event(message)
            current_record = LogEntry(timestamp, event_type, message, line_stripped)
            current_extra_lines = []

        elif line_stripped.startswith("Test ") and ":" in line_stripped:
            # ENET test result line (no timestamp) - part of ENET block
            if (
                current_record is not None
                and current_record.event_type == EventType.ENET_TEST_START
            ):
                current_extra_lines.append(line_stripped)
            else:
                # Standalone test line
                yield LogEntry(None, EventType.ENET_TEST_RESULT, line_stripped, line_stripped)

        elif current_record is not None:
            # Continuation line for Error or ENET description
            current_extra_lines.append(line_stripped)

    # Don't forget the last record
    if current_record is not None:
        records.append((current_record, current_extra_lines))

    # Now yield records, enriching Error and ENET entries with their extra lines
    for record, extra_lines in records:
        if record.event_type == EventType.ENET_TEST_START:
            # Yield the ENET test start
            yield record
            # Yield each test result as separate entry
            for extra in extra_lines:
                if extra.startswith("Test ") and ":" in extra:
                    yield LogEntry(None, EventType.ENET_TEST_RESULT, extra, extra)

        elif record.event_type == EventType.ERROR:
            # Enrich error message with full description
            full_message = record.message
            if extra_lines:
                full_description = "\n".join(extra_lines)
                full_message = f"{record.message}\n{full_description}"
            yield LogEntry(record.timestamp, record.event_type, full_message, record.raw_line)

        else:
            # Regular record - just yield it
            # But check if extra lines contain useful info for defauts
            if record.event_type in (
                EventType.DEFAUT_SPIKE,
                EventType.DEFAUT_NUMS,
                EventType.DEFAUT_50HZ,
            ):
                if extra_lines:
                    full_message = f"{record.message}\n" + "\n".join(extra_lines)
                    yield LogEntry(
                        record.timestamp,
                        record.event_type,
                        full_message,
                        record.raw_line,
                    )
                else:
                    yield record
            else:
                yield record


def classify_event(message: str) -> EventType:
    """Classify the event type based on message content."""
    # Normalize for comparison (handle encoding issues)
    msg_lower = message.lower()

    # ENET test - handle various encodings of "présence"
    if "test de pr" in msg_lower and ("sence" in msg_lower or "enet" in msg_lower):
        return EventType.ENET_TEST_START

    # Acquisition events - handle encoding variants
    # "Acquisition_démarrée" or "Acquisition_d�marr�e" or "Acquisition_demarree"
    # "Acquisition_arrêtée" or "Acquisition_arr�t�e" or "Acquisition_arretee"
    elif "acquisition_" in msg_lower:
        # Check for stop first (contains "arr" after "acquisition")
        if "_arr" in msg_lower or "arrêtée" in message or "arretee" in msg_lower:
            return EventType.ACQUISITION_STOP
        # Then check for start (contains "d" after underscore, not followed by arr)
        elif "_d" in msg_lower or "démarrée" in message or "demarree" in msg_lower:
            return EventType.ACQUISITION_START

    elif message.startswith("File created"):
        return EventType.FILE_CREATED
    elif "trigger manuel" in msg_lower:
        return EventType.MANUAL_TRIGGER
    elif "Error" in message or "Erreur" in message:
        return EventType.ERROR

    # Fault detection events - "Default detected" or "Défaut détecté"
    elif "default detected" in msg_lower or "faut detect" in msg_lower:
        if "spike" in msg_lower:
            return EventType.DEFAUT_SPIKE
        elif "defautnums" in msg_lower or "i_max" in msg_lower:
            return EventType.DEFAUT_NUMS
        elif "courants50hz" in msg_lower or "50hz" in msg_lower:
            return EventType.DEFAUT_50HZ
        return EventType.DEFAUT_SPIKE  # Default to spike if unclear
    elif "SpikeAimant" in message or "Spike" in message:
        return EventType.DEFAUT_SPIKE
    elif "DefautNums" in message or "I_MAX" in message:
        return EventType.DEFAUT_NUMS
    elif "Courants50Hz" in message or "50Hz" in message or "50 Hz" in message:
        return EventType.DEFAUT_50HZ
    elif "enregistrement du d" in msg_lower and "faut" in msg_lower:
        return EventType.DEFAUT_RECORDING
    return EventType.UNKNOWN


class LogParser:
    """Main parser class for processing log files."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.entries: list[LogEntry] = []
        self.enet_tests: list[ENETTestResult] = []
        self.acquisitions: list[AcquisitionEvent] = []
        self.files_created: list[FileCreatedEvent] = []
        self.errors: list[ErrorEvent] = []
        self.defauts: list[DefautEvent] = []
        self.defaut_files: dict[
            str, dict
        ] = {}  # filename -> {type, description, timestamp, details}
        self.files_with_errors: dict[str, dict] = {}  # filename -> {file_info, error_info}

    def parse(self) -> "LogParser":
        """Parse the log file and populate structured data."""
        content = None
        # Try different encodings
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                with open(self.filepath, encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            # Fallback: read with errors='replace'
            with open(self.filepath, encoding="utf-8", errors="replace") as f:
                content = f.read()

        self._parse_entries(content.splitlines())
        self._extract_structured_data()
        return self

    def parse_string(self, content: str) -> "LogParser":
        """Parse log content from a string."""
        self._parse_entries(content.splitlines())
        self._extract_structured_data()
        return self

    def _parse_entries(self, lines: list[str]) -> None:
        """Parse raw lines into LogEntry objects."""
        self.entries = list(parse_log_lines(lines))

    def _extract_structured_data(self) -> None:
        """Extract structured data from parsed entries."""
        current_enet_test = None
        pending_defaut: DefautEvent | None = None  # Track defaut waiting for file association
        recent_files: list[
            tuple[datetime, str, str]
        ] = []  # (timestamp, filepath, file_type) - track recent files for error association

        for entry in self.entries:
            if entry.event_type == EventType.ENET_TEST_START:
                assert entry.timestamp is not None
                if current_enet_test:
                    self.enet_tests.append(current_enet_test)
                current_enet_test = ENETTestResult(timestamp=entry.timestamp)

            elif entry.event_type == EventType.ENET_TEST_RESULT and current_enet_test:
                # Parse "Test M3 : KO" format
                match = re.match(r"Test\s+(\w+)\s*:\s*(OK|KO)", entry.message)
                if match:
                    device, status = match.groups()
                    current_enet_test.results[device] = status == "OK"

            elif entry.event_type == EventType.ACQUISITION_START:
                acq = self._parse_acquisition(entry, "start")
                if acq:
                    self.acquisitions.append(acq)
                # Clear recent files on new acquisition start
                recent_files.clear()

            elif entry.event_type == EventType.ACQUISITION_STOP:
                acq = self._parse_acquisition(entry, "stop")
                if acq:
                    self.acquisitions.append(acq)

            elif entry.event_type == EventType.FILE_CREATED:
                assert entry.timestamp is not None
                match = re.search(r"File created\s*:\s*(.+)$", entry.message)
                if match:
                    filepath = match.group(1).strip()
                    file_event = FileCreatedEvent(timestamp=entry.timestamp, filepath=filepath)
                    self.files_created.append(file_event)

                    # Track for error association (Archive, Overview, Stats files)
                    file_type = file_event.file_type
                    if file_type in ("Archive", "Overview", "Stats"):
                        recent_files.append((entry.timestamp, filepath, file_type))

                    # Check if this file is associated with a pending defaut
                    if pending_defaut and self._is_defaut_file(filepath):
                        pending_defaut.associated_file = filepath
                        self._register_defaut_file(pending_defaut, filepath)
                        pending_defaut = None
                    # Also check if file itself indicates a defaut type
                    elif self._is_defaut_file(filepath):
                        self._register_defaut_file_standalone(entry.timestamp, filepath)

            elif entry.event_type == EventType.ERROR:
                error = self._parse_error(entry)
                if error:
                    self.errors.append(error)
                    # Associate error with recent files (within 60 seconds)
                    self._associate_error_with_files(error, recent_files)

            # Handle defaut events
            elif entry.event_type in (
                EventType.DEFAUT_SPIKE,
                EventType.DEFAUT_NUMS,
                EventType.DEFAUT_50HZ,
            ):
                defaut = self._parse_defaut(entry)
                if defaut:
                    self.defauts.append(defaut)
                    pending_defaut = defaut

            elif entry.event_type == EventType.DEFAUT_RECORDING:
                assert entry.timestamp is not None
                # "Enregistrement du défaut..." - next file will be the defaut file
                if pending_defaut is None:
                    # Create a generic defaut event if we don't have one pending
                    pending_defaut = DefautEvent(
                        timestamp=entry.timestamp,
                        defaut_type=DefautType.UNKNOWN,
                        message=entry.message,
                    )
                    self.defauts.append(pending_defaut)

        # Don't forget the last ENET test
        if current_enet_test:
            self.enet_tests.append(current_enet_test)

    def _associate_error_with_files(
        self, error: ErrorEvent, recent_files: list[tuple[datetime, str, str]]
    ) -> None:
        """Associate a DAQ error with recently created files."""
        if not recent_files:
            return

        error_info = get_error_info(error.error_code)

        for file_ts, filepath, file_type in recent_files:
            # Associate if file was created within 60 seconds before the error
            time_diff = (error.timestamp - file_ts).total_seconds()
            if 0 <= time_diff <= 60:
                filename = filepath.replace("\\", "/").split("/")[-1]

                # Extract magnet from filepath
                magnet_match = re.search(r"\\(M\d+|HT\d+|A\d+)\\", filepath)
                magnet = magnet_match.group(1) if magnet_match else None

                # If file already has an error, append to the list
                if filename in self.files_with_errors:
                    self.files_with_errors[filename]["errors"].append(
                        {
                            "error_code": error.error_code,
                            "error_short": error_info["short"],
                            "error_description": error_info["description"],
                            "error_solutions": error_info["solutions"],
                            "error_location": error.location,
                            "error_timestamp": error.timestamp,
                        }
                    )
                else:
                    self.files_with_errors[filename] = {
                        "filepath": filepath,
                        "file_type": file_type,
                        "magnet": magnet,
                        "file_timestamp": file_ts,
                        "errors": [
                            {
                                "error_code": error.error_code,
                                "error_short": error_info["short"],
                                "error_description": error_info["description"],
                                "error_solutions": error_info["solutions"],
                                "error_location": error.location,
                                "error_timestamp": error.timestamp,
                            }
                        ],
                    }

    def _is_defaut_file(self, filepath: str) -> bool:
        """Check if a filepath is a defaut-related file."""
        defaut_folders = ["Fichiers_Spike", "Fichiers_Default", "Default", "Spike"]
        return any(folder in filepath for folder in defaut_folders)

    def _parse_defaut(self, entry: LogEntry) -> DefautEvent | None:
        """
        Parse a defaut detection event.

        Message formats:
        - Courants50Hz_Courant_A1:val,Courant_A2:val,..._HH:MM:SS  MM/DD[_SpikeAimant_<sensors>_HH:MM:SS  MM/DD]
        - SpikeAimant_<sensor1>_<sensor2>_..._HH:MM:SS  MM/DD
        - DefautNums_<fault1>,<fault2>,..._HH:MM:SS  MM/DD
        """
        assert entry.timestamp is not None
        defaut_type = DefautType.UNKNOWN
        details: dict[str, Any] = {}

        # Check for "Default detected" format
        if "default detected" in entry.message.lower() or "faut detect" in entry.message.lower():
            # Extract the part after "Default detected : "
            match = re.search(r"Default detected\s*:\s*(.+)$", entry.message, re.IGNORECASE)
            if not match:
                return None

            payload = match.group(1).strip()

            # Determine fault type from start of payload
            if payload.startswith("Courants50Hz"):
                defaut_type = DefautType.COURANTS_50HZ
                details = self._parse_courants50hz_payload(payload, entry.timestamp)

            elif payload.startswith("SpikeAimant"):
                defaut_type = DefautType.SPIKE_AIMANT
                details = self._parse_spike_aimant_payload(payload, entry.timestamp)

            elif payload.startswith("DefautNums"):
                defaut_type = DefautType.DEFAUT_NUMS
                details = self._parse_defaut_nums_payload(payload, entry.timestamp)

        # Handle non "Default detected" patterns (legacy support)
        elif entry.event_type == EventType.DEFAUT_SPIKE:
            defaut_type = DefautType.SPIKE_AIMANT
            match = re.search(
                r"Spike(?:Aimant)?[_\s]*(M\d+|HT\d+|A\d+)[_\s]*(CH\d+)?", entry.message
            )
            if match:
                details["magnet"] = match.group(1)
                if match.group(2):
                    details["channel"] = match.group(2)

        elif entry.event_type == EventType.DEFAUT_NUMS:
            defaut_type = DefautType.DEFAUT_NUMS
            match = re.search(r"DefautNums[_\s]*(A\d+|M\d+|HT\d+)[_\s]*(I_MAX)?", entry.message)
            if match:
                details["axis"] = match.group(1)
                if match.group(2):
                    details["trigger"] = "I_MAX"

        elif entry.event_type == EventType.DEFAUT_50HZ:
            defaut_type = DefautType.COURANTS_50HZ
            current_matches = re.findall(r"Courant_(\w+):(\d+\.?\d*)", entry.message)
            if current_matches:
                details["currents"] = {axis: float(value) for axis, value in current_matches}

        return DefautEvent(
            timestamp=entry.timestamp,
            defaut_type=defaut_type,
            message=entry.message,
            details=details,
        )

    def _extract_detection_timestamp(
        self, timestamp_str: str, record_timestamp: datetime | None
    ) -> datetime | None:
        """
        Extract detection timestamp from string like "HH:MM:SS  MM/DD".
        Handles 12-hour format conversion using record timestamp.
        """
        match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})\s+(\d{2})/(\d{2})", timestamp_str)
        if not match:
            return None

        hour, minute, second, month, day = match.groups()
        hour = int(hour)

        if record_timestamp:
            year = record_timestamp.year
            record_hour = record_timestamp.hour
            # Convert 12-hour to 24-hour format
            if record_hour >= 12 and hour < 12:
                hour += 12
            elif record_hour < 12 and hour == 12:
                hour = 0
        else:
            year = 2025

        try:
            return datetime(year, int(month), int(day), hour, int(minute), int(second))
        except ValueError:
            return None

    def _parse_courants50hz_payload(self, payload: str, record_timestamp: datetime | None) -> dict:
        """
        Parse Courants50Hz payload.

        Format: Courants50Hz_Courant_A1:val,Courant_A2:val,..._HH:MM:SS  MM/DD[_SpikeAimant_<sensors>_HH:MM:SS  MM/DD]
        """
        details: dict[str, Any] = {}

        # Check if there's a SpikeAimant part
        spike_match = re.search(r"_SpikeAimant_(.+)$", payload)
        if spike_match:
            # Parse the SpikeAimant part
            spike_payload = "SpikeAimant_" + spike_match.group(1)
            details["spike_aimant"] = self._parse_spike_aimant_payload(
                spike_payload, record_timestamp
            )
            # Remove SpikeAimant part from payload for current parsing
            payload = payload[: spike_match.start()]

        # Extract currents: "Courant_A1:10924.6,Courant_A2:10922.9,..."
        # Can be comma-separated or underscore-separated
        current_matches = re.findall(r"Courant_(\w+):(\d+\.?\d*)", payload)
        if current_matches:
            details["currents"] = {axis: float(value) for axis, value in current_matches}

        # Extract detection timestamp (at the end before _SpikeAimant if present)
        # Pattern: "_HH:MM:SS  MM/DD" at end of the currents section
        ts_match = re.search(r"_(\d{1,2}:\d{2}:\d{2}\s+\d{2}/\d{2})\s*$", payload)
        if ts_match:
            detection_ts = self._extract_detection_timestamp(ts_match.group(1), record_timestamp)
            if detection_ts:
                details["detection_timestamp"] = detection_ts

        return details

    def _parse_spike_aimant_payload(self, payload: str, record_timestamp: datetime | None) -> dict:
        """
        Parse SpikeAimant payload.

        Format: SpikeAimant_<sensor1>_<sensor2>_..._HH:MM:SS  MM/DD
        Sensors: Interne1-2, Interne3, Interne4, Interne5, Interne6, Interne7,
                 Externe1, Externe2, ALL_internes, ALL_externes
        """
        details: dict[str, Any] = {}

        # Remove "SpikeAimant_" prefix
        content = re.sub(r"^SpikeAimant_", "", payload)

        # Extract detection timestamp from end
        ts_match = re.search(r"_?(\d{1,2}:\d{2}:\d{2}\s+\d{2}/\d{2})\s*$", content)
        if ts_match:
            detection_ts = self._extract_detection_timestamp(ts_match.group(1), record_timestamp)
            if detection_ts:
                details["detection_timestamp"] = detection_ts
            # Remove timestamp from content
            content = content[: ts_match.start()]

        # Parse sensors - split by underscore, but handle special cases
        # Known sensors: Interne1-2, Interne3-7, Externe1-2, ALL_internes, ALL_externes
        sensors = []

        # First, handle ALL_internes and ALL_externes (contain underscore)
        if "ALL_internes" in content:
            sensors.append("ALL_internes")
            content = content.replace("ALL_internes", "")
        if "ALL_externes" in content:
            sensors.append("ALL_externes")
            content = content.replace("ALL_externes", "")

        # Now split remaining by underscore and filter empty strings
        parts = [p for p in content.split("_") if p]

        # Add individual sensors
        for part in parts:
            if part.startswith("Interne") or part.startswith("Externe"):
                sensors.append(part)

        if sensors:
            details["sensors"] = sensors

            # Categorize sensors
            details["internes"] = [
                s for s in sensors if s.startswith("Interne") or s == "ALL_internes"
            ]
            details["externes"] = [
                s for s in sensors if s.startswith("Externe") or s == "ALL_externes"
            ]

        return details

    def _parse_defaut_nums_payload(self, payload: str, record_timestamp: datetime | None) -> dict:
        """
        Parse DefautNums payload.

        Format: DefautNums_<fault1>,<fault2>,..._HH:MM:SS  MM/DD
        Faults: A3_I_MAX, A4_Def3, A4_10_Eps, etc.
        """
        details: dict[str, Any] = {}

        # Remove "DefautNums_" prefix
        content = re.sub(r"^DefautNums_", "", payload)

        # Extract detection timestamp from end
        ts_match = re.search(r"_(\d{1,2}:\d{2}:\d{2}\s+\d{2}/\d{2})\s*$", content)
        if ts_match:
            detection_ts = self._extract_detection_timestamp(ts_match.group(1), record_timestamp)
            if detection_ts:
                details["detection_timestamp"] = detection_ts
            # Remove timestamp from content
            content = content[: ts_match.start()]

        # Faults are comma-separated: "A3_I_MAX,A4_I_MAX" or "A4_Def3,A4_I_MAX,A4_10_Eps"
        if content:
            faults_str = content.strip("_")
            faults = [f.strip() for f in faults_str.split(",") if f.strip()]

            if faults:
                details["faults"] = faults

                # Parse individual faults to extract axis and fault type
                parsed_faults_by_axis: dict[str, list[str]] = {}
                axes = set()
                fault_types = set()

                for fault in faults:
                    # Pattern: "A3_I_MAX", "A4_Def3", "A4_10_Eps"
                    fault_match = re.match(r"(A\d+|M\d+|HT\d+)_(.+)", fault)
                    if fault_match:
                        axis = fault_match.group(1)
                        fault_type = fault_match.group(2)
                        axes.add(axis)
                        fault_types.add(fault_type)

                        # Group by axis
                        if axis not in parsed_faults_by_axis:
                            parsed_faults_by_axis[axis] = []
                        parsed_faults_by_axis[axis].append(fault_type)

                if parsed_faults_by_axis:
                    details["parsed_faults"] = parsed_faults_by_axis
                    details["axes"] = sorted(list(axes))
                    details["fault_types"] = sorted(list(fault_types))

        return details

    def _register_defaut_file(self, defaut: DefautEvent, filepath: str) -> None:
        """Register a defaut file with its associated defaut event."""
        # Handle both Unix and Windows paths
        filename = filepath.replace("\\", "/").split("/")[-1]
        self.defaut_files[filename] = {
            "filepath": filepath,
            "type": defaut.defaut_type.value,
            "description": defaut.description,
            "timestamp": defaut.timestamp,
            "details": defaut.details,
            "original_message": defaut.message,
        }

    def _register_defaut_file_standalone(self, timestamp: datetime, filepath: str) -> None:
        """Register a defaut file detected by folder name (no explicit defaut event)."""
        # Handle both Unix and Windows paths
        filename = filepath.replace("\\", "/").split("/")[-1]

        # Infer defaut type from folder
        if "Spike" in filepath:
            defaut_type = DefautType.SPIKE_AIMANT
        elif "Default" in filepath:
            # Could be DefautNums or Courants50Hz - mark as unknown without more context
            defaut_type = DefautType.UNKNOWN
        else:
            defaut_type = DefautType.UNKNOWN

        self.defaut_files[filename] = {
            "filepath": filepath,
            "type": defaut_type.value,
            "description": defaut_type.description,
            "timestamp": timestamp,
            "details": {},
            "original_message": None,
        }

    def _parse_acquisition(self, entry: LogEntry, action: str) -> AcquisitionEvent | None:
        """Parse acquisition start/stop event."""
        assert entry.timestamp is not None
        # Pattern: "Acquisition_démarrée_sur_Groupe 1&2_aimant_M9"
        # Also handle encoding variants: "Acquisition_d�marr�e_sur_Groupe"
        pattern = r"Acquisition_[^_]+_sur_Groupe\s+([\d&]+)_aimant_(\w+)"
        match = re.search(pattern, entry.message)
        if match:
            return AcquisitionEvent(
                timestamp=entry.timestamp,
                action=action,
                group=match.group(1),
                magnet=match.group(2),
            )
        return None

    def _parse_error(self, entry: LogEntry) -> ErrorEvent | None:
        """Parse error event, including multi-line description."""
        assert entry.timestamp is not None
        # Pattern: "Error -200220 occurred at ..." or "Erreur démarrage de ..."
        match = re.search(r"(?:Error|Erreur)\s+(-?\d+)\s+occurred at\s+([^\n,]+)", entry.message)
        if match:
            error_code = int(match.group(1))
            location = match.group(2).strip()

            # Extract task name if present: "Nom de tâche : Task_Magnet_G2"
            task_match = re.search(r"Nom de t[aâ]che\s*:\s*(\S+)", entry.message)
            _task_name = task_match.group(1) if task_match else None

            # Build description from the full message
            description = entry.message

            return ErrorEvent(
                timestamp=entry.timestamp,
                error_code=error_code,
                location=location,
                description=description,
                raw_message=entry.raw_line,
            )

        # Also handle "Erreur démarrage de <device>" pattern
        match_startup = re.search(r"Erreur d[ée]marrage de\s+([^,\n]+)", entry.message)
        if match_startup:
            # Look for error code in the rest of the message
            code_match = re.search(r"Error\s+(-?\d+)", entry.message)
            error_code = int(code_match.group(1)) if code_match else 0

            return ErrorEvent(
                timestamp=entry.timestamp,
                error_code=error_code,
                location=f"Startup: {match_startup.group(1).strip()}",
                description=entry.message,
                raw_message=entry.raw_line,
            )

        return None

    def summary(self) -> dict:
        """Generate a summary of the parsed log."""
        return {
            "total_entries": len(self.entries),
            "date_range": self._get_date_range(),
            "enet_tests": {
                "count": len(self.enet_tests),
                "consistently_failing": self._get_consistently_failing_devices(),
            },
            "acquisitions": {
                "started": sum(1 for a in self.acquisitions if a.action == "start"),
                "stopped": sum(1 for a in self.acquisitions if a.action == "stop"),
                "magnets": list(set(a.magnet for a in self.acquisitions)),
            },
            "files_created": len(self.files_created),
            "files_with_errors": len(self.files_with_errors),
            "defauts": {
                "count": len(self.defauts),
                "by_type": self._count_defauts_by_type(),
                "files": len(self.defaut_files),
            },
            "errors": {
                "count": len(self.errors),
                "by_code": self._count_errors_by_code(),
            },
        }

    def get_defaut_files_dict(self) -> dict[str, dict]:
        """
        Get dictionary of defaut files with their descriptions.

        Returns:
            Dict mapping filename to:
                - filepath: full path to the file
                - type: DefautType value (SpikeAimant, DefautNums, Courants50Hz)
                - description: human-readable description of the fault
                - timestamp: when the defaut was detected
                - details: additional parsed details (magnet, channel, axis, etc.)
        """
        return self.defaut_files

    def get_files_with_errors_dict(self) -> dict[str, dict]:
        """
        Get dictionary of TDMS files (Archive, Overview, Stats) with associated DAQ errors.

        Returns:
            Dict mapping filename to:
                - filepath: full path to the file
                - file_type: Archive, Overview, or Stats
                - magnet: associated magnet (M9, HT1, etc.)
                - file_timestamp: when the file was created
                - errors: list of associated errors, each containing:
                    - error_code: DAQ error code (e.g., -200279, -2532)
                    - error_short: short description
                    - error_description: detailed description
                    - error_solutions: suggested solutions
                    - error_location: LabVIEW VI location
                    - error_timestamp: when the error occurred
        """
        return self.files_with_errors

    def _count_defauts_by_type(self) -> dict[str, int]:
        """Count defauts grouped by type."""
        counts: dict[str, int] = {}
        for defaut in self.defauts:
            type_name = defaut.defaut_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def _get_date_range(self) -> tuple[datetime, datetime] | None:
        """Get the date range of entries with timestamps."""
        timestamps = [e.timestamp for e in self.entries if e.timestamp]
        if timestamps:
            return (min(timestamps), max(timestamps))
        return None

    def _get_consistently_failing_devices(self) -> list[str]:
        """Get devices that fail in all ENET tests."""
        if not self.enet_tests:
            return []

        failing_in_all = None
        for test in self.enet_tests:
            failed = set(test.failed_devices)
            if failing_in_all is None:
                failing_in_all = failed
            else:
                failing_in_all &= failed

        return sorted(failing_in_all) if failing_in_all else []

    def _count_errors_by_code(self) -> dict[int, int]:
        """Count errors grouped by error code."""
        counts: dict[int, int] = {}
        for error in self.errors:
            counts[error.error_code] = counts.get(error.error_code, 0) + 1
        return counts


def main() -> None:
    """Demo usage with sample data."""
    import json
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        parser = LogParser(filepath).parse()
    else:
        # Demo with inline sample including defauts
        sample = """jeu. 19-09-2019 16:14:33: Test de présence des boitiers ENET
Test A1 : OK
Test A2 : OK
Test M3 : KO
Test M7 : KO

mer. 02-10-2019 14:45:54: Acquisition_démarrée_sur_Groupe 1&2_aimant_M9
mer. 02-10-2019 14:46:22: File created : F:\\Fichiers_Data\\M9\\Fichiers_Archive\\M9_Archive_191002-1445.tdms
mer. 02-10-2019 14:46:24: Acquisition_arrêtée_sur_Groupe 1&2_aimant_M9

mer. 02-10-2019 15:30:00: SpikeAimant_M9_CH3 détecté
mer. 02-10-2019 15:30:01: Enregistrement du défaut...
mer. 02-10-2019 15:30:02: File created : F:\\Fichiers_Data\\M9\\Fichiers_Spike\\M9_Spike_191002-153000.tdms

mer. 02-10-2019 16:45:00: DefautNums_A2_I_MAX déclenché
mer. 02-10-2019 16:45:01: File created : F:\\Fichiers_Data\\A2\\Fichiers_Default\\A2_Default_191002-164500.tdms

mer. 02-10-2019 17:00:00: Courants50Hz détecté sur M8
mer. 02-10-2019 17:00:01: File created : F:\\Fichiers_Data\\M8\\Fichiers_Default\\M8_Default_191002-170000.tdms

jeu. 03-10-2019 09:02:49: Error -200279 occurred at GR1&2.lvlib:Acquisition.lvlib:Acquire DAQ NUM.vi
"""
        parser = LogParser("/dev/null")
        parser.parse_string(sample)

    # Print summary
    summary = parser.summary()
    logger.info("=" * 70)
    logger.info("LOG ANALYSIS SUMMARY")
    logger.info("=" * 70)

    if summary["date_range"]:
        start, end = summary["date_range"]
        logger.info(f"\nDate range: {start} → {end}")

    logger.info(f"\nTotal entries: {summary['total_entries']}")

    logger.info(f"\n--- ENET Tests ({summary['enet_tests']['count']}) ---")
    if summary["enet_tests"]["consistently_failing"]:
        logger.info(
            f"Consistently failing devices: {', '.join(summary['enet_tests']['consistently_failing'])}"
        )

    for i, test in enumerate(parser.enet_tests, 1):
        logger.info(f"  Test {i} ({test.timestamp}): {test.ok_count} OK, {test.ko_count} KO")
        if test.failed_devices:
            logger.info(f"    Failed: {', '.join(test.failed_devices)}")

    logger.info("\n--- Acquisitions ---")
    logger.info(
        f"Started: {summary['acquisitions']['started']}, Stopped: {summary['acquisitions']['stopped']}"
    )
    if summary["acquisitions"]["magnets"]:
        logger.info(f"Magnets: {', '.join(summary['acquisitions']['magnets'])}")

    logger.info(f"\n--- Files Created ({summary['files_created']}) ---")
    for fc in parser.files_created[:5]:  # Show first 5
        logger.info(f"  {fc.timestamp}: {fc.file_type} for {fc.magnet}")
    if len(parser.files_created) > 5:
        logger.info(f"  ... and {len(parser.files_created) - 5} more")

    # === FILES WITH ERRORS DICTIONARY ===
    logger.info(f"\n{'=' * 70}")
    logger.info(f"FILES WITH DAQ ERRORS ({summary['files_with_errors']} files)")
    logger.info("=" * 70)

    files_with_errors = parser.get_files_with_errors_dict()

    if files_with_errors:
        for filename, info in files_with_errors.items():
            logger.info(f"\n{filename}")
            logger.info(f"   Type: {info['file_type']} | Magnet: {info['magnet']}")
            logger.info(f"   Created: {info['file_timestamp']}")
            logger.info(f"   Path: {info['filepath']}")
            logger.info(f"   Errors ({len(info['errors'])}):")
            for err in info["errors"]:
                logger.info(f"      [{err['error_code']}] {err['error_short']}")
                logger.info(f"         {err['error_description']}")
                logger.info(f"         Solutions: {err['error_solutions']}")
                logger.info(f"         Location: {err['error_location']}")
    else:
        logger.info("\nNo files with DAQ errors detected.")

    # === DEFAUT FILES DICTIONARY ===
    logger.info(f"\n{'=' * 70}")
    logger.info(f"DEFAUT FILES DICTIONARY ({summary['defauts']['files']} files)")
    logger.info("=" * 70)

    defaut_files = parser.get_defaut_files_dict()

    if defaut_files:
        for filename, info in defaut_files.items():
            logger.info(f"\n{filename}")
            logger.info(f"   Type: {info['type']}")
            logger.info(f"   Description: {info['description']}")
            logger.info(f"   Record timestamp: {info['timestamp']}")
            if info["details"]:
                # Format details nicely
                details_str = []
                for key, value in info["details"].items():
                    if isinstance(value, datetime | dict):
                        details_str.append(f"{key}: {value}")
                    else:
                        details_str.append(f"{key}: {value}")
                logger.info(f"   Details: {', '.join(details_str)}")
            logger.info(f"   Full path: {info['filepath']}")
    else:
        logger.info("\nNo defaut files detected in this log.")

    # Summary by defaut type
    if summary["defauts"]["by_type"]:
        logger.info("\n--- Defauts by Type ---")
        for dtype, count in summary["defauts"]["by_type"].items():
            logger.info(f"  {dtype}: {count}")

    logger.info(f"\n--- Errors ({summary['errors']['count']}) ---")
    for code, count in summary["errors"]["by_code"].items():
        logger.info(f"  Error {code}: {count} occurrence(s)")

    # Export defaut_files as JSON for programmatic use
    logger.info(f"\n{'=' * 70}")
    logger.info("JSON EXPORTS")
    logger.info("=" * 70)

    # Convert datetime to string for JSON serialization
    logger.info("\n--- Files with DAQ Errors ---")
    export_files_errors = {}
    for filename in natsorted(files_with_errors.keys()):
        info = files_with_errors[filename]
        export_files_errors[filename] = {
            # "filepath": info["filepath"],
            "file_type": info["file_type"],
            "magnet": info["magnet"],
            "file_timestamp": (
                info["file_timestamp"].isoformat() if info["file_timestamp"] else None
            ),
            "errors": [
                {
                    **err,
                    "error_timestamp": (
                        err["error_timestamp"].isoformat() if err["error_timestamp"] else None
                    ),
                }
                for err in info["errors"]
            ],
        }
    logger.info(json.dumps(export_files_errors, indent=2, ensure_ascii=False))

    # Save export_files_errors to file
    errors_output_file = "files_with_errors.json"
    with open(errors_output_file, "w", encoding="utf-8") as f:
        json.dump(export_files_errors, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved files with errors to: {errors_output_file}")

    logger.info("\n--- Defaut Files ---")
    export_defaut = {}
    for filename in natsorted(defaut_files.keys()):
        info = defaut_files[filename]

        # Recursively serialize details, converting any datetime objects
        def serialize_value(value: Any, exclude_keys: set | None = None) -> Any:
            if exclude_keys is None:
                exclude_keys = set()
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, dict):
                return {
                    k: serialize_value(v, exclude_keys)
                    for k, v in value.items()
                    if k not in exclude_keys
                }
            elif isinstance(value, list):
                return [serialize_value(v, exclude_keys) for v in value]
            else:
                return value

        serialized_details = serialize_value(
            info["details"], exclude_keys={"axes", "internes", "externes"}
        )

        export_defaut[filename] = {
            # "filepath": info["filepath"],
            "type": info["type"],
            "description": info["description"],
            "timestamp": info["timestamp"].isoformat() if info["timestamp"] else None,
            "details": serialized_details,
            # "original_message": info["original_message"],
        }
    logger.info(json.dumps(export_defaut, indent=2, ensure_ascii=False))

    # Save export_defaut to file
    defaut_output_file = "defaut_files.json"
    with open(defaut_output_file, "w", encoding="utf-8") as f:
        json.dump(export_defaut, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved defaut files to: {defaut_output_file}")


if __name__ == "__main__":
    main()
