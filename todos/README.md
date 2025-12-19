# convertxml.py - XML Configuration Parser

A Python tool for parsing CIRRUS_CP XML configuration files from the LNCMI magnet control system.

## Overview

This parser extracts PID controller parameters from XML configuration files used by the CIRRUS_CP analog control system. It specifically focuses on current regulation settings for various magnets (M7, M8, M9, M10).

## Purpose

The tool reads XML configuration files from the CIRRUS control system and extracts:

- PID controller parameters (Kp, Ki, Kd)
- Threshold settings (numero_seuil)
- Magnet-specific configurations
- Loop coupling ratios (rapport_entre_boucle)
- Configuration version and date

## Installation

Requires Python 3.x and the xml2dict library:

```bash
pip install xml2dict
```

## Usage

### Basic Usage

Parse a single XML configuration file:

```bash
python convertxml.py config.xml
```

Parse multiple XML files:

```bash
python convertxml.py config1.xml config2.xml config3.xml
```

### Command Line Options

```bash
python convertxml.py [OPTIONS] INPUT_FILE [INPUT_FILE ...]
```

**Arguments:**
- `input_file` - One or more XML configuration files to parse (required)

**Options:**
- `--datadir DIR` - Directory containing the XML files (default: current directory)
- `--debug` - Enable debug mode to print full JSON configuration

### Examples

Parse config from a specific directory:

```bash
python convertxml.py A1_config.xml --datadir /srv-data/cirrus/A1/xml/
```

Parse with debug output:

```bash
python convertxml.py config.xml --debug
```

## XML Configuration Structure

The parser expects XML files with the following structure:

```xml
<configuration version_config="X.X" date="YYYY-MM-DD">
  <CIRRUS_CP>
    <analogique>
      <regs>
        <regulateur numero="1" numero_jeu="19" numero_seuil="1" 
                    Ki="100" Kp="50" Kd="10" 
                    rapport_entre_boucle="...">
        </regulateur>
        <!-- More regulateur entries -->
      </regs>
      <rampes>
        <!-- Ramp configurations -->
      </rampes>
    </analogique>
  </CIRRUS_CP>
</configuration>
```

## Configuration Parameters

### Magnet Mapping

The tool maps `numero_jeu` values to magnet identifiers:

| numero_jeu | Magnet |
| ---------- | ------ |
| 17         | M7     |
| 18         | M8     |
| 19         | M9     |
| 20         | M10    |

### Regulateur Parameters

- **numero** - Controller type (1 = Current, 2 = Voltage, 3-6 = Pont1-4)
- **numero_jeu** - Magnet identifier (see mapping above)
- **numero_seuil** - Threshold range (1-3):
  - 1: 0 to 60A
  - 2: 60 to 1000A
  - 3: Above 1000A
- **Ki** - Integral gain (PID parameter)
- **Kp** - Proportional gain (PID parameter)
- **Kd** - Derivative gain (PID parameter)
- **rapport_entre_boucle** - Loop coupling ratio (links between groups)

**Note:** A value of -1 for any PID parameter (Ki, Kp, Kd) is treated as 0.

## Output

### Standard Output

For each XML file, the tool prints:

1. Configuration version and date
2. Current PID parameters for each recognized magnet

Example output:

```
A1_config.xml
Version: 2.1, Date: 2023-05-15
Current PID M9: Numero Seuil: 1, Ki: 100, Kp: 50, Kd: 10, Rapport Entre Boucle: 0.5
Current PID M9: Numero Seuil: 2, Ki: 150, Kp: 75, Kd: 15, Rapport Entre Boucle: 0.5
Current PID M10: Numero Seuil: 1, Ki: 120, Kp: 60, Kd: 12, Rapport Entre Boucle: 0.6
```

### Debug Output

With `--debug` flag, the tool prints the complete parsed configuration as formatted JSON.

## Configuration Source

XML configuration files are typically retrieved from:

```
https://srv-data-install.lncmi.cnrs.fr/cirrus/A1/xml/
```

Or from the local data server at:

```
/srv-data/cirrus/A1/xml/
```

## Current Filtering

The parser currently only extracts parameters for:
- **Current controllers** (numero = 1)
- **Recognized magnets** (M7, M8, M9, M10)

Voltage controllers and bridge controllers (Pont1-4) are parsed but not displayed.

## Future Enhancements

Potential areas for extension:
- Extract voltage controller parameters (numero = 2)
- Parse bridge controller settings (numero = 3-6)
- Process ramp configurations from `configuration.CIRRUS_CP.analogique.rampes.rampe`
- Export results to JSON/CSV format
- Add validation for PID parameter ranges

## Notes

- Values of -1 in PID parameters are normalized to 0
- The tool preserves the original working directory after execution
- Multiple threshold configurations can exist for the same magnet
- Loop coupling ratios define relationships between magnet groups (GR1: A1-A2, GR2: A3-A4)

## See Also

Related documentation:
- CIRRUS control system manual
- R14L09 p13 (threshold definitions)
- Magnet group configurations (GR1, GR2)

## License

See the main project LICENSE file.
