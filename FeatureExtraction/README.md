

# Scripts for Feature Extraction
This directory contains scripts designed for automating feature extraction from network traffic data. The scripts support processing of PCAP files, extracting flow-level statistics using YAF, Super Mediator, and `tshark`, and converting the results into structured formats for analysis.


## Features
- **PCAP to YAF Conversion**: Extracts flow-level statistics from PCAP files using YAF.
- **YAF to JSON Conversion**: Converts YAF output to JSON format using Super Mediator.
- **PCAP Field Extraction with `tshark`**: Extracts detailed packet-level fields such as IP and TCP/UDP headers.


## Requirements

Ensure you have the following installed on your system:
- Python (>=3.7)
- YAF (Yet Another Flowmeter)
- Super Mediator
- Wireshark (`tshark` command-line tool)
- Required Python Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`

Install Python dependencies using:
```bash
pip install -r requirements.txt
```

## Directory Structure

```plaintext
scripts/
├── YAF-features.py           # Script for feature extraction (PCAP to YAF/JSON). These features use in Okui22 paper. 
├── Yang-features.py          # Script for extracting packet-level fields with tshark. These features used in Yang19 paper
├── requirements.txt          # Required Python libraries
├── README.md                 # Documentation for the directory
```

## Usage

### 1. Setup
Ensure the necessary tools (yaf, super_mediator, tshark) are installed and accessible in your system's PATH. Update any required paths in the scripts to point to your data directories.

### 2. Running the Scripts

#### Convert PCAP to YAF
Run the following command to process PCAP files and generate YAF output:
```bash
python extract_features.py --input-dir <pcap_directory> --output-dir <yaf_directory>
```

#### Convert YAF to JSON
Run the following command to process YAF files and convert them to JSON format:
```bash
python extract_features.py --yaf-dir <yaf_directory> --json-dir <json_directory>
```

#### Extract Packet Fields with tshark

Run the following command to extract fields like IP and TCP/UDP headers from PCAP files:
```bash
python extract_tshark_features.py pcap_list.txt
```
pcap_list.txt should contain the paths of PCAP files, one per line.

### 3. Batch Processing
The scripts are designed to automatically process all files within the specified directories. Ensure input files are organised accordingly.

## Customisation

### Extend tshark Fields
To include additional fields during packet extraction, update the command variable in extract_tshark_features.py.

### Update Paths
Modify paths in the scripts (extract_features.py and extract_tshark_features.py) to suit your data directory structure.

### Password Handling
For operations requiring elevated permissions (e.g., `sudo`), update the `sudo_password` variable in the script or configure your system for password-less execution of required commands.


## Example Output

- **YAF File**: Contains flow-level statistics extracted from PCAP files.
- **JSON File**: Provides structured data suitable for further analysis.
- **CSV File**: Stores extracted packet fields for tasks like feature engineering and model training.

## Contributing
Feel free to submit issues or pull requests to improve the scripts or documentation.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

