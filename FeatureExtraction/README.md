

# Scripts for Feature Extraction

This directory contains scripts designed for extracting features from PCAP files for further analysis and machine learning tasks. The workflow processes network traffic data and converts it into formats used in IoT device identification solutions. For example, YAF-Features.py use YAF tool and Super Mediator. The scripts enable processing of PCAP files, generating YAF files, and converting them into JSON format for further analysis.

## Features
- **PCAP to YAF Conversion**: Extracts flow-level statistics from PCAP files using YAF.
- **YAF to JSON Conversion**: Converts YAF output to JSON format using Super Mediator for better compatibility and processing.
- **Automated Workflow**: Efficiently processes multiple files in batches, ensuring scalability.

## Requirements

Ensure you have the following installed on your system:
- Python (>=3.7)
- YAF (Yet Another Flowmeter)
- Super Mediator
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
├── YAF-features.py           # Script for YAF feature extraction
├── Yang-features.py          # Script for features used in Yang19 paper
├── requirements.txt          # Required Python libraries
├── README.md                 # Documentation for the directory
```

## Usage

### 1. Setup
Ensure the necessary tools (`yaf` and `super_mediator`) are installed and accessible in your system's PATH. Update any required paths in the scripts to point to your data directories.

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

### 3. Batch Processing
The scripts are designed to automatically process all files within the specified directories. Ensure input files are organised accordingly.

## Customisation

### Update Paths
Modify paths in the script (`extract_features.py`) to suit your data directory structure.

### Password Handling
For operations requiring elevated permissions (e.g., `sudo`), update the `sudo_password` variable in the script or configure your system for password-less execution of required commands.

## Example Output

- **YAF File**: Contains flow-level statistics extracted from PCAP files.
- **JSON File**: Provides structured data suitable for further analysis.

## Contributing
Feel free to submit issues or pull requests to improve the scripts or documentation.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

