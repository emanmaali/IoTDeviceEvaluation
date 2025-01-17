# after installing YAF you can use this script to extract features
from pathlib import Path
import os
import subprocess
from subprocess import Popen, PIPE
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def execute_command(command, sudo_password=None):
    """Execute a shell command with optional sudo privilege."""
    try:
        if sudo_password:
            command = f"echo {sudo_password} | sudo -S {command}"
        with Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True) as process:
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                logging.info(f"Command succeeded: {command}")
                return stdout
            else:
                logging.error(f"Command failed: {command}\nError: {stderr}")
                return None
    except Exception as e:
        logging.error(f"Exception occurred while executing command: {command}\n{e}")
        return None

def process_pcap_files(input_dir, output_dir, sudo_password=None):
    """Process PCAP files into YAF and JSON formats."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pcap") and not file.startswith(".") and "pcappatternto process-" in file:
                pcap_file = Path(root) / file
                if pcap_file.stat().st_size > 144:  # Check file size
                    file_stem = pcap_file.stem
                    yaf_file = output_dir / f"{file_stem}.yaf"
                    json_file = output_dir / f"{file_stem}.json"

                    # Generate YAF file
                    cmd_yaf = f"yaf --in {pcap_file} --out {yaf_file} --flow-stats --mac"
                    execute_command(cmd_yaf)

                    # Generate JSON file from YAF
                    cmd_json = f"super_mediator {yaf_file} --out {json_file} --output-mode json"
                    execute_command(cmd_json, sudo_password)

def process_yaf_files(yaf_dir, sudo_password=None):
    """Convert YAF files to JSON using super_mediator."""
    yaf_dir = Path(yaf_dir)

    for yaf_file in yaf_dir.glob("*.yaf"):
        json_file = yaf_file.with_suffix(".json")
        cmd = f"super_mediator {yaf_file} --out {json_file} --output-mode json"
        execute_command(cmd, sudo_password)

if __name__ == "__main__":
    # Define paths and sudo password
    input_directory = "your_input_directory_path"  # Replace with the path to your PCAP files
    output_directory = "your_output_directory_path"  # Replace with the path for output files
    sudo_password = "yourpassword"  # Replace with your sudo password

    # Process PCAP files
    logging.info("Starting PCAP file processing...")
    process_pcap_files(input_directory, output_directory, sudo_password)

    # Process YAF files
    logging.info("Starting YAF to JSON conversion...")
    process_yaf_files(output_directory, sudo_password)

    logging.info("Processing completed.")
