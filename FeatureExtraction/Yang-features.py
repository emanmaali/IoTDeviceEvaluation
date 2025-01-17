import os
import csv
import sys
import subprocess
from scapy.all import TCP, IP
import glob
import numpy as np
import os
import re
import warnings
from subprocess import Popen, PIPE


tcp_order = [0, 1, 2, 3,5, 6, 8, 9, 10, 11, 12,13,14]
udp_order = [0, 1, 2, 4,5, 7, 8, 9, 10, 11, 12,13,14]

def extend_array_to_length(arr, n):
    """
    Extend an array to a certain length by appending empty strings ('') to it.
    
    Args:
    - arr (list): The input array.
    - n (int): The desired length of the array.
    
    Returns:
    - extended_arr (list): The array extended to the desired length.
    """
    # Check if the array length is already greater than or equal to the desired length
    if len(arr) >= n:
        return arr
    
    # Calculate the number of empty strings to append
    num_empty_strings = n - len(arr)
    
    # Extend the array by appending empty strings
    extended_arr = arr + [''] * num_empty_strings
    
    return extended_arr

def extract_fields_from_packets(pcap_file):
    print(pcap_file)
    # Run tshark command to extract fields from the pcap file
    command = f"tshark -r {pcap_file} -T fields -e frame.time_epoch -e ip.proto -e ip.src -e tcp.srcport -e udp.srcport -e ip.dst -e tcp.dstport -e udp.dstport -e ip.tos -e ip.ttl -e ip.flags.df -e tcp.window_size_value -e tcp.analysis.rto -e tcp.options.mss -e tcp.options"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, _ = process.communicate()
    result = []
    # "frame.time_epoch\tproto\tip.src\tsrcport\tip.dst\tdstport\tip.tos\tip.ttl\tip.flags.df\ttcp.window_size_value\ttcp.analysis.ack_rtt\ttcp.options.mss\ttcp.options"
    field_names_with_values = output.strip().split('\n')
    for filed_item in field_names_with_values:
        fields = re.split(r'\t', filed_item)  # Regular expression to split the string by tabs
        parsed_fields = list(fields)
        if len(parsed_fields) == 1: 
            continue
        elif len(parsed_fields) > 1:
            # Extract fields from the output
            if parsed_fields[1] == '1' or parsed_fields[1] == '': 
                continue 
            else: 
                if parsed_fields[1] == '6':
                    if (len(parsed_fields) == 15):
                        result.append([parsed_fields[i] for i in tcp_order])
                    else:
                        extended_arr = extend_array_to_length(parsed_fields, 15)
                        result.append([extended_arr [i] for i in tcp_order])
                elif parsed_fields[1] == '17':
                     if (len(parsed_fields) == 15):
                        result.append([parsed_fields[i] for i in udp_order])
                     else:                 
                        extended_arr = extend_array_to_length(parsed_fields, 15)
                        result.append([extended_arr [i] for i in udp_order])
    result = np.asarray(result)
    return result

# type of service field (TOS), 
# time to live field (TTL) and don’t
# segment fields (DF). 
# Receiver window field (WIN),
# RTO sequence (RTO), 
# Max segment size field (MSS) 
# TCP options field (OPT)

def save_to_csv(rows, csv_file):
    # Write the extracted fields to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # "ip.tos\tip.ttl\tip.flags.df\ttcp.window_size_value\ttcp.analysis.ack_rtt\ttcp.options.mss\ttcp.options"
        writer.writerow(['timestamp', 'proto', 'src', 'srcport', 'dst' ,'dstport' , 
                          'IP TOS', 'IP TTL', 'IP DF',
                          'TCP WIN', 'TCP RTO', 
                          'TCP MSS', 'TCP OPT'])
        writer.writerows(rows)

    print("Extracted fields saved to:", csv_file)

def extract_and_save(pcap_files):
    # Iterate over each pcap file in the list
    for pcap_file in pcap_files:
        # Extract fields from the pcap file
        feature_extracted  = extract_fields_from_packets(pcap_file)
        # Create output CSV filename by replacing the extension with .csv
        # csv_file = os.path.splitext(pcap_file)[0] + ".csv"
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(pcap_file))[0]
        # Construct the new file path for the CSV file
        csv_file = os.path.join('/data/eman/observation/Extracted-Features/YANG-Features/', filename + ".csv")
        # Save extracted fields to CSV file
        save_to_csv(feature_extracted, csv_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python YAF-Features.py pcap_list.txt")
        sys.exit(1)

    # Get the input argument
    pcap_list_file = sys.argv[1]

    # Read the list of pcap files from the text file
    with open(pcap_list_file, 'r') as file:
        pcap_files = file.read().splitlines()

        extract_and_save(pcap_files)
