import warnings
import datetime 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math 
import os 
import logging
import tldextract
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

baseline_dict_path = '/data/eman/observation/Extracted-Features/TF-IDF-Features/minitestbed/baseline-minitestbed-device-dict.txt'
IDF_dict_path = '/data/eman/observation/Extracted-Features/TF-IDF-Features/minitestbed/baseline-minitestbed-IDF-dict.txt'

class TF_IDF:
    def __init__(self):
        pass

    def load_dns_data(self, file_path):
        data = pd.read_pickle(file_path)
        return data

    def preprocess_dns_data(self, data):
        grouped_data = data.groupby('device_ip')['domain'].apply(list).reset_index(name='queries')
        return grouped_data
    
    def compute_similarity(self,if_idf_baseline_dict,enviroment_tf_idf_vector):
        # Create vectors from dictionaries
        keys = set(if_idf_baseline_dict.keys()).union(set(enviroment_tf_idf_vector.keys()))
        vector1 = [if_idf_baseline_dict.get(key, 0) for key in keys]
        vector2 = [enviroment_tf_idf_vector.get(key, 0) for key in keys]
        # Compute cosine similarity
        cosine_sim = cosine_similarity([vector1], [vector2])[0][0]
        # print("Cosine Similarity:", cosine_sim)
        return cosine_sim
    
    def compute_tfidf_features(self, dict1, dict2):
        # Convert dictionaries to dataframes
        df1 = pd.DataFrame.from_dict(dict1, orient='index', columns=['IDF'])
        df2 = pd.DataFrame.from_dict(dict2, orient='index', columns=['frequency'])

        # Reset index to make 'domain' a column
        df1.reset_index(inplace=True)
        df2.reset_index(inplace=True)
        df1.rename(columns={'index': 'domain'}, inplace=True)
        df2.rename(columns={'index': 'domain'}, inplace=True)

        # Merge dataframe2 with dataframe1 based on the 'domain' column
        merged_df = pd.merge(df2, df1, on='domain', how='left')
        # Multiply IDF with frequency
        merged_df['TF-IDF'] = merged_df['IDF'] * merged_df['frequency']
        return merged_df

    def compute_tf_idf_vector(self,device_queries_dict, device_idf, observation_period):
        # Compute the number of time windows Nt
        time_window = 24  # Assuming time window is 1 hour
        num_time_windows = observation_period / time_window

        device_queries = pd.DataFrame.from_dict(device_queries_dict, orient='index', columns=['Probability'])
        device_queries.index.name = 'domain'
        device_queries = device_queries.reset_index()
        device_queries['Probability'] = device_queries['Probability'].astype(float)
        device_queries['domain'] = device_queries['domain'].astype(str)
        tf_idf_vector = {}
        for index, row in device_queries.iterrows():
            domain = row['domain']
            query_probability = row['Probability']
            fkj = query_probability * num_time_windows
            # Multiply fkj by IDF
            tf_idf_vector[domain] = fkj * device_idf.get(domain, 0)
        return tf_idf_vector

    def compute_enviroment_tf_idf_vector(self, iot_tf_idf_vector, enviroment_queries, client_idf):
        """
        Compute the similarity between an IoT TF-IDF vector and the behavior of a client.
        ------
        Parameters:
            iot_tf_idf_vector (dict): TF-IDF vector for the IoT device.
            enviroment_queries (dict): Dictionary containing domain names queried by the client and their frequencies.
            enviroment_idf (dict): IDF values for each domain.
        Returns:
            dict: enviroment_tf_idf vector for the enviroment.
        """
        enviroment_tf_idf_vector = {}
        # Project enviroment queries into the space of IoT TF-IDF vector
        for domain, frequency in enviroment_queries.items():
            # Check if domain exists in IoT TF-IDF vector
            if domain in iot_tf_idf_vector:
                enviroment_tf_idf_vector[domain] = frequency * client_idf.get(domain,0)
        return enviroment_tf_idf_vector
    
    def inverse_document_frequency(self, all_clients, num_clinets_with_domain):
        # num_clinets_with_domain = sum(1 for doc in all_clients if domain in doc)
        if num_clinets_with_domain > 0:
            return math.log(all_clients / num_clinets_with_domain)
        else:
            return 0.0  # If the term doesn't appear in any document, IDF is 0
    
    def get_classification(self,similarity,threshold):
        return 1 if similarity > threshold else 0

    def count_0_1_for_device_pairs(self,file_path):
        counts = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                device_pair = parts[1]
                count = int(parts[2])
                if device_pair not in counts:
                    counts[device_pair] = {'0': 0, '1': 0}
                counts[device_pair][str(count)] += 1
    
        # Find the count with the highest '1'
        max_count_1 = 0
        for device_pair, count_dict in counts.items():
            if count_dict['1'] > max_count_1:
                max_count_1 = count_dict['1']
    
        # Return only the count with the highest '1'
        result = {}
        for device_pair, count_dict in counts.items():
            if count_dict['1'] == max_count_1:
                result[device_pair] = count_dict['1']
    
        return result 

class Preprocessing:

    def __init__(self):
        pass

    def chunk_dataframe_by_day(self,df, timestamp_column):
        new_df = df.copy()
        # Convert the timestamp column to datetime if it's not already
        new_df[timestamp_column] = pd.to_datetime( new_df[timestamp_column])
        # Set the timestamp column as the index
        new_df.set_index(timestamp_column, inplace=True)
        # Resample the DataFrame to daily intervals
        daily_chunks = [(chunk.index.date[0], chunk) for _, chunk in new_df.resample('D')]
        return daily_chunks

    def preprocess_dataframe(self,df):
        logging.info("converting column data types and adding a 'start_time' column...")
        # Define column data type conversions
        column_type_conversions = {
            'src_ip': 'string',
            'dst_ip': 'string',
            'device_name': 'string',
            'dns_query':'string',
            'domain': 'string'}
        # Convert column data types
        df = df.astype(column_type_conversions)
        # Convert 'timestamp' column to datetime and assign to 'start_time'
        df['start_time'] = pd.to_datetime(df['timestamp'], unit='ns')
        return df
    
    def count_dns_queries_per_device(self, df):
        logging.info('calculate dns query per device ..')
        # Group the DataFrame by 'device' column and count the number of DNS queries
        dns_query_counts = df.groupby('device_name')['dns_query'].count().reset_index()
        dns_query_counts.columns = ['device_name', 'num_dns_queries']
        return dns_query_counts

    def calculate_dns_query_probability(self, dns_queries_df):
        logging.info('calculate dns query probability..')
        # Group the DataFrame by device name and DNS query and count the frequency
        dns_query_frequency = dns_queries_df.groupby(['device_name', 'domain']).size().reset_index(name='frequency')
        # Calculate total DNS queries count for each device name
        total_queries_per_device = dns_query_frequency.groupby('device_name')['frequency'].sum().reset_index(name='total_queries')
        # Merge the total_queries_per_device DataFrame with dns_query_frequency
        dns_query_frequency = dns_query_frequency.merge(total_queries_per_device, on='device_name')
        # Calculate probability
        dns_query_frequency['probability'] = dns_query_frequency['frequency'] / dns_query_frequency['total_queries']
        return dns_query_frequency
    
    def count_devices_per_domain(self, dns_queries_df):
        # logging.info('count devices per domain..')
        # Group the DataFrame by domain and device name, count the number of unique devices
        devices_per_domain = dns_queries_df.groupby(['domain', 'device_name']).size().reset_index(name='device_count')
        # Group by domain and count the number of unique devices
        devices_per_domain = devices_per_domain.groupby('domain')['device_count'].count().reset_index(name='num_devices')
        return devices_per_domain
    
    def count_dns_query_frequency(self, dns_queries_df):
        # logging.info('we count dns query frequency..')
        # Group the DataFrame by device name and dns_query, and count the frequency
        dns_query_frequency = dns_queries_df.groupby(['device_name', 'domain']).size().reset_index(name='frequency')
        return dns_query_frequency

    def get_domain(self, domain):
        extracted = tldextract.extract(domain)
        return extracted.domain + '.' + extracted.suffix
    
    def read_dict(self,file_path):
        with open(file_path, 'r') as file:
            dict_str = file.read()
        return eval(dict_str)
    
    def preprocessing_baseline(self,baseline_path):
        logging.info(f'we are processing the baseline {baseline_path}..')
        try: 
            dns_queries_df = pd.read_pickle(baseline_path)
            if not dns_queries_df.empty:
                dns_queries_df ['domain'] = dns_queries_df ['dns_query'].apply(self.get_domain)
                dns_queries_df = self.preprocess_dataframe(dns_queries_df)
                dns_queries_df = dns_queries_df.replace('TPLINK_SmartPlug_2', 'TPLINK_SmartPlug2')
                dns_queries_df['device_name'].unique()
                return dns_queries_df
        except Exception as e:
            print(f"An error occurred with processing {baseline_path}: {e}") 

    def flush_text_file(self,file_path):
        with open(file_path, 'w') as file:
            pass 

    def get_output(self, filename):
        file_name, _= os.path.splitext(filename)
        new_file = f"{file_name}.txt"
        return os.path.join('/data/eman/observation/TF-IDF-results',new_file) 

# Main function 
def main(): 
    preprocess = Preprocessing()
    tfidf = TF_IDF()

    baseline_path = '/data/eman/observation/Extracted-Features/TF-IDF-Features/minitestbed/minitestbed_new_pcaps_202212-tf.pkl'
    testcase_path = ['/data/eman/observation/Extracted-Features/TF-IDF-Features/minitestbed/minitestbed_new_pcaps_202309-tf.pkl',
                     '/data/eman/observation/Extracted-Features/TF-IDF-Features/minitestbed/minitestbed_new_pcaps_202306-tf.pkl',
                     '/data/eman/observation/Extracted-Features/TF-IDF-Features/minitestbed/minitestbed_new_pcaps_202401-tf.pkl',
                     '/data/eman/observation/Extracted-Features/TF-IDF-Features/minitestbed/minitestbed_new_pcaps_202402-tf.pkl',
                     '/data/eman/observation/Extracted-Features/TF-IDF-Features/minitestbed/minitestbed_new_pcaps_202312-tf.pkl']
    
    dns_queries_df = preprocess.preprocessing_baseline(baseline_path)
    # dns_query_frequency = preprocess.count_dns_query_frequency(dns_queries_df)
    # dns_query_frequency = preprocess.calculate_dns_query_probability(dns_queries_df)
    domain_device_frequency = preprocess.count_devices_per_domain(dns_queries_df)
    IDF_info = []
    for index, row in domain_device_frequency.iterrows():
        # domain = row['domain']
        num_devices = row['num_devices']
        IDF_info.append({'domain': row['domain'], 'IDF': tfidf.inverse_document_frequency(9, num_devices)})       
   
    # Convert to DataFrame
    idf_df = pd.DataFrame(IDF_info)
    idf_dict = {key: value for key, value in zip(idf_df['domain'], idf_df['IDF'])}
    
    idf_df_dic  = preprocess.read_dict(IDF_dict_path)
    device_dict  = preprocess.read_dict(baseline_dict_path)

    df_test = tfidf.compute_tfidf_features(idf_dict,device_dict.get('Amazon_Echo'))
    if_idf_baseline_dict = dict(zip(df_test['domain'], df_test['TF-IDF']))

    baseline_device_queries = device_dict.get('Amazon_Echo')
    observation_period = 504  # 21 days = 24x21
    iot_tf_idf_vector = tfidf.compute_tf_idf_vector(baseline_device_queries, idf_df_dic, observation_period)
    
    for test_case in testcase_path:
        logging.info(f"processing case {test_case}...")
        out_path = preprocess.get_output(os.path.basename(test_case))
        testcase_df = pd.read_pickle(test_case)
        testcase_df ['domain'] = testcase_df ['dns_query'].apply(preprocess.get_domain)
        testcase_df= preprocess.preprocess_dataframe(testcase_df)
        for device in testcase_df['device_name'].unique():
            dns_device_df = testcase_df.loc[testcase_df['device_name'] == device ]
            dns_device_df = dns_device_df.reset_index()
            daily_chunks = preprocess.chunk_dataframe_by_day(dns_device_df, 'timestamp')
            for date, chunk in daily_chunks:
                dns_amazon_freq_df = preprocess.count_dns_query_frequency(chunk)
                domain_freq_dict = dict(zip(dns_amazon_freq_df['domain'], dns_amazon_freq_df['frequency']))
                result_df = tfidf.compute_tfidf_features(idf_dict,domain_freq_dict)
                domain_tfif_dict = dict(zip( result_df['domain'],  result_df['TF-IDF']))
                enviroment_tf_idf_vector = tfidf.compute_enviroment_tf_idf_vector(iot_tf_idf_vector, domain_tfif_dict, idf_df_dic)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'a') as file:
                    similarity_cosine = tfidf.compute_similarity(if_idf_baseline_dict,enviroment_tf_idf_vector)
                    # classification = tfidf.get_classification(similarity_cosine,0.75)
                    file.write(f'{date}:{device}xAmazon_Echo:{similarity_cosine}\n')
        # print(tfidf.count_0_1_for_device_pairs(out_path))
     

if __name__ == "__main__":
      main()
