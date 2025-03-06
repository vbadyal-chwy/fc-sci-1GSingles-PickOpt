import os
import pandas as pd
import vertica_python
from typing import Dict, Any
from utility.config_loader import load_config, get_creds
from utility.logging_config import setup_logging

class DataPuller:
    def __init__(self, config_path: str, input_mode: int):
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config, 'DataPuller')
        self.config_path = config_path
        self.input_mode = input_mode
        self.project_root = os.path.dirname(os.path.dirname(self.config_path))
        self.input_dir = os.path.join(self.project_root, "input")
        os.makedirs(self.input_dir, exist_ok=True)

        if input_mode == 1:
            vertica_creds = get_creds(self.config['database']['vertica']['credentials_file'])
            self.vertica_conn_info = {
                'host': self.config['database']['vertica']['host'],
                'port': self.config['database']['vertica']['port'],
                'database': self.config['database']['vertica']['database'],
                'user': vertica_creds[0],
                'password': vertica_creds[1],
                'use_prepared_statements': False,
                'autocommit': True,
                'tlsmode': 'disable'
            }
            self.mssql_conn_info = self.config['database']['mssql']

    def read_sql_file(self, file_name: str) -> str:
        sql_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'sql')
        file_path = os.path.join(sql_dir, file_name)
        with open(file_path, 'r') as file:
            return file.read()

    def execute_vertica_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        self.logger.debug(f"Executing Vertica query: {query[:100]}...")
        with vertica_python.connect(**self.vertica_conn_info) as connection:
            with connection.cursor() as cursor:
                if params:
                    query = query.format(**params)
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                return pd.DataFrame(data, columns=columns)

    def get_container_data(self, fc: str, start_time: str, end_time: str) -> pd.DataFrame:
        if self.input_mode == 1:
            query = self.read_sql_file('container_data.sql')
            result_df = self.execute_vertica_query(query, {'fc': fc, 'start_time': start_time, 'end_time': end_time})
            result_df.to_csv(os.path.join(self.input_dir, "container_data.csv"), index=False)
        else:
            csv_path = os.path.join(self.input_dir, "container_data-manual.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Container data file not found at {csv_path}")
            result_df = pd.read_csv(csv_path)
            result_df['arrive_datetime'] = pd.to_datetime(result_df['arrive_datetime'])
            result_df['cut_datetime'] = pd.to_datetime(result_df['cut_datetime'])
            result_df['pull_datetime'] = pd.to_datetime(result_df['pull_datetime'])
        return result_df

    def get_slotbook_data(self, fc: str) -> pd.DataFrame:
        if self.input_mode == 1:
            query = self.read_sql_file('slotbook_data.sql')
            result_df = self.execute_vertica_query(query, {'fc': fc})
            result_df.to_csv(os.path.join(self.input_dir, "slotbook_data.csv"), index=False)
        else:
            csv_path = os.path.join(self.input_dir, "slotbook_data-manual.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Slotbook data file not found at {csv_path}")
            result_df = pd.read_csv(csv_path)
        return result_df