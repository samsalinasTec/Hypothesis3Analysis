import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


rollup_df = pd.read_csv(r'/home/sam.salinas/PythonProjects/H3Canibalization/rollup_canibalizacion.csv')
rollup_df = rollup_df.loc[rollup_df["hist_ediciones_distintas"] >= 2]

credentials_path = r'/home/sam.salinas/PythonProjects/H3Canibalization/sorteostec-ml-5f178b142b6f.json'  # PLACEHOLDER
bq_project = 'sorteostec-ml'  # PLACEHOLDER
dataset_id = 'h3'  # PLACEHOLDER
table_id = 'rollup_canibalizacion_filter_clients'  # PLACEHOLDER

credentials = service_account.Credentials.from_service_account_file(credentials_path)
client = bigquery.Client(credentials=credentials, project=bq_project)
full_table_id = f"{client.project}.{dataset_id}.{table_id}"


job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
)
job = client.load_table_from_dataframe(rollup_df, full_table_id, job_config=job_config)
job.result()
