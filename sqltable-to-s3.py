import argparse
import pandas as pd
import sqlalchemy
import boto3
import sys

def export_to_parquet(db_name, table_name, s3_bucket, s3_key):
    try:
        # Connect to MariaDB
        print(f"Connecting to database '{db_name}'...")
        engine = sqlalchemy.create_engine(
            f'mysql+pymysql://user-1:password@localhost/{db_name}'
        )

        # Read table into dataframe
        print(f"Reading table '{table_name}'...")
        df = pd.read_sql(f'SELECT * FROM {table_name}', engine)
        print(f"Found {len(df)} rows")

        # Save as parquet locally
        local_path = f'/tmp/{table_name}.parquet'
        print(f"Converting to Parquet...")
        df.to_parquet(local_path, index=False)

        # Upload to S3
        print(f"Uploading to s3://{s3_bucket}/{s3_key}...")
        s3 = boto3.client('s3')
        s3.upload_file(local_path, s3_bucket, s3_key)

        print(f"Done! File uploaded to s3://{s3_bucket}/{s3_key}")

    except sqlalchemy.exc.OperationalError as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a MariaDB table to S3 as a Parquet file"
    )
    parser.add_argument("--db",       required=True, help="Database name")
    parser.add_argument("--table",    required=True, help="Table name")
    parser.add_argument("--bucket",   required=True, help="S3 bucket name")
    parser.add_argument("--key",      required=True, help="S3 key (file path in bucket)")

    args = parser.parse_args()

    export_to_parquet(args.db, args.table, args.bucket, args.key)
