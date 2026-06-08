from utils import Sql, train_val_test_split, get_sm_service_role_arn
import numpy as np
import pandas as pd
import sagemaker, boto3
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris

role = get_sm_service_role_arn()
region = 'us-east-1'
s3_client = boto3.client('s3')

data_bucket='omm-test-bucket'
model_data_path = 'models/abalone/data'

sql_user='user-1'
sql_password='password'
sql_db='db_1'

if __name__ == '__main__':

    # DATA INGESTION
    sql=Sql(sql_user, sql_password, sql_db)

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    abalone_df = sql.query('SELECT * FROM abalone;')

    # FEATURE ENGINEERING

    # DATA FORMATTING
    y = abalone_df['rings']
    X = abalone_df.drop(columns=['rings'])

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, train_size=0.7, val_size=0.15, random_state=42)

    train_s3_path=f's3://{data_bucket}/{model_data_path}/input/train'
    validation_s3_path=f's3://{data_bucket}/{model_data_path}/input/validation'
    test_s3_path=f's3://{data_bucket}/{model_data_path}/input/test'

    pd.concat([y_train, X_train], axis=1).to_csv(f'{train_s3_path}/train.csv', index=False, header=False)
    pd.concat([y_val, X_val], axis=1).to_csv(f'{validation_s3_path}/validation.csv', index=False, header=False)
    pd.concat([y_test, X_test], axis=1).to_csv(f'{test_s3_path}/test.csv', index=False, header=False)

    input_data = {
        'train': sagemaker.inputs.TrainingInput(train_s3_path+'/', content_type='text/csv'),
        'validation': sagemaker.inputs.TrainingInput(validation_s3_path+'/', content_type='text/csv'),
        # xgboost only accepts train and validation 'test': sagemaker.inputs.TrainingInput(test_s3_path+'/', content_type='text/csv')
    }

    # TRAINING
    estimator = sagemaker.estimator.Estimator(
        image_uri=sagemaker.image_uris.retrieve('xgboost', region, version='1.5-1'),
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=f's3://{data_bucket}/{model_data_path}/output',
        sagemaker_session=sagemaker_session,
        subnets=['subnet-001be661bcef4b615', 'subnet-003ad32933ca43e74'],
        security_group_ids=['sg-00f14515abe1e47e8']
    )

    # Set hyperparameters
    estimator.set_hyperparameters(
        objective='reg:squarederror',
        num_round=100
    )

    # Train
    estimator.fit(input_data)

    # EVALUATION
    print(estimator.model_data)
