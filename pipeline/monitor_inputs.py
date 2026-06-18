def get_monitoring_job_input(
        deploy_type, 
        endpoint_name=None, 
        data_cature_dir=None, 
        dataset_format={'Csv': {'Header': True|False}}
        ):

    if deploy_type == 'realtime':
        return {
            'EndpointInput': {
                'EndpointName': endpoint_name,
                'LocalPath': '/opt/ml/processing/input/endpoint'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }
    else:
        return {
            'BatchTransformInput': {
                'DataCapturedDestinationS3Uri': f'{data_cature_dir}/',
                'DatasetFormat': dataset_format, # {'Csv': {'Header': True|False}#, # 'Json': {'Line': True|False}, # 'Parquet': {}}
                'LocalPath': '/opt/ml/processing/input'#,
                # 'S3InputMode': 'Pipe'|'File',
                # 'S3DataDistributionType': 'FullyReplicated'|'ShardedByS3Key',
                # 'FeaturesAttribute': 'string',
                # 'InferenceAttribute': 'string',
                # 'ProbabilityAttribute': 'string',
                # 'ProbabilityThresholdAttribute': 123.0,
                # 'StartTimeOffset': 'string',
                # 'EndTimeOffset': 'string',
                # 'ExcludeFeaturesAttribute': 'string'
            }
        }