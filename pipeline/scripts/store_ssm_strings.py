import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    sm_client = boto3.client('sagemaker')
    
    # Store baseline paths in SSM for inference pipeline to use
    ssm = boto3.client('ssm')
    
    ssm.put_parameter(
        Name='/abalone/baseline/data_quality_statistics',
        Value=event['dq_statistics_path'],
        Type='String',
        Overwrite=True
    )
    
    ssm.put_parameter(
        Name='/abalone/baseline/data_quality_constraints',
        Value=event['dq_constraints_path'],
        Type='String',
        Overwrite=True
    )
    
    ssm.put_parameter(
        Name='/abalone/baseline/model_bias_statistics',
        Value=event['mb_statistics_path'],
        Type='String',
        Overwrite=True
    )
    
    ssm.put_parameter(
        Name='/abalone/baseline/model_bias_constraints',
        Value=event['mb_constraints_path'],
        Type='String',
        Overwrite=True
    )
    
    ssm.put_parameter(
        Name='/abalone/model_name',
        Value=event['model_name'],
        Type='String',
        Overwrite=True
    )
    
    logger.info("Monitoring configurations registered")
    return {'status': 'success'}