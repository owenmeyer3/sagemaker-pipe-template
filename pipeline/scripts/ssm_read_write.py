import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_params(ssm, scope):
    return ssm.get_parameters_by_path(Path=f'/{scope}/', Recursive=True) 

def set_param(ssm, scope, name, value, type='String'):
    ssm.put_parameter(Name=f"/{scope}/{name}", Value=value, Type=type, Overwrite=True) # update if already exists
    return ssm.get_parameter(Name=name)['Parameter']['Value']

def handler(event, context):
    ssm = boto3.client('ssm')

    scope = event['scope']

    # Set input vars
    if 'writes' in event:
        input_dicts = event['writes']
        for name, value in input_dicts.items():
            set_param(ssm, scope, name, value, type='String')

    # Return project data
    params = get_params(ssm, scope)  # includes nested paths
    logger.info(f"Read from SSM: {params}")
    return {'params':params}