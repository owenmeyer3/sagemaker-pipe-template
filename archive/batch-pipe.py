from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
import utils, boto3, sagemaker

# What you create once (setup):
# suggest_baseline()           → one-time job to establish baseline statistics
# create_monitoring_schedule() → creates the schedule/configuration
# Pipeline definition  → upsert() registers the pipeline

# What runs during inference:
# MonitorBatchTransformStep → runs every time the pipeline executes
#                             transform job + monitoring analysis
#                             compares inference data against baseline
#                             generates violation reports

# Parameters
# model_package_arn = ParameterString(name='ModelPackageArn', default_value='arn:aws:sagemaker:us-east-1:088461143167:model-package/abalone/1')
# input_data = ParameterString(name='InputData', default_value='s3://omm-test-bucket/test/test_X.csv')
# output_path = ParameterString(name='OutputPath',default_value='s3://omm-test-bucket/batch-output/')
model_package_group_name = ParameterString(name='ModelPackageGroupName')
model_package_version = ParameterString(name='ModelPackageVersion', default_value='latest')
project_bucket = ParameterString(name='ProjectBucket', default_value='my-bucket')
project_path = ParameterString(name='ProjectPath', default_value='models/project')
project_path = ParameterString(name='RunType', default_value='Deploy', enum_values=['Deploy','Inference'])

project_dir=       f's3://{project_bucket}/{project_path}'
data_dir=          f"{project_dir}/data"
model_dir=         f"{project_dir}/model"
data_capture_dir=  f'{data_dir}/capture'
monitors_dir=      f'{data_dir}/monitors'
ground_truth_dir=  f'{data_dir}/ground-truth'
baseline_file=     f'{data_dir}/baseline/baseline.csv'
baseline_pred_file=f'{data_dir}/baseline/baseline_pred.csv'
dq_monitor_dir=    f'{monitors_dir}/data-quality'
mq_monitor_dir=    f'{monitors_dir}/model-quality'
mb_monitor_dir=    f'{monitors_dir}/model-bias'
me_monitor_dir=    f'{monitors_dir}/model-explainability'

boto_session=boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=boto_session)

role = sagemaker.get_execution_role(sagemaker_session)
region=boto_session.region_name

model_name, model_package_arn = utils.create_model_object_from_registry(boto_session, model_package_group_name, role, model_package_version=model_package_version)

# Transformer
transformer = sagemaker.transformer.Transformer(
    model_name=model_name,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=p.batch_out_dir+'/',
    accept='text/csv',
    assemble_with='Line',
    sagemaker_session=sagemaker_session
)


# Monitor transform step
monitor_transform_step = MonitorBatchTransformStep(
    name='AbaloneMonitorTransformStep',
    transform_step_args=transformer.transform(
        data=input_data.default_value,
        content_type='text/csv',
        split_type='Line'
    ),
    monitor_configuration=data_quality_monitor,
    baseline_statistics=data_quality_monitor.baseline_statistics(),
    baseline_constraints=data_quality_monitor.suggested_constraints(),
    output_s3_uri='s3://omm-test-bucket/monitoring-reports/',
    fail_on_violation=False
)

# Pipeline
pipeline = Pipeline(
    name='AbaloneInferencePipeline',
    parameters=[
        model_package_arn,
        input_data,
        output_path
    ],
    steps=[monitor_transform_step],
    sagemaker_session=sagemaker_session
)

pipeline.upsert(role_arn=role)

# Run with specific model version
execution = pipeline.start(
    parameters={
        'ModelPackageArn': 'arn:aws:sagemaker:us-east-1:088461143167:model-package/abalone/1',
        'InputData': 's3://omm-test-bucket/test/test_X.csv',
        'OutputPath': 's3://omm-test-bucket/batch-output/'
    }
)

execution.wait()
print("Pipeline complete")