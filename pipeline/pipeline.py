from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.lambda_helper import Lambda
from sagemaker.core.image_uris import retrieve as retrieve_image
from sagemaker.core.resources import Model
from sagemaker.core.lambda_helper import Lambda
from sagemaker.core.transformer import Transformer
from sagemaker.core.inputs import TransformInput
from sagemaker.core.shapes.shapes import ScheduleConfig, ContainerDefinition

from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.steps import TransformStep
from sagemaker.mlops.workflow.lambda_step import LambdaOutputTypeEnum, LambdaStep, LambdaOutput
from sagemaker.mlops.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
import utils, baseline, boto3, argparse, monitor, json
import pandas as pd

class CPipeline(Pipeline):
    def __init__(self, sagemaker_session, name, model_name_param, model_package_arn_param, model_package_group_name_param, model_package_version_param, target_name, prediction_name, project_bucket, project_path, role_param, action, deployment_type, monitor_instance_type='ml.m5.large', endpoint_instance_type='ml.m5.large'):

        self.name=name
        self.target_name=target_name
        self.prediction_name=prediction_name
        self.sagemaker_session=sagemaker_session

        self.model_name_param=model_name_param
        self.model_package_arn_param=model_package_arn_param
        self.model_package_group_name_param=model_package_group_name_param
        self.model_package_version_param=model_package_version_param

        self.project_bucket=project_bucket
        self.project_path=project_path
        self.role_param=role_param
        self.monitor_instance_type=monitor_instance_type
        self.endpoint_instance_type=endpoint_instance_type
        self.model_image_uri=retrieve_image('xgboost', 'us-east-1', version='1.5-1')

        project_dir=       f's3://{project_bucket}/{project_path}'
        data_dir=          f"{project_dir}/data"
        self.model_dir=         f"{project_dir}/model"
        self.data_capture_dir=  f'{data_dir}/capture'
        self.transforms_dir=    f'{data_dir}/transforms'
        self.ground_truth_dir=  f'{data_dir}/ground-truth'
    
        self.baseline_file=     f'{data_dir}/baseline/baseline.csv'
        self.baseline_pred_file=f'{data_dir}/baseline/baseline_pred.csv'
        self.baseline_X_file   =f'{data_dir}/input/test/test_X.csv'
        self.train_file=        f'{data_dir}/input/train/train.csv'

   

        self.baseliner = baseline.Baseliner(model_name_param.to_string(), data_dir, self.baseline_file, self.train_file, monitor_instance_type, sagemaker_session)
        self.monitor_maker = monitor.MonitorMaker(model_name_param.to_string(), data_dir, self.baseline_file, self.train_file, monitor_instance_type, sagemaker_session)

        self.steps = []
        self.parameters = []
        if action == 'deploy':
            self.steps = self.get_deploy_steps(        
                deployment_type=deployment_type
            )
            self.parameters=[
                self.model_package_arn_param,
                self.model_name_param,
                self.model_package_group_name_param,
                self.model_package_version_param,
                self.project_bucket,
                self.project_path,
                self.role_param
            ]

        elif action == 'inference':
            self.steps = self.get_inference_steps(        
                deployment_type=deployment_type
            )
            self.parameters=[
                self.model_package_group_name_param,
                self.model_package_version_param,
                self.project_bucket,
                self.project_path,
                self.role_param
            ]
        else:
            pass

        super().__init__(
            name=self.name,
            parameters=self.parameters,
            steps=self.steps,
            sagemaker_session=self.sagemaker_session
            )


    def get_deploy_steps(self, deployment_type):
        # baseline data prep
        self.baseliner.make_baseline_sets(self.target_name, self.prediction_name, target_type=float)

        if deployment_type == 'realtime':
            
            get_model_step = self.get_model_from_registry_step()

            deploy_endpoint_step=self.get_deploy_endpoint_step(depends_on=[get_model_step])

            data_quality_step = self.baseliner.get_data_quality_step(
                self.role_param.default_value, depends_on=[deploy_endpoint_step]
            )
            data_bias_step = self.baseliner.get_data_bias_step(
                self.role_param.default_value, depends_on=[deploy_endpoint_step]
            )
            model_quality_step = self.baseliner.get_model_quality_step(
                self.role_param.default_value, depends_on=[deploy_endpoint_step]
            )
            model_bias_step = self.baseliner.get_model_bias_step(
                self.role_param.default_value, depends_on=[deploy_endpoint_step]
            )
            model_explainability_step = self.baseliner.get_model_explainability_step(
                self.role_param.default_value, depends_on=[deploy_endpoint_step]
            )

            # ssm_step = self.get_ssm_step(self.name, writes={}, depends_on=[])

            return [get_model_step, deploy_endpoint_step, data_quality_step, data_bias_step, model_quality_step, model_bias_step, model_explainability_step]#, ssm_step]
        
        elif deployment_type == 'batch':

            get_model_step = self.get_model_from_registry_step()

            batch_transform_step=self.get_batch_transform_step(self.baseline_X_file, depends_on=[get_model_step])

            job_definition=self.baseline.get_job_definition(self.sagemaker_session, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None)

            schedule_config = ScheduleConfig(
                schedule_expression='cron(0 * ? * * *)', 
                data_analysis_start_time="-PT1H", 
                data_analysis_end_time="-PT2H"
            )

            model_quality_step=self.get_batch_model_quality_step(self.sagemaker_session, batch_transform_step, schedule_config, depends_on=[])


            return [get_model_step]
        else:
            return []


    def get_model_from_registry_step(self):

        # make create model step
        lambda_function = Lambda(
            function_name='GetModelFromRegistry',
            execution_role_arn=self.role_param.default_value,
            script='scripts/get_model_from_registry.py',  # path to your file
            handler='get_model_from_registry.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )
        create_model_step = LambdaStep(
            name='GetModelStep',
            lambda_func=lambda_function,
            inputs={
                'model_package_group_name': self.model_package_group_name_param,
                'model_package_version': self.model_package_version_param,
                'role': self.role_param
            },
            outputs=[
                LambdaOutput(output_name='model_name', output_type=LambdaOutputTypeEnum.String),
                LambdaOutput(output_name='model_package_arn', output_type=LambdaOutputTypeEnum.String)
            ]
        )
        return create_model_step


    def get_batch_create_step(self):

        # model = Model(
        #     image_uri=self.model_image_uri,
        #     model_data=self.model_package_arn,  # from registry
        #     role=self.role,
        #     sagemaker_session=self.sagemaker_session
        # )

            # model_name: StrPipeVar
            # primary_container: Optional[ContainerDefinition] = Unassigned()
            # containers: Optional[List[ContainerDefinition]] = Unassigned()
            # inference_execution_config: Optional[InferenceExecutionConfig] = Unassigned()
            # execution_role_arn: Optional[StrPipeVar] = Unassigned()
            # vpc_config: Optional[VpcConfig] = Unassigned()
            # creation_time: Optional[datetime.datetime] = Unassigned()
            # model_arn: Optional[StrPipeVar] = Unassigned()
            # enable_network_isolation: Optional[bool] = Unassigned()
            # deployment_recommendation: Optional[DeploymentRecommendation] = Unassigned()

        
        _model_name=self.model_name_param.default_value
        _role=self.role_param.default_value
        _model_package_arn=self.model_package_arn_param.default_value

        print(_model_name)
        print(_role)
        print(_model_package_arn)


        model = Model.create(
            model_name=self.model_name_param.default_value,
            execution_role_arn=self.role_param.default_value,
            containers=[
                ContainerDefinition(model_package_name=self.model_package_group_name_param)
            ]
        )

        print(model)

        create_model_step = ModelStep(
            name='CreateModelStep',
            step_args=model,
            sagemaker_session=self.sagemaker_session,
            region='us-east-1'
        )
        
        # make create model step
        create_model_from_registry = Lambda(
            function_name='GetModelFromRegistry',
            execution_role_arn=self.role_param.default_value,
            script='scripts/create_model_from_registry.py',  # path to your file
            handler='create_model_from_registry.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )
        create_model_step = LambdaStep(
            name='CreateModelStep',
            lambda_func=create_model_from_registry,
            inputs={
                'model_package_group_name': self.model_package_group_name_param,
                'model_package_version': self.model_package_version_param,
                'role': self.role_param
            },
            outputs=[
                LambdaOutput(
                    output_name='model_name',
                    output_type=LambdaOutputTypeEnum.String
                ),
                LambdaOutput(
                    output_name='model_package_arn',
                    output_type=LambdaOutputTypeEnum.String
                )
            ]
        )
        # create_model_step=None
        return create_model_step 


    def get_deploy_endpoint_step(self, depends_on=[]):
            
        deploy_endpoint_lambda = Lambda(
            function_name='DeployEndpoint',
            execution_role_arn=self.role_param.default_value,
            script='scripts/deploy_endpoint.py',
            handler='deploy_endpoint.handler',
            timeout=600,  # endpoints take time to deploy
        )

        deploy_endpoint_step = LambdaStep(
            name='DeployEndpointStep',
            lambda_func=deploy_endpoint_lambda,
            inputs={
                'model_name': self.model_name_param,
                'endpoint_name': f'{self.model_package_group_name_param.default_value}-{self.model_package_version_param.default_value}-abalone-endpoint',
                'instance_type': self.endpoint_instance_type,
                'data_capture_path':self.data_capture_dir
            },
            outputs=[
                LambdaOutput(
                    output_name='endpoint_name',
                    output_type=LambdaOutputTypeEnum.String
                )
            ],
            depends_on=depends_on
        )
        return deploy_endpoint_step  

    def get_batch_transform_step(self, input_data, depends_on=[]):

        transformer = Transformer(
            model_name=self.model_name,
            instance_count=1,
            instance_type=self.monitor_instance_type,
            output_path=f'{self.data_capture_dir}/transformations',
            accept='text/csv',
            assemble_with='Line'
        )

        # transform_step = TransformStep(
        #     name="TransformStep",
        #     step_args=transformer.transform(
        #         data='s3://omm-test-bucket/models/abalone/data/input/test/test_X.csv',
        #         content_type='text/csv',
        #         split_type='Line'
        #     )
        # )
        transform_step = TransformStep(
            name="TransformStep",
            transformer=transformer,
            inputs=TransformInput(
                data=input_data,
                content_type='text/csv',
                split_type='Line'
            ),
            depends_on=depends_on,
            sagemaker_session=self.sagemaker_session
        )

        return transform_step

    # def get_ssm_step(self, scope, writes={}, depends_on=[]):
            
    #     read_write_lambda = Lambda(
    #         function_name='SSMReadWrite',
    #         execution_role_arn=self.role_param.default_value,
    #         script='scripts/ssm_read_write.py',
    #         handler='ssm_read_write.handler',
    #         timeout=30,  # endpoints take time to deploy
    #     )

    #     ssm_step = LambdaStep(
    #         name='SSMReadWriteStep',
    #         lambda_func=read_write_lambda,
    #         inputs={
    #             'writes':writes,
    #             'scope': scope
    #         },
    #         outputs=[
    #             LambdaOutput(
    #                 output_name='params',
    #                 output_type=LambdaOutputTypeEnum.String
    #             )
    #         ],
    #         depends_on=depends_on
    #     )
    #     return ssm_step    
    

    

    def get_inference_steps(self, deployment_type):

        inference_step=None
        if deployment_type == 'realtime':
            inference_step = self.get_realtime_inference_step()
        elif deployment_type == 'batch':
            inference_step = self.run_batch_inference_step()
        else:
            pass

        return [inference_step]


    def get_realtime_inference_step(self):
        pass

    def run_batch_inference_pipeline(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action',                   type=str, choices=['deploy', 'inference'], required=True)
    parser.add_argument('--deployment-type',          type=str, choices=['realtime',   'batch'], required=True)
    parser.add_argument('--model-package-group-name', type=str,                                  required=True)
    parser.add_argument('--model-package-version',    type=str,                                  required=True)
    parser.add_argument('--pipe-name',                type=str,                                  required=True)
    parser.add_argument('--target-name',              type=str,                                  required=True) # rings
    parser.add_argument('--prediction-name',          type=str,                                  required=True) # rings_prediction
    parser.add_argument('--project-bucket',           type=str,                                  required=True) # omm-test-bucket
    parser.add_argument('--project-path',             type=str,                                  required=True) # models/abalone'
    parser.add_argument('--monitor-instance-type',    type=str,            default='ml.m5.large') # ml.m5.large'
    parser.add_argument('--wait',                     action='store_true', default=False) # False
    args = parser.parse_args()

    print('INIT')
    sagemaker_session = PipelineSession(boto_session=boto3.Session(region_name='us-east-1'))

    iam_role = get_execution_role(sagemaker_session)
    role_param = ParameterString(name='Role', default_value=iam_role)

    print(f"args.model_package_group_name {args.model_package_group_name}")
    model_package_group_name_param = ParameterString(name='ModelPackageGroupName', default_value=args.model_package_group_name)
    model_package_version_param = ParameterString(name='ModelPackageVersion', default_value=args.model_package_version)
    
    model_name, model_package_arn = utils.create_model_object_from_registry(sagemaker_session.boto_session, model_package_group_name_param.default_value, role_param.default_value, model_package_version=model_package_version_param.default_value)
    model_name_param = ParameterString(name='ModelName', default_value=model_name) 
    model_package_arn_param = ParameterString(name='ModelPackageArn', default_value=model_package_arn) 


    pipeline = CPipeline(
        sagemaker_session, 
        args.pipe_name,
        model_name_param, # param
        model_package_arn_param, # param
        model_package_group_name_param,  # param
        model_package_version_param,   # param
        args.target_name, 
        args.prediction_name, 
        args.project_bucket, 
        args.project_path, 
        role_param, 
        args.action, 
        args.deployment_type, 
        monitor_instance_type=args.monitor_instance_type
    )

    print(f"role_param.default_value {role_param.default_value}")

    pipeline_definition = json.loads(pipeline.definition())
    print(json.dumps(pipeline_definition, indent=2))
    pipeline.upsert(role_arn=role_param.default_value)
    
    execution = pipeline.start(
        # Override
        parameters={
            'ModelPackageGroupName': model_package_group_name_param.default_value,
            'ModelPackageVersion': model_package_version_param.default_value,
            'ModelPackageArn': model_package_arn_param.default_value,
            'ModelName': model_name_param.default_value,
            'Role': role_param.default_value
        }
    )
    
    print(f"Execution started: {execution.arn}")
    
    if args.wait:
        execution.wait()
        print("Execution complete")

def test_main():
    sm_client = boto3.client('sagemaker', region_name='us-east-1')

    # List all pipelines
    response = sm_client.list_pipelines()
    for pipeline in response['PipelineSummaries']:
        print(pipeline['PipelineName'])
    
    # delet monitors and endpoint
    import boto3
    sm_client = boto3.client('sagemaker', region_name='us-east-1')
    endpoint_name='abalone-endpoint'
    # List and delete monitoring schedules
    schedules = sm_client.list_monitoring_schedules(
        EndpointName=endpoint_name
    )

    for schedule in schedules['MonitoringScheduleSummaries']:
        sm_client.delete_monitoring_schedule(
            MonitoringScheduleName=schedule['MonitoringScheduleName']
        )
        print(f"Deleted schedule: {schedule['MonitoringScheduleName']}")

    # Then delete endpoint
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print("Endpoint deleted")


    exit()

# python3 pipeline/pipeline.py --action deploy --deployment-type realtime --model-package-group-name abalone --model-package-version 1 --pipe-name sagemaker-pipe-template --target-name rings --prediction-name rings_prediction --project-bucket omm-test-bucket --project-path "models/abalone" --monitor-instance-type "ml.m5.large"

# python3 pipeline/pipeline.py --action deploy --deployment-type batch --model-package-group-name abalone --model-package-version 1 --pipe-name sagemaker-pipe-template --target-name rings --prediction-name rings_prediction --project-bucket omm-test-bucket --project-path "models/abalone" --monitor-instance-type "ml.m5.large"

if __name__ == '__main__':
    main()





