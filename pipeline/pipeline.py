from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.image_uris import retrieve as retrieve_image
from sagemaker.core.resources import Model
from sagemaker.core.lambda_helper import Lambda
from sagemaker.core.transformer import Transformer
from sagemaker.core.inputs import TransformInput
from sagemaker.core.shapes.shapes import ScheduleConfig, ContainerDefinition
from sagemaker.core.workflow.conditions import ConditionEquals
from sagemaker.core.workflow.functions import Join

from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.steps import TransformStep
from sagemaker.mlops.workflow.lambda_step import LambdaOutputTypeEnum, LambdaStep, LambdaOutput
from sagemaker.mlops.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.mlops.workflow.condition_step import ConditionStep
import utils, baseline, boto3, argparse, monitor, json, paths, datetime, time
import pandas as pd

class CPipeline(Pipeline):
    def __init__(self, sagemaker_session, build_role_arn, name, deployment_type, target_name, target_type, prediction_name):

        self.sagemaker_session = sagemaker_session
        self.build_role_arn=build_role_arn
        self.name = name
        self.deployment_type=deployment_type
        self.target_name=target_name
        self.target_type = target_type
        self.prediction_name_param=prediction_name
        self.model_package_group_name_param = ParameterString(name='ModelPackageGroupName', default_value='abalone')
        self.model_package_version_param = ParameterInteger(name='ModelPackageVersion', default_value=1)
        self.runtime_role_param = ParameterString(name='RuntimeRole', default_value='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1')
        self.action_param = ParameterString(name='Action', default_value='deploy', enum_values=['deploy', 'inference'])
        self.project_bucket_param = ParameterString(name='ProjectBucket', default_value='abalone')
        self.project_path_param = ParameterString(name='ProjectPath', default_value='abalone')
        self.monitor_instance_type_param = ParameterString(name='MonitorInstanceType', default_value='ml.m5.large')
        self.endpoint_instance_type_param = ParameterString(name='EndpointInstanceType', default_value='ml.m5.large')
        self.transform_instance_type_param = ParameterString(name='EndpointInstanceType', default_value='ml.m5.large')
        self.training_bucket_param = ParameterString(name='TrainingBucket', default_value='omm-test-bucket')
        self.training_dir_param = ParameterString(name='TrainingDir', default_value='projects/abalone/models/abalone')
        self.pipeline_bucket_param = ParameterString(name='PipelineBucket', default_value='omm-test-bucket')
        self.p_params = paths.PathParams(self.training_bucket_param, self.training_dir_param, self.pipeline_bucket_param, self.name)

        

        # GET / CREATE MODEL
        get_or_create_model_from_registry_step = LambdaStep(
            name='GetModelStep',
            lambda_func=Lambda(
                function_name='GetOrCreateModelFromRegistry',
                execution_role_arn=build_role_arn,
                script='scripts/get_or_create_model_from_registry.py',  # path to your file
                handler='get_or_create_model_from_registry.handler',    # filename.function_name
                timeout=60,
                memory_size=128
            ),
            inputs={
                'model_package_group_name': self.model_package_group_name_param,
                'model_package_version': self.model_package_version_param,
                'role': self.runtime_role_param
            },
            outputs=[
                LambdaOutput(output_name='model_name', output_type=LambdaOutputTypeEnum.String),
                LambdaOutput(output_name='model_package_arn', output_type=LambdaOutputTypeEnum.String)
            ],
            depends_on=[]
        )
        self.model_name_param = get_or_create_model_from_registry_step.properties.Outputs['model_name']
        self.baseliner = baseline.Baseliner(self.model_name_param, self.p_params, self.monitor_instance_type_param, self.sagemaker_session)

        ###########################
        ## Deploy Branch
        deploy_steps=self.get_deploy_steps(depends_on=[get_or_create_model_from_registry_step])

        ###########################
        ## Inference Branch
        inference_steps=self.get_inference_steps(depends_on=[get_or_create_model_from_registry_step])
        
        ###########################
        ## Choice
        is_inference = ConditionEquals(left=self.action_param, right='inference')
        deploy_or_inference_steps = ConditionStep(name='ActionTypeCheck', conditions=[is_inference], if_steps=[deploy_steps], else_steps=[inference_steps], depends_on=[])

        ###########################
        ## Full Pipe
        steps = [get_or_create_model_from_registry_step, deploy_or_inference_steps]
        super().__init__(name=self.name, parameters=self.parameters, steps=steps, sagemaker_session=self.sagemaker_session)

        ###########################


    def get_deploy_steps(self, depends_on=[]):
        # baseline data prep
        make_baseline_sets_step = self.baseliner.get_make_baseline_sets_step(self.role_param, self.target_name_param, self.prediction_name_param, self.build_role_arn, self.target_name_param, depends_on=depends_on)

        ###########################
        ## Realtime Deploy Branch
        realtime_deploy_steps=self.get_realtime_deploy_steps()
        
        ###########################
        ## Batch Deploy Branch
        batch_deploy_steps=self.get_batch_deploy_steps()

        ###########################
        ## Choice
        is_realtime = ConditionEquals(left=self.action_param, right='realtime')
        realtime_or_batch_steps = ConditionStep(name='ActionTypeCheck', conditions=[is_realtime], if_steps=[realtime_deploy_steps], else_steps=[batch_deploy_steps], depends_on=make_baseline_sets_step)

        return [realtime_or_batch_steps]

    def get_inference_steps(self, depends_on=[]):

        ###########################
        ## Realtime inference Branch
        realtime_inference_steps=self.get_realtime_inference_steps()
        
        ###########################
        ## Batch inference Branch
        batch_inference_steps=self.get_batch_inference_steps()

        ###########################
        ## Choice
        is_realtime = ConditionEquals(left=self.action_param, right='realtime')
        realtime_or_batch_steps = ConditionStep(name='ActionTypeCheck', conditions=[is_realtime], if_steps=[realtime_inference_steps], else_steps=[batch_inference_steps])

        return [realtime_or_batch_steps]

    def get_realtime_deploy_steps(self, depends_on=[]):
    
        deploy_endpoint_step=self.get_deploy_endpoint_step(depends_on=depends_on)
        data_quality_step = self.baseliner.get_data_quality_baseline_step(self.role_param.default_value, depends_on=[deploy_endpoint_step])
        model_quality_step = self.baseliner.get_model_quality_baseline_step(self.role_param.default_value, depends_on=[deploy_endpoint_step])
        model_bias_step = self.baseliner.get_model_bias_baseline_step(self.role_param.default_value, depends_on=[deploy_endpoint_step])
        model_explainability_step = self.baseliner.get_model_explainability_baseline_step(self.role_param.default_value, depends_on=[deploy_endpoint_step])

        return [deploy_endpoint_step, data_quality_step, model_quality_step, model_bias_step, model_explainability_step]

    def get_batch_deploy_steps(self, depends_on=[]):

        batch_transform_step=self.get_batch_transform_step(self.p.baseline_X_file, depends_on=depends_on)
        data_quality_step = self.baseliner.get_data_quality_baseline_step(self.role_param.default_value, depends_on=[batch_transform_step])
        model_quality_step = self.baseliner.get_model_quality_baseline_step(self.role_param.default_value, depends_on=[batch_transform_step])
        model_bias_step = self.baseliner.get_model_bias_baseline_step(self.role_param.default_value, depends_on=[batch_transform_step])
        model_explainability_step = self.baseliner.get_model_explainability_baseline_step(self.role_param.default_value, depends_on=[batch_transform_step])

        return [batch_transform_step, data_quality_step, model_quality_step, model_bias_step, model_explainability_step]


    def get_realtime_inference_steps(self, depends_on=[]):
        return []


    def get_batch_inference_steps(self, depends_on=[]):
        return []
    
    def get_deploy_endpoint_step(self, depends_on=[]):
            
        deploy_endpoint_lambda = Lambda(
            function_name='DeployEndpoint',
            execution_role_arn=self.build_role_arn,
            script='scripts/deploy_endpoint.py',
            handler='deploy_endpoint.handler',
            timeout=600,  # endpoints take time to deploy
        )

        deploy_endpoint_step = LambdaStep(
            name='DeployEndpointStep',
            lambda_func=deploy_endpoint_lambda,
            inputs={
                'model_name': self.model_name_param,
                'model_package_group_name_param': self.model_package_group_name_param,
                'model_package_version_param': self.model_package_version_param,
                'instance_type': self.endpoint_instance_type,
                'data_capture_path':self.p.data_capture_dir
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

        transform_step = TransformStep(
            name="TransformStep",
            transformer=Transformer(
                model_name=self.model_name,
                instance_count=1,
                instance_type=self.monitor_instance_type,
                output_path=f'{self.p.data_capture_dir}/transformations',
                accept='text/csv',
                assemble_with='Line'
            ),
            inputs=TransformInput(data=input_data, content_type='text/csv', split_type='Line'),
            depends_on=depends_on,
            sagemaker_session=self.sagemaker_session
        )

        return transform_step








        # Deploy batch & inference
        make_baseline_sets_step = self.baseliner.get_make_baseline_sets_step(self.role_param, self.target_name, self.prediction_name, target_type=float)
        get_model_step = self.get_model_from_registry_step()
        pre_deploy_steps = [make_baseline_sets_step, get_model_step]




        deploy_endpoint_step=self.get_deploy_endpoint_step(depends_on=[get_model_step])








        is_inference = ConditionEquals(left=deploy_type_param, right='inference')
        deploy_or_inference_step = ConditionStep(name='ActionTypeCheck', conditions=[is_inference], if_steps=[make_baseline_sets_step], else_steps=[inferencestep], depends_on=[])





        

        if action == 'deploy':
            self.steps = self.get_deploy_steps(deployment_type=deployment_type)
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
            self.steps = self.get_inference_steps(deployment_type=deployment_type)
            self.parameters=[
                self.model_package_group_name_param,
                self.model_package_version_param,
                self.project_bucket,
                self.project_path,
                self.role_param
            ]
        else:
            pass

        steps = [get_or_create_model_from_registry_step, realtime_or_batch_step, deploy_or_inference_step]
        super().__init__(name=self.name, parameters=self.parameters, steps=steps, sagemaker_session=self.sagemaker_session)














    def get_deploy_steps(self, deployment_type):
        # baseline data prep
        make_baseline_sets_step = self.baseliner.get_make_baseline_sets_step(self.role_param, self.target_name, self.prediction_name, target_type=float, depends_on=[])

        get_model_step = self.get_model_from_registry_step()

        if deployment_type == 'realtime':
            deploy_endpoint_step=self.get_deploy_endpoint_step(depends_on=[get_model_step])

            data_quality_step = self.baseliner.get_data_quality_baseline_step(self.role_param.default_value, depends_on=[deploy_endpoint_step])
            model_quality_step = self.baseliner.get_model_quality_baseline_step(self.role_param.default_value, depends_on=[deploy_endpoint_step])
            model_bias_step = self.baseliner.get_model_bias_baseline_step(self.role_param.default_value, depends_on=[deploy_endpoint_step])
            model_explainability_step = self.baseliner.get_model_explainability_baseline_step(self.role_param.default_value, depends_on=[deploy_endpoint_step])

            # ssm_step = self.get_ssm_step(self.name, writes={}, depends_on=[])

            return [make_baseline_sets_step, get_model_step, deploy_endpoint_step, data_quality_step, model_quality_step, model_bias_step, model_explainability_step]#, ssm_step]
        
        elif deployment_type == 'batch':

            batch_transform_step=self.get_batch_transform_step(self.p.baseline_X_file, depends_on=[get_model_step])

            job_definition=self.baseline.get_job_definition(self.sagemaker_session, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None)

            schedule_config = ScheduleConfig(
                schedule_expression='cron(0 * ? * * *)', 
                data_analysis_start_time="-PT1H", 
                data_analysis_end_time="-PT2H"
            )

            model_quality_step=self.get_batch_model_quality_step(self.sagemaker_session, batch_transform_step, schedule_config, depends_on=[])


            return [make_baseline_sets_step, get_model_step]
        else:
            return []


    def get_model_from_registry_step(self, depends_on=[]):

        # make create model step
        lambda_function = Lambda(
            function_name='GetModelFromRegistry',
            execution_role_arn=self.build_role_arn,
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
            ],
            depends_on=depends_on
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
        get_or_create_model_from_registry = Lambda(
            function_name='GetModelFromRegistry',
            execution_role_arn=self.role_param.default_value,
            script='scripts/get_or_create_model_from_registry.py',  # path to your file
            handler='get_or_create_model_from_registry.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )
        create_model_step = LambdaStep(
            name='CreateModelStep',
            lambda_func=get_or_create_model_from_registry,
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
                'data_capture_path':self.p.data_capture_dir
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
            output_path=f'{self.p.data_capture_dir}/transformations',
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


def create_pipe(name, deployment_type, target_name, target_type, prediction_name, region='us-east-1', build_role_arn='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1'):

    print('INIT')
    sagemaker_session = PipelineSession(boto_session=boto3.Session(region_name='us-east-1'))

    pipeline = CPipeline(
        sagemaker_session, 
        build_role_arn=build_role_arn,
        name=name,
        deployment_type=deployment_type,
        target_name=target_name,
        target_type=target_type,
        prediction_name=prediction_name
    )

    pipeline_definition = json.loads(pipeline.definition())
    print(json.dumps(pipeline_definition, indent=2))
    pipeline.upsert(role_arn=build_role_arn)
    


def run(pipeline_name, wait=False, pipeline_parameters={}):

    # pipeline_parameters=[
    #     {'Name': 'InputData', 'Value': 's3://omm-test-bucket/data/test.csv'},
    #     {'Name': 'InstanceType', 'Value': 'ml.m5.large'}
    # ]

    sm_client = boto3.client('sagemaker', region_name='us-east-1')

    ex_response = sm_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=f"{pipeline_name}-execution-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        # Override
        PipelineParameters=pipeline_parameters
    )
    ex_arn=ex_response['PipelineExecutionArn']

    if not wait:
        print(f'started pipe execution {ex_arn}')
        return


    while True:
        ex_details = sm_client.describe_pipeline_execution(PipelineExecutionArn=ex_arn)
        status = ex_details['PipelineExecutionStatus']
        print(f"Status: {status}")
        if status in ['Succeeded', 'Failed', 'Stopped']:
            break
        time.sleep(30)

def test_main():
    sm_client = boto3.client('sagemaker', region_name='us-east-1')

    # List all pipelines
    response = sm_client.list_pipelines()
    for pipeline in response['PipelineSummaries']:
        print(pipeline['PipelineName'])
    
    # delete monitors and endpoint
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

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--build-role',               type=str, default='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1')
    # args = parser.parse_args()
    create_pipe('abalone-pipe', 'realtime', 'rings', 'float', 'rings-prediction', region='us-east-1', build_role_arn='arn:aws:iam::088461143167:role/SageMakerExecutionRole-1')