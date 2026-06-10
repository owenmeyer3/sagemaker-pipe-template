from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.lambda_helper import Lambda
from sagemaker.core.image_uris import retrieve as retrieve_image
from sagemaker.core.resources import Model
from sagemaker.core.lambda_helper import Lambda
from sagemaker.core.transformer import Transformer
from sagemaker.core.inputs import TransformInput

from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.lambda_step import LambdaOutputTypeEnum, LambdaStep, LambdaOutput
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.steps import TransformStep
from sagemaker.mlops.workflow.lambda_step import LambdaOutputTypeEnum, LambdaStep, LambdaOutput
from sagemaker.mlops.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
import utils, baseline, boto3, argparse





class CPipeline(Pipeline):
    def __init__(self, sagemaker_session, name, model_package_group_name, model_package_version, target_name, prediction_name, project_bucket, project_path, role, action, deployment_type, monitor_instance_type='ml.m5.large', endpoint_instance_type='ml.m5.large'):

        self.target_name=target_name
        self.prediction_name=prediction_name

        self.model_name, self.model_package_arn = utils.create_model_object_from_registry(sagemaker_session.boto_session, model_package_group_name, role, model_package_version=model_package_version)
        self.sagemaker_session=sagemaker_session
        self.name=name
        self.model_package_group_name=model_package_group_name
        self.model_package_version=model_package_version
        self.target_name=target_name
        self.prediction_name=prediction_name
        self.project_bucket=project_bucket
        self.project_path=project_path
        self.role=role
        self.monitor_instance_type=monitor_instance_type
        self.endpoint_instance_type=endpoint_instance_type
        self.model_image_uri=retrieve_image('xgboost', 'us-east-1', version='1.5-1')

        # model_package_group_name = ParameterString(name='ModelPackageGroupName')
        # model_package_version = ParameterString(name='ModelPackageVersion', default_value='latest')
        # role = ParameterString(name='Role')

        project_dir=       f's3://{project_bucket}/{project_path}'
        data_dir=          f"{project_dir}/data"
        self.model_dir=         f"{project_dir}/model"
        self.data_capture_dir=  f'{data_dir}/capture'
        self.transforms_dir=    f'{data_dir}/transforms'
        self.ground_truth_dir=  f'{data_dir}/ground-truth'
        self.train_file=        f'{data_dir}/input/train/train.csv'
        self.baseline_file=     f'{data_dir}/baseline/baseline.csv'
        self.baseline_pred_file=f'{data_dir}/baseline/baseline_pred.csv'
        self.train_X_file=      f'{data_dir}/monitors/model-explainability/train_X.csv'

        self.baseliner = baseline.Baseliner(self.model_name, data_dir, self.baseline_file, self.train_file, monitor_instance_type, sagemaker_session)

        self.steps = []
        self.parameters = []
        if action == 'deploy':
            self.steps = self.get_deploy_steps(        
                deployment_type=deployment_type
            )
            self.parameters=[]

        elif action == 'inference':
            self.steps = self.get_inference_steps(        
                deployment_type=deployment_type
            )
            self.parameters=[
                self.model_package_group_name,
                self.model_package_version,
                self.project_bucket,
                self.project_path,
                self.role
            ]
        else:
            pass

        super.__init(
            name=self.name,
            parameters=self.parameters,
            region='us-east-1',
            steps=self.steps,
            sagemaker_session=self.sagemaker_session
            )


    def get_deploy_steps(self, deployment_type):
        # baseline data prep
        self.baseliner.make_baseline_sets(self.target_name, self.prediction_name, target_type=float)

        if deployment_type == 'realtime':
            create_model_step = self.get_realtime_create_step()

            deploy_endpoint_step=self.get_deploy_endpoint_step(depends_on=[create_model_step])

            data_quality_step = self.baseliner.get_data_quality_step(
                self.sagemaker_session, self.role, depends_on=[deploy_endpoint_step]
            )
            model_quality_step = self.baseliner.get_model_quality_step(
                self.sagemaker_session, self.role, self.target_name, self.prediction_name, depends_on=[deploy_endpoint_step]
            )
            model_bias_step = self.baseliner.get_model_bias_step(
                self.sagemaker_session, self.role, self.target_name, self.prediction_name, depends_on=[deploy_endpoint_step]
            )
            model_explainability_step = self.baseliner.get_model_explainability_step(
                self.sagemaker_session, self.role, depends_on=[deploy_endpoint_step]
            )

            ssm_step = self.get_ssm_step(self.name, writes={}, depends_on=[])

            return [create_model_step, deploy_endpoint_step, data_quality_step, model_quality_step, model_bias_step, model_explainability_step, ssm_step]
        
        elif deployment_type == 'batch':
            create_model_step = self.get_batch_create_step()

            batch_transform_step=self.get_batch_transform_step(self.train_X_file, depends_on=[create_model_step])

            job_definition=self.baseline.get_job_definition(self.sagemaker_session, probability_attribute=None, probability_threshold_attribute=None, exclude_features_attribute=None)

            schedule_config = ScheduleConfig(
                schedule_expression='cron(0 * ? * * *)', 
                data_analysis_start_time="-PT1H", 
                data_analysis_end_time="-PT2H"
            )

            model_quality_step=self.get_batch_model_quality_step(self.sagemaker_session, batch_transform_step, schedule_config, depends_on=[])


            return [create_model_step]
        else:
            return []


    def get_realtime_create_step(self):

        # make create model step
        create_model_from_registry = Lambda(
            function_name='CreateModelFromRegistry',
            execution_role_arn=self.role,
            script='scripts/create_model_from_registry.py',  # path to your file
            handler='create_model_from_registry.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )
        create_model_step = LambdaStep(
            name='CreateModelStep',
            lambda_func=create_model_from_registry,
            inputs={
                'model_package_group_name': self.model_package_group_name,
                'model_package_version': self.model_package_version,
                'role': self.role
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
        return create_model_step


    def get_batch_create_step(self):

        model = Model(
            image_uri=self.model_image_uri,
            model_data=self.model_package_arn,  # from registry
            role=self.role,
            sagemaker_session=self.sagemaker_session
        )

        create_model_step = ModelStep(
            name='CreateModelStep',
            step_args=model.create(
                instance_type=self.endpoint_instance_type
            ),
            sagemaker_session=self.sagemaker_session
        )
        
        # make create model step
        create_model_from_registry = Lambda(
            function_name='CreateModelFromRegistry',
            execution_role_arn=self.role,
            script='scripts/create_model_from_registry.py',  # path to your file
            handler='create_model_from_registry.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )
        create_model_step = LambdaStep(
            name='CreateModelStep',
            lambda_func=create_model_from_registry,
            inputs={
                'model_package_group_name': self.model_package_group_name,
                'model_package_version': self.model_package_version,
                'role': self.role
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
            ],
            sagemaker_session=self.sagemaker_session
        )
        create_model_step=None
        return create_model_step 


    def get_deploy_endpoint_step(self, role, depends_on=[]):
            
        deploy_endpoint_lambda = Lambda(
            function_name='DeployEndpoint',
            execution_role_arn=role,
            script='scripts/deploy_endpoint.py',
            handler='deploy_endpoint.handler',
            timeout=600,  # endpoints take time to deploy
        )

        deploy_endpoint_step = LambdaStep(
            name='DeployEndpointStep',
            lambda_func=deploy_endpoint_lambda,
            inputs={
                'model_name': self.model_name,
                'endpoint_name': f'{self.model_package_group_name}-{self.model_package_version}-abalone-endpoint',
                'instance_type': self.endpoint_instance_type,
                'data_capture_path':self.data_capture_dir
            },
            outputs=[
                LambdaOutput(
                    output_name='endpoint_name',
                    output_type=LambdaOutputTypeEnum.String
                )
            ],
            depends_on=depends_on,
            sagemaker_session=self.sagemaker_session
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

    def get_ssm_step(self, role, scope, writes={}, depends_on=[]):
            
        read_write_lambda = Lambda(
            function_name='SSMReadWrite',
            execution_role_arn=role,
            script='scripts/ssm_read_write.py',
            handler='ssm_read_write.handler',
            timeout=30,  # endpoints take time to deploy
        )

        ssm_step = LambdaStep(
            name='SSMReadWriteStep',
            lambda_func=read_write_lambda,
            inputs={
                'writes':writes,
                'scope': scope
            },
            outputs=[
                LambdaOutput(
                    output_name='params',
                    output_type=LambdaOutputTypeEnum.String
                )
            ],
            depends_on=depends_on,
            sagemaker_session=self.sagemaker_session
        )
        return ssm_step    
    

    

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

    # sagemaker_session = sagemaker.Session(boto_session=boto3.Session())
    print('INIT')
    sagemaker_session = PipelineSession(boto_session=boto3.Session(region_name='us-east-1'))

    model_package_group_name = ParameterString(name='ModelPackageGroupName')
    model_package_version = ParameterString(name='ModelPackageVersion', default_value='latest')
    role = ParameterString(name='Role', default_value=get_execution_role(sagemaker_session)) 
    print('INIT VARS')

    pipeline = CPipeline(
        sagemaker_session, 
        args.pipe_name, 
        args.model_package_group_name, 
        args.model_package_version, 
        args.target_name, 
        args.prediction_name, 
        args.project_bucket, 
        args.project_path, 
        role, 
        args.action, 
        args.deployment_type, 
        monitor_instance_type=args.monitor_instance_type
    )

    # pipeline.upsert(role_arn=role)
    
    # execution = pipeline.start(
        # Override
    #     parameters={
    #         'ModelPackageGroupName': pipeline.model_package_group_name,
    #         'ModelPackageVersion': pipeline.model_package_version,
    #         'Role': pipeline.role
    #     }
    # )
    
    # print(f"Execution started: {execution.arn}")
    
    # if args.wait:
    #     execution.wait()
    #     print("Execution complete")

def test_main():

    exit()

# python3 pipeline/pipeline.py --action deploy --deployment-type batch --model-package-group-name abalone --model-package-version 1 --pipe-name sagemaker-pipe-template --target-name rings --prediction-name rings_prediction --project-bucket omm-test-bucket --project-path 'models/abalone' --monitor-instance-type ml.m5.large

if __name__ == '__main__':
    main()





