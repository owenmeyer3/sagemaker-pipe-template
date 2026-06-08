from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
import utils, baseline, boto3, sagemaker, argparse

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
        self.model_image_uri=sagemaker.image_uris.retrieve('xgboost', 'us-east-1', version='1.5-1')

        # model_package_group_name = ParameterString(name='ModelPackageGroupName')
        # model_package_version = ParameterString(name='ModelPackageVersion', default_value='latest')
        # role = ParameterString(name='Role')

        project_dir=       f's3://{project_bucket.default_value}/{project_path.default_value}'
        data_dir=          f"{project_dir}/data"
        self.model_dir=         f"{project_dir}/model"
        self.data_capture_dir=  f'{data_dir}/capture'
        self.ground_truth_dir=  f'{data_dir}/ground-truth'
        self.train_file=        f'{data_dir}/input/train/train.csv'
        self.baseline_file=     f'{data_dir}/baseline/baseline.csv'
        self.baseline_pred_file=f'{data_dir}/baseline/baseline_pred.csv'

        self.baseliner = baseline.Baseliner(self.model_name, data_dir, self.baseline_file, self.train_file, monitor_instance_type)

        self.steps = []
        self.parameters = []
        if action == 'deploy':
            self.steps = self.get_deploy_steps(        
                data_dir, 
                target_name, 
                prediction_name,
                role,
                monitor_instance_type,
                self.model_name,
                project_bucket,
                project_path
            )
            self.parameters=[]

        elif action == 'inference':
            self.steps = self.get_inference_steps(        
                sagemaker_session, 
                deployment_type,
                model_package_group_name, 
                model_package_version, 
                data_dir, 
                target_name, 
                prediction_name,
                role,
                monitor_instance_type,
                self.model_name,
                project_bucket,
                project_path
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
            steps=self.steps,
            sagemaker_session=self.sagemaker_session
            )


    def get_deploy_steps(self, deployment_type):
        # baseline data prep
        self.baseline.make_baseline_sets(self.target_name, self.prediction_name, target_type=float)

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
            create_model_step = self.get_batch_create_step(sagemaker_session)


            

            return [create_model_step]
        else:
            return []


    def get_realtime_create_step(self):

        # make create model step
        create_model_from_registry = sagemaker.lambda_helper.Lambda(
            function_name='CreateModelFromRegistry',
            execution_role_arn=self.role,
            script='scripts/create_model_from_registry.py',  # path to your file
            handler='create_model_from_registry.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )
        create_model_step = sagemaker.workflow.lambda_step.LambdaStep(
            name='CreateModelStep',
            lambda_func=create_model_from_registry,
            inputs={
                'model_package_group_name': self.model_package_group_name,
                'model_package_version': self.model_package_version,
                'role': self.role
            },
            outputs=[
                sagemaker.workflow.lambda_step.LambdaOutput(
                    output_name='model_name',
                    output_type=sagemaker.workflow.lambda_step.LambdaOutputTypeEnum.String
                ),
                sagemaker.workflow.lambda_step.LambdaOutput(
                    output_name='model_package_arn',
                    output_type=sagemaker.workflow.lambda_step.LambdaOutputTypeEnum.String
                )
            ]
        )
        return create_model_step


    def get_batch_create_step(self, sagemaker_session):

        model = sagemaker.model.Model(
            image_uri=self.model_image_uri,
            model_data=self.model_package_arn,  # from registry
            role=self.role,
            sagemaker_session=sagemaker_session
        )

        create_model_step = sagemaker.workflow.model_stepModelStep(
            name='CreateModelStep',
            step_args=model.create(
                instance_type=instance_type.default_value
            )
        )
        
        # make create model step
        create_model_from_registry = sagemaker.lambda_helper.Lambda(
            function_name='CreateModelFromRegistry',
            execution_role_arn=self.role,
            script='scripts/create_model_from_registry.py',  # path to your file
            handler='create_model_from_registry.handler',    # filename.function_name
            timeout=60,
            memory_size=128
        )
        create_model_step = sagemaker.workflow.lambda_step.LambdaStep(
            name='CreateModelStep',
            lambda_func=create_model_from_registry,
            inputs={
                'model_package_group_name': self.model_package_group_name,
                'model_package_version': self.model_package_version,
                'role': self.role
            },
            outputs=[
                sagemaker.workflow.lambda_step.LambdaOutput(
                    output_name='model_name',
                    output_type=sagemaker.workflow.lambda_step.LambdaOutputTypeEnum.String
                ),
                sagemaker.workflow.lambda_step.LambdaOutput(
                    output_name='model_package_arn',
                    output_type=sagemaker.workflow.lambda_step.LambdaOutputTypeEnum.String
                )
            ]
        )
        create_model_step=None
        return create_model_step 


    def get_deploy_endpoint_step(self, depends_on=[]):
            
        deploy_endpoint_lambda = sagemaker.lambda_helper.Lambda(
            function_name='DeployEndpoint',
            execution_role_arn=role,
            script='scripts/deploy_endpoint.py',
            handler='deploy_endpoint.handler',
            timeout=600,  # endpoints take time to deploy
            depends_on=depends_on
        )

        deploy_endpoint_step = sagemaker.workflow.LambdaStep(
            name='DeployEndpointStep',
            lambda_func=deploy_endpoint_lambda,
            inputs={
                'model_name': self.model_name,
                'endpoint_name': f'{self.model_package_group_name}-{self.model_package_version}-abalone-endpoint',
                'instance_type': self.endpoint_instance_type,
                'data_capture_path':self.data_capture_dir
            },
            outputs=[
                sagemaker.workflow.lambda_step.LambdaOutput(
                    output_name='endpoint_name',
                    output_type=sagemaker.workflow.lambda_step.LambdaOutputTypeEnum.String
                )
            ],
            depends_on=[depends_on]
        )
        return deploy_endpoint_step  


    def get_ssm_step(self, scope, writes={}, depends_on=[]):
            
        read_write_lambda = sagemaker.lambda_helper.Lambda(
            function_name='SSMReadWrite',
            execution_role_arn=role,
            script='scripts/ssm_read_write.py',
            handler='ssm_read_write.handler',
            timeout=30,  # endpoints take time to deploy
            depends_on=depends_on
        )

        ssm_step = sagemaker.workflow.LambdaStep(
            name='SSMReadWriteStep',
            lambda_func=read_write_lambda,
            inputs={
                'writes':writes,
                'scope': scope
            },
            outputs=[
                sagemaker.workflow.lambda_step.LambdaOutput(
                    output_name='params',
                    output_type=sagemaker.workflow.lambda_step.LambdaOutputTypeEnum.String
                )
            ],
            depends_on=[depends_on]
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


if __name__ == '__main__':
    import argparse
    import boto3
    import sagemaker

    parser = argparse.ArgumentParser()
    parser.add_argument('--action',                   type=str, choices=['deploy', 'inference'], required=True)
    parser.add_argument('--deployment-type',          type=str, choices=['realtime',   'batch'], required=True)
    parser.add_argument('--model-package-group-name', type=str, choices=['realtime',   'batch'], required=True)
    parser.add_argument('--model-package-version',    type=str, choices=['realtime',   'batch'], required=True)
    parser.add_argument('--pipe-name',                type=str,                                  required=True)
    parser.add_argument('--target-name',              type=str,                                  required=True) # rings
    parser.add_argument('--prediction-name',          type=str,                                  required=True) # rings_prediction
    parser.add_argument('--project-bucket',           type=str,                                  required=True) # omm-test-bucket
    parser.add_argument('--project-path',             type=str,                                  required=True) # models/abalone'
    parser.add_argument('--monitor-instance-type',    type=str,            default='ml.m5.large') # ml.m5.large'
    parser.add_argument('--wait',                     action='store_true', default=False) # False
    args = parser.parse_args()

    boto_session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    role = sagemaker.get_execution_role(sagemaker_session)

    model_package_group_name = ParameterString(name='ModelPackageGroupName')
    model_package_version = ParameterString(name='ModelPackageVersion', default_value='latest')
    role = ParameterString(name='Role')

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

    pipeline.upsert(role_arn=role)
    
    execution = pipeline.start(
        # Override
        parameters={
            'ModelPackageGroupName': pipeline.model_package_group_name,
            'ModelPackageVersion': pipeline.model_package_version,
            'Role': pipeline.role
        }
    )
    
    print(f"Execution started: {execution.arn}")
    
    if args.wait:
        execution.wait()
        print("Execution complete")