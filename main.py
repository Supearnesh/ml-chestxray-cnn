import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()

bucket = 's3://chest-xrays'

role = 'arn:aws:iam::XXXXXXXXXXXX:role/service-role/AmazonSageMaker-ExecutionRole'  

estimator = PyTorch(
    entry_point='train.py',
    source_dir='train',
    role=role,
    framework_version='1.4.0',
    py_version='py3',
    instance_count=1,
    instance_type='ml.p2.xlarge',
    hyperparameters={
        'epochs': 6,
        'batch-size': 128
    }
)

estimator.fit({
    'train': bucket+'/train',
    'test': bucket+'/test'
})