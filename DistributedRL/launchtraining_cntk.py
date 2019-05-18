import os
import sys
import uuid
import json

#Azure batch. To install, run 'pip install cryptography azure-batch azure-storage'
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batchauth
import azure.batch.models as batchmodels

os.chdir(".\\DistributedRL")

with open('notebook_config.json', 'r') as f:
    NOTEBOOK_CONFIG = json.loads(f.read())

batch_credentials = batchauth.SharedKeyCredentials(NOTEBOOK_CONFIG['batch_account_name'], NOTEBOOK_CONFIG['batch_account_key'])
batch_client = batch.BatchServiceClient(batch_credentials, batch_url=NOTEBOOK_CONFIG['batch_account_url'])
job_id = 'distributed_rl_{0}'.format(str(uuid.uuid4()))

job = batch.models.JobAddParameter(id=job_id, pool_info=batch.models.PoolInformation(pool_id=NOTEBOOK_CONFIG['batch_pool_name']))
batch_client.job.add(job)

tasks = []
# Trainer task
tasks.append(batchmodels.TaskAddParameter(
        id='TrainerTask',
        command_line=r'call C:\\prereq\\mount.bat && C:\\ProgramData\\Anaconda3\\Scripts\\activate.bat py36 && python -u Z:\\scripts_downpour\\app\\cntk_agent.py data_dir=Z: experiment_name={0}'.format(job_id),
        display_name='Trainer',
        user_identity=batchmodels.UserIdentity(user_name=NOTEBOOK_CONFIG['batch_job_user_name']),
        multi_instance_settings = batchmodels.MultiInstanceSettings(number_of_instances=1, coordination_command_line='cls')
    ))
    
batch_client.task.add_collection(job_id, tasks)
print('')