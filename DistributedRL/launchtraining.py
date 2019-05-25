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

batch_update_frequency = 25000
max_epoch_runtime_sec = 30
per_iter_epsilon_reduction=0.000001
min_epsilon = 0.1
batch_size = 32
replay_memory_size = 50000
weights_path = ''
train_conv_layers = 'false'

batch_credentials = batchauth.SharedKeyCredentials(NOTEBOOK_CONFIG['batch_account_name'], NOTEBOOK_CONFIG['batch_account_key'])
batch_client = batch.BatchServiceClient(batch_credentials, batch_url=NOTEBOOK_CONFIG['batch_account_url'])
job_id = 'distributed_rl_{0}'.format(str(uuid.uuid4()))

job = batch.models.JobAddParameter(id=job_id, pool_info=batch.models.PoolInformation(pool_id=NOTEBOOK_CONFIG['batch_pool_name']))
batch_client.job.add(job)

tasks = []
# Trainer task
tasks.append(batchmodels.TaskAddParameter(
        id='TrainerTask',
        command_line=r'call C:\\prereq\\mount.bat && C:\\ProgramData\\Anaconda3\\Scripts\\activate.bat py36 && python -u Z:\\scripts_downpour\\manage.py runserver 0.0.0.0:80 data_dir=Z:\\\\ role=trainer experiment_name={0} batch_update_frequency={1} weights_path={2} train_conv_layers={3} per_iter_epsilon_reduction={4} min_epsilon={5}'.format(job_id, batch_update_frequency, weights_path, train_conv_layers, per_iter_epsilon_reduction, min_epsilon),
        display_name='Trainer',
        user_identity=batchmodels.UserIdentity(user_name=NOTEBOOK_CONFIG['batch_job_user_name']),
        multi_instance_settings = batchmodels.MultiInstanceSettings(number_of_instances=1, coordination_command_line='cls')
    ))

# Agent tasks
agent_cmd_line = r'call C:\\prereq\\mount.bat && C:\\ProgramData\\Anaconda3\\Scripts\\activate.bat py36 && python -u Z:\\scripts_downpour\\app\\distributed_agent.py data_dir=Z: role=agent max_epoch_runtime_sec={0} per_iter_epsilon_reduction={1:f} min_epsilon={2:f} batch_size={3} replay_memory_size={4} experiment_name={5} weights_path={6} train_conv_layers={7}'.format(max_epoch_runtime_sec, per_iter_epsilon_reduction, min_epsilon, batch_size, replay_memory_size, job_id, weights_path, train_conv_layers) 
for i in range(0, NOTEBOOK_CONFIG['batch_pool_size'] - 1, 1):
    tasks.append(batchmodels.TaskAddParameter(
            id='AgentTask_{0}'.format(i),
            command_line = agent_cmd_line,
            display_name='Agent_{0}'.format(i),
            user_identity=batchmodels.UserIdentity(user_name=NOTEBOOK_CONFIG['batch_job_user_name']),
            multi_instance_settings=batchmodels.MultiInstanceSettings(number_of_instances=1, coordination_command_line='cls')
        ))
    
batch_client.task.add_collection(job_id, tasks)
print('')