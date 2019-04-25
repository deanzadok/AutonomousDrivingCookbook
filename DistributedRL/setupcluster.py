#Standard python libraries
import json
import os
import re
import datetime
import time

from IPython.display import clear_output

#Azure file storage. To install, run 'pip install azure-storage-file'
from azure.storage.file import FileService
from azure.storage.file import ContentSettings

#Azure blob. To install, run 'pip install azure-storage-blob'
from azure.storage.blob import BlockBlobService
from azure.storage.blob import PublicAccess

#Azure batch. To install, run 'pip install cryptography azure-batch azure-storage'
import azure.storage.blob as azureblob
import azure.batch.models as batchmodels
import azure.batch.batch_auth as batchauth
import azure.batch as batch

os.chdir(".\\DistributedRL")

with open('notebook_config.json', 'r') as f:
    NOTEBOOK_CONFIG = json.loads(f.read())

#Generate mount.bat
with open('Template\\mount_bat.template', 'r') as f:
    mount_bat_cmd = f.read()
    
mount_bat_cmd = mount_bat_cmd\
                    .replace('{storage_account_name}', NOTEBOOK_CONFIG['storage_account_name'])\
                    .replace('{file_share_name}', NOTEBOOK_CONFIG['file_share_name'])\
                    .replace('{storage_account_key}', NOTEBOOK_CONFIG['storage_account_key'])

with open('Blob\\mount.bat', 'w') as f:
    f.write(mount_bat_cmd)
    
#Generate setup_machine.py
with open('Template\\setup_machine_py.template', 'r') as f:
    setup_machine_py = f.read()

setup_machine_py = setup_machine_py\
                    .replace('{storage_account_name}', NOTEBOOK_CONFIG['storage_account_name'])\
                    .replace('{file_share_name}', NOTEBOOK_CONFIG['file_share_name'])\
                    .replace('{storage_account_key}', NOTEBOOK_CONFIG['storage_account_key'])\
                    .replace('{batch_job_user_name}', NOTEBOOK_CONFIG['batch_job_user_name'])\
                    .replace('{batch_job_user_password}', NOTEBOOK_CONFIG['batch_job_user_password'])

with open('Blob\\setup_machine.py', 'w') as f:
    f.write(setup_machine_py)
    
#Generate run_airsim_on_user_login.xml
with open('Template\\run_airsim_on_user_login_xml.template', 'r', encoding='utf-16') as f:
    startup_task_xml = f.read()
    
startup_task_xml = startup_task_xml\
                    .replace('{batch_job_user_name}', NOTEBOOK_CONFIG['batch_job_user_name'])

with open('Share\\scripts_downpour\\run_airsim_on_user_login.xml', 'w', encoding='utf-16') as f:
    f.write(startup_task_xml)

# create file share
file_service = FileService(account_name = NOTEBOOK_CONFIG['storage_account_name'], account_key=NOTEBOOK_CONFIG['storage_account_key'])
file_service.create_share(NOTEBOOK_CONFIG['file_share_name'], fail_on_exist=False)

# upload all files to share
def create_directories(path, file_service):
    split_dir = path.split('\\')
    for i in range(1, len(split_dir)+1, 1):
        combined_dir = '\\'.join(split_dir[:i])
        file_service.create_directory(NOTEBOOK_CONFIG['file_share_name'], combined_dir, fail_on_exist=False)

for root, directories, files in os.walk('Share'):
    for file in files:
        regex_pattern = '{0}[\\\\]?'.format('Share').replace('\\', '\\\\')
        upload_directory = re.sub(regex_pattern, '', root)
        print('Uploading {0} to {1}...'.format(os.path.join(root, file), upload_directory))
        if (len(upload_directory) == 0):
            upload_directory = None
        if (upload_directory != None):
            create_directories(upload_directory, file_service)
        file_service.create_file_from_path(          
            NOTEBOOK_CONFIG['file_share_name'], 
            upload_directory,                   
            file,                               
            os.path.join(root, file)            
            )

block_blob_service = BlockBlobService(account_name = NOTEBOOK_CONFIG['storage_account_name'], account_key = NOTEBOOK_CONFIG['storage_account_key'])
block_blob_service.create_container('prereq', public_access = PublicAccess.Container)

for root, directories, files in os.walk('Blob'):
    for file in files:
        block_blob_service.create_blob_from_path( 
            'prereq',                             
            file,                                 
            os.path.join(root, file)              
            )

# create image
os.system('powershell.exe ".\\CreateImage.ps1 -subscriptionId {0} -storageAccountName {1} -storageAccountKey {2} -resourceGroupName {3}'\
          .format(NOTEBOOK_CONFIG['subscription_id'], NOTEBOOK_CONFIG['storage_account_name'], NOTEBOOK_CONFIG['storage_account_key'], NOTEBOOK_CONFIG['resource_group_name']))


with open('Template\\pool.json.template', 'r') as f:
    pool_config = f.read()
    
pool_config = pool_config\
                .replace('{batch_pool_name}', NOTEBOOK_CONFIG['batch_pool_name'])\
                .replace('{subscription_id}', NOTEBOOK_CONFIG['subscription_id'])\
                .replace('{resource_group_name}', NOTEBOOK_CONFIG['resource_group_name'])\
                .replace('{storage_account_name}', NOTEBOOK_CONFIG['storage_account_name'])\
                .replace('{batch_job_user_name}', NOTEBOOK_CONFIG['batch_job_user_name'])\
                .replace('{batch_job_user_password}', NOTEBOOK_CONFIG['batch_job_user_password'])\
                .replace('{batch_pool_size}', str(NOTEBOOK_CONFIG['batch_pool_size']))

with open('pool.json', 'w') as f:
    f.write(pool_config)
    
create_cmd = 'powershell.exe ".\ProvisionCluster.ps1 -subscriptionId {0} -resourceGroupName {1} -batchAccountName {2}"'\
    .format(NOTEBOOK_CONFIG['subscription_id'], NOTEBOOK_CONFIG['resource_group_name'], NOTEBOOK_CONFIG['batch_account_name'])
    
print('Executing command. Check the terminal output for authentication instructions.')

os.system(create_cmd)