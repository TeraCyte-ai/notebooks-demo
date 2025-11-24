import requests
from .config import get_service_ip


def get_experiment_metadata(exp_id: str) -> dict:
    url = f'http://{get_service_ip()}/api/experiment/get_experiment_metadata'
    params = {
        'exp_id': exp_id
    }
    response = requests.get(url, params=params)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in get_experiment_metadata for {exp_id}: {response.status_code} - {error_detail}")
    exp_metadata = response.json()
    return exp_metadata

def get_sample_metadata(serial_number: str, exp_id: str) -> dict:
    url = f'http://{get_service_ip()}/api/sample/get_sample_metadata'
    params = {
        'sample_id': serial_number,
        'exp_id': exp_id
    }
    response = requests.get(url, params=params)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in get_sample_metadata for {serial_number}, {exp_id}: {response.status_code} - {error_detail}")
    sample_metadata = response.json()
    return sample_metadata

def get_user_samples(user_id: str) -> dict:
    url = f'http://{get_service_ip()}/api/user/get_user_samples'
    params = {
        'user_id': user_id
    }
    response = requests.get(url, params=params)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in get_user_samples for {user_id}: {response.status_code} - {error_detail}")
    user_samples = response.json()
    return user_samples

def get_sample_datapath(serial_number: str) -> str:
    url = f'http://{get_service_ip()}/api/sample/get_sample_datapath'
    params = {
        'sample_id': serial_number
    }
    response = requests.get(url, params=params)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in get_sample_datapath for {serial_number}: {response.status_code} - {error_detail}")
    sample_datapath = response.json()
    return sample_datapath

def get_sample_records(serial_number: str) -> dict:
    url = f'http://{get_service_ip()}/api/sample/get_sample_records'
    params = {
        'sample_id': serial_number
    }
    response = requests.get(url, params=params)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in get_sample_records for {serial_number}: {response.status_code} - {error_detail}")
    sample_records = response.json()
    return sample_records

def generate_token(serial_number: str) -> str:
    url = f'http://{get_service_ip()}/api/user/generate_token'
    params = {
        'sample_id': serial_number
    }
    response = requests.get(url, params=params)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in generate_token for {serial_number}: {response.status_code} - {error_detail}")
    sample_token = response.json()
    return sample_token

def get_wells_groups(serial_number: str, assay_id: str) -> dict:
    url = f'http://{get_service_ip()}/api/sample/get_wells_query'
    params = {
        'sn': serial_number,
        'assay_id': assay_id
    }
    response = requests.get(url, params=params)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in get_wells_groups for {serial_number}, {assay_id}: {response.status_code} - {error_detail}")
    wells_groups = response.json()
    return wells_groups

def save_wells_group(serial_number: str, assay_id: str, group_name: str, group_description: str, global_indexes: list) -> dict:
    url = f'http://{get_service_ip()}/api/sample/save_group'
    payload = {
        'sn': serial_number,
        'assay_id': assay_id,
        'query_data': {
            'query_name': group_name,
            'query_description': group_description,
            'global_indexes': global_indexes
        }
    }
    response = requests.post(url, json=payload)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in save_wells_group for {serial_number}, {assay_id}: {response.status_code} - {error_detail}")
    result = response.json()
    return result

def remove_wells_group(serial_number: str, assay_id: str, group_name: str) -> dict:
    url = f'http://{get_service_ip()}/api/sample/remove_group'
    payload = {
        'sn': serial_number,
        'assay_id': assay_id,
        'query_name': group_name
    }
    response = requests.post(url, json=payload)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in remove_wells_group for {serial_number}, {assay_id}: {response.status_code} - {error_detail}")
    result = response.json()
    return result

def get_assay_workflows(assay_id: str) -> dict:
    url = f'http://{get_service_ip()}/workflows/get_assay_workflows'
    params = {
        'assay_id': assay_id
    }
    response = requests.get(url, params=params)
    if not response.status_code == 200:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
        except:
            error_detail = response.text if response.text else 'Unknown error'
        print(f"Error in get_assay_workflows for {assay_id}: {response.status_code} - {error_detail}")
    workflows = response.json()
    return workflows

def get_vizarr_url() -> dict:
    url = f'http://{get_service_ip()}/api/get_vizarr_url'
    try:
        response = requests.get(url)
        if not response.status_code == 200:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except:
                error_detail = response.text if response.text else 'Unknown error'
            print(f"Error in get_vizarr_url: {response.status_code} - {error_detail}")
    
        vizarr_url = response.json()
    except:  # Fallback to default URL if service is unreachable
        vizarr_url = 'https://green-wave-0a25ef503.1.azurestaticapps.net'
    return vizarr_url
