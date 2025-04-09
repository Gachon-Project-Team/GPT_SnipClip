def load_config(config_file='/ssd2/home/swpark/swpark/GPT_SnipClip/.env'):
    config = {}
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

def get_config():
  return load_config().get('config', '')

def get_OPENAI_API_KEY():
  return load_config().get('OPENAI_API_KEY', '')

def get_SSH_HOST():
  return load_config().get('SSH_HOST', '')

def get_SSH_PORT():
  return load_config().get('SSH_PORT', '')

def get_SSH_USERNAME():
  return load_config().get('SSH_USERNAME', '')

def get_SSH_PASSWORD():
  return load_config().get('SSH_PASSWORD', '')

def get_CONDA_ENV_NAME():
  return load_config().get('CONDA_ENV_NAME', '')

def get_GENERATE_SCRIPT():
  return load_config().get('GENERATE_SCRIPT', '')

def get_OUTPUT_DIR():
  return load_config().get('OUTPUT_DIR', '')

def get_LOCAL_SAVE_DIR():
  return load_config().get('LOCAL_SAVE_DIR', '')

def get_HOST():
    return load_config().get('HOST', '')

def get_naver_key():
  client_id = load_config().get('NAVER_CLIENT_ID', '')
  client_secret = load_config().get('NAVER_CLIENT_SECRET', '')
  return client_id, client_secret

def get_gpt_key():
  return load_config().get('OPENAI_API_KEY', '')


def get_naver_key():
  client_id="oIPVUEZLM7U1n_IoWVv6"
  client_secret="a5lh4ogQIl"
  return client_id, client_secret

def get_gpt_key():
  #api_key= "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" # bye
  api_key = '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' # swp
  return api_key





