def load_config(config_file='.env'):
    config = {}
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

def get_config()
  return load_config().get('config', '')

def get_OPENAI_API_KEY()
  return load_config().get('OPENAI_API_KEY', '')

def get_SSH_HOST()
  return load_config().get('SSH_HOST', '')

def get_SSH_PORT()
  return load_config().get('SSH_PORT', '')

def get_SSH_USERNAME()
  return load_config().get('SSH_USERNAME', '')

def get_SSH_PASSWORD()
  return load_config().get('SSH_PASSWORD', '')

def get_CONDA_ENV_NAME()
  return load_config().get('CONDA_ENV_NAME', '')

def get_GENERATE_SCRIPT()
  return load_config().get('GENERATE_SCRIPT', '')

def get_OUTPUT_DIR()
  return load_config().get('OUTPUT_DIR', '')

def get_LOCAL_SAVE_DIR()
  return load_config().get('LOCAL_SAVE_DIR', '')


def get_naver_key():
  client_id = load_config().get('NAVER_CLIENT_ID', '')
  client_secret = load_config().get('NAVER_CLIENT_SECRET', '')
  return client_id, client_secret

def get_gpt_key():
  return load_config().get('OPENAI_API_KEY', '')