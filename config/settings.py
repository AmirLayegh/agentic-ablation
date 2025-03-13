import os
import getpass

def set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Initialize environment variables
set_env("OPENAI_API_KEY") 