import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class EnvConfig:
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists() and load_dotenv:
        load_dotenv(dotenv_path=env_path)

    GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", 44))
#Việc cố định số này đảm bảo tính tái lập (Reproducibility). nếu chạy lại code này nhiều lần, Accuracy, F1 Score sẽ luôn giống hệt nhau, không bị thay đổi do sự ngẫu nhiên.
    TRANSFORMERS_OFFLINE = int(os.environ.get("TRANSFORMERS_OFFLINE", 0))
    HUGGING_LOCAL_MODEL_FILES_PATH = os.environ.get(
        "HUGGING_LOCAL_MODEL_FILES_PATH", "/home/ngochv/Dynamic_Feature/models"
    )


env_config = EnvConfig()
print(f"SEED: {env_config.GLOBAL_SEED}")
print(f"MODEL PATH: {env_config.HUGGING_LOCAL_MODEL_FILES_PATH}")