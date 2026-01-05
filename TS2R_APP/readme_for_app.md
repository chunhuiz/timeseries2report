# 🔋 Battery LLM Assistant

## Installation

```bash
pip install -r requirements.txt
```

## Notice
1. First, `.env`environment file needs to be modified according to your own LLM API.

```bash
BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" # API address
CHAT_API_KEY="your-api" # API Key (OpenAI format)
MODEL_NAME="qwen3-max" # LLM name
TEMPERATURE=0.3 # temperature
MAX_TOKENS=8192 # max token
```

2. Run `splite_single_data.py` to prepare test data (as 'test.pickle') from the original data file 'TJU_raw_data.pickle'

## Run
After configuration is completed, simply run the following command on the `bash` terminal.

```bash
streamlit run app.py
```
