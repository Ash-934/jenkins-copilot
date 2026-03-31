## Prerequisites

- **Jenkins Server** - A running Jenkins server (e.g., via Docker `docker compose up -d`). Make sure to get the Jenkins initial admin password and log in.
- **LLM Endpoint** - An OpenAI-compatible API endpoint with a model that supports tool calling. You can use local platforms like [lmstudio.ai](https://lmstudio.ai/) running Qwen 3.5, or a hosted/cloud LLM endpoint.
- **Python 3.11+**

## Quick Start

### 1. Get Jenkins API Token

1. Open your running Jenkins instance (e.g., `http://localhost:8080`) → Log in
2. Click your **username** (top-right corner) → **Configure**
3. Scroll to **API Token** → **Add new Token** → Give it a name → **Generate**
4. Copy the token

### 2. Configure Environment variables

```bash
cp .env.example .env
# Edit .env with your LLM endpoint and Jenkins connection details
```
*Note: Provide `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY` in your `.env` file. All LangChain ChatOpenAI models that support tool calling will work seamlessly.*

### 3. Run the App

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py
# Open http://localhost:8000
```