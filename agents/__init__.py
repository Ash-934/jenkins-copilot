import json
from jenkins_client import JenkinsClient

_jenkins: JenkinsClient | None = None


def get_client() -> JenkinsClient:
    global _jenkins
    if _jenkins is None:
        _jenkins = JenkinsClient()
    return _jenkins


def to_json(data) -> str:
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, default=str)
