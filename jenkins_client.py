"""Jenkins REST API client - fetches build logs, job configs, plugin lists, etc."""

import httpx
from config import settings


class JenkinsClient:

    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        api_token: str | None = None,
    ):
        self.base_url = (base_url or settings.JENKINS_URL)
        self.auth = (
            username or settings.JENKINS_USER,
            api_token or settings.JENKINS_API_TOKEN,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=self.auth,
            timeout=30.0,
            follow_redirects=True,
        )

    # -- helpers --

    async def _get_json(self, path: str, **params) -> dict:
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _get_text(self, path: str) -> str:
        resp = await self._client.get(path)
        resp.raise_for_status()
        return resp.text

    # -- jobs --

    async def list_jobs(self) -> list[dict]:
        """List all top-level jobs with name, url, color (status)."""
        data = await self._get_json("/api/json", tree="jobs[name,url,color]")
        return data.get("jobs", [])

    async def get_job_info(self, job_name: str) -> dict:
        data = await self._get_json(
            f"/job/{job_name}/api/json",
            tree="name,url,color,buildable,lastBuild[number,result,timestamp,duration],lastSuccessfulBuild[number],lastFailedBuild[number],healthReport[description,score]",
        )
        return data

    async def get_job_config_xml(self, job_name: str) -> str:
        return await self._get_text(f"/job/{job_name}/config.xml")

    # -- builds --

    async def get_build_info(self, job_name: str, build_number: int | str = "lastBuild") -> dict:
        data = await self._get_json(
            f"/job/{job_name}/{build_number}/api/json",
            tree="number,result,timestamp,duration,estimatedDuration,actions[causes[shortDescription]],changeSets[items[msg,author[fullName]]]",
        )
        return data

    async def get_build_log(self, job_name: str, build_number: int | str = "lastBuild") -> str:
        return await self._get_text(f"/job/{job_name}/{build_number}/consoleText")

    async def get_build_log_tail(
        self, job_name: str, build_number: int | str = "lastBuild", lines: int = 80
    ) -> str:
        full_log = await self.get_build_log(job_name, build_number)
        log_lines = full_log.strip().splitlines()
        tail = log_lines[-lines:] if len(log_lines) > lines else log_lines
        return "\n".join(tail)

    async def get_pipeline_stages(
        self, job_name: str, build_number: int | str = "lastBuild"
    ) -> dict:
        # TODO: works only for pipeline jobs, handle error
        return await self._get_json(f"/job/{job_name}/{build_number}/wfapi/describe")

    async def get_recent_builds(self, job_name: str, count: int = 10) -> list[dict]:
        data = await self._get_json(
            f"/job/{job_name}/api/json",
            tree=f"builds[number,result,timestamp,duration]{{0,{count}}}",
        )
        return data.get("builds", [])

    # -- plugins --

    async def list_plugins(self) -> list[dict]:
        data = await self._get_json(
            "/pluginManager/api/json",
            depth="1",
            tree="plugins[shortName,longName,version,active,hasUpdate]",
        )
        return data.get("plugins", [])

    # -- queue & agents --

    async def get_queue(self) -> list[dict]:
        data = await self._get_json(
            "/queue/api/json", tree="items[id,task[name],why,inQueueSince]"
        )
        return data.get("items", [])

    async def get_agents(self) -> list[dict]:
        data = await self._get_json(
            "/computer/api/json",
            tree="computer[displayName,idle,offline,numExecutors,assignedLabels[name]]",
        )
        return data.get("computer", [])

    # -- system info --

    async def get_system_info(self) -> dict:
        resp = await self._client.head("/")
        return {
            "version": resp.headers.get("X-Jenkins", "unknown"),
            "url": self.base_url,
        }

    # -- lifecycle --

    async def close(self):
        await self._client.aclose()

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/json", params={"tree": "mode"})
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
