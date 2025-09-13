from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EventStreamError


@dataclass
class AgentConfig:
    region: str
    agent_name: str
    foundation_model: str
    instruction: str
    role_arn: str
    description: str = "Sample agent created by aws_strand_sdk"
    idle_ttl_seconds: int = 600


class AgentManager:
    """
    Thin wrapper around Amazon Bedrock Agents (control plane + runtime).

    - Control plane client name: 'bedrock-agent'
    - Runtime client name: 'bedrock-agent-runtime'
    """

    def __init__(self, region: str, profile: Optional[str] = None, *, access_key: Optional[str] = None, secret_key: Optional[str] = None):
        session_kwargs: Dict[str, Any] = {}
        if profile:
            session_kwargs["profile_name"] = profile
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
        self._session = boto3.Session(**session_kwargs)
        config = Config(region_name=region, retries={"max_attempts": 10, "mode": "standard"})
        self.agents = self._session.client("bedrock-agent", config=config)
        self.runtime = self._session.client("bedrock-agent-runtime", config=config)
        self.region = region

    def get_caller_identity(self) -> Dict[str, Any]:
        """Return the AWS STS caller identity for the current session.

        Useful for diagnosing which principal is used for API calls.
        """
        sts = self._session.client("sts", region_name=self.region)
        return sts.get_caller_identity()

    # ---------- Control plane ----------
    def list_agents(self, max_results: int = 100) -> list[Dict[str, Any]]:
        agents: list[Dict[str, Any]] = []
        token: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {"maxResults": max_results}
            if token:
                kwargs["nextToken"] = token
            resp = self.agents.list_agents(**kwargs)
            # Some SDKs use 'agents', others 'agentSummaries'
            items = resp.get("agents") or resp.get("agentSummaries") or []
            agents.extend(items)
            token = resp.get("nextToken")
            if not token:
                break
        return agents

    def find_agent_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for a in self.list_agents():
            if a.get("agentName") == name:
                return a
        return None

    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        return self.agents.get_agent(agentId=agent_id)["agent"]
    def create_agent(self, cfg: AgentConfig) -> Dict[str, Any]:
        try:
            resp = self.agents.create_agent(
                agentName=cfg.agent_name,
                instruction=cfg.instruction,
                foundationModel=cfg.foundation_model,
                description=cfg.description,
                idleSessionTTLInSeconds=cfg.idle_ttl_seconds,
                agentResourceRoleArn=cfg.role_arn,
                clientToken=str(uuid.uuid4()),
            )
            return resp["agent"]
        except ClientError as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code")
            if error_code == "ConflictException":
                existing = self.find_agent_by_name(cfg.agent_name)
                if existing and existing.get("agentId"):
                    # Return full details
                    return self.get_agent(existing["agentId"])
            raise

    def prepare_agent(self, agent_id: str) -> Dict[str, Any]:
        resp = self.agents.prepare_agent(
            agentId=agent_id,
        )
        return resp

    def wait_for_status(self, agent_id: str, desired: str = "PREPARED", timeout_s: int = 600, poll_s: int = 10) -> str:
        start = time.time()
        while time.time() - start < timeout_s:
            agent = self.agents.get_agent(agentId=agent_id)["agent"]
            status = agent.get("agentStatus")
            if status == desired:
                return status
            if status in {"FAILED", "DELETING"}:
                raise RuntimeError(f"Agent moved to terminal state: {status}")
            time.sleep(poll_s)
        raise TimeoutError(f"Timed out waiting for agent {agent_id} to reach {desired}")

    def create_alias(self, agent_id: str, alias_name: str = "prod") -> Dict[str, Any]:
        resp = self.agents.create_agent_alias(
            agentId=agent_id,
            agentAliasName=alias_name,
            description=f"Alias {alias_name}",
            clientToken=str(uuid.uuid4()),
        )
        return resp["agentAlias"]

    def list_aliases(self, agent_id: str, max_results: int = 100) -> list[Dict[str, Any]]:
        aliases: list[Dict[str, Any]] = []
        token: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {"agentId": agent_id, "maxResults": max_results}
            if token:
                kwargs["nextToken"] = token
            resp = self.agents.list_agent_aliases(**kwargs)
            items = resp.get("agentAliasSummaries") or resp.get("agentAliases") or []
            aliases.extend(items)
            token = resp.get("nextToken")
            if not token:
                break
        return aliases

    def find_alias_by_name(self, agent_id: str, alias_name: str) -> Optional[Dict[str, Any]]:
        for a in self.list_aliases(agent_id):
            if a.get("agentAliasName") == alias_name:
                return a
        return None

    def update_agent_role(self, agent_id: str, role_arn: str) -> Dict[str, Any]:
        """Update the agent's execution role ARN.

        After updating, re-run prepare to propagate changes to the alias.
        """
        resp = self.agents.update_agent(
            agentId=agent_id,
            agentResourceRoleArn=role_arn,
            clientToken=str(uuid.uuid4()),
        )
        return resp["agent"]

    # ---------- Runtime ----------
    def invoke(self, agent_id: str, alias_id: str, text: str, session_id: Optional[str] = None, enable_trace: bool = False) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        try:
            resp = self.runtime.invoke_agent(
                agentId=agent_id,
                agentAliasId=alias_id,
                inputText=text,
                sessionId=session_id,
                enableTrace=enable_trace,
            )
            # The response is an EventStream; collect final chunk text
            chunks = []
            for event in resp.get("completion", []):
                if "chunk" in event:
                    data = event["chunk"].get("bytes") or b""
                    try:
                        decoded = data.decode("utf-8", errors="ignore")
                    except Exception:
                        decoded = str(data)
                    chunks.append(decoded)
                elif "finalResponse" in event:
                    # Some SDK versions use finalResponse.text
                    txt = event["finalResponse"].get("text")
                    if txt:
                        chunks.append(txt)
            return "".join(chunks).strip()
        except ClientError as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code in {"AccessDeniedException", "accessDeniedException"}:
                raise RuntimeError(
                    "Access denied invoking agent. Ensure your caller has bedrock:InvokeAgent and (for streaming) bedrock:InvokeModelWithResponseStream permissions."
                ) from e
            raise
        except EventStreamError as e:
            # Surface a cleaner message for stream-time auth/perm failures
            msg = str(e)
            if "accessDeniedException" in msg or "Access denied" in msg:
                raise RuntimeError(
                    "Access denied during agent streaming. Grant bedrock:InvokeAgent and bedrock:InvokeModelWithResponseStream to the invoking identity and retry."
                ) from e
            raise


def load_config_from_env() -> AgentConfig:
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    agent_name = os.environ.get("AGENT_NAME", "strand-demo-agent")
    foundation_model = os.environ.get("FOUNDATION_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
    instruction = os.environ.get("AGENT_INSTRUCTION", "You are a helpful assistant.")
    role_arn = os.environ.get("AGENT_ROLE_ARN", "")
    if not role_arn:
        raise ValueError("AGENT_ROLE_ARN env var is required to create an agent.")
    return AgentConfig(
        region=region,
        agent_name=agent_name,
        foundation_model=foundation_model,
        instruction=instruction,
        role_arn=role_arn,
    )
