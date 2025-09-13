# aws_strand_agent

Starter Python project to create and operate an Amazon Bedrock Agent (aka "agent SDK").

## Features
- Create an agent (`create_agent`)
- Prepare the agent (`prepare_agent` -> status PREPARED)
- Create an alias (`create_agent_alias`)
- Invoke the agent via runtime streaming (`invoke_agent`)
- Simple CLI with quickstart path

## Prereqs
- Python 3.10+
- AWS account with Amazon Bedrock Agents enabled in your region
- An IAM role ARN with required permissions for `agentResourceRoleArn`
- AWS credentials configured (profile or env vars)

## Install

Create a venv and install deps:

```powershell
# From repo root
cd "d:/ptyhon and angular developement/awsstarndsagent/aws_strand_agent"
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure
Create a `.env` file (or use environment variables):

```
# .env example
AWS_REGION=us-east-1
AGENT_NAME=strand-demo-agent
FOUNDATION_MODEL=anthropic.claude-3-haiku-20240307-v1:0
AGENT_INSTRUCTION=You are a helpful assistant.
AGENT_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/BedrockAgentRole
```

The role must trust bedrock and have needed service permissions (S3, Lambda, etc. as your agent requires).

## CLI Usage

Activate your venv first, then run:

```powershell
# Show commands
python -m src.cli --help

# 1) Create the agent
python -m src.cli create-agent --env-file .env

# 2) Prepare the agent
python -m src.cli prepare <AGENT_ID>

# 3) Create an alias
python -m src.cli alias <AGENT_ID> --name prod

# 4) Invoke via runtime
python -m src.cli invoke <AGENT_ID> <ALIAS_ID> --text "Hello"

# Or one-shot quickstart (create -> prepare -> alias -> invoke)
python -m src.cli quickstart --env-file .env --alias-name prod --prompt "Say hello"
```

## Notes
- After `create_agent`, status might be `NOT_PREPARED`. Run `prepare` and wait for `PREPARED`.
- `invoke_agent` returns streaming events; the CLI collects final text from chunks.
- Ensure Bedrock Agents is available in your chosen region.

## Uninstall
To remove the venv on Windows PowerShell:

```powershell
deactivate ; Remove-Item -Recurse -Force .venv
```
