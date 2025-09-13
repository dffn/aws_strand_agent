import os
import uuid
import json
import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

from .aws_strand_sdk import AgentManager, AgentConfig

console = Console()

def setup_env(env_file: str | None):
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        load_dotenv()

def build_manager(ctx) -> AgentManager:
    region = ctx.obj.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    profile = ctx.obj.get("profile") or os.getenv("AWS_PROFILE")
    access_key = ctx.obj.get("access_key") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = ctx.obj.get("secret_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    return AgentManager(region, profile, access_key=access_key, secret_key=secret_key)

def _extract_alias_id(value: str) -> str:
    """Return alias ID if given an alias ID or an alias ARN.

    Accepts values like 'abcd1234' or
    'arn:aws:bedrock:us-east-1:123456789012:agent/AGENTID/alias/ALIASID'.
    """
    if value and value.startswith("arn:aws:bedrock:") and "/alias/" in value:
        try:
            return value.split("/alias/")[-1]
        except Exception:
            return value
    return value

def _extract_agent_id(value: str) -> str:
    """Return agent ID if given an agent ID or an agent/alias ARN.

    Accepts values like '5YVSEANCUS' or
    'arn:aws:bedrock:us-east-1:123456789012:agent/5YVSEANCUS' or
    'arn:aws:bedrock:us-east-1:123456789012:agent/5YVSEANCUS/alias/ALIASID'.
    """
    if value and value.startswith("arn:aws:bedrock:") and "/agent/" in value:
        try:
            after = value.split("/agent/")[-1]
            return after.split("/")[0]
        except Exception:
            return value
    return value

@click.group()
@click.option("--env-file", type=click.Path(exists=True), help="Path to .env with config values (loaded before running)")
@click.option("--region", envvar="AWS_REGION", help="AWS region, defaults to env AWS_REGION/AWS_DEFAULT_REGION")
@click.option("--profile", envvar="AWS_PROFILE", help="AWS named profile to use")
@click.option("--access-key", envvar="AWS_ACCESS_KEY_ID", help="AWS access key id")
@click.option("--secret-key", envvar="AWS_SECRET_ACCESS_KEY", help="AWS secret access key")
@click.pass_context
def cli(ctx, env_file, region, profile, access_key, secret_key):
    setup_env(env_file)
    ctx.ensure_object(dict)
    ctx.obj["region"] = region
    ctx.obj["profile"] = profile
    ctx.obj["access_key"] = access_key
    ctx.obj["secret_key"] = secret_key

@cli.command("create-agent")
@click.option("--env-file", type=click.Path(exists=True), help="Path to .env with config values")
@click.option("--instruction", help="Agent instruction text (min 40 chars). Overrides AGENT_INSTRUCTION env if provided.")
@click.pass_context
def create_agent_cmd(ctx, env_file, instruction):
    """Create a new Bedrock Agent (control plane)."""
    setup_env(env_file)
    chosen_instruction = instruction or os.getenv("AGENT_INSTRUCTION", "You are a helpful assistant.")
    if len(chosen_instruction) < 40:
        console.print("[red]AGENT instruction must be at least 40 characters (AWS requirement).[/red]")
        console.print("Tip: pass --instruction or set AGENT_INSTRUCTION in .env to a longer description of the agent's role.")
        raise SystemExit(2)

    cfg = AgentConfig(
        region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or ctx.obj.get("region") or "us-east-1",
        agent_name=os.getenv("AGENT_NAME", "strand-demo-agent"),
        foundation_model=os.getenv("FOUNDATION_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"),
        instruction=chosen_instruction,
        role_arn=os.getenv("AGENT_ROLE_ARN", ""),
    )
    if not cfg.role_arn:
        console.print("[red]AGENT_ROLE_ARN is required.[/red]")
        raise SystemExit(2)

    m = build_manager(ctx)
    agent = m.create_agent(cfg)
    table = Table(title="Agent Created")
    table.add_column("Key")
    table.add_column("Value")
    for k in ["agentId", "agentArn", "agentName", "agentStatus", "foundationModel"]:
        table.add_row(k, str(agent.get(k)))
    console.print(table)

@cli.command("prepare")
@click.argument("agent_id")
@click.pass_context
def prepare_cmd(ctx, agent_id):
    """Prepare an agent to make it invocable."""
    m = build_manager(ctx)
    console.print("Preparing agent...")
    m.prepare_agent(agent_id)
    status = m.wait_for_status(agent_id, "PREPARED")
    console.print(f"Status: {status}")

@cli.command("alias")
@click.argument("agent_id")
@click.option("--name", default="prod", help="Alias name")
@click.pass_context
def alias_cmd(ctx, agent_id, name):
    m = build_manager(ctx)
    alias = m.create_alias(agent_id, name)
    console.print(alias)

@cli.command("whoami")
@click.pass_context
def whoami_cmd(ctx):
    """Show AWS caller identity, region, and profile used by this CLI."""
    m = build_manager(ctx)
    ident = m.get_caller_identity()
    table = Table(title="AWS Caller Identity")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Account", str(ident.get("Account")))
    table.add_row("UserId", str(ident.get("UserId")))
    table.add_row("Arn", str(ident.get("Arn")))
    table.add_row("Region", m.region)
    table.add_row("Profile", str(ctx.obj.get("profile") or os.getenv("AWS_PROFILE") or "(default)"))
    console.print(table)

@cli.command("quickstart")
@click.option("--env-file", type=click.Path(exists=True), help="Path to .env with config values")
@click.option("--alias-name", default="prod", help="Alias name")
@click.option("--prompt", default="Say hello", help="Prompt to send after setup")
@click.option("--instruction", help="Agent instruction text (min 40 chars). Overrides AGENT_INSTRUCTION env if provided.")
@click.pass_context
def quickstart_cmd(ctx, env_file, alias_name, prompt, instruction):
    """One-shot: create -> prepare -> alias -> invoke"""
    setup_env(env_file)
    chosen_instruction = instruction or os.getenv("AGENT_INSTRUCTION", "You are a helpful assistant.")
    if len(chosen_instruction) < 40:
        console.print("[red]AGENT instruction must be at least 40 characters (AWS requirement).[/red]")
        console.print("Tip: pass --instruction or set AGENT_INSTRUCTION in .env to a longer description of the agent's role.")
        raise SystemExit(2)
    cfg = AgentConfig(
        region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or ctx.obj.get("region") or "us-east-1",
        agent_name=os.getenv("AGENT_NAME", "strand-demo-agent"),
        foundation_model=os.getenv("FOUNDATION_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"),
        instruction=chosen_instruction,
        role_arn=os.getenv("AGENT_ROLE_ARN", ""),
    )
    if not cfg.role_arn:
        console.print("[red]AGENT_ROLE_ARN is required.[/red]")
        raise SystemExit(2)

    m = build_manager(ctx)
    agent = m.create_agent(cfg)
    agent_id = agent["agentId"]
    console.print(f"Created agent {agent_id}, preparing...")
    m.prepare_agent(agent_id)
    m.wait_for_status(agent_id, "PREPARED")
    alias = m.create_alias(agent_id, alias_name)
    alias_id = alias["agentAliasId"]
    console.print(f"Invoking agent via alias {alias_id}...")
    out = m.invoke(agent_id, alias_id, prompt)
    console.print(out)

@cli.command("list-agents")
@click.pass_context
def list_agents_cmd(ctx):
    """List Bedrock agents in the account/region."""
    m = build_manager(ctx)
    agents = m.list_agents()
    table = Table(title="Agents")
    table.add_column("agentId")
    table.add_column("name")
    table.add_column("status")
    for a in agents:
        table.add_row(str(a.get("agentId")), str(a.get("agentName")), str(a.get("agentStatus")))
    console.print(table)

@cli.command("list-aliases")
@click.argument("agent_id")
@click.pass_context
def list_aliases_cmd(ctx, agent_id):
    """List aliases for a specific agent."""
    m = build_manager(ctx)
    normalized_agent_id = _extract_agent_id(agent_id)
    aliases = m.list_aliases(normalized_agent_id)
    table = Table(title=f"Aliases for {normalized_agent_id}")
    table.add_column("aliasId")
    table.add_column("name")
    for al in aliases:
        table.add_row(str(al.get("agentAliasId")), str(al.get("agentAliasName")))
    console.print(table)

@cli.command("set-role")
@click.argument("agent_id")
@click.option("--role-arn", envvar="AGENT_ROLE_ARN", required=True, help="Execution role ARN to attach to the agent")
@click.pass_context
def set_role_cmd(ctx, agent_id, role_arn):
    """Update the agent's execution role ARN, then advise to prepare again."""
    m = build_manager(ctx)
    agent = m.update_agent_role(agent_id, role_arn)
    table = Table(title="Agent Updated")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("agentId", str(agent.get("agentId")))
    table.add_row("agentResourceRoleArn", str(agent.get("agentResourceRoleArn")))
    console.print(table)
    console.print("Now run 'prepare' to propagate the new role: e.g., prepare AGENT_ID")
@cli.command("invoke")
@click.argument("agent_id")
@click.argument("alias_id", required=False)
@click.option("--alias-name", default="prod", help="Alias name to resolve if alias_id not provided")
@click.option("--create-alias-if-missing/--no-create-alias-if-missing", default=False, help="Create the alias if it does not exist (requires CreateAgentAlias)")
@click.option("--text", required=True, help="Input text to send to agent")
@click.option("--session-id", default=None, help="Session id to continue conversations")
@click.option("--trace/--no-trace", default=False, help="Enable trace events")
@click.pass_context
def invoke_cmd(ctx, agent_id, alias_id, alias_name, create_alias_if_missing, text, session_id, trace):
    m = build_manager(ctx)
    # Normalize inputs (accept IDs or ARNs)
    agent_id = _extract_agent_id(agent_id)
    if alias_id:
        alias_id = _extract_alias_id(alias_id)
    else:
        try:
            found = m.find_alias_by_name(agent_id, alias_name)
        except Exception:
            console.print("[red]Cannot list aliases. Pass alias ID/ARN directly or grant bedrock:ListAgentAliases.[/red]")
            raise SystemExit(2)
        if not found:
            if create_alias_if_missing:
                created = m.create_alias(agent_id, alias_name)
                alias_id = created.get("agentAliasId")
                console.print(f"Created alias '{alias_name}' with id {alias_id} for agent {agent_id}.")
            else:
                console.print(f"[red]Alias '{alias_name}' not found for agent {agent_id}. Use list-aliases, pass alias ID/ARN, or use --create-alias-if-missing.[/red]")
                raise SystemExit(2)
        else:
            alias_id = found.get("agentAliasId")
    out = m.invoke(agent_id, alias_id, text, session_id=session_id, enable_trace=trace)
    console.print("[bold]Response:[/bold]")
    console.print(out)
