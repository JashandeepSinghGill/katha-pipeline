"""The one seam every LLM call goes through (locked decision from the
transcript-slice plan). Two backends, chosen by Settings.llm_backend:

  claude_cli    `claude -p` on the Max subscription. The host user must have
                run `claude login`. Verified flags (2026-09-05):
                  --safe-mode         no CLAUDE.md / hooks / plugins / MCP; OAuth still works
                  --output-format json --json-schema S   -> stdout JSON with `structured_output`
                  --system-prompt-file F                  replaces the default system prompt
                  --tools ""                               no built-in tools
                  --no-session-persistence
                NEVER --bare: it disables OAuth ("auth is strictly ANTHROPIC_API_KEY").
                The user prompt is piped on stdin as UTF-8 (Windows caps argv at 32 KB).
  anthropic_api The official SDK; pay-as-you-go. Structured output via
                output_config.format = json_schema (SDK 1.x).

Nothing outside this module may learn which backend is live.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Protocol

from katha.config import Settings

log = logging.getLogger("katha.llm")


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict: ...


def parse_cli_result(payload: object) -> dict:
    if not isinstance(payload, dict) or payload.get("type") != "result":
        raise LLMError(f"claude -p wrote an unexpected top-level object: {str(payload)[:200]!r}")
    if payload.get("is_error"):
        raise LLMError(f"claude -p reported an error: {payload.get('result')}")
    out = payload.get("structured_output")
    if not isinstance(out, dict):
        raise LLMError(
            "claude -p returned no structured_output — was --json-schema passed and "
            f"did the model comply? result={str(payload.get('result'))[:200]!r}"
        )
    log.info(
        "claude -p ok: cost_usd=%s models=%s turns=%s",
        payload.get("total_cost_usd"), list((payload.get("modelUsage") or {}).keys()),
        payload.get("num_turns"),
    )
    return out


class ClaudeCliProvider:
    name = "claude_cli"

    def __init__(self, settings: Settings, model: str, bin_path: str | None = None):
        wanted = bin_path or settings.claude_bin
        found = shutil.which(wanted)
        if found is None:
            raise LLMError(
                f"claude CLI not found at {wanted!r} (KATHA_CLAUDE_BIN); install Claude "
                "Code and run `claude login` as the worker's user"
            )
        self.bin = found
        self.model = model
        self.settings = settings

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict:
        cwd = self.settings.work_dir
        cwd.mkdir(parents=True, exist_ok=True)
        fd, sys_path = tempfile.mkstemp(dir=cwd, prefix=".sysprompt.", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(system)
            cmd = [
                self.bin, "-p", "--safe-mode",
                "--output-format", "json",
                "--json-schema", json.dumps(schema, ensure_ascii=False),
                "--system-prompt-file", sys_path,
                "--tools", "",
                "--no-session-persistence",
                "--model", self.model,
            ]
            timeout = self.settings.llm_timeout_seconds
            try:
                proc = subprocess.run(
                    cmd, input=user.encode("utf-8"), capture_output=True,
                    timeout=timeout, cwd=cwd,
                )
            except subprocess.TimeoutExpired:
                raise LLMError(f"claude -p timed out after {timeout}s") from None
            except OSError as exc:
                raise LLMError(f"could not run claude -p: {exc}") from exc
        finally:
            try:
                os.unlink(sys_path)
            except OSError:
                pass

        stdout = proc.stdout.decode("utf-8", "replace")
        stderr = proc.stderr.decode("utf-8", "replace")
        if proc.returncode != 0:
            raise LLMError(
                f"claude -p exited {proc.returncode}: {(stderr or stdout).strip()[:500]}"
            )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            raise LLMError(f"claude -p wrote non-JSON output: {stdout[:300]!r}") from None
        return parse_cli_result(payload)


class AnthropicApiProvider:
    name = "anthropic_api"

    def __init__(self, settings: Settings, model: str, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key or None,
                timeout=float(settings.llm_timeout_seconds),
            )
        self.client = client
        self.model = model
        self.max_tokens = settings.llm_max_tokens

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict:
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Anthropic API unreachable: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Anthropic API call failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Anthropic API refused the request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise LLMError(
                f"Anthropic API output truncated at max_tokens={self.max_tokens}; "
                "raise KATHA_LLM_MAX_TOKENS or lower correct_window_segments"
            )
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("Anthropic API returned no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise LLMError(f"Anthropic API returned non-JSON text: {text[:300]!r}") from None


def make_provider(settings: Settings, model: str) -> LLMProvider:
    if settings.llm_backend == "claude_cli":
        return ClaudeCliProvider(settings, model)
    if settings.llm_backend == "anthropic_api":
        return AnthropicApiProvider(settings, model)
    raise LLMError(
        f"unknown llm_backend {settings.llm_backend!r}; expected claude_cli or anthropic_api"
    )
