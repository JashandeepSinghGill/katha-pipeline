import json
import os
import sys
from pathlib import Path

import pytest

from katha.config import Settings
from katha.llm import (
    AnthropicApiProvider,
    ClaudeCliProvider,
    LLMError,
    make_provider,
    parse_cli_result,
)

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}

STUB_PY = r'''
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
mode = open(os.path.join(here, "mode.txt"), encoding="utf-8").read().strip()
stdin = sys.stdin.buffer.read().decode("utf-8")
with open(os.path.join(here, "captured.json"), "w", encoding="utf-8") as f:
    json.dump({"argv": sys.argv[1:], "stdin": stdin, "cwd": os.getcwd()}, f, ensure_ascii=False)
if mode == "sleep":
    time.sleep(3); sys.exit(0)
if mode == "exit2":
    sys.stderr.write("boom"); sys.exit(2)
if mode == "garbage":
    sys.stdout.buffer.write(b"not json"); sys.exit(0)
with open(os.path.join(here, "response.json"), encoding="utf-8") as f:
    sys.stdout.buffer.write(f.read().encode("utf-8"))
'''


class CliStub:
    """A fake `claude` binary: a .bat/.sh launcher around a Python script so
    the stub's stdout bytes are exact UTF-8 regardless of the console code page."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        (root / "stub.py").write_text(STUB_PY, encoding="utf-8")
        if os.name == "nt":
            self.bin = root / "claude-stub.bat"
            self.bin.write_text(f'@echo off\r\n"{sys.executable}" "{root / "stub.py"}" %*\r\n')
        else:
            self.bin = root / "claude-stub"
            self.bin.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{root / "stub.py"}" "$@"\n')
            self.bin.chmod(0o755)
        self.mode("ok")
        self.respond({"type": "result", "is_error": False,
                      "structured_output": {"ok": True}, "total_cost_usd": 0.01,
                      "modelUsage": {"claude-sonnet-5": {}}})

    def mode(self, m: str) -> None:
        (self.root / "mode.txt").write_text(m, encoding="utf-8")

    def respond(self, payload: dict) -> None:
        (self.root / "response.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def captured(self) -> dict:
        return json.loads((self.root / "captured.json").read_text(encoding="utf-8"))


@pytest.fixture
def stub(tmp_path):
    return CliStub(tmp_path / "stub")


@pytest.fixture
def cli_settings(tmp_path):
    return Settings(_env_file=None, work_dir=tmp_path / "work", llm_timeout_seconds=1)


def test_cli_provider_returns_structured_output_and_pipes_prompt_on_stdin(stub, cli_settings):
    p = ClaudeCliProvider(cli_settings, model="claude-sonnet-5", bin_path=str(stub.bin))
    out = p.complete_json(system="SYS ਸਿਸਟਮ", user="USER ਵਾਹਿਗੁਰੂ", schema=SCHEMA)
    assert out == {"ok": True}
    cap = stub.captured()
    assert cap["stdin"] == "USER ਵਾਹਿਗੁਰੂ"
    argv = cap["argv"]
    assert argv[:2] == ["-p", "--safe-mode"]
    assert "--bare" not in argv
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5"
    assert argv[argv.index("--tools") + 1] == ""
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    sys_file = Path(argv[argv.index("--system-prompt-file") + 1])
    assert not sys_file.exists()  # cleaned up after the call
    assert Path(cap["cwd"]).resolve() == cli_settings.work_dir.resolve()


def test_cli_provider_system_prompt_file_contained_the_prompt(stub, cli_settings, monkeypatch):
    import katha.llm as llm_mod
    seen = {}
    real_run = llm_mod.subprocess.run

    def spy(cmd, **kw):
        seen["system"] = Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        return real_run(cmd, **kw)

    monkeypatch.setattr(llm_mod.subprocess, "run", spy)
    ClaudeCliProvider(cli_settings, model="m", bin_path=str(stub.bin)).complete_json(
        system="SYS ਸਿਸਟਮ", user="u", schema=SCHEMA)
    assert seen["system"] == "SYS ਸਿਸਟਮ"


@pytest.mark.parametrize("mode, needle", [
    ("exit2", "exited 2"),
    ("garbage", "non-JSON"),
    ("sleep", "timed out"),
])
def test_cli_provider_failure_modes_raise_llm_error(stub, cli_settings, mode, needle):
    stub.mode(mode)
    p = ClaudeCliProvider(cli_settings, model="m", bin_path=str(stub.bin))
    with pytest.raises(LLMError, match=needle):
        p.complete_json(system="s", user="u", schema=SCHEMA)


def test_cli_provider_surfaces_the_clis_own_error_text(stub, cli_settings):
    stub.respond({"type": "result", "is_error": True, "result": "Not logged in · Please run /login"})
    p = ClaudeCliProvider(cli_settings, model="m", bin_path=str(stub.bin))
    with pytest.raises(LLMError, match="Not logged in"):
        p.complete_json(system="s", user="u", schema=SCHEMA)


def test_cli_provider_missing_binary_fails_at_construction(cli_settings, tmp_path):
    with pytest.raises(LLMError, match="claude"):
        ClaudeCliProvider(cli_settings, model="m", bin_path=str(tmp_path / "nope"))


def test_parse_cli_result_requires_structured_output():
    with pytest.raises(LLMError, match="structured_output"):
        parse_cli_result({"type": "result", "is_error": False, "result": "{}"})
    with pytest.raises(LLMError):
        parse_cli_result({"type": "assistant"})
    assert parse_cli_result({"type": "result", "is_error": False,
                             "structured_output": {"a": 1}}) == {"a": 1}


class _Block:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block("text", text)]
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


class FakeClient:
    def __init__(self, resp):
        self.messages = FakeMessages(resp)


def test_api_provider_uses_structured_output_format_and_parses_text(cli_settings):
    client = FakeClient(_Resp('{"ok": true}'))
    p = AnthropicApiProvider(cli_settings, model="claude-sonnet-5", client=client)
    assert p.complete_json(system="S", user="U", schema=SCHEMA) == {"ok": True}
    kw = client.messages.calls[0]
    assert kw["model"] == "claude-sonnet-5"
    assert kw["system"] == "S"
    assert kw["messages"] == [{"role": "user", "content": "U"}]
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert kw["max_tokens"] == cli_settings.llm_max_tokens


@pytest.mark.parametrize("resp, needle", [
    (_Resp("not json"), "non-JSON"),
    (_Resp("{}", stop_reason="max_tokens"), "max_tokens"),
    (_Resp("{}", stop_reason="refusal"), "refus"),
])
def test_api_provider_failure_modes(cli_settings, resp, needle):
    p = AnthropicApiProvider(cli_settings, model="m", client=FakeClient(resp))
    with pytest.raises(LLMError, match=needle):
        p.complete_json(system="S", user="U", schema=SCHEMA)


def test_api_provider_wraps_sdk_errors(cli_settings):
    import anthropic
    import httpx2
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    p = AnthropicApiProvider(cli_settings, model="m", client=FakeClient(err))
    with pytest.raises(LLMError, match="Anthropic API"):
        p.complete_json(system="S", user="U", schema=SCHEMA)


def test_make_provider_selects_backend(cli_settings, stub):
    api = make_provider(Settings(_env_file=None, llm_backend="anthropic_api",
                                 anthropic_api_key="k"), model="m")
    assert api.name == "anthropic_api"
    cli = make_provider(Settings(_env_file=None, claude_bin=str(stub.bin),
                                 work_dir=cli_settings.work_dir), model="m")
    assert cli.name == "claude_cli"
    with pytest.raises(LLMError, match="llm_backend"):
        make_provider(Settings(_env_file=None, llm_backend="gpt"), model="m")
