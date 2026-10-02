"""Agent loop tests with scripted fake clients: no network, no API key, no cost."""

import copy
import json
from types import SimpleNamespace

import pytest

import sp500_agent.llm_agent as llm
from sp500_agent.llm_agent import ClaudeResearchAgent, DeepSeekResearchAgent, create_agent
from sp500_agent.research_tools import TOOL_DEFINITIONS, ResearchToolkit


@pytest.fixture(scope="module")
def toolkit(research_data):
    return ResearchToolkit(research_data)


# -- DeepSeek / OpenAI-compatible fakes ---------------------------------------------------------
class FakeChatMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        out = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in self.tool_calls
            ]
        return {k: v for k, v in out.items() if v is not None or not exclude_none}


def chat_response(finish_reason, content=None, calls=()):
    tool_calls = [SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=args)) for i, (name, args) in enumerate(calls)]
    message = FakeChatMessage(content, tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


class ScriptedChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


def test_deepseek_runs_tools_and_answers(toolkit):
    client = ScriptedChatClient([
        chat_response("tool_calls", "Checking.", [("stock_snapshot", '{"ticker": "NVDA"}'), ("rank_stocks", '{"n": 3}')]),
        chat_response("stop", "NVDA ranks well."),
    ])
    agent = DeepSeekResearchAgent(toolkit, model="deepseek-test", client=client)
    reply = agent.ask("How is NVDA?")
    assert reply.text == "NVDA ranks well." and reply.stop_reason == "stop"
    assert [c.name for c in reply.tool_calls] == ["stock_snapshot", "rank_stocks"]
    tool_messages = [m for m in agent.history if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_0", "call_1"]
    assert json.loads(tool_messages[0]["content"])["ticker"] == "NVDA"
    first = client.requests[0]
    assert first["messages"][0] == {"role": "system", "content": llm.SYSTEM_PROMPT}
    assert {t["function"]["name"] for t in first["tools"]} == {t["name"] for t in TOOL_DEFINITIONS}
    assert reply.usage == {"input_tokens": 200, "output_tokens": 40}


def test_deepseek_history_is_append_only_across_turns(toolkit):
    client = ScriptedChatClient([
        chat_response("tool_calls", None, [("dataset_overview", "{}")]),
        chat_response("stop", "First answer."),
        chat_response("stop", "Second answer."),
    ])
    agent = DeepSeekResearchAgent(toolkit, model="m", client=client)
    agent.ask("one")
    before = copy.deepcopy(agent.history)
    agent.ask("two")
    assert agent.history[: len(before)] == before
    assert client.requests[-1]["messages"][1:] == agent.history[:-1]


def test_deepseek_bad_arguments_are_reported_to_the_model(toolkit):
    client = ScriptedChatClient([
        chat_response("tool_calls", None, [("stock_snapshot", "{not json"), ("stock_snapshot", '{"ticker": "ZZZZ"}'), ("buy", "{}")]),
        chat_response("stop", "Sorry."),
    ])
    agent = DeepSeekResearchAgent(toolkit, model="m", client=client)
    reply = agent.ask("?")
    assert all(call.error for call in reply.tool_calls)
    contents = [m["content"] for m in agent.history if m["role"] == "tool"]
    assert contents[0].startswith("Error: Arguments were not valid JSON")
    assert "not among the" in contents[1] and "Unknown tool" in contents[2]


def test_deepseek_stops_calling_tools_after_the_round_limit(toolkit):
    client = ScriptedChatClient([chat_response("tool_calls", None, [("dataset_overview", "{}")])] * 2 + [chat_response("stop", "Done.")])
    agent = DeepSeekResearchAgent(toolkit, model="m", client=client, max_tool_rounds=2)
    assert agent.ask("loop").text == "Done."
    assert [r["tool_choice"] for r in client.requests] == ["auto", "auto", "none"]


def test_deepseek_closes_tool_calls_cut_off_by_length(toolkit):
    client = ScriptedChatClient([chat_response("length", "Partial", [("rank_stocks", '{"n": 3')])])
    agent = DeepSeekResearchAgent(toolkit, model="m", client=client)
    reply = agent.ask("?")
    assert "cut off" in reply.text
    assert agent.history[-1] == {"role": "tool", "tool_call_id": "call_0", "content": "Error: not run because the response was cut off."}


# -- Claude fakes ---------------------------------------------------------------------------------
def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def claude_response(stop_reason, *blocks):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=50, output_tokens=10, cache_read_input_tokens=5))


class ScriptedClaudeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_claude_request_shape_and_tool_loop(toolkit):
    client = ScriptedClaudeClient([
        claude_response("tool_use", text_block("Looking."), tool_block("tu_1", "stock_snapshot", {"ticker": "AAPL"}), tool_block("tu_2", "sector_summary", {})),
        claude_response("end_turn", text_block("AAPL summary.")),
    ])
    agent = ClaudeResearchAgent(toolkit, client=client)
    reply = agent.ask("AAPL?")
    assert reply.text == "AAPL summary." and len(reply.tool_calls) == 2
    first, second = client.requests
    assert first["model"] == llm.DEFAULT_CLAUDE_MODEL == "claude-opus-5-5"
    assert first["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in first["betas"]
    assert first["output_config"] == {"effort": "medium"} and first["cache_control"] == {"type": "ephemeral"}
    # The cached prefix (system + tools) must be byte-identical between requests.
    assert first["system"] == second["system"] and first["tools"] == second["tools"]
    results = agent.history[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"] and not any(r["is_error"] for r in results)


def test_claude_marks_tool_errors(toolkit):
    client = ScriptedClaudeClient([
        claude_response("tool_use", tool_block("tu_1", "stock_snapshot", {"ticker": "NOPE"})),
        claude_response("end_turn", text_block("Not found.")),
    ])
    agent = ClaudeResearchAgent(toolkit, client=client)
    agent.ask("?")
    assert agent.history[2]["content"][0]["is_error"] is True


def test_claude_refusal_without_content_keeps_history_valid(toolkit):
    client = ScriptedClaudeClient([claude_response("refusal")])
    agent = ClaudeResearchAgent(toolkit, client=client)
    reply = agent.ask("?")
    assert "declined" in reply.text
    assert agent.history[-1]["role"] == "assistant" and agent.history[-1]["content"]


def test_claude_max_tokens_closes_pending_tool_use(toolkit):
    client = ScriptedClaudeClient([claude_response("max_tokens", tool_block("tu_9", "rank_stocks", {}))])
    agent = ClaudeResearchAgent(toolkit, client=client)
    agent.ask("?")
    assert agent.history[-1]["content"][0]["tool_use_id"] == "tu_9" and agent.history[-1]["content"][0]["is_error"]


# -- provider selection ---------------------------------------------------------------------------
@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr(llm, "load_environment", lambda: None)
    for key in ["LLM_PROVIDER", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_MODEL", "CLAUDE_AGENT_MODEL", "CLAUDE_AGENT_EFFORT"]:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_auto_provider_prefers_configured_key(toolkit, no_dotenv):
    no_dotenv.setenv("ANTHROPIC_API_KEY", "test")
    assert llm.configured_providers() == ["anthropic"]
    no_dotenv.setenv("DEEPSEEK_API_KEY", "test")
    assert llm.configured_providers() == ["deepseek", "anthropic"]
    agent = create_agent(toolkit, client=ScriptedChatClient([]))
    assert isinstance(agent, DeepSeekResearchAgent) and agent.model == llm.DEFAULT_DEEPSEEK_MODEL
    no_dotenv.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(create_agent(toolkit, client=ScriptedClaudeClient([])), ClaudeResearchAgent)


def test_missing_keys_give_a_clear_error(toolkit, no_dotenv):
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY or ANTHROPIC_API_KEY"):
        create_agent(toolkit)
