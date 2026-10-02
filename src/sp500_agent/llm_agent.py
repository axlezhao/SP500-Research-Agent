"""LLM research agent: a language model that answers questions by calling the research tools.

Two backends share the same tools and system prompt:

- DeepSeek, through its OpenAI-compatible API (`DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL`).
- Claude, through the Anthropic SDK (`ANTHROPIC_API_KEY`, optional `CLAUDE_AGENT_MODEL`).

`LLM_PROVIDER` picks one explicitly (`deepseek` or `anthropic`); otherwise the first configured key wins,
DeepSeek first. Keys can live in a `.env` file at the project root (git-ignored).

Conversation history is append-only: each turn adds the model's messages exactly as returned, plus the
tool results. Rewriting earlier turns would break prompt caching and, on newer Claude models, invalidate
the model's earlier reasoning.

Run `python -m sp500_agent.llm_agent` for an interactive chat in the terminal.
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import TARGET_DESCRIPTION, load_environment
from .research_tools import TOOL_DEFINITIONS, ResearchData, ResearchToolkit, ToolInputError

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 8
MAX_TOKENS = 16_000

SYSTEM_PROMPT = f"""You are the research assistant in an educational S&P 500 data science project. You answer questions about stocks, sectors, the project's machine-learning model and its backtest by calling the tools provided. The tools are your only source of market data.

How to work:
- Ground every number you state in a tool result from this conversation. Never invent prices, returns, headlines or statistics. If the tools don't cover something (intraday prices, options, macro data, events after the data's as-of date), say so plainly.
- Early in a conversation, call dataset_overview so you know the data's as-of date and how reliable the model is. Mention the as-of date when you quote prices or predictions.
- When several lookups are independent, request them together in one step.
- The model estimates the probability that a stock will {TARGET_DESCRIPTION}. It is a ranking signal, not a forecast of where the price goes, and its out-of-sample edge is small, so present probabilities as noisy model output. When the user asks whether to trust the model, use model_performance and backtest_results and explain the evidence honestly, including the caveats.
- Tools return decimals: 0.034 means 3.4%. Volatility is annualised. Convert to percentages in your answer.
- Depending on the data source, you may also have point-in-time fundamentals from SEC filings (fundamentals_history), market conditions from FRED (macro_snapshot), S&P 500 additions and removals (index_changes), and live headlines and SEC filings (recent_news, recent_filings). If a tool says its data isn't available, say so rather than guessing.
- Headlines and filings are context, not proof: say when a headline is only a title, and don't infer an event's market impact from the title alone.

Answer style:
- Lead with the answer, then the supporting numbers. Use a compact markdown table when comparing several stocks.
- Your answer is shown in a chat window: use bold labels or small headings (####), never # or ## headings.
- The app draws a chart from every price_history call, so describe the trend and key levels in words instead of listing the price series.
- Briefly explain quant terms (AUC, information coefficient, Sharpe ratio, drawdown) the first time they matter.
- This is an educational tool. You can describe what the signals and model say, compare stocks and explain risks, but do not tell the user to buy, sell or size a position. If they ask for a personal decision, give the relevant evidence and note once, in one sentence, that this is not financial advice."""


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: dict | None = None
    error: str | None = None


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)


def configured_providers() -> list[str]:
    load_environment()
    providers = []
    if os.environ.get("DEEPSEEK_API_KEY"):
        providers.append("deepseek")
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        providers.append("anthropic")
    return providers


class ResearchAgent(ABC):
    provider = ""

    def __init__(self, toolkit: ResearchToolkit, model: str, max_tool_rounds: int = MAX_TOOL_ROUNDS):
        self.toolkit = toolkit
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.history: list = []

    def reset(self) -> None:
        self.history = []

    def run_tool(self, name: str, arguments) -> tuple[str, ToolCall, bool]:
        """Execute one tool call. Returns (content for the model, record for display, is_error)."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                call = ToolCall(name, {"raw": arguments}, error="Arguments were not valid JSON.")
                return f"Error: {call.error} Send the arguments as a JSON object.", call, True
        call = ToolCall(name, dict(arguments or {}))
        try:
            call.result = self.toolkit.call(name, call.arguments)
        except ToolInputError as exc:
            call.error = str(exc)
            return f"Error: {exc}", call, True
        except Exception as exc:  # a bug in a tool should not end the conversation
            call.error = f"{type(exc).__name__}: {exc}"
            return f"Error: the tool failed ({call.error}).", call, True
        return json.dumps(call.result, separators=(",", ":"), ensure_ascii=False), call, False

    @abstractmethod
    def ask(self, question: str) -> AgentReply: ...


class DeepSeekResearchAgent(ResearchAgent):
    """Chat Completions tool loop against DeepSeek's OpenAI-compatible endpoint."""

    provider = "deepseek"

    def __init__(self, toolkit: ResearchToolkit, model: str | None = None, client=None, **kwargs):
        load_environment()
        super().__init__(toolkit, model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL), **kwargs)
        if client is None:
            from openai import OpenAI

            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("Set DEEPSEEK_API_KEY (in the environment or .env) to use the DeepSeek agent.")
            client = OpenAI(api_key=api_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
        self.client = client
        self.tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in TOOL_DEFINITIONS
        ]

    def ask(self, question: str) -> AgentReply:
        reply = AgentReply(text="", provider=self.provider, model=self.model, usage={"input_tokens": 0, "output_tokens": 0})
        self.history.append({"role": "user", "content": question})
        for round_number in range(self.max_tool_rounds + 1):
            final_round = round_number == self.max_tool_rounds
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *self.history],
                tools=self.tools,
                tool_choice="none" if final_round else "auto",
            )
            if response.usage:
                reply.usage["input_tokens"] += response.usage.prompt_tokens or 0
                reply.usage["output_tokens"] += response.usage.completion_tokens or 0
            choice = response.choices[0]
            message = choice.message
            self.history.append(message.model_dump(exclude_none=True))
            reply.stop_reason = choice.finish_reason or ""
            if message.content:
                reply.text = message.content

            tool_calls = message.tool_calls or []
            if choice.finish_reason == "tool_calls" and tool_calls:
                for tool_call in tool_calls:
                    content, record, _ = self.run_tool(tool_call.function.name, tool_call.function.arguments)
                    reply.tool_calls.append(record)
                    self.history.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})
                continue

            # Any other finish leaves no tool call pending; close stray ones so the next request is valid.
            for tool_call in tool_calls:
                self.history.append({"role": "tool", "tool_call_id": tool_call.id, "content": "Error: not run because the response was cut off."})
            if choice.finish_reason == "length":
                reply.text += "\n\n_(The answer was cut off at the length limit.)_"
            elif choice.finish_reason == "content_filter":
                reply.text = reply.text or "The provider's content filter blocked this answer."
            return reply
        return reply


class ClaudeResearchAgent(ResearchAgent):
    """Messages API tool loop with Claude."""

    provider = "anthropic"

    def __init__(self, toolkit: ResearchToolkit, model: str | None = None, client=None, effort: str | None = None, **kwargs):
        load_environment()
        super().__init__(toolkit, model or os.environ.get("CLAUDE_AGENT_MODEL", DEFAULT_CLAUDE_MODEL), **kwargs)
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        # Claude Opus 5.5 defaults to medium effort; set it explicitly so behaviour doesn't shift with defaults.
        self.effort = effort or os.environ.get("CLAUDE_AGENT_EFFORT", "medium")
        # system and tools stay byte-identical across requests so the cached prefix keeps matching.
        self.system = [{"type": "text", "text": SYSTEM_PROMPT}]

    def _create(self, final_round: bool):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            tools=TOOL_DEFINITIONS,
            tool_choice={"type": "none"} if final_round else {"type": "auto"},
            messages=self.history,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            # If a safety classifier declines, retry server-side on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def ask(self, question: str) -> AgentReply:
        reply = AgentReply(text="", provider=self.provider, model=self.model, usage={"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0})
        self.history.append({"role": "user", "content": question})
        for round_number in range(self.max_tool_rounds + 1):
            response = self._create(final_round=round_number == self.max_tool_rounds)
            usage = response.usage
            reply.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            reply.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
            reply.usage["cache_read_input_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
            reply.stop_reason = response.stop_reason or ""

            content = list(response.content)
            if not content:
                # A refusal before any output has no content; an empty assistant turn is invalid, so record a note.
                content = [{"type": "text", "text": "(No answer was produced.)"}]
            self.history.append({"role": "assistant", "content": content})
            text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
            if text:
                reply.text = text

            tool_uses = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
            if response.stop_reason == "tool_use" and tool_uses:
                results = []
                for block in tool_uses:
                    result, record, is_error = self.run_tool(block.name, block.input)
                    reply.tool_calls.append(record)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": is_error})
                # All results for one assistant turn go back in a single user message.
                self.history.append({"role": "user", "content": results})
                continue
            if response.stop_reason == "pause_turn":
                continue

            if tool_uses:
                # Cut off mid-call (max_tokens): every tool_use still needs a matching result.
                self.history.append(
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": b.id, "content": "Not run: the response was cut off.", "is_error": True} for b in tool_uses]}
                )
            if response.stop_reason == "refusal":
                reply.text = reply.text or "Claude declined to answer this request."
            elif response.stop_reason == "max_tokens":
                reply.text += "\n\n_(The answer was cut off at the length limit.)_"
            return reply
        return reply


AGENTS = {"deepseek": DeepSeekResearchAgent, "anthropic": ClaudeResearchAgent}


def create_agent(toolkit: ResearchToolkit, provider: str | None = None, **kwargs) -> ResearchAgent:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "auto").lower()
    if provider == "auto":
        available = configured_providers()
        if not available:
            raise RuntimeError("No LLM key found. Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY (environment or .env).")
        provider = available[0]
    if provider not in AGENTS:
        raise ValueError(f"Unknown provider {provider!r}. Use one of: {', '.join(AGENTS)}.")
    return AGENTS[provider](toolkit, **kwargs)


def _print_reply(reply: AgentReply) -> None:
    for call in reply.tool_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
        print(f"  · {call.name}({args})" + (f"  [error: {call.error}]" if call.error else ""))
    print(f"\n{reply.text}\n")
    print(f"  [{reply.provider}:{reply.model} · {reply.usage.get('input_tokens', 0):,} in / {reply.usage.get('output_tokens', 0):,} out tokens]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the S&P 500 research agent.")
    parser.add_argument("question", nargs="*", help="Ask one question and exit. Omit for an interactive chat.")
    parser.add_argument("--provider", choices=list(AGENTS), help="Override LLM_PROVIDER / auto-detection.")
    parser.add_argument("--model", help="Override the provider's default model.")
    args = parser.parse_args()

    agent = create_agent(ResearchToolkit(ResearchData.load()), provider=args.provider, model=args.model)
    if args.question:
        _print_reply(agent.ask(" ".join(args.question)))
        return
    print(f"S&P 500 research agent ({agent.provider}:{agent.model}). Type 'reset' to start over, 'quit' to exit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "reset":
            agent.reset()
            continue
        if question:
            _print_reply(agent.ask(question))


if __name__ == "__main__":
    main()
