import asyncio
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from log_1 import (
    EdgeGroundingConfig,
    EdgeQueryGraphGrounder,
    build_class_inventory,
    build_parse_prompt,
    load_prediction_records_from_csv,
    make_client,
    parse_response_json,
)


load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))

app = FastAPI(title="LangChain Async Delay Demo")

grounder = EdgeQueryGraphGrounder(
    EdgeGroundingConfig(
        default_pred_csv=Path("/home/nvidia/.hydra/orbbec/backend/objects.csv"),
        disable_rerank=True,
    )
)


class LLMInvokeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query for the agent")


class DelayToolInput(BaseModel):
    seconds: float = Field(
        ...,
        ge=0,
        le=10,
        description="Number of seconds to wait before returning the tool result.",
    )


class GuideToolInput(BaseModel):
    query: str = Field(..., description="original navigation description")


def get_openai_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )


@tool("async_delay_tool", args_schema=DelayToolInput)
async def async_delay_tool(seconds: float) -> dict[str, Any]:
    """Use this tool when the user asks to wait, pause, or delay for a number of seconds."""
    started_at = time.perf_counter()
    await asyncio.sleep(seconds)
    elapsed = time.perf_counter() - started_at
    return {
        "ok": True,
        "message": f"Delay completed in {elapsed:.3f}s",
        "elapsed": elapsed,
        "input": {"seconds": seconds},
    }


@tool("guide_tool", args_schema=GuideToolInput)
async def guide_tool(query: str) -> dict[str, Any]:
    """Use this tool when the user asks to find, guide, navigate, go to, or locate an object."""
    records = load_prediction_records_from_csv(grounder.config.default_pred_csv, scene_id_override=grounder.config.default_scene_id)
    class_inventory = build_class_inventory(records)
    client = make_client(grounder.config.llm_base_url, grounder.config.llm_api_key)
    parse_answer = client.chat.completions.create(
        model=grounder.config.llm_model,
        messages=[{"role": "user", "content": build_parse_prompt(query, class_inventory)}],
        stream=False,
        max_tokens=512,
        temperature=0.0,
    )
    answer = ""
    if parse_answer.choices and parse_answer.choices[0].message:
        answer = parse_answer.choices[0].message.content or ""
    query_graph = parse_response_json(answer)
    if not isinstance(query_graph, dict):
        return {
            "ok": False,
            "message": "query_graph parse failed",
            "input": {"query": query},
            "parse_raw_response": answer.strip(),
        }
    result = grounder.find_with_query_graph(query, query_graph, log=False)
    return {
        "ok": True,
        "result": result,
        "input": {"query": query},
        "query_graph": query_graph,
    }


def build_agent() -> Any:
    llm = get_openai_llm()
    return create_agent(
        model=llm,
        tools=[guide_tool, async_delay_tool],
        system_prompt=(
            "You are a helpful assistant. Use tools when needed. "
            "For waiting requests, call async_delay_tool. "
            "For navigation/find/guide/locate/go-to requests, call guide_tool exactly once. "
            "Pass the original user navigation request as the `query` argument to guide_tool. "
            "Do not parse the navigation request yourself in the system prompt. "
            "Do not call guide_tool twice."
        ),
        debug=False,
    )


def _format_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    pending_calls: dict[str, dict[str, Any]] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                pending_calls[tool_call["id"]] = {
                    "tool": tool_call["name"],
                    "tool_input": tool_call.get("args", {}),
                    "tool_output": None,
                }
        elif isinstance(message, ToolMessage):
            existing = pending_calls.get(message.tool_call_id, {})
            existing["tool_output"] = message.content
            if "tool" in existing:
                formatted.append(existing)

    return formatted


async def default_agent_runner(query: str) -> dict[str, Any]:
    agent = build_agent()
    raw_result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    messages = raw_result.get("messages", [])
    intermediate_steps = _format_messages(messages)
    output = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            output = str(message.content)
            break
    return {
        "output": output,
        "intermediate_steps": intermediate_steps,
    }


async def run_agent_query(
    query: str,
    agent_runner: Any | None = None,
) -> dict[str, Any]:
    runner = agent_runner or getattr(app.state, "agent_runner", default_agent_runner)
    agent_result = await runner(query)
    return {
        "ok": True,
        "query": query,
        "output": agent_result.get("output", ""),
        "tool_calls": agent_result.get("intermediate_steps", []),
    }


@app.post("/llm/invoke")
async def llm_invoke(payload: LLMInvokeRequest) -> dict[str, Any]:
    return await run_agent_query(payload.query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
