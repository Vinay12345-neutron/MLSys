import os
import json
import itertools
from google import genai
from google.genai import types
from typing import List, Dict, TypedDict, Optional, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# 1. API ROTATION SETUP
load_dotenv()
# Fetches keys from .env like GOOGLE_API_KEY=key1,key2,key3
raw_keys = os.getenv("GOOGLE_API_KEY")
if raw_keys:
    api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    key_cycle = itertools.cycle(api_keys)
else:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")


def call_gemini_with_rotation(prompt: str):
    """Configures the SDK with the next key and returns a response."""
    current_key = next(key_cycle)

    # Initialize the new Client with the rotated key
    client = genai.Client(api_key=current_key)

    # Call generate_content using the client object
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",  # Ensures structured output
        ),
    )
    return response.text


# 2. THE LATENCY & OOM CALCULATOR (THE "TOOL")
def calculate_latency_and_oom(graph_spec: dict, schedule: dict) -> dict:
    """Calculates hardware latency and checks for OOM."""
    cap = graph_spec["fast_memory_capacity"]
    bw = graph_spec["slow_memory_bandwidth"]
    native_w, native_h = graph_spec["native_granularity"]

    total_latency = 0
    resident_tensors = set()

    try:
        for i, (nodes, gran, retain) in enumerate(
            zip(
                schedule["subgraphs"],
                schedule["granularities"],
                schedule["tensors_to_retain"],
            )
        ):
            w, h, k = gran

            # Simplified Working Set check
            # (Inputs + Outputs) in the current slice must fit in FM
            input_tensors = []
            output_tensors = []
            for n in nodes:
                input_tensors.extend(graph_spec["inputs"][n])
                output_tensors.extend(graph_spec["outputs"][n])

            unique_tensors = set(input_tensors) | set(output_tensors)
            current_working_set = len(unique_tensors) * (w * h)

            if current_working_set > cap:
                return {"error": f"OOM at Subgraph {i}: {current_working_set} > {cap}"}

            # Latency = max(Compute, Memory_In + Memory_Out)
            # Compute padded to native granularity
            comp_cost = (
                sum(graph_spec["base_costs"][n] for n in nodes)
                * (native_w / w)
                * (native_h / h)
            )

            # Memory In (only load what's not resident)
            needed_inputs = set(input_tensors) - resident_tensors
            mem_in = (
                sum(
                    graph_spec["widths"][t] * graph_spec["heights"][t]
                    for t in needed_inputs
                )
                / bw
            )

            # Memory Out
            mem_out = (
                sum(
                    graph_spec["widths"][t] * graph_spec["heights"][t]
                    for t in output_tensors
                )
                / bw
            )

            total_latency += max(comp_cost, mem_in + mem_out)
            resident_tensors = set(retain)

        return {"status": "success", "latency": total_latency}
    except Exception as e:
        return {"error": f"Logic error in schedule: {str(e)}"}


# 3. LANGGRAPH STATE & NODES
class AgentState(TypedDict):
    graph_spec: dict
    current_schedule: Optional[dict]
    feedback: str
    best_latency: float
    iterations: int


def optimizer_node(state: AgentState):
    """The Brain - Uses the new google-genai SDK directly."""
    prompt = f"""
    Solve this hardware scheduling problem: {json.dumps(state["graph_spec"])}
    Constraints: Minimize latency, stay under memory capacity.
    Last Feedback: {state["feedback"]}
    
    Return a JSON with:
    'subgraphs': [[node_indices], ...]
    'granularities': [[w, h, k], ...]
    'tensors_to_retain': [[tensor_indices], ...]
    """

    raw_json = call_gemini_with_rotation(prompt)
    try:
        # Gemini might wrap the JSON in markdown blocks (```json ... ```)
        # Stripping them ensures json.loads doesn't fail
        clean_json = (
            raw_json.strip().removeprefix("```json").removesuffix("```").strip()
        )
        schedule = json.loads(clean_json)
        return {"current_schedule": schedule, "iterations": state["iterations"] + 1}
    except json.JSONDecodeError:
        return {
            "feedback": f"Invalid JSON returned. Raw output: {raw_json}. Please retry."
        }


def validator_node(state: AgentState):
    """The Logic - Runs the local calculation."""
    res = calculate_latency_and_oom(state["graph_spec"], state["current_schedule"])

    if "error" in res:
        return {"feedback": res["error"]}

    new_latency = res["latency"]
    if new_latency < state["best_latency"]:
        return {
            "best_latency": new_latency,
            "feedback": f"Success! New best: {new_latency}",
        }
    return {"feedback": f"Valid but worse: {new_latency}"}


# 4. ORCHESTRATION
def run_agent(problem_json):
    builder = StateGraph(AgentState)
    builder.add_node("optimizer", optimizer_node)
    builder.add_node("validator", validator_node)

    builder.set_entry_point("optimizer")
    builder.add_edge("optimizer", "validator")

    def loop_check(state):
        return END if state["iterations"] >= 3 else "optimizer"

    builder.add_conditional_edges("validator", loop_check)

    graph = builder.compile()

    init_state = {
        "graph_spec": problem_json,
        "current_schedule": None,
        "feedback": "Initial attempt",
        "best_latency": float("inf"),
        "iterations": 0,
    }

    for event in graph.stream(init_state):
        print(event)


if __name__ == "__main__":
    sample_dag = {
        "widths": [128, 128, 128],
        "heights": [128, 128, 128],
        "inputs": [[0, 1]],
        "outputs": [[2]],
        "base_costs": [1000],
        "op_types": ["MatMul"],
        "fast_memory_capacity": 20000,
        "slow_memory_bandwidth": 10,
        "native_granularity": [128, 128],
    }
    run_agent(sample_dag)
