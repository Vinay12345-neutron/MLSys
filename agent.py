import os
import sys
import json
import time
import requests
from simulator import Problem, evaluate

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
keys = os.getenv("GOOGLE_API_KEY").split(",")
client = genai.Client(api_key=keys[0])


def format_prompt(problem_json):
    with open("prompts/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    user_prompt = f"Here is the hardware specification and graph topology:\n\n{json.dumps(problem_json, indent=2)}\n\nPlease provide the optimal schedule in JSON format."
    return system_prompt + "\n\n" + user_prompt


def call_gemini_api(api_key, full_prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            # Optional: Use thinking_config for better logic on complex schedules
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.thought:
            print("\n--- MODEL THINKING PROCESS ---")
            print(part.text)
            print("------------------------------\n")

    if response.text:
        return response.text
    else:
        raise Exception("Model returned an empty response.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python agent.py <path_to_input.json> <path_to_output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    start_time = time.time()
    TIMEOUT = 580  # 9 minutes and 40 seconds to be safe before the 10-minute constraint

    api_key = keys[0]
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    with open(input_path, "r") as f:
        problem_data = json.load(f)

    problem = Problem(problem_data)
    prompt = format_prompt(problem_data)

    best_solution = None
    best_latency = float("inf")

    attempts = 0
    max_attempts = 10

    history = []
    plateaus = 0
    i = 0

    while time.time() - start_time < TIMEOUT and attempts < max_attempts:
        attempts += 1
        api_key = keys[i % len(keys)]
        print(f"--- Attempt {attempts} ---", flush=True)
        try:
            print("Sending request to Gemini...", flush=True)

            # Construct the full prompt context
            full_prompt = prompt
            for i in range(0, len(history), 2):
                full_prompt += f"\n\nModel Response:\n{history[i]}\n\nUser Feedback:\n{history[i + 1]}"

            raw_text = call_gemini_api(api_key, full_prompt).strip()

            # Clean up the markdown block formatting if the model still outputs it
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            print("Model output:", flush=True)
            print(raw_text, flush=True)

            try:
                candidate_data = json.loads(raw_text)
                candidate_solution = Solution(candidate_data)

                # We attempt to evaluate it locally
                valid, msg, lat = evaluate(problem, candidate_solution)

                if valid:
                    print(f"Valid schedule found! Latency: {lat}")
                    if lat < best_latency:
                        best_latency = lat
                        best_solution = candidate_data
                        plateaus = 0
                        # Provide feedback to optimize further if there's time
                        feedback = f"Your schedule is VALID and has a latency of {lat}. Can you optimize the granularity, grouping, or data residency (tensors_to_retain) to reduce the latency further? Specifically check if you can use split-K or larger groupings without OOM."
                        history.extend([raw_text, feedback])
                    else:
                        plateaus += 1
                        if plateaus >= 3:
                            print(
                                "Optimal schedule confirmed. Stopping early!",
                                flush=True,
                            )
                            break
                        feedback = f"Your new valid schedule did NOT improve the latency. It is stuck at {lat}. Try a radically different grouping or approach. If you cannot improve it, output the exact same JSON."
                        history.extend([raw_text, feedback])

                else:
                    print(f"Invalid schedule: {msg}")
                    # Provide exact feedback
                    feedback = f"Your schedule was INVALID for the following reason:\n{msg}\n\nPlease fix the errors and generate a new JSON. Remember to carefully check memory limits and execution rules."
                    history.extend([raw_text, feedback])

            except json.JSONDecodeError:
                print("Failed to parse JSON")
                feedback = (
                    "Your output was not valid JSON. Please provide ONLY valid JSON."
                )
                history.extend([raw_text, feedback])
            except Exception as e:
                print(f"Validation error: {e}")
                feedback = f"A validation error occurred: {e}. Please ensure your JSON strictly matches the requested format."
                history.extend([raw_text, feedback])

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                wait_time = 65
                print(
                    f"Hit the RPM limit (429). Sleeping for {wait_time} seconds to clear the bucket...",
                    flush=True,
                )
                time.sleep(wait_time)
            else:
                print(f"API Error: {error_str}")
                time.sleep(15)
            continue

    if best_solution:
        with open(output_path, "w") as f:
            json.dump(best_solution, f, indent=2)
        print(
            f"Successfully wrote best solution to {output_path} with latency {best_latency}"
        )
    else:
        print("Failed to find a valid solution within constraints and timeout.")
        # Write out an empty invalid JSON if entirely failed, just to not crash the runner
        with open(output_path, "w") as f:
            json.dump(
                {
                    "subgraphs": [],
                    "granularities": [],
                    "tensors_to_retain": [],
                    "traversal_orders": [],
                    "subgraph_latencies": [],
                },
                f,
            )


if __name__ == "__main__":
    main()
