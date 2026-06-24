import os
import sys

# Forces Python, CrewAI, and the OS to use UTF-8 for all internal
# logging, file operations, and console outputs to prevent charmap crashes
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import time
import argparse
import warnings
import re
import json
from dotenv import load_dotenv
from evalplus.data import get_human_eval_plus
from openai import OpenAI
import litellm

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Intercepts every API call from CrewAI's engine directly to capture
# tokens regardless of how deeply nested the Crews are in the Flow
global_task_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

orig_completion = litellm.completion
orig_acompletion = litellm.acompletion

def extract_usage(usage_obj):
    global global_task_usage
    if usage_obj is not None:
        try:
            if isinstance(usage_obj, dict):
                global_task_usage["prompt_tokens"] += usage_obj.get("prompt_tokens", 0)
                global_task_usage["completion_tokens"] += usage_obj.get("completion_tokens", 0)
                global_task_usage["total_tokens"] += usage_obj.get("total_tokens", 0)
            else:
                global_task_usage["prompt_tokens"] += getattr(usage_obj, "prompt_tokens", 0)
                global_task_usage["completion_tokens"] += getattr(usage_obj, "completion_tokens", 0)
                global_task_usage["total_tokens"] += getattr(usage_obj, "total_tokens", 0)
        except Exception:
            pass

def hooked_completion(*args, **kwargs):
    if kwargs.get("stream") is True:
        kwargs["stream_options"] = {"include_usage": True}
    response = orig_completion(*args, **kwargs)
    if kwargs.get("stream") is True:
        def generator_wrapper():
            for chunk in response:
                extract_usage(getattr(chunk, "usage", None))
                yield chunk
        return generator_wrapper()
    else:
        extract_usage(getattr(response, "usage", None))
        return response

async def hooked_acompletion(*args, **kwargs):
    if kwargs.get("stream") is True:
        kwargs["stream_options"] = {"include_usage": True}
    response = await orig_acompletion(*args, **kwargs)
    if kwargs.get("stream") is True:
        async def async_generator_wrapper():
            async for chunk in response:
                extract_usage(getattr(chunk, "usage", None))
                yield chunk
        return async_generator_wrapper()
    else:
        extract_usage(getattr(response, "usage", None))
        return response

litellm.completion = hooked_completion
litellm.acompletion = hooked_acompletion


# Generate code for baseline
def generate_baseline_code(prompt: str) -> tuple[str, dict]:
    client = OpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    clean_model_name = os.getenv("OPENAI_MODEL_NAME", "").replace("openrouter/", "")

    try:
        response = client.chat.completions.create(
            model=clean_model_name,
            messages=[
                {"role": "system", "content": "You are an expert Python programmer. Complete the provided Python function. Return ONLY the raw Python code. Do not include markdown formatting, backticks, or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        usage = response.usage
        usage_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

        raw_output = response.choices[0].message.content
        ticks = "`" * 3
        pattern = rf"{ticks}(?:python|py)?\s*(.*?)\s*{ticks}"
        code_blocks = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)

        final_code = code_blocks[-1].strip() if code_blocks else raw_output.strip()
        return final_code, usage_data

    except Exception as e:
        print(f"[!] API Error during baseline generation: {str(e)}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# Main evaluation loop for the multi-agent frameworks
def run_evaluation(system_name: str, mode: str):
    print(f"\n--- Starting {system_name.capitalize()} Generation ---")
    global global_task_usage

    # Dynamically import the requested framework
    if system_name != "baseline":
        if system_name == "agentcoder":
            from agentcoder.src.agentcoder.flow import AgentCoderFlow as FlowClass
        elif system_name == "agilecoder":
            from agilecoder.src.agilecoder.flow import AgileCoderFlow as FlowClass
        elif system_name == "codecor":
            from codecor.src.codecor.flow import CodeCorFlow as FlowClass
        else:
            raise ValueError(f"Unknown system: {system_name}")

    problems = get_human_eval_plus()
    task_ids = list(problems.keys()) if mode == "full" else [list(problems.keys())[0]]
    print(f"[*] Mode: {mode.upper()} (Executing {len(task_ids)} tasks)")

    output_dir = f"outputs/{system_name}"
    os.makedirs(output_dir, exist_ok=True)
    code_output_path = os.path.join(output_dir, "samples.jsonl")
    usage_output_path = os.path.join(output_dir, "usage_metrics.jsonl")

    # Identify already processed tasks to see if it is possible to resume
    completed_tasks = set()
    if os.path.exists(code_output_path):
        with open(code_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_tasks.add(json.loads(line)["task_id"])
                except json.JSONDecodeError:
                    continue
        print(f"[*] Resume Check: Found {len(completed_tasks)} already completed tasks. Skipping them.")

    # Execution Loop
    for task_id in task_ids:
        if task_id in completed_tasks:
            print(f"[*] Skipping {task_id} (Already completed)")
            continue

        prompt = problems[task_id]["prompt"]
        max_retries = 5
        retry_delay = 30
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                print(f"\n[*] Executing {system_name.capitalize()} for {task_id} (Attempt {attempt}/{max_retries})...")

                # Reset token counters for this task
                global_task_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                if system_name == "baseline":
                    final_code, task_usage = generate_baseline_code(prompt)
                else:
                    flow = FlowClass()

                    # Execute framework flow
                    if system_name == "codecor":
                        raw_result = flow.kickoff(inputs={"requirement": prompt})
                    else:
                        flow.state["requirement"] = prompt
                        raw_result = flow.kickoff()

                    # State extraction (Bypass routing strings like "repair")
                    result_str = str(raw_result).strip().lower()

                    if result_str in ["finish", "repair", "success", "end", "fail", "done"]:
                        # CodecoR specific code extraction
                        if system_name == "codecor":
                            if flow.state.get("ranked_code_set"):
                                best = sorted(flow.state["ranked_code_set"], key=lambda x: ("FAILED" in x["feedback"], x["repair_rounds"]))[0]
                                raw_code = best["code"]
                            elif flow.state.get("failed_code_snippets"):
                                raw_code = flow.state["failed_code_snippets"][0]["code"]
                            elif flow.state.get("surviving_code_snippets"):
                                raw_code = flow.state["surviving_code_snippets"][0]["code"]
                            else:
                                raw_code = "" # Fallback in case of complete failure

                        # AgentCoder and AgileCoder fallback
                        else:
                            if isinstance(flow.state, dict):
                                raw_code = flow.state.get("best_code", flow.state.get("code", flow.state.get("final_code", raw_result)))
                            else:
                                raw_code = getattr(flow.state, "best_code", getattr(flow.state, "code", getattr(flow.state, "final_code", raw_result)))
                    else:
                        # Direct return
                        raw_code = raw_result

                    # Output Sanitiser (Strips emojis and bad characters to prevent JSON/Eval crashes)
                    final_code = str(raw_code).encode('ascii', 'ignore').decode('ascii')
                    task_usage = dict(global_task_usage)

                print(f"[DEBUG] Extracted pure code length: {len(str(final_code))} characters")
                print(f"[DEBUG] Usage for {task_id}: {task_usage}")
                success = True
                break

            except Exception as e:
                print(f"\n[!] Error on {task_id}: {str(e)}")
                if attempt < max_retries:
                    print(f"[*] Network drop or timeout detected. Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"[!] Failed {task_id} entirely after {max_retries} attempts.")

        # Maintain exactly 164 records even if all retries crash so EvalPlus still works
        if not success:
            print(f"\n[!] Task {task_id} completely failed. Writing blank fallback to maintain dataset count.")
            final_code = ""
            task_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Write incrementally
        with open(code_output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"task_id": task_id, "completion": str(final_code)}) + '\n')

        with open(usage_output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"task_id": task_id, "usage": task_usage}) + '\n')

    print("\n--- Generation complete ---")
    print(f"[*] Code saved to: {code_output_path}")
    print(f"[*] Usage metrics saved to: {usage_output_path}")

# Entrypoint
def main():
    parser = argparse.ArgumentParser(description="Run Multi-Agent Code Generation Evaluation")
    parser.add_argument("system", choices=["agentcoder", "agilecoder", "codecor", "baseline"])
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    args = parser.parse_args()

    run_evaluation(args.system, args.mode)

if __name__ == "__main__":
    main()