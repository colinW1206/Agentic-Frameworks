import os
import subprocess
import sys

def evaluate_all_systems():
    outputs_dir = "outputs"

    if not os.path.exists(outputs_dir):
        print(f"[!] No '{outputs_dir}' directory found. Run your generation scripts first.")
        return

    print("\n--- Starting Universal Benchmark Evaluation (HumanEval) ---")

    frameworks = ["baseline", "agentcoder", "agilecoder", "codecor"]

    for system_name in frameworks:
        system_path = os.path.join(outputs_dir, system_name)
        sample_file = os.path.join(system_path, "samples.jsonl")

        if os.path.exists(sample_file):
            print(f"\n--- Evaluating system: {system_name.upper()} ---")

            # Use sys.executable to ensure use of the active uv virtual environment
            command = [
                sys.executable,
                "-m", "evalplus.evaluate",
                "--dataset", "humaneval",
                "--samples", sample_file
            ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[*] Evaluation for {system_name} finished (Exit Code: {e.returncode}).")
            except Exception as e:
                print(f"\n[!] Failed to run evaluation for {system_name}: {e}")
        else:
            print(f"\n[*] Skipping {system_name.upper()}: No samples.jsonl found at '{sample_file}'.")

if __name__ == "__main__":
    evaluate_all_systems()