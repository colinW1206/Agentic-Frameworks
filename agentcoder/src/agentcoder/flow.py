import os
import re
import sys
import subprocess
import tempfile
import asyncio
from crewai.flow.flow import Flow, listen, start, router, or_
from .crew import AgentCoderCrew

class AgentCoderFlow(Flow):
    MAX_RETRIES = 3

    # Puts the code and tests together for execution
    def execute_code_deterministically(self, code: str, tests: str) -> str:
        full_script = f"{code}\n\n{tests}\n\nif __name__ == '__main__':\n    import unittest\n    unittest.main(exit=False)\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(full_script)
            temp_path = temp_file.name

        try:
            result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return "PASSED"
            else:
                return f"FAILED\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "FAILED\nTimeoutError: Code execution exceeded 5 seconds."
        except Exception as e:
            return f"FAILED\nSystemError: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Extracts code from markdown blocks
    def clean_code(self, raw_text: str):
        ticks = "`" * 3
        pattern = rf"{ticks}(?:python|py)?\s*(.*?)\s*{ticks}"
        code_blocks = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        return code_blocks[-1].strip() if code_blocks else raw_text.strip()

    @start()
    async def run_double_blind_sprint(self):
        print("\n--- Starting AgentCoder: Parallel Code & Test Generation ---")
        self.state["retry_count"] = 0

        self.state["programmer_instruction"] = (
            f"Write the Python implementation for the following requirement:\n\n"
            f"REQUIREMENT:\n{self.state['requirement']}\n\n"
            f"Do NOT write any tests. Only output the function signature, docstring, and logic."
        )

        self.state["test_instruction"] = (
            f"Write a comprehensive `unittest` suite for the following requirement:\n\n"
            f"REQUIREMENT:\n{self.state['requirement']}\n\n"
            f"Do not implement the solution. Assume the function signature provided in the "
            f"requirement exists. Focus on edge cases, large inputs, and boundary conditions."
        )

        code_coroutine = AgentCoderCrew().coding_crew().akickoff(inputs={"programmer_instruction": self.state["programmer_instruction"]})
        test_coroutine = AgentCoderCrew().testing_crew().akickoff(inputs={"test_instruction": self.state["test_instruction"]})

        code_result, test_result = await asyncio.gather(code_coroutine, test_coroutine)

        self.state["code"] = self.clean_code(code_result.raw)
        self.state["tests"] = self.clean_code(test_result.raw)

    @listen("trigger_refine")
    async def refine(self):
        self.state["retry_count"] += 1
        print(f"[*] Refining implementation and tests based on execution failures (Attempt {self.state['retry_count']}/{self.MAX_RETRIES})...")

        feedback_msg = (
            f"Original Requirement: {self.state['requirement']}\n\n"
            f"The previous execution failed with this error/traceback:\n"
            f"{self.state['feedback']}\n\n"
            f"Please refine your work to fix these errors. Fix any logic bugs or incorrect test expectations."
        )

        self.state["programmer_instruction"] = feedback_msg + " Output ONLY the raw Python code."
        self.state["test_instruction"] = feedback_msg + " Output ONLY the raw Python unittest code."

        code_coroutine = AgentCoderCrew().coding_crew().akickoff(inputs={"programmer_instruction": self.state["programmer_instruction"]})
        test_coroutine = AgentCoderCrew().testing_crew().akickoff(inputs={"test_instruction": self.state["test_instruction"]})

        # Gathering after the asynchronous execution
        code_result, test_result = await asyncio.gather(code_coroutine, test_coroutine)

        self.state["code"] = self.clean_code(code_result.raw)
        self.state["tests"] = self.clean_code(test_result.raw)

    @router(or_(run_double_blind_sprint, refine))
    def check_results(self):
        print("[*] Testing Software in Deterministic Sandbox...")

        feedback = self.execute_code_deterministically(self.state["code"], self.state["tests"])
        self.state["feedback"] = feedback

        if feedback == "PASSED":
            print("[+] All independent tests passed! Wrapping up...")
            return "wrap_up"

        if self.state["retry_count"] < self.MAX_RETRIES:
            print(f"[!] Tester found errors. Routing to retry loop (Attempt {self.state['retry_count'] + 1}/{self.MAX_RETRIES})...")
            return "trigger_refine"

        print("[!] Max retries reached. Forcing wrap up with current code...")
        return "wrap_up"

    @listen("wrap_up")
    def final_code_extraction(self):
        return self.state.get("code", "")