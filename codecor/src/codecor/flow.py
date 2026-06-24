import asyncio
import os
import re
import ast
import sys
import subprocess
from crewai.flow.flow import Flow, listen, start, router
from .crew import CodecorCrew

class CodeCorFlow(Flow):
    POOL_SIZE = 3
    MAX_REPAIR_ROUNDS = 3

    def extract_array(self, text: str):
        match = re.search(r'\[.*?\]', text)
        if match:
            try:
                return ast.literal_eval(match.group(0))
            except (ValueError, SyntaxError):
                pass
        return []

    def clean_code(self, raw_text: str):
        ticks = "`" * 3
        pattern = rf"{ticks}(?:python|py)?\s*(.*?)\s*{ticks}"
        code_blocks = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        return code_blocks[-1].strip() if code_blocks else raw_text.strip()

    # Generate and prune Chain of Thought prompts
    @start()
    async def phase_1_prompt_generation(self):
        print(f"\n--- Phase I: Generating {self.POOL_SIZE} CoT Prompts ---")
        self.state["requirement"] = self.state.get("requirement", "")

        gen_tasks = [CodecorCrew().cot_gen_crew().akickoff(inputs={"requirement": self.state["requirement"]}) for _ in range(self.POOL_SIZE)]
        cot_results = await asyncio.gather(*gen_tasks)
        cot_pool = [res.raw for res in cot_results]

        eval_tasks = [CodecorCrew().cot_eval_crew().akickoff(inputs={"cot_candidate": cot}) for cot in cot_pool]
        eval_results = await asyncio.gather(*eval_tasks)

        surviving_cots = []
        for cot, eval_res in zip(cot_pool, eval_results):
            score = self.extract_array(eval_res.raw)
            if score == [1, 1, 1, 1]:
                surviving_cots.append(cot)

        self.state["selected_cot"] = surviving_cots[0] if surviving_cots else cot_pool[0]
        print(f"[*] CoT Pruning Complete. {len(surviving_cots)}/{self.POOL_SIZE} survived. Selected best CoT.")

    # Parallel generation of test cases and code snippets based on the selected chain of thought
    @listen("phase_1_prompt_generation")
    async def phase_2_and_3_generation(self):
        print(f"\n--- Phase II & III: Generating {self.POOL_SIZE} Code Snippets and {self.POOL_SIZE} Test Suites ---")

        inputs = {"requirement": self.state["requirement"], "selected_cot": self.state["selected_cot"]}

        test_gen_tasks = [CodecorCrew().test_gen_crew().akickoff(inputs=inputs) for _ in range(self.POOL_SIZE)]
        code_gen_tasks = [CodecorCrew().code_gen_crew().akickoff(inputs=inputs) for _ in range(self.POOL_SIZE)]

        test_results = await asyncio.gather(*test_gen_tasks)
        code_results = await asyncio.gather(*code_gen_tasks)

        test_pool = [self.clean_code(res.raw) for res in test_results]
        code_pool = [self.clean_code(res.raw) for res in code_results]

        # Prune tests
        test_eval_tasks = [CodecorCrew().test_eval_crew().akickoff(inputs={"test_candidate": t}) for t in test_pool]
        test_eval_res = await asyncio.gather(*test_eval_tasks)

        surviving_tests = []
        for t, e_res in zip(test_pool, test_eval_res):
            if self.extract_array(e_res.raw) == [1, 1, 1]:
                surviving_tests.append(t)

        self.state["master_test_suite"] = "\n\n".join(surviving_tests) if surviving_tests else test_pool[0]

        # Prune code
        code_eval_tasks = [CodecorCrew().code_eval_crew().akickoff(inputs={"code_candidate": c}) for c in code_pool]
        code_eval_res = await asyncio.gather(*code_eval_tasks)

        self.state["surviving_code_snippets"] = []
        for c, e_res in zip(code_pool, code_eval_res):
            if self.extract_array(e_res.raw) == [1]:
                self.state["surviving_code_snippets"].append({"code": c, "feedback": "", "repair_rounds": 0})

        if not self.state["surviving_code_snippets"]:
            self.state["surviving_code_snippets"].append({"code": code_pool[0], "feedback": "", "repair_rounds": 0})

        self.state["ranked_code_set"] = []

    # Executes all surviving snippets natively (not using docker for this framework)
    @listen("phase_2_and_3_generation")
    async def phase_4_result_checking(self):
        print("\n--- Phase IV: Result Checking (Native Windows Execution) ---")

        snippets_to_test = self.state["surviving_code_snippets"]
        self.state["surviving_code_snippets"] = []

        # Anchor the workspace to this package, not the caller's cwd, so test files never
        # leak into the project root when run.py is launched from a different directory.
        workspace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        self.state["failed_code_snippets"] = []

        for idx, snippet in enumerate(snippets_to_test):
            combined_code = (
                f"{snippet['code']}\n\n"
                f"{self.state['master_test_suite']}\n\n"
                f"if __name__ == '__main__':\n"
                f"    import unittest\n"
                f"    unittest.main(exit=False)\n"
            )

            file_name = f"task_{idx}_test.py"
            host_file_path = os.path.join(workspace_dir, file_name)

            with open(host_file_path, "w", encoding="utf-8") as f:
                f.write(combined_code)

            try:
                result = subprocess.run(
                    [sys.executable, host_file_path],
                    capture_output=True,
                    text=True,
                    timeout=15
                )

                # Combine standard output and errors to feed back to the LLM
                full_output = result.stdout + "\n" + result.stderr

                # Unittest returns exit code 0 if all tests pass, 1 if any fail
                if result.returncode == 0:
                    feedback = "PASSED\n" + full_output
                else:
                    feedback = "FAILED\n" + full_output

            except subprocess.TimeoutExpired:
                feedback = "FAILED\nTimeout: The code took too long to execute (possible infinite loop)."
            except Exception as e:
                feedback = f"FAILED\nExecution Error: {str(e)}"
            finally:
                # Don't let scratch test files accumulate in the source tree across runs.
                if os.path.exists(host_file_path):
                    os.remove(host_file_path)

            snippet["feedback"] = feedback

            if "FAILED" not in feedback:
                print("[+] Snippet PASSED! Adding to Ranked Code Set.")
                self.state["ranked_code_set"].append(snippet)
            else:
                if snippet["repair_rounds"] < self.MAX_REPAIR_ROUNDS:
                    print("[!] Snippet FAILED. Queuing for repair.")
                    self.state["failed_code_snippets"].append(snippet)
                else:
                    print("[!] Snippet FAILED. Max repairs reached. Adding to Ranked Code Set as fallback.")
                    self.state["ranked_code_set"].append(snippet)

    @router("phase_4_result_checking")
    def route_to_repair_or_finish(self):
        if self.state["failed_code_snippets"]:
            return "repair"
        return "finish"

    # Repair the failed snippets
    @listen("repair")
    async def phase_5_code_repairing(self):
        print(f"\n--- Phase V: Code Repairing ({len(self.state['failed_code_snippets'])} snippets) ---")

        repaired_snippets = []

        for failed_snippet in self.state["failed_code_snippets"]:
            failed_snippet["repair_rounds"] += 1
            inputs = {"code": failed_snippet["code"], "feedback": failed_snippet["feedback"]}

            # Generate pool of repair advice
            advice_tasks = [CodecorCrew().repair_advice_crew().akickoff(inputs=inputs) for _ in range(self.POOL_SIZE)]
            advice_results = await asyncio.gather(*advice_tasks)
            advice_pool = [res.raw for res in advice_results]

            # Prune it
            eval_tasks = [CodecorCrew().repair_eval_crew().akickoff(inputs={"advice_candidate": adv}) for adv in advice_pool]
            eval_results = await asyncio.gather(*eval_tasks)

            surviving_advice = []
            for adv, e_res in zip(advice_pool, eval_results):
                if self.extract_array(e_res.raw) == [1, 1, 1, 1]:
                    surviving_advice.append(adv)

            selected_advice = surviving_advice[0] if surviving_advice else failed_snippet["feedback"]

            # Apply the repair
            apply_inputs = {
                "requirement": self.state["requirement"],
                "code": failed_snippet["code"],
                "feedback": failed_snippet["feedback"],
                "selected_advice": selected_advice
            }
            # Using akickoff for consistency since it is inside an async loop
            repair_result = await CodecorCrew().repair_apply_crew().akickoff(inputs=apply_inputs)

            repaired_snippets.append({
                "code": self.clean_code(repair_result.raw),
                "feedback": "",
                "repair_rounds": failed_snippet["repair_rounds"]
            })

        self.state["surviving_code_snippets"] = repaired_snippets

        # Loop back to phase 4
        await self.phase_4_result_checking()
        return self.route_to_repair_or_finish()

    # Selecting highest ranked code
    @listen("finish")
    def wrap_up(self):
        print("\n--- Finalizing CodeCoR Output ---")

        ranked_set = self.state.get("ranked_code_set", [])
        if not ranked_set:
            return ""

        # Sort by fewest repair rounds, preferring those that actually passed
        ranked_set.sort(key=lambda x: ("FAILED" in x["feedback"], x["repair_rounds"]))

        best_snippet = ranked_set[0]
        print(f"[*] Selected best snippet (Repairs: {best_snippet['repair_rounds']}, Status: {'FAILED' if 'FAILED' in best_snippet['feedback'] else 'PASSED'})")

        return best_snippet["code"]