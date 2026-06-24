import json
import re

# Used to clean the code samples within the output files to make sure that EvalPlus can evaluate them
def clean_code(code: str) -> str:
    code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL | re.IGNORECASE)

    code = code.replace("</think>", "").replace("<think>", "")

    if code.strip().lower() in ["repair", "finish", "fail", "done", "success"]:
        return ""

    block_pattern = r'```[^\n]*\n(.*?)\n```'
    blocks = re.findall(block_pattern, code, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        code = max(blocks, key=len)
    else:
        code = re.sub(r'```[^\n]*', '', code)

    if "if __name__ == '__main__':" in code:
        code = code.split("if __name__ == '__main__':")[0]
    elif 'if __name__ == "__main__":' in code:
        code = code.split('if __name__ == "__main__":')[0]

    code = code.lstrip('\r\n').rstrip()

    if re.search(r'^(?:async\s+)?def\s+', code, flags=re.MULTILINE):
        code = "    pass\n\n" + code

    return code

def sanitise_jsonl(input_file: str, output_file: str):
    print(f"Cleaning {input_file}...")
    cleaned_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            data = json.loads(line)
            original_code = data["completion"]

            clean = clean_code(original_code)
            data["completion"] = clean

            outfile.write(json.dumps(data) + '\n')
            cleaned_count += 1

    print(f"Successfully cleaned {cleaned_count} samples. Saved to {output_file}")

if __name__ == "__main__":
    input_path = "outputs/agilecoder/samples.jsonl"
    output_path = "outputs/agilecoder/samples.jsonl"

    sanitise_jsonl(input_path, output_path)