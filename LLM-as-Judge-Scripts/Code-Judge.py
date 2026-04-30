
import pandas as pd
from openai import OpenAI
import time
import glob
import os

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r""   #Use base directory here
MODEL_NAME = "gpt-5.1"
TEMPERATURE = 0.0
SLEEP_SEC = 0.5

client = OpenAI(api_key="")  # uses OPENAI_API_KEY

# ============================================================
# CODE JUDGE PROMPT
# ============================================================
JUDGE_PROMPT = """
You are an expert evaluator for code-related tasks.

Given:
- Question (which defines the problem and its assumptions)
- Reference Answer (one possible correct solution)
- Generated Answer (which may contain explanations, markdown, headers, or extra text)

Your task is to determine whether the Generated Answer correctly solves the problem stated in the Question.

IMPORTANT INSTRUCTIONS (STRICT):
- Focus ONLY on executable code in the Generated Answer.
- Ignore all surrounding text, explanations, markdown, headers (e.g., ###), comments, or conversational content.
- If code appears inside markdown blocks (```), consider ONLY the code inside those blocks.
- If multiple code blocks are present:
  - IGNORE any code block that is syntactically incomplete or truncated.
  - Evaluate ONLY the first complete and syntactically valid code block that solves the Question.
  - Do NOT penalize the answer for the presence of later incomplete or unrelated code blocks.
- Do NOT mark an answer incorrect solely because an incomplete or truncated code fragment appears after a complete solution.

Additional rules:
- Use the Reference Answer only as guidance; the Generated Answer does NOT need to match it.
- Judge correctness ONLY with respect to the stated problem and its assumptions.
- Do NOT introduce additional requirements not stated in the Question.
- Do NOT penalize missing input validation if the problem already constrains the inputs.
- Different valid implementations are acceptable.
- Logical or functional errors in the evaluated code make the solution incorrect.
- Partial or incomplete implementations of the evaluated solution are incorrect.

Output:
Correct: Yes or No
Explanation: <brief justification focusing on the evaluated code logic>
"""

# ============================================================
# FIND ALL CODE CSV FILES
# ============================================================
csv_files = glob.glob(os.path.join(BASE_DIR, "accuracy_Code_*.csv"))
print(f" Found {len(csv_files)} code CSV files\n")

# ============================================================
# PROCESS EACH FILE (ALL ROWS)
# ============================================================
for csv_path in csv_files:
    file_name = os.path.basename(csv_path)
    print(f" Starting file: {file_name}")

    df = pd.read_csv(csv_path)
    total_questions = len(df)

    df_out = df.copy()
    df_out["judge_correct"] = None
    df_out["judge_explanation"] = None

    for i, (idx, row) in enumerate(df_out.iterrows(), start=1):
        user_message = f"""
Question:
{row['question']}

Reference Answer:
{row['reference answer']}

Generated Answer:
{row['answer']}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        output = response.choices[0].message.content.strip()

        correct, explanation = "", ""

        for line in output.splitlines():
            if line.lower().startswith("correct"):
                correct = line.split(":")[1].strip()
            elif line.lower().startswith("explanation"):
                explanation = line.split(":", 1)[1].strip()

        # Safety cap on explanation length
        explanation = " ".join(explanation.split()[:25])

        df_out.at[idx, "judge_correct"] = correct
        df_out.at[idx, "judge_explanation"] = explanation

        # PRINT AFTER EACH QUESTION
        print(f"   ➤ Question {i}/{total_questions} processed (Correct: {correct})")

        time.sleep(SLEEP_SEC)

    # ========================================================
    # SAVE OUTPUT
    # ========================================================
    out_path = csv_path.replace(".csv", "_Judge_Output.csv")
    df_out.to_csv(out_path, index=False)

    print(f" Finished file: {file_name}")
    print(f" Output saved to: {out_path}\n")

print(" Code generation evaluation completed for all files.")


# In[ ]:




