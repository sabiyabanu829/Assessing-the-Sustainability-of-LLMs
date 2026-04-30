
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
# MATH JUDGE PROMPT
# ============================================================
JUDGE_PROMPT = """
You are an expert evaluator for mathematical and reasoning tasks.

Inputs:
- Question
- Reference Answer (one possible correct solution)
- Generated Answer

Determine whether the Generated Answer correctly solves the mathematical problem stated in the Question.

Use the Reference Answer only to verify correctness of the final result;
the Generated Answer does NOT need to match the reference in method or notation.

Rules:
- The final numerical or symbolic answer must be correct.
- Minor differences in notation, formatting, or algebraic form are acceptable.
- Correct reasoning with an incorrect final answer is incorrect.
- Partial solutions are considered incorrect.

Output format (STRICT):
Correct: Yes or No
Explanation: <ONE concise sentence, maximum 25 words>
"""

# ============================================================
# FIND ALL MATH CSV FILES
# ============================================================
csv_files = glob.glob(os.path.join(BASE_DIR, "accuracy_Math_*.csv"))
print(f" Found {len(csv_files)} math CSV files\n")

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

        # Absolute safety: cap explanation length
        explanation = " ".join(explanation.split()[:25])

        df_out.at[idx, "judge_correct"] = correct
        df_out.at[idx, "judge_explanation"] = explanation

        #  PRINT AFTER EACH QUESTION
        print(f"   ➤ Question {i}/{total_questions} processed (Correct: {correct})")

        time.sleep(SLEEP_SEC)

    # ========================================================
    # SAVE OUTPUT
    # ========================================================
    out_path = csv_path.replace(".csv", "_Judge_Output.csv")
    df_out.to_csv(out_path, index=False)

    print(f" Finished file: {file_name}")
    print(f" Output saved to: {out_path}\n")

print(" Mathematical reasoning evaluation completed for all files.")

