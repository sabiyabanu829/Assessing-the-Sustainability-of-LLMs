
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
# QA JUDGE PROMPT
# ============================================================
JUDGE_PROMPT = """
You are an expert evaluator for question-answering tasks.

Given:
- Question
- Reference Answer (one possible correct answer)
- Generated Answer

Evaluate how well the Generated Answer answers the Question.
Use the Reference Answer only as guidance; the Generated Answer does NOT need to match it exactly.

Scoring rules:
5 = Fully correct and factually accurate answer to the question
4 = Mostly correct, but slightly incomplete or imprecise
3 = Partially correct; answers the question but misses key details
2 = Mostly incorrect or vague, with limited correct information
1 = Incorrect, misleading, or irrelevant

Additional rules:
- Semantically equivalent answers are acceptable.
- Different wording or level of detail is acceptable.
- Extra information is acceptable if it is correct and relevant.
- Any factual error or contradiction should result in a score of 1.

Output format (STRICT):
Score: <1–5>
Explanation: <ONE concise sentence, maximum 25 words>
"""

# ============================================================
# FIND ALL QA CSV FILES
# ============================================================
csv_files = glob.glob(os.path.join(BASE_DIR, "accuracy_QA_*.csv"))
print(f" Found {len(csv_files)} QA CSV files\n")

# ============================================================
# PROCESS EACH FILE (ALL ROWS)
# ============================================================
for csv_path in csv_files:
    file_name = os.path.basename(csv_path)
    print(f" Starting file: {file_name}")

    df = pd.read_csv(csv_path)
    total_questions = len(df)

    df_out = df.copy()
    df_out["judge_score"] = None
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

        score, explanation = "", ""

        for line in output.splitlines():
            if line.lower().startswith("score"):
                score = line.split(":")[1].strip()
            elif line.lower().startswith("explanation"):
                explanation = line.split(":", 1)[1].strip()

        # Hard cap explanation length
        explanation = " ".join(explanation.split()[:25])

        df_out.at[idx, "judge_score"] = score
        df_out.at[idx, "judge_explanation"] = explanation

        #  PRINT AFTER EACH QUESTION
        print(f"   ➤ Question {i}/{total_questions} processed (Score: {score})")

        time.sleep(SLEEP_SEC)

    # ========================================================
    # SAVE OUTPUT
    # ========================================================
    out_path = csv_path.replace(".csv", "_Judge_Output.csv")
    df_out.to_csv(out_path, index=False)

    print(f" Finished file: {file_name}")
    print(f" Output saved to: {out_path}\n")

print("QA evaluation completed for all files.")



