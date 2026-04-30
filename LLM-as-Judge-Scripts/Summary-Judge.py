
import pandas as pd
from openai import OpenAI
import time
import glob
import os

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r""  #put base directory here
MODEL_NAME = "gpt-5.1"
TEMPERATURE = 0.0
SLEEP_SEC = 0.5

client = OpenAI(api_key="")  # uses OPENAI_API_KEY

# ============================================================
# JUDGE PROMPT
# ============================================================
JUDGE_PROMPT = """
You are an expert evaluator for text summarization tasks.

Given:
- Source Text
- Reference Summary
- Generated Summary

Evaluate how well the Generated Summary captures the key information from the Source Text.
Use the Reference Summary only as guidance; the Generated Summary does NOT need to match it exactly.

Scoring rules:
5 = Excellent summary that is faithful, accurate, and covers all key points
4 = Good summary that is mostly accurate but misses minor details
3 = Adequate summary that captures some key points but omits important information
2 = Poor summary with major omissions or inaccuracies
1 = Incorrect, misleading, or largely irrelevant summary

Additional rules:
- Different wording or structure from the reference is acceptable.
- Minor omissions are acceptable at higher scores.
- Extra information is acceptable only if it is factually correct and supported by the source text.
- Any hallucinated fact, contradiction, or unsupported claim should result in a score of 1.
- Summaries that substantially repeat or closely mirror the source text or reference summary should be penalized, even if factually correct.

Output format:
Score: <1–5>
Explanation: <ONE concise sentence, maximum 25 words>
"""

# ============================================================
# FIND ALL CSV FILES
# ============================================================
csv_files = glob.glob(os.path.join(BASE_DIR, "accuracy_Summ_*.csv"))
print(f" Found {len(csv_files)} CSV files\n")

# ============================================================
# PROCESS EACH FILE (ALL ROWS)
# ============================================================
for csv_path in csv_files:
    file_name = os.path.basename(csv_path)
    print(f" Starting file: {file_name}")

    df = pd.read_csv(csv_path)
    total_questions = len(df)

    df_subset = df.copy()
    df_subset["judge_score"] = None
    df_subset["judge_explanation"] = None

    for i, (idx, row) in enumerate(df_subset.iterrows(), start=1):
        user_message = f"""
Source Text:
{row['question']}

Reference Summary:
{row['reference answer']}

Generated Summary:
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

        explanation = " ".join(explanation.split()[:25])

        df_subset.at[idx, "judge_score"] = score
        df_subset.at[idx, "judge_explanation"] = explanation

        #  PRINT AFTER EACH QUESTION
        print(f"   ➤ Question {i}/{total_questions} processed (Score: {score})")

        time.sleep(SLEEP_SEC)

    # ========================================================
    # SAVE OUTPUT
    # ========================================================
    out_path = csv_path.replace(".csv", "_Judge_Output.csv")
    df_subset.to_csv(out_path, index=False)

    print(f" Finished file: {file_name}")
    print(f" Output saved to: {out_path}\n")

print(" Full evaluation completed for all files.")





