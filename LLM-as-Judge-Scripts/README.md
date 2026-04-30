#  LLM-as-a-Judge Code Evaluation Pipeline

This repository provides an **LLM-as-a-Judge evaluation pipeline** for automatically assessing the **correctness of generated code** produced by large language models (LLMs).

The pipeline uses a **strong proprietary LLM (GPT-5.1)** as an expert judge to compare:
- the original programming **question**,
- a **reference answer**, and
- the **generated answer**,

and determines whether the generated code **correctly solves the problem**.

The evaluation is performed on previously generated CSV files (using answer-generation scripts in the LLM Answer Generation - Accuracy Folder) and produces **augmented CSV outputs** with correctness labels and short explanations.

---

##  Judge Model

The evaluation uses the following model as the judge:

```
gpt-5.1
```

The judge model is accessed via the OpenAI API and is run in **deterministic mode** (`temperature = 0.0`) to ensure reproducibility.

---

##  Judging Prompt Strategy

For each task, we used different judge prompt. These prompts can be found in the respective .py files


##  Requirements

### Hardware
- No GPU required
- Stable internet connection

### Software
- Python ≥ 3.10.0
- pandas
- openai (official OpenAI client)

Install dependencies:

```bash
pip install pandas openai
```

Set the OpenAI API key in the .py files:

```bash
client = OpenAI(api_key="")  # uses OPENAI_API_KEY
```

---

##  Running

Set the base directory containing the previously generated CSV files CSV files:

```python
BASE_DIR = "path/to/csv/files"
```

Run all the evaluation script:

```bash
python CodeJudge.py
```

These scripts will generate separate csv files for each .py file

##  Reproducibility

To ensure consistent results:
- Temperature is fixed at `0.0`
- The same judge prompt is used across all evaluations
- Sleep intervals are applied to avoid API throttling
