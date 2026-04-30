
import time
import gc
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

MODELS = [
    "Llama-2-13b",
    "Yi-34B",
    "qwen2-32B",
    "Deepseek-7b",
    "Falcon3-7B",
    "Llama-2-7b",
    "Llama-3-8B",
    "Phi-3-14B",
    "Qwen2-7B",
    "Yi-6B",
    "gemma-2-9b",
    "gemma-2-9b",
    "Mistral-7B",
    "Qwen2-14B",
]

def reset_between_questions(model):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    if hasattr(model, "past_key_values"):
        model.past_key_values = None

    torch.cuda.synchronize()
    time.sleep(1)  # cooldown before next question

# ============================================================
# GPU CLEANUP
# ============================================================

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    time.sleep(12)   # 10–15s cooldown


def build_prompt(question):
    return f"""### Instruction:
Solve the following problem and provide only the final answer.

### Problem:
{question}

### Answer:
Final Answer: """


    
# ============================================================
# RUN SINGLE MODEL
# ============================================================
    
def run_model(model_name, input_csv):

    print("\n====================================")
    print(f"Running model: {model_name}")
    print("====================================")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()

    df = pd.read_csv(input_csv)
    results = []
    total = len(df)

    for idx, row in df.iterrows():

        print(f"[{model_name}] Question {idx+1}/{total} | ID={row['id']}")

        prompt = build_prompt(row["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                use_cache=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        results.append({
            "model": model_name,
            "dataset": row["dataset"],
            "id": row["id"],
            "question": row["question"],
            "answer": answer,
            "tokens_generated": output.shape[-1] - inputs.input_ids.shape[-1]
        })

        #  HARD CLEANUP
        del inputs, output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        time.sleep(2) 

        #  PERIODIC RELOAD
        if idx > 0 and idx % 150 == 0:
            print(f"[{model_name}] Reloading model at {idx}")
            del model
            cleanup()
            time.sleep(15) 

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="eager"
            )
            model.eval()

    out_csv = f"accuracy_Math_{model_name}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    del model
    cleanup()
    time.sleep(15) 


# ============================================================
# RUN ALL MODELS
# ============================================================

def run_all_models(input_csv):

    # ============================================================
    # Check if a model is already in GPU memory
    # ============================================================
    print("Checking for existing model in GPU...")

    try:
        # If CUDA memory is allocated from a previous model, clear it
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.synchronize()

        # Try to delete any leftover model variable
        if 'model' in globals():
            print("Existing model found → deleting...")
            del globals()['model']
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            print("No previous model found.")
    except Exception as e:
        print(f"Warning while clearing old model: {e}")

    time.sleep(3)
    print("GPU cleared. Loading new model...\n")
    
    for model_name in MODELS:
        try:
            run_model(model_name, input_csv)
        except Exception as e:
            print(f" Error with {model_name}: {e}")
            cleanup()

# ============================================================
# EXECUTION
# ============================================================

run_all_models("input-math.csv")



