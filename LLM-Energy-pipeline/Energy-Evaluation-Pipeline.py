#!/usr/bin/env python
# coding: utf-8

import time
import gc
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from pynvml import *
import sys
import threading

torch.manual_seed(0)

# ============================================================
# NVML INITIALIZATION
# ============================================================

def init_nvml():
    try:
        nvmlInit()
        return nvmlDeviceGetHandleByIndex(0)
    except NVMLError as e:
        print(f"NVML Error: {str(e)}")
        return None


# ============================================================
# POWER SAMPLER CLASS
# ============================================================

class PowerSampler:
    def __init__(self, handle, interval=0.1):
        self.handle = handle
        self.interval = interval
        self.running = False
        self.samples = []

    def _read_power(self):
        try:
            return nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Watts
        except:
            return None

    def _sample_loop(self):
        while self.running:
            p = self._read_power()
            t = time.perf_counter()
            if p is not None:
                self.samples.append((t, p))
            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def compute_energy(self):
        if len(self.samples) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.samples) - 1):
            t1, p1 = self.samples[i]
            t2, p2 = self.samples[i + 1]
            dt = t2 - t1
            total += ((p1 + p2) / 2) * dt
        return total


# ============================================================
# RESET ENVIRONMENT
# ============================================================

def reset_between_questions(model):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    if hasattr(model, "past_key_values"):
        model.past_key_values = None

    torch.cuda.synchronize()
    time.sleep(5)  # cooldown before next question


# ============================================================
# IDLE GPU POWER BASELINE
# ============================================================

def measure_idle_power(handle, duration=10):
    print(f"Measuring idle power for {duration}s...")
    readings = []
    start = time.perf_counter()

    while time.perf_counter() - start < duration:
        p = nvmlDeviceGetPowerUsage(handle) / 1000.0
        readings.append(p)
        time.sleep(0.1)

    avg = sum(readings) / len(readings)
    print(f"Idle Power = {avg:.2f} W\n")
    return avg


# ============================================================
# SINGLE INFERENCE MEASUREMENT
# ============================================================

def measure_once(prompt, model, tokenizer, handle, idle_power, sample_rate=0.1, max_new_tokens=100):

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    sampler = PowerSampler(handle, interval=sample_rate)
    sampler.start()

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,              # deterministic
            temperature=0.0,              # deterministic
            top_p=1.0,
            top_k=0,
            use_cache=False,              # clears KV cache every call
            eos_token_id=None,            # NEVER stop early
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    sampler.stop()

    total_energy = sampler.compute_energy()
    time_taken = end - start
    marginal_energy = max(0, total_energy - idle_power * time_taken)

    tokens_in = inputs.input_ids.shape[-1]
    tokens_out = max(output.shape[-1] - tokens_in, 1)

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "answer": answer,
        "tokens_generated": tokens_out,
        "time": round(time_taken, 8),
        "energy_total": round(total_energy, 8),
        "energy_marginal": round(marginal_energy, 8),
        "energy_per_token": round(marginal_energy / tokens_out, 8),
        "latency_per_token": round(time_taken / tokens_out, 8),
        "avg_power": round(total_energy / time_taken, 8),
        "vram_peak": round(torch.cuda.max_memory_allocated() / 1e9, 8)
    }


# ============================================================
# RUN 10 REPEATS PER QUESTION
# ============================================================

def evaluate_question(prompt, model, tokenizer, handle, idle_power):

    results = []
    for i in range(10):
        print(f"  Run {i+1}/5...")
        r = measure_once(prompt, model, tokenizer, handle, idle_power)
        results.append(r)

        print("  Cooling down for 10s...\n")
        time.sleep(10)

    avg = {
        "answer": results[0]["answer"],
        "tokens_generated": results[0]["tokens_generated"],
        "time": round(sum(r["time"] for r in results) / 5, 8),
        "energy_total": round(sum(r["energy_total"] for r in results) / 5, 8),
        "energy_marginal": round(sum(r["energy_marginal"] for r in results) / 5, 8),
        "energy_per_token": round(sum(r["energy_per_token"] for r in results) / 5, 8),
        "latency_per_token": round(sum(r["latency_per_token"] for r in results) / 5, 8),
        "avg_power": round(sum(r["avg_power"] for r in results) / 5, 8),
        "vram_peak": round(sum(r["vram_peak"] for r in results) / 5, 8),
    }

    return avg


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_experiment(input_csv, output_csv, model_name):

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

    # ============================================================
    # Load tokenizer and model fresh
    # ============================================================
    print("Loading tokenizer...")

    

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
        trust_remote_code=True,
        use_safetensors=True
    )

    # ============================================================
    # GPU power tools
    # ============================================================
    handle = init_nvml()
    idle_power = measure_idle_power(handle)

    df = pd.read_csv(input_csv)
    results = []

    for idx, row in df.iterrows():
        print(f"\n============================================")
        print(f"Processing {row['id']}")

        reset_between_questions(model)

        avg = evaluate_question(
            row["question"],
            model,
            tokenizer,
            handle,
            idle_power
        )

        results.append({
            "dataset": row["dataset"],
            "id": row["id"],
            "question": row["question"],
            **avg
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    print(f"\nSaved results → {output_csv}")

    nvmlShutdown()



# ============================================================
# EXECUTION
# ============================================================

run_experiment(
    input_csv="input.csv",   #input dataset
    output_csv="results-modelname-7b.csv",  #output file name
    model_name="modelname-7b"  #write model name here
)

