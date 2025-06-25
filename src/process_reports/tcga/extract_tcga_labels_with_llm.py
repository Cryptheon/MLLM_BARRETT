import argparse
import torch
import json
import pandas as pd
import os
from llama_cpp import Llama
from utils.util_functions import load_config


def load_tcga_labels(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_prompt(prompt_template, report):
    return [
        {"role": "system", "content": "You are a pathology assistant that extracts standardized cancer types from clinical reports."},
        {"role": "user", "content": prompt_template.format(report=report)}
    ]


def extract_label(model, report, prompt_template, config):
    messages = get_prompt(prompt_template, report)
    output = model.create_chat_completion(
        messages=messages,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repeat_penalty=config.repetition_penalty,
        stream=False
    )
    return output['choices'][0]['message']['content'].strip()


def main():
    parser = argparse.ArgumentParser(description="Extract cancer types from reports using a language model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True, help="JSON file with [{'patient_id': ..., 'generated_report': ...}]")
    parser.add_argument("--tcga_json", type=str, required=True, help="Path to TCGA label json")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["LLAMA_VISIBLE_DEVICES"] = str(args.gpu)
    config = load_config(args.config)

    model = Llama(model_path=config.model_path,
                  n_gpu_layers=config.n_gpu_layers,
                  flash_attn=True,
                  n_ctx=8192*2,
                  chat_format=config.chat_format)

    with open(args.prompt_path, "r") as f:
        prompt_template = f.read()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    tcga_labels = load_tcga_labels(args.tcga_json)
    label_set = set(label.lower() for label in tcga_labels.values())

    output_data = []
    for entry in data:
        report = entry["generated_report"]
        pid = entry["patient_id"]

        response = extract_label(model, report, prompt_template, config)
        normalized_response = response.lower().strip()

        match = next((tcga for tcga, full in tcga_labels.items() if full.lower() in normalized_response), "UNKNOWN")

        output_data.append({
            "patient_id": pid,
            "extracted_label": response,
            "tcga_code": match
        })
        print(f"{pid} => {response} [{match}]")

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved output to {args.output_json}")


if __name__ == "__main__":
    main()

