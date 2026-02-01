import asyncio
import json
import argparse
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

async def process_item(client, item, args, semaphore):
    # Determine input text
    raw_text = item.get("prompt") or item.get("text") or ""
    
    # Simple Mode switching
    if args.mode == "think":
        # Qwen specific thinking trigger if supported
        final_prompt = f"{raw_text}" 
        # Note: Actual "thinking" usually requires specific chat template handling
        # For base processing, we just send the prompt.
    else:
        final_prompt = raw_text

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body={"top_k": args.top_k, "min_p": 0.05}, # min_p added for Qwen stability
                max_tokens=args.max_tokens,
                timeout=300
            )
            item["response"] = response.choices[0].message.content
            item["finish_reason"] = response.choices[0].finish_reason
        except Exception as e:
            item["error"] = str(e)
            item["response"] = None
        return item

async def run_batch(args):
    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm")
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # 1. Resume Logic
    processed_prompts = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    # Use a hash or exact string match of prompt to identify done items
                    if 'prompt' in res and 'response' in res:
                        processed_prompts.add(res['prompt'])
                except: continue
        print(f"Resuming: Found {len(processed_prompts)} items already processed.")

    # 2. Load Data
    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line:
                item = json.loads(clean_line)
                if item.get('prompt') not in processed_prompts:
                    data.append(item)

    if not data:
        print("No new items to process.")
        return

    # 3. Process
    print(f"Processing {len(data)} items...")
    tasks = [process_item(client, item, args, semaphore) for item in data]
    
    with open(args.output, 'a', encoding='utf-8') as f:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inference"):
            result = await future
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", type=str, default="no_think")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=100) # vLLM can handle high concurrency
    parser.add_argument("--temperature", type=float, default=0.1) # Low temp for cleaning tasks
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    
    args = parser.parse_args()
    asyncio.run(run_batch(args))
