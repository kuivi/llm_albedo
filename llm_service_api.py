# llama_service_api.py
"""
Universal LLM API Server supporting multiple model families (Gemma, Llama, DeepSeek)
"""
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import asyncio
from starlette.responses import StreamingResponse
import json
import argparse
import os


app = FastAPI()

# Global variables for model and tokenizer
model = None
tokenizer = None
model_family = None

def load_model(model_path, model_type):
    """Load the model and tokenizer based on the provided path"""
    global model, tokenizer, model_family
    
    print(f"Loading {model_type} model from {model_path}")
    model_family = model_type.lower()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    # Make sure tokenizer has pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    print(f"{model_type} model loaded successfully")

def format_prompt(messages, model_type):
    """Format messages based on the model family"""
    prompt = ""
    
    if model_type == "deepseek":
        # DeepSeek format
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"
        
    elif model_type == "llama":
        # Llama format
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                prompt += f"<|system|>\n{content}\n<|end|>\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n<|end|>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n<|end|>\n"
        prompt += "<|assistant|>\n"
        
    else:
        # Generic format (works for Gemma and others)
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        
    return prompt

# Define the /v1/chat/completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Endpoint to handle chat completions for various model types.
    """
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    
    # Format prompt based on model family
    prompt = format_prompt(messages, model_family)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    # Generate text
    generate_kwargs = {
        "max_length": inputs.input_ids.shape[1] + max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if stream:
        # Stream the response
        async def generate():
            with torch.no_grad():
                output_ids = inputs.input_ids
                for _ in range(max_tokens):
                    outputs = model(output_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature and top_p sampling
                    if temperature > 0:
                        scaled_logits = next_token_logits / temperature
                        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                        if top_p < 1.0:
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                            )
                            probs = probs.masked_fill(indices_to_remove, 0.0)
                            probs = probs / probs.sum(dim=-1, keepdim=True)
                            
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy decoding
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    output_ids = torch.cat([output_ids, next_token], dim=-1)
                    
                    # Check if EOS token was generated
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    # Decode and stream new token
                    new_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    
                    if new_text:
                        data = {
                            "choices": [{
                                "delta": {"content": new_text},
                                "index": 0,
                                "finish_reason": None
                            }],
                            "model": body.get("model", "local-model"),
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                
                # End of generation
                data = {
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    "model": body.get("model", "local-model"),
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Return the full response
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "model": body.get("model", "local-model"),
            }
            return response

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM API Server")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the local model directory")
    parser.add_argument("--model_type", type=str, default="generic",
                        choices=["gemma", "llama", "deepseek", "generic"],
                        help="Model family type (determines prompt formatting)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    
    args = parser.parse_args()
    
    # Load the model
    load_model(args.model_path, args.model_type)
    
    # Start the server
    print(f"Starting API server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)