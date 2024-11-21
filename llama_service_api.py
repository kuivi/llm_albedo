# llama_service_api.py
"""
This module provides an API endpoint for generating chat completions using a language model.
It uses FastAPI to expose a /v1/chat/completions endpoint for handling chat messages and returning generated responses.
The API supports both streaming and non-streaming outputs, allowing real-time interaction or full-text responses. 
It leverages a pre-trained model from Hugging Face, optimized for GPU usage, to process conversation prompts and generate completions.
The file is designed for deployment in high-performance environments, such as Albedo, using GPU acceleration.
"""
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import asyncio
from starlette.responses import StreamingResponse
import json


app = FastAPI()

model_path = "./model/models--BSCES-LLMS--Gemma-7B-FT/snapshots/2a890271ae6a0cf58da22925ff73559aa98ee9f8/"


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# Load the model with device_map
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"  # Requires accelerate
)



model.eval()

# Define the /v1/chat/completions endpoint
@app.post("/v1/chat/completions")
"""
Endpoint to handle chat completions.
This endpoint receives a POST request with a JSON body containing chat messages and various parameters for text generation.
It processes the messages to create a prompt, generates a response using a language model, and returns the generated text.
Args:
    request (Request): The incoming HTTP request containing the JSON body with chat messages and parameters.
Returns:
    StreamingResponse or dict: If streaming is enabled, returns a StreamingResponse that streams the generated text.
                               Otherwise, returns a dictionary with the generated text.
JSON Body Parameters:
    messages (list): A list of message objects, each containing a "role" (e.g., "user" or "assistant") and "content".
    stream (bool): Whether to stream the response. Defaults to False.
    max_tokens (int): The maximum number of tokens to generate. Defaults to 256.
    temperature (float): Sampling temperature for text generation. Defaults to 0.7.
    top_p (float): Nucleus sampling probability for text generation. Defaults to 1.0.
    n (int): The number of response sequences to generate. Defaults to 1.
    stop (list or None): A list of stop tokens to end the generation. Defaults to None.
Example:
    POST /v1/chat/completions
    {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"}
        ],
        "stream": false,
        "max_tokens": 150,
        "temperature": 0.8,
        "top_p": 0.9,
        "n": 1,
        "stop": ["\n"]
    Response:
    {
                "content": "I'm here to assist you with any questions you have."
        "model": "custom-model"
"""
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    n = body.get("n", 1)
    stop = body.get("stop", None)
    
    # Extract the conversation so far
    prompt = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
        else:
            prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant:"

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Generate text
    generate_kwargs = {
        "max_length": inputs.input_ids.shape[1] + max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": n,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "no_repeat_ngram_size": 2,
    }

    if stop is not None:
        # Implement stop tokens if necessary
        pass

    if stream:
        # Stream the response
        async def generate():
            with torch.no_grad():
                output_ids = inputs.input_ids
                for _ in range(max_tokens):
                    outputs = model(output_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    output_ids = torch.cat([output_ids, next_token], dim=-1)
                    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    delta = generated_text[len(prompt):]
                    if delta:
                        data = {
                            "choices": [{
                                "delta": {"content": delta},
                                "index": 0,
                                "finish_reason": None
                            }],
                            "model": body.get("model", "custom-model"),
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0.01)  # Slight delay to simulate streaming
                # Signal the end of the stream
                data = {
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    "model": body.get("model", "custom-model"),
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Return the full response
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated_text[len(prompt):]
            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": completion
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "model": body.get("model", "custom-model"),
            }
            return response