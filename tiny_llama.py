from transformers import pipeline
import torch
import itertools
import threading
import time
import sys
import datetime


def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f'Task time = {difference} \n')
    sys.stdout.write('\rDone!     \n')


pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

model_description = {
    "role": "system",
    "content": ""
    }
user_content = {
    "role": "user",
    "content": ""
}

#print(model_description)
#print(user_content)

model_description["content"] = input("Describe the model you want: ")

input_user_content = "-"

messages = []

messages.append(model_description)

while input_user_content:
    time.sleep(1)
    input_user_content = input("User Prompt: ")
    if not input_user_content:
        break
    user_content["content"] = input_user_content 
    #messages.append(
    #    {
    #        "role":"user",
    #        "content":input_user_content
    #   })
    messages.append(user_content)

    print(user_content)

    done = False
    t = threading.Thread(target=animate)
    job_start_time = datetime.datetime.now() # storing the current time in the variable
    t.start() # Start the thread

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    # <|system|>
    # You are a friendly chatbot who always responds in the style of a pirate.</s>
    # <|user|>
    # How many helicopters can a human eat in one sitting?</s>
    # <|assistant|>
    # ...

    job_finish_time = datetime.datetime.now()
    difference = job_finish_time - job_start_time
    # t.join() # Wait for the thread to finish executing
    done = True

