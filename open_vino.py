# https://docs.openvino.ai/2023.2/notebooks/124-hugging-face-hub-with-output.html#installing-requirements

import ipywidgets as widgets
import openvino as ov

from pathlib import Path

import numpy as np
import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import itertools
import sys
import time
import threading
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

## 2. Initializing a Model Using the HF Transformers Package

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL, return_dict=True)
# The torchscript=True flag is used to ensure the model outputs are tuples
# instead of ModelOutput (which causes JIT errors).
model = AutoModelForSequenceClassification.from_pretrained(MODEL, torchscript=True)

## 2.

### 3. Original Model inference

text = "HF models run perfectly with OpenVINO!"

encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0]
scores = torch.softmax(scores, dim=0).numpy(force=True)

def print_prediction(scores):
    for i, descending_index in enumerate(scores.argsort()[::-1]):
        label = model.config.id2label[descending_index]
        score = np.round(float(scores[descending_index]), 4)
        print(f"{i+1}) {label} {score}")

#print_prediction(scores)
### 3.
        
#### 4. Converting the Model to OpenVINO IR format¶

save_model_path = Path('./models/model.xml')

if not save_model_path.exists():
    ov_model = ov.convert_model(model, example_input=dict(encoded_input))
    ov.save_model(ov_model, save_model_path)

#### 4.

##### 5. Converted Model Inference¶

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

print(device)

##### 5.

done = False
t = threading.Thread(target=animate)
job_start_time = datetime.datetime.now() # storing the current time in the variable
t.start() # Start the thread

###### 6. OpenVINO model IR must be compiled for a specific device prior to the model inference.

compiled_model = core.compile_model(save_model_path, device.value)

# Compiled model call is performed using the same parameters as for the original model
scores_ov = compiled_model(encoded_input.data)[0]

scores_ov = torch.softmax(torch.tensor(scores_ov[0]), dim=0).detach().numpy()

print_prediction(scores_ov)

###### 6.

job_finish_time = datetime.datetime.now()
difference = job_finish_time - job_start_time
# t.join() # Wait for the thread to finish executing
done = True

