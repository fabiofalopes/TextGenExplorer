# HF 

## OpenVino
1. Load a text generation model from Hugging Face's Transformers library (specifically, the TinyLlama model)
2. Tokenize a given text input using the model's tokenizer.
3. Convert the loaded model to OpenVINO IR format for inference on Intel hardware.
4. Perform inference with the converted OpenVINO model.
5. Print the decoded output from the inference.

### Reads
[reddit1](https://www.reddit.com/r/LocalLLaMA/comments/17gf824/llamacpp_is_slower_on_iris_xe_gpu_than_on_cpu/)

[llamaHF](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)

[TinyLlama Repo](https://github.com/jzhang38/TinyLlama)

[TinyLlama HF Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

[Hugging Face Model Hub with OpenVINO](https://docs.openvino.ai/2023.2/notebooks/124-hugging-face-hub-with-output.html)

[Hugging Face Model Hub with OpenVINO -> Requirements](https://docs.openvino.ai/2023.2/notebooks/124-hugging-face-hub-with-output.html#installing-requirements)

[Twitter-roBERTa-base for Sentiment Analysis](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)