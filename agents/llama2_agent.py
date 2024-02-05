import os
from typing import Tuple

from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline

from .agent import Agent

HUGGINGFACE_MODEL_NAME = "AhmadPython/llama2_textual_entailment_graph"

class LLama2Agent(Agent):
    '''
    The LLama2Agent is a finetuned on a GBRL dataset, which was put together from various sources of
    entailment tasks and common sense, legal, ethics and natural science reasoning.
    '''

    def __init__(self, log_file = None, feedback_file = None, layout = "planar", font_size = 12, node_size = 300, access_token = None):
        '''
        The access token to the model is added to the parameter list.
        If None, the access token needs to be passed as an environment variable.
        With the access token, the huggingface client will be logged in and gives access to the model.
        Parameters
        __________
        access_token:   str (optional)
                        An access token for read acces on my huggingface hub.
        '''
        super(LLama2Agent, self).__init__(log_file, feedback_file, layout, font_size, node_size)
        access_token
        if not access_token:
            access_token = os.environ['HUGGINGFACE_ACCESS_TOKEN']                 #raise KeyError if no HUGGINGFACE_ACCESS_TOKEN environ set.
        login(access_token)
        self.pipeline = self._load_generator()

    def _load_generator(self):
        '''
        Loads the LLama2 model, that was trained on base of the DeepSpeed example application "DeepSpeed-Chat", which suggests the following
        initialization on inference.
        see https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/inference/chatbot.py
        '''
        tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAME, fast_tokenizer = True)
        tokenizer.pad_token = tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(HUGGINGFACE_MODEL_NAME)
        model_class = AutoModelForCausalLM.from_config(model_config)
        model = model_class.from_pretrained(HUGGINGFACE_MODEL_NAME, from_tf=bool(".ckpt" in HUGGINGFACE_MODEL_NAME), config=model_config)
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
        generator = pipeline("text-generation",
                            model=model,
                            tokenizer=tokenizer)
        return generator
    

    def _get_response(self, query, **kwargs) -> Tuple[str,str]:
        '''
        The response is produced by the fine tuned LLama2.
        There is a high chance, the model does not respond in the expected syntax.
        The user can control the max new tokens produced by the model.

        Parameters
        __________
        max_new_tokens:         int (default 128)
                                Sets the max new tokens produced by the model.
        '''
        max_new_tokens = kwargs["max_new_tokens"] if "max_new_tokens" in kwargs else 128
        response = self.pipeline(query, max_new_tokens=max_new_tokens)
        actual_response = str(response[0]["generated_text"])
        clean_response = actual_response[len(query):].replace("<|endoftext|></s>", "")
        if "<;>" in clean_response:
            clean_response = clean_response[:clean_response.index("<;>")+3]
        return clean_response, actual_response