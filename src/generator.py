from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from langchain.llms import HuggingFacePipeline

class MedicalLLM:
    def __init__(self, base_model_name, adapter_path):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        self.pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, max_new_tokens=512)
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def get_llm(self):
        return self.llm
