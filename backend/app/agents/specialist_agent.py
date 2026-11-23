import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.agents.base import Agent
from app.config import settings


class SpecialistAgent(Agent):
    """Run the fine-tuned Llama adapter locally on the CUDA GPU."""

    name = "Specialist Agent"
    color = Agent.RED

    BASE_MODEL = "meta-llama/Llama-3.2-3B"
    FINETUNED_MODEL = "ed-donner/price-2025-11-30_15.10.55-lite"
    ADAPTER_REVISION = "259f6f974c511cd87c0b7af78e894e0507a87ecf"
    PREFIX = "Price is $"
    QUESTION = "What does this cost to the nearest dollar?"

    def __init__(self):
        """Load the base model and adapter once when the ensemble is initialized."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The local Specialist Agent requires a CUDA GPU. "
                "Install a CUDA-enabled PyTorch build and check the NVIDIA driver."
            )

        self.device = torch.device("cuda:0")
        token = (
            settings.hf_token.get_secret_value()
            if settings.hf_token is not None
            else None
        )
        self.log(f"Loading {self.BASE_MODEL} and its fine-tuned adapter on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL,
            torch_dtype=torch.float16,
            device_map={"": self.device.index or 0},
            low_cpu_mem_usage=True,
            token=token,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            self.FINETUNED_MODEL,
            revision=self.ADAPTER_REVISION,
            token=token,
        )
        self.model.eval()
        self.log("Specialist Agent is ready on CUDA")

    def price(self, description: str) -> float:
        """Generate a price estimate locally with the fine-tuned model."""
        prompt = f"{self.QUESTION}\n\n{description}\n\n{self.PREFIX}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        price_text = decoded.rsplit(self.PREFIX, maxsplit=1)[-1].replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+", price_text)
        result = float(match.group()) if match else 0.0
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
