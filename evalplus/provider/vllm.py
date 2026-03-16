from typing import List, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class VllmDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        force_base_prompt: bool = False,
        tensor_parallel_size: int = 1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gguf_file: str = None,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "enable_prefix_caching": enable_prefix_caching,
            "enable_chunked_prefill": enable_chunked_prefill,
        }

        self.force_base_prompt = force_base_prompt
        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        tokenizer_kwargs = {}
        if gguf_file is not None:
            tokenizer_kwargs["gguf_file"] = gguf_file
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, **tokenizer_kwargs)
        if self.is_direct_completion():
            self.eos += extra_eos_for_direct_completion(dataset)
        else:
            self.eos += ["\n```\n"]
        self.llm = LLM(model=name, max_model_len=2048, **kwargs)

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    def _format_prompt(self, prompt: str) -> str:
        """Format a single prompt based on model type."""
        if self.is_direct_completion():
            return prompt
        return make_raw_chat_prompt(
            prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
        )

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        """Generate code for a single prompt."""
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        prompt = self._format_prompt(prompt)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs

    def codegen_batch(
        self, 
        prompts: List[Tuple[str, int]], 
        do_sample: bool = True
    ) -> List[List[str]]:
        """Generate code for multiple prompts using vLLM's continuous batching.
        
        This method prepares all prompts at once and lets vLLM optimize
        the scheduling across different prompts.
        
        Args:
            prompts: List of (prompt, num_samples) tuples
            do_sample: Whether to use sampling
            
        Returns:
            List of lists of generated code strings, one list per prompt
        """
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        
        # Build the list of all prompts to generate
        # Format: [prompt1, prompt1, ..., prompt2, prompt2, ...]
        all_formatted_prompts = []
        prompt_indices = []  # Track which original prompt each item belongs to
        
        for idx, (prompt, num_samples) in enumerate(prompts):
            formatted = self._format_prompt(prompt)
            all_formatted_prompts.extend([formatted] * num_samples)
            prompt_indices.extend([idx] * num_samples)
        
        if not all_formatted_prompts:
            return [[] for _ in prompts]
        
        # Generate all at once - vLLM will use continuous batching
        vllm_outputs = self.llm.generate(
            all_formatted_prompts,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )
        
        # Group outputs by original prompt
        results = [[] for _ in prompts]
        for output_idx, vllm_output in enumerate(vllm_outputs):
            prompt_idx = prompt_indices[output_idx]
            gen_text = vllm_output.outputs[0].text.replace("\t", "    ")
            results[prompt_idx].append(gen_text)
        
        return results
