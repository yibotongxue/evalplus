from abc import ABC, abstractmethod
from typing import List, Tuple

from evalplus.provider.utility import EOS


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 768,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = False,
        instruction_prefix: str = None,
        response_prefix: str = None,
        **kwargs,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        """Generate code for a single prompt.
        
        Args:
            prompt: The input prompt
            do_sample: Whether to use sampling
            num_samples: Number of samples to generate
            
        Returns:
            List of generated code strings
        """
        pass

    def codegen_batch(
        self, 
        prompts: List[Tuple[str, int]], 
        do_sample: bool = True
    ) -> List[List[str]]:
        """Generate code for multiple prompts in batch.
        
        This is the preferred method for bulk generation as it allows
        the backend to optimize scheduling across different prompts.
        
        Args:
            prompts: List of (prompt, num_samples) tuples
            do_sample: Whether to use sampling
            
        Returns:
            List of lists of generated code strings, one list per prompt
        """
        # Default implementation: fall back to single prompt generation
        results = []
        for prompt, num_samples in prompts:
            outputs = self.codegen(prompt, do_sample=do_sample, num_samples=num_samples)
            results.append(outputs)
        return results

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
