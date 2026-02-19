import asyncio
import logging
import os
from typing import Dict, List

import openai
from dotenv import load_dotenv

load_dotenv()

from openevolve import OpenEvolve
from openevolve.config import load_config
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class ResponsesLLM(LLMInterface):
    """LLM interface using OpenAI Responses API."""

    def __init__(self, model_cfg):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        self.client = openai.OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.retries or 0,
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        input_messages = []
        if system_message:
            input_messages.append({"role": "system", "content": system_message})
        input_messages.extend(messages)

        params = {
            "model": self.model,
            "input": input_messages,
        }

        if self.max_tokens:
            params["max_output_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        reasoning_prefixes = ("o1", "o3", "o4", "gpt-5", "gpt-oss")
        is_reasoning = self.model.lower().startswith(reasoning_prefixes)

        if self.temperature is not None and not is_reasoning:
            params["temperature"] = kwargs.get("temperature", self.temperature)

        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}

        retries = kwargs.get("retries", self.retries) or 0
        retry_delay = kwargs.get("retry_delay", self.retry_delay) or 5

        for attempt in range(retries + 1):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.responses.create(**params)
                )
                return response.output_text
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed: {e}")
                    raise


def create_responses_llm(model_cfg):
    return ResponsesLLM(model_cfg)


async def main():
    config = load_config("config.yaml")

    for model in config.llm.models:
        model.init_client = create_responses_llm
    for model in config.llm.evaluator_models:
        model.init_client = create_responses_llm

    openevolve = OpenEvolve(
        initial_program_path="initial_program.py",
        evaluation_file="evaluator.py",
        config=config,
    )

    best_program, best_score = await openevolve.run()
    print(f"\nBest score: {best_score}")
    print(f"Best program saved to: {openevolve.output_dir}/best_program.py")


if __name__ == "__main__":
    asyncio.run(main())
