from pathlib import Path
from typing import Generator, List, Union
from hayhooks import streaming_generator
from haystack import Pipeline
from hayhooks.server.pipelines.utils import get_last_user_message
from hayhooks.server.utils.base_pipeline_wrapper import BasePipelineWrapper
from hayhooks.server.logger import log


URLS = [
    "https://haystack.deepset.ai",
    "https://www.redis.io",
    "https://ssi.inc",
    "https://www.deepset.ai",
]


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        pipeline_yaml = (Path(__file__).parent / "chat_with_website.yml").read_text()
        self.pipeline = Pipeline.loads(pipeline_yaml)

    def run_api(self, urls: List[str], question: str) -> str:
        """
        Ask a question to a list of websites.

        Args:
            urls (List[str]): List of URLs to ask the question to.
            question (str): Question to ask the websites.

        Returns:
            str: Answer to the question.
        """
        result = self.pipeline.run(
            {"fetcher": {"urls": urls}, "prompt": {"query": question}}
        )

        # result["llm"]["replies"][0] is a ChatMessage instance
        return result["llm"]["replies"][0].text

    def run_chat_completion(
        self, model: str, messages: list[dict], body: dict
    ) -> Union[str, Generator]:
        log.trace(
            f"Running pipeline with model: {model}, messages: {messages}, body: {body}"
        )

        question = get_last_user_message(messages)
        log.trace(f"Question: {question}")

        # Streaming pipeline run, will return a generator
        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "fetcher": {"urls": URLS},
                "prompt": {"query": question},
            },
        )
