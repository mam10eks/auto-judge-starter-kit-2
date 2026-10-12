from autojudge_base import Leaderboard, LeaderboardBuilder, LeaderboardSpec, MeasureSpec
from sentence_transformers import SentenceTransformer
from statistics import mean

import time
import torch

BERT_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("SIM_SCORE", description="Average Dot Product similarity between topic and individual sentences"),
))

class GemmaJudge:
    def evaluate(self, model, topic, sentences):
        if not topic or not sentences:
            return 0.0

        # Encode the problem statement
        base_embed = model.encode(
            [topic],
            prompt_name='QuestionAnswering',
            convert_to_tensor=True,
            #normalize_embeddings=True
        )

        # Encode the sentences individually
        sent_embed = model.encode(
            sentences,
            prompt_name="Document",
            convert_to_tensor=True,
            #normalize_embeddings=True
        )

        # Normalized dot prod same as cosine sim
        sims = base_embed @ sent_embed.T;

        return sims.mean().item()

    def judge(self, rag_responses, rag_topics, llm_config, **kwargs) -> Leaderboard:
        builder = LeaderboardBuilder(BERT_SPEC)

        topics: Dict[str, str] = {t.request_id: t.problem_statement or t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topics.keys())

        model = SentenceTransformer(
            'google/embeddinggemma-2',
            device='cuda',
        )

        for response in rag_responses:
            topic = topics.get(response.metadata.topic_id)
            sentences = [s.text for s in response.responses] if response.responses else []

            score = self.evaluate(model, topic, sentences) 

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                values={"SIM_SCORE": score},
            )
        topic_ids = [t.request_id for t in rag_topics]
        return builder.build(expected_topic_ids=topic_ids, on_missing="fix_aggregate")
