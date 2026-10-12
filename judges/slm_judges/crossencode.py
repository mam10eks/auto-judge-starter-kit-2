from autojudge_base import Leaderboard, LeaderboardBuilder, LeaderboardSpec, MeasureSpec
from sentence_transformers import CrossEncoder 
from statistics import mean

import time
import torch

CROSS_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("PRECISION", description="Estimated precision for all results"),
    MeasureSpec("PRECISION_AT_K", description="Estimated precision for top K results"),
    MeasureSpec("MEAN_PRECISION", description="Mean of PRECISION & PRECISION_AT_K"),
    MeasureSpec("NUM_RESULTS", description="Number of sentences/citations"),
    MeasureSpec("MEAN_SIGMOID", description="Mean of raw scores from sigmoid"),
))

K = 10                      # How many results to evaluate for SIM_AT_K_SCORE
RELEVANCE_THRESHOLD = 0.5   # Threshold for relevance docs between 0-1
BATCH_SIZE = 64

# todo: maybe swap out model/eval for earlier benchmark test if time, the scores for rag26 are very low with msmarco ce
class CrossencoderJudge:
    # This attempts to get a precision score instead of just similarity
    def evaluate(self, model, topic, sentences):
        if not topic or not sentences:
            return {
                "PRECISION": 0.0,
                "PRECISION_AT_K": 0.0,
                "MEAN_PRECISION": 0.0,
                "NUM_RESULTS": 0,
                "MEAN_SIGMOID": 0
            }

        # Score each (topic, sentence) pair jointly; sigmoid maps logits to 0-1
        scores = model.predict(
            [(topic, s) for s in sentences],
            activation_fn=torch.nn.Sigmoid(),
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        # Keep original response order so "top K" means the first K the run returned
        relevant = [float(s > RELEVANCE_THRESHOLD) for s in scores]

        precision = mean(relevant)
        precision_at_k = mean(relevant[:K])

        return {
            "PRECISION": precision,
            "PRECISION_AT_K": precision_at_k,
            "MEAN_PRECISION": (precision + precision_at_k) / 2,
            "NUM_RESULTS": len(sentences),
            "MEAN_SIGMOID": mean(scores)
        }

    def judge(self, rag_responses, rag_topics, llm_config, **kwargs) -> Leaderboard:
        builder = LeaderboardBuilder(CROSS_SPEC)

        topics: Dict[str, str] = {t.request_id: t.problem_statement or t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topics.keys())

        model = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            device='cuda',
        )

        for response in rag_responses:
            topic = topics.get(response.metadata.topic_id)
            sentences = [s.text for s in response.responses] if response.responses else []

            values = self.evaluate(model, topic, sentences) 

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                values=values
            )
        topic_ids = [t.request_id for t in rag_topics]
        return builder.build(expected_topic_ids=topic_ids, on_missing="fix_aggregate")
