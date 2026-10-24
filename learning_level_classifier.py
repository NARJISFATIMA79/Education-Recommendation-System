
# import re

# from transformers import pipeline


# MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

# _classifier = None

# CANDIDATE_LABELS = [
#     "beginner-level educational instruction",
#     "intermediate-level educational instruction",
#     "advanced-level educational instruction",
# ]

# LEVEL_NAMES = {
#     "beginner-level educational instruction": "Beginner",
#     "intermediate-level educational instruction": "Intermediate",
#     "advanced-level educational instruction": "Advanced",
# }


# def get_classifier():
#     """Load the local NLP model once per program run."""
#     global _classifier

#     if _classifier is None:
#         _classifier = pipeline(
#             "zero-shot-classification",
#             model=MODEL_NAME,
#             device=-1,
#         )

#     return _classifier


# def split_transcript(text, max_words=100):
#     """Divide transcript text into manageable chunks."""
#     words = re.findall(r"\S+", text)

#     return [
#         " ".join(words[i:i + max_words])
#         for i in range(0, len(words), max_words)
#     ]


# def score_text(text, model):
#     """Return the model's relative scores for the three levels."""
#     result = model(
#         text,
#         candidate_labels=CANDIDATE_LABELS,
#         hypothesis_template="This transcript is an example of {}.",
#         multi_label=False,
#     )

#     return dict(zip(result["labels"], result["scores"]))


# def classify_transcript(
#     transcript,
#     transcript_language=None,
#     title="",
#     description="",
# ):
#     """
#     Estimate learning level using transcript, title, and description.

#     The combined score is a provisional ranking score, not a
#     calibrated probability of the video's true learning level.
#     """

#     transcript = (transcript or "").strip()
#     title = (title or "").strip()
#     description = (description or "").strip()
#     language = (transcript_language or "").lower()

#     model = get_classifier()

#     source_scores = {}
#     source_weights = {}

#     # 1. Transcript evidence: main source.
#     if transcript and (
#         not language or "english" in language
#     ):
#         chunks = split_transcript(transcript)

#         # Sample at most 12 chunks to limit processing time.
#         max_chunks = 12

#         if len(chunks) > max_chunks:
#             indexes = [
#                 round(i * (len(chunks) - 1) / (max_chunks - 1))
#                 for i in range(max_chunks)
#             ]
#             chunks = [chunks[i] for i in indexes]

#         if chunks:
#             chunk_scores = [
#                 score_text(chunk, model)
#                 for chunk in chunks
#             ]

#             source_scores["transcript"] = {
#                 label: sum(scores[label] for scores in chunk_scores)
#                 / len(chunk_scores)
#                 for label in CANDIDATE_LABELS
#             }
#             source_weights["transcript"] = 0.70

#     # 2. Title evidence: supporting source.
#     if title:
#         source_scores["title"] = score_text(title, model)
#         source_weights["title"] = 0.20

#     # 3. Description evidence: supporting source.
#     if description:
#         # Limit processing time for very long descriptions.
#         description_text = description[:2000]
#         source_scores["description"] = score_text(
#             description_text, model
#         )
#         source_weights["description"] = 0.10

#     if not source_scores:
#         return {
#             "learning_level": "Unknown",
#             "learning_level_score": None,
#             "learning_level_reason": (
#                 "No usable transcript, title, or description "
#                 "was available."
#             ),
#         }

#     # Normalize weights when one or more sources are unavailable.
#     total_weight = sum(source_weights.values())

#     combined_scores = {
#         label: sum(
#             source_scores[source][label] * weight
#             for source, weight in source_weights.items()
#         ) / total_weight
#         for label in CANDIDATE_LABELS
#     }

#     best_label = max(combined_scores, key=combined_scores.get)
#     best_score = float(combined_scores[best_label])

#     # This is a provisional cutoff and must be validated.
#     if best_score < 0.45:
#         level = "Unknown"
#         reason = (
#             "The combined model score did not meet the provisional "
#             "cutoff. Validate this cutoff against manually labelled "
#             "videos."
#         )
#     else:
#         level = LEVEL_NAMES[best_label]

#         sources_used = ", ".join(source_scores.keys())

#         reason = (
#             f"Provisional {level} estimate using: {sources_used}. "
#             "Transcript evidence receives the highest weight when "
#             "an English transcript is available."
#         )

#     return {
#         "learning_level": level,
#         "learning_level_score": round(best_score, 4),
#         "learning_level_reason": reason,
#     }
