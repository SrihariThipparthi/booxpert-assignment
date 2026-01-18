import json
import logging
import os

import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


class NameMatcher:
    def __init__(self):
        self.names = self._load_names()

        logger.info("Loading semantic similarity model...")
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu"
        )

        logger.info("Pre-computing name embeddings...")
        self.name_embeddings = self.model.encode(
            self.names, convert_to_tensor=True, show_progress_bar=False
        )
        logger.info(f"Loaded {len(self.names)} names with embeddings")

    def _load_names(self):
        data_path = os.path.join(os.path.dirname(__file__), "data", "names.json")

        if not os.path.exists(data_path):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            default_names = self._get_default_names()
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump({"names": default_names}, f, indent=2, ensure_ascii=False)
            return default_names

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
            return data["names"]

    def _get_default_names(self):
        return [
            "Geetha",
            "Geeta",
            "Gita",
            "Gitu",
            "Gitanjali",
            "Priya",
            "Priyanka",
            "Priyam",
            "Priyanshu",
            "Amit",
            "Amitabh",
            "Amith",
            "Amita",
            "Rajesh",
            "Raja",
            "Raju",
            "Rajeev",
            "Rajiv",
            "Suresh",
            "Suri",
            "Suraj",
            "Surya",
            "Mohammed",
            "Mohammad",
            "Muhammad",
            "Muhammed",
            "Aisha",
            "Ayesha",
            "Aysha",
            "Aiesha",
            "Krishna",
            "Krish",
            "Krishnan",
            "Karthik",
            "Kartik",
            "Lakshmi",
            "Laxmi",
            "Lakshman",
            "Deepak",
            "Dipak",
            "Deepika",
            "Dipika",
            "Sandeep",
            "Sandip",
            "Sanjay",
            "Sanjiv",
            "Ramesh",
            "Ram",
            "Raman",
            "Rama",
        ]

    def find_similar_names(self, input_name, top_k=5):
        input_embedding = self.model.encode(input_name, convert_to_tensor=True)

        semantic_scores = (
            util.cos_sim(input_embedding, self.name_embeddings)[0].cpu().numpy()
        )

        fuzzy_scores = np.array(
            [
                fuzz.ratio(input_name.lower(), name.lower()) / 100.0
                for name in self.names
            ]
        )

        combined_scores = (0.6 * semantic_scores) + (0.4 * fuzzy_scores)

        sorted_indices = np.argsort(combined_scores)[::-1]

        all_matches = []
        for idx in sorted_indices[:top_k]:
            match = {
                "name": self.names[idx],
                "score": round(float(combined_scores[idx]), 3),
                "semantic_score": round(float(semantic_scores[idx]), 3),
                "fuzzy_score": round(float(fuzzy_scores[idx]), 3),
            }
            all_matches.append(match)

        best_match = {"name": all_matches[0]["name"], "score": all_matches[0]["score"]}

        return {"best_match": best_match, "all_matches": all_matches}


if __name__ == "__main__":
    matcher = NameMatcher()

    test_names = ["Gita", "Mohammad", "Prya", "Kris"]

    for test_name in test_names:
        logger.info(f"Testing: {test_name}")

        results = matcher.find_similar_names(test_name)

        logger.info(
            f"\nBest Match: {results['best_match']['name']} "
            f"(Score: {results['best_match']['score']:.3f})"
        )

        logger.info("\nTop 5 Matches:")
        for i, match in enumerate(results["all_matches"][:5], 1):
            logger.info(
                f"{i}. {match['name']:<15} - Score: {match['score']:.3f} "
                f"(Semantic: {match['semantic_score']:.3f}, "
                f"Fuzzy: {match['fuzzy_score']:.3f})"
            )
