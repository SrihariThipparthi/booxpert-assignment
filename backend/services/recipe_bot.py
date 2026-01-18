import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeBot:
    def __init__(self, model_path: str = None):
        if model_path is None:
            script_dir = Path(__file__).parent
            model_path = str(script_dir / "recipe-bot-finetuned")

        self.model_path = model_path
        self.base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Loading base model + LoRA adapter from {self.model_path}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            self.model = PeftModel.from_pretrained(base_model, self.model_path)

            self.model.eval()
            self.model.config.use_cache = True

            logger.info("LoRA model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading LoRA model: {e}")
            logger.error("Falling back to base model...")
            self._load_base_model()

    def _load_base_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info("Base model loaded")

    def generate_recipe(
        self, ingredients: str, use_fallback: bool = True
    ) -> dict[str, str]:
        try:
            ingredient_list = [ing.strip() for ing in ingredients.split(",")]
            ingredients_text = ", ".join(ingredient_list)

            prompt = f"""<s>[INST] Suggest a recipe using the following ingredients:
{ingredients_text} [/INST]"""

            logger.info(f" Generating recipe for: {ingredients_text}")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,  # org 400
                    min_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "[/INST]" in full_response:
                recipe_text = full_response.split("[/INST]", 1)[1].strip()
            else:
                recipe_text = full_response.strip()

            if not self._is_valid_recipe(recipe_text):
                logger.info(
                    "Generated response doesn't look like a recipe, using fallback..."
                )
                if use_fallback:
                    return self.get_recipe_by_keywords(ingredients)
                else:
                    return {
                        "success": False,
                        "ingredients": ingredients_text,
                        "recipe": None,
                        "error": "Generated invalid recipe",
                    }

            return {
                "success": True,
                "ingredients": ingredients_text,
                "recipe": recipe_text,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            if use_fallback:
                return self.get_recipe_by_keywords(ingredients)
            return {
                "success": False,
                "ingredients": ingredients,
                "recipe": None,
                "error": str(e),
            }

    def _is_valid_recipe(self, text: str) -> bool:
        if len(text) < 100:
            return False

        text_lower = text.lower()

        has_ingredients = "ingredients:" in text_lower or "ingredient" in text_lower
        has_instructions = "instructions:" in text_lower or "instruction" in text_lower

        cooking_words = [
            "cook",
            "heat",
            "add",
            "mix",
            "stir",
            "serve",
            "bake",
            "fry",
            "boil",
        ]
        has_cooking_actions = (
            sum(1 for word in cooking_words if word in text_lower) >= 2
        )

        return (has_ingredients or has_instructions) and has_cooking_actions


_recipe_bot_instance = None


def get_recipe_bot(model_path: str = None) -> RecipeBot:
    global _recipe_bot_instance
    if _recipe_bot_instance is None:
        _recipe_bot_instance = RecipeBot(model_path)
    return _recipe_bot_instance


if __name__ == "__main__":
    logger.info("Testing Recipe Bot")

    bot = RecipeBot()

    test_ingredients = ["chicken, rice"]

    for ingredients in test_ingredients:
        result = bot.generate_recipe(ingredients, use_fallback=True)

        if result["success"]:
            logger.info("\nGenerated Recipe:\n")
            logger.info(result["recipe"])
        else:
            logger.error(f"\nError: {result['error']}")
