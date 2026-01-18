import sys
from pathlib import Path

import torch


def check_environment():
    """Check if environment is set up correctly"""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    # Check PyTorch
    print(f"\nlogger.info PyTorch version: {torch.__version__}")
    print(f"logger.info CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("  CUDA detected but not needed for CPU inference")
    else:
        print("   logger.info CPU-only setup (perfect for deployment)")

    # Check required packages
    try:
        import transformers

        print(f"logger.info Transformers version: {transformers.__version__}")
    except ImportError:
        print("logger.infoTransformers not installed!")
        print("   Run: pip install transformers")
        return False

    try:
        import peft

        print(f"logger.info PEFT version: {peft.__version__}")
    except ImportError:
        print("logger.infoPEFT not installed!")
        print("   Run: pip install peft")
        return False

    return True


def check_model_files():
    """Check if model files exist"""
    print("\n" + "=" * 70)
    print("Model Files Check")
    print("=" * 70)

    model_path = Path("recipe-bot-finetuned")

    if not model_path.exists():
        print(f"\nlogger.infoModel directory not found: {model_path}")
        print("\n Steps to fix:")
        print("   1. Download model from Colab")
        print("   2. Extract to backend/ directory")
        print("   3. Ensure folder name is 'recipe-bot-finetuned-final'")
        return False

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
    ]

    print(f"\nlogger.info Model directory found: {model_path.absolute()}")
    print("\nChecking required files:")

    all_found = True
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   logger.info {file} ({size:.2f} MB)")
        else:
            print(f"   logger.info{file} - MISSING!")
            all_found = False

    return all_found


def test_model_loading():
    """Test if model loads correctly"""
    print("\n" + "=" * 70)
    print(" Model Loading Test")
    print("=" * 70)

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\n1Loading tokenizer...")
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("   logger.info Tokenizer loaded")

        print("\n2Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print("   logger.info Base model loaded on CPU")

        print("\n3Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, "recipe-bot-finetuned")
        print("   logger.info LoRA adapters loaded successfully!")

        # Count parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        print("\n Model Statistics:")
        print(f"   • Total parameters: {total:,}")
        print(f"   • Trainable (LoRA): {trainable:,}")
        print(f"   • Device: {next(model.parameters()).device}")
        print(f"   • Dtype: {next(model.parameters()).dtype}")

        return model, tokenizer

    except Exception as e:
        print(f"\nlogger.infoError loading model: {e}")
        return None, None


def test_inference(model, tokenizer):
    """Test actual recipe generation"""
    print("\n" + "=" * 70)
    print(" Inference Test")
    print("=" * 70)

    test_cases = [
        "eggs, onions",
        "tomatoes, pasta",
    ]

    model.eval()
    model.config.use_cache = True

    for ingredients in test_cases:
        print(f"\nTesting: {ingredients}")
        print("-" * 70)

        prompt = f"""<s>[INST] Suggest a recipe using the following ingredients:
{ingredients} [/INST]"""

        inputs = tokenizer(prompt, return_tensors="pt")

        print("   Generating recipe...")
        import time

        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - start

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "[/INST]" in full_response:
            recipe = full_response.split("[/INST]", 1)[1].strip()
        else:
            recipe = full_response

        print(f"\n    Generation time: {elapsed:.2f}s")
        print("\n    Generated Recipe:\n")
        print("   " + recipe[:300].replace("\n", "\n   "))
        if len(recipe) > 300:
            print("   ...")
        print("-" * 70)


def main():
    print("\n" + "=" * 70)
    print(" Recipe Bot Model Verification")
    print("=" * 70)

    # Step 1: Check environment
    if not check_environment():
        print("\nlogger.infoEnvironment check failed!")
        print("Please install missing packages and try again.")
        sys.exit(1)

    # Step 2: Check model files
    if not check_model_files():
        print("\nlogger.infoModel files check failed!")
        print("Please download and extract the model first.")
        sys.exit(1)

    # Step 3: Load model
    model, tokenizer = test_model_loading()
    if model is None:
        print("\nlogger.infoModel loading failed!")
        sys.exit(1)

    # Step 4: Test inference
    test_inference(model, tokenizer)

    print("\n" + "=" * 70)
    print("logger.info All checks passed!")
    print("=" * 70)
    print("\nlogger.infoYour model is ready to use!")
    print("\nNext steps:")
    print("   1. Run: python recipe_bot.py")
    print("   2. Run: python api.py")
    print("   3. Run: streamlit run ../frontend/app.py")


if __name__ == "__main__":
    main()
