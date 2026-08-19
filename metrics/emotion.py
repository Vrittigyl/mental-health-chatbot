from transformers import pipeline

print("Loading Emotion Model...")
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=3
)

def get_primary_emotion(text: str) -> str:
    """
    Analyzes the text and returns the primary emotion.
    """
    result = emotion_classifier(text)
    
    # Extract the top label from the pipeline output
    if result and isinstance(result, list):
        if isinstance(result[0], list):
            return result[0][0]["label"]
        return result[0]["label"]
    
    return "neutral"

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 40)
    print("  Emotion Detector Test CLI")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
        emotion = get_primary_emotion(test_text)
        print(f"\nText: '{test_text}'")
        print(f"Detected Emotion: {emotion}\n")
    else:
        while True:
            try:
                test_text = input("\nEnter text to analyze (or 'quit' to exit): ").strip()
                if test_text.lower() in ['quit', 'exit', 'q']:
                    break
                if test_text:
                    emotion = get_primary_emotion(test_text)
                    print(f"Detected Emotion: -> {emotion}")
            except KeyboardInterrupt:
                break
        print("\nGoodbye!")
