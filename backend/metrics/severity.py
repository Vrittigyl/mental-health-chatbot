from transformers import pipeline

print("Loading Severity Model...")
# Load the zero-shot classification pipeline globally
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=-1  # Use -1 for CPU, 0 for first GPU
)

def detect_severity(text):

    # Descriptive candidate labels
    candidate_labels = [
        "mild emotional distress that does not require immediate professional help",
        "moderate mental health concern that should be evaluated by a mental health professional",
        "mental health crisis requiring immediate medical or emergency intervention"
    ]

    # Mapping descriptive labels to simple severity levels
    label_mapping = {
        candidate_labels[0]: "Low",
        candidate_labels[1]: "Medium",
        candidate_labels[2]: "High"
    }

    # Perform zero-shot classification
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template="This statement describes {}.",
        multi_label=False
    )

    print("\nSeverity Prediction")
    print("-" * 40)

    for label, score in zip(result["labels"], result["scores"]):
        print(f"{label_mapping[label]:<8}: {score:.4f}")

    return label_mapping[result["labels"][0]], result['scores'][0]