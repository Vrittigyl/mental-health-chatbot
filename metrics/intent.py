from transformers import pipeline

print("Loading Intent Model...")
# Load the zero-shot classification pipeline globally
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=-1  # Use -1 for CPU, 0 for first GPU
)

def detect_intent(text):

    # Descriptive candidate labels
    candidate_labels = [
        "the user is seeking guidance, advice, or a practical solution to their problem",
        "the user is seeking emotional support, comfort, empathy, or someone to listen to them",
        "the user is seeking reassurance that their feelings, thoughts, reactions, or situation are normal or okay",
        "the user is seeking factual information or an explanation about their mental health or emotional experience",
        "the user is expressing emotional distress without explicitly asking for advice or information",
        "the user is trying to understand the reasons behind their thoughts, feelings, or behavior",
        "the user is seeking validation that their feelings or experiences are legitimate",
        "the user is seeking connection, companionship, or someone to talk to because they feel alone",
        "the user is seeking immediate help because they may be in a mental health or safety crisis",
        "the user's intention is unclear or does not fit the other categories"
    ]

    # Mapping descriptive labels to simple intent categories
    label_mapping = {
        candidate_labels[0]: "Guidance",
        candidate_labels[1]: "Emotional Support",
        candidate_labels[2]: "Reassurance",
        candidate_labels[3]: "Information Seeking",
        candidate_labels[4]: "Expressing Distress",
        candidate_labels[5]: "Self Reflection",
        candidate_labels[6]: "Validation Seeking",
        candidate_labels[7]: "Connection Seeking",
        candidate_labels[8]: "Crisis / Immediate Help",
        candidate_labels[9]: "Other"
    }

    # Perform zero-shot classification
    result = classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template="This statement describes {}.",
        multi_label=False
    )

    print("\nIntent Prediction")
    print("-" * 40)

    for label, score in zip(result["labels"], result["scores"]):
        print(f"{label_mapping[label]:<8}: {score:.4f}")

    return label_mapping[result["labels"][0]], result['scores'][0]

# import spacy
# nlp = spacy.load("en_core_web_trf")  

# def extract_intent(text):
#     doc = nlp(text)
#     for token in doc:
#         if token.dep_ == "dobj":
#             action = token.head.lemma_        # verb governing the object
#             obj = token.text                  # or expand to full noun phrase via token.subtree
#             return action, obj
#     return None, None
