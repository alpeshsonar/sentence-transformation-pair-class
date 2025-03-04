import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd


def cosine_similarity(emb1, emb2):
    """Compute Cosine Similarity between two embeddings."""
    return F.cosine_similarity(emb1, emb2, dim=0).item()


def predict(model, sentence1, sentence2, threshold=0.85):
    """Predict the class label using Cosine Similarity."""
    # ✅ Encode both sentences using model
    embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)

    # ✅ Compute Cosine Similarity
    similarity = cosine_similarity(embeddings[0], embeddings[1])

    # ✅ Print similarity score for debugging
    print(f"Cosine Similarity: {similarity:.4f}")

    # ✅ Set a threshold (0.5 for binary classification)
    predicted_class = 1 if similarity > threshold else 0

    return predicted_class, similarity


def load_model(model_path="st-pair-class-own"):
    """Load the fine-tuned sentence transformer model."""
    return SentenceTransformer(model_path)


def validate_test_data(model):
    """Validate model predictions against labeled test data."""
    df_test = pd.df = pd.read_csv("train_balanced.csv")

    test_data = list(zip(df_test["sentence1"], df_test["sentence2"], df_test["label"]))

    # Limit to first 20 rows for debugging
    # test_data = test_data[:20]
    total, correct = 0, 0

    print("\nValidating test data using Cosine Similarity...\n")
    for sentence1, sentence2, label in test_data:
        predicted_class, similarity = predict(model, sentence1, sentence2)
        is_correct = predicted_class == label
        total += 1
        correct += is_correct
        if not is_correct:
            print(f"❌ Misclassified: [{sentence1}] & [{sentence2}] | Expected: {label}, Predicted: {predicted_class}, Similarity: {similarity:.4f}")

    accuracy = (correct / total) * 100
    print(f"\n✅ Cosine Similarity Accuracy: {accuracy:.2f}% ({correct}/{total} correct)\n")


if __name__ == "__main__":
    # Load model
    model = load_model()

    # Validate test dataset using Cosine Similarity
    validate_test_data(model)
