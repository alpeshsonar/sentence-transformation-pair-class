import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, losses
import os

# Set device consistently
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_model_with_classifier(model_path, classifier_path, device=device):
    # Load base model and move to device
    model = SentenceTransformer(model_path)
    model.to(device)
    
    # Load the classifier state dict to check dimensions
    classifier_state_dict = torch.load(classifier_path, map_location=device)
    
    # Get the weight shape to determine input dimension
    weight_shape = classifier_state_dict['weight'].shape
    input_dim = weight_shape[1]  # Second dimension is input size
    output_dim = weight_shape[0]  # First dimension is output size
    
    print(f"Loaded classifier dimensions: input={input_dim}, output={output_dim}")
    print(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Create classifier with correct dimensions and move to device
    classifier = nn.Linear(input_dim, output_dim)
    classifier.load_state_dict(classifier_state_dict)
    classifier.to(device)
    
    return model, classifier

def predict(model, classifier, premise, hypothesis, device=device):
    # Get embeddings and ensure they're on the right device
    embeddings = model.encode([premise, hypothesis], convert_to_tensor=True, device=device)
    
    # Check if we need to concatenate instead of average
    classifier_input_dim = classifier.weight.shape[1]
    model_dim = model.get_sentence_embedding_dimension()
    
    if classifier_input_dim == model_dim * 2:
        # Classifier expects concatenated embeddings
        print("Using concatenated embeddings for prediction")
        embedding = torch.cat([embeddings[0], embeddings[1]])
    elif classifier_input_dim == model_dim * 3:
        # Classifier might expect [sent1, sent2, |sent1-sent2|]
        print("Using enhanced concatenated embeddings for prediction")
        embedding = torch.cat([embeddings[0], embeddings[1], torch.abs(embeddings[0] - embeddings[1])])
    else:
        # Default to averaging if dimensions still don't match
        print("Using averaged embeddings for prediction")
        embedding = (embeddings[0] + embeddings[1]) / 2
    
    # Ensure embedding is on the same device as classifier
    embedding = embedding.to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = classifier(embedding.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    
    return prediction, probs.squeeze().tolist()

# Map numerical labels to text labels
label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

# First try to load from expected path
model, classifier = load_model_with_classifier("with-class-header/final_model", "with-class-header/final_model/classifier_head.pth")

# Test cases
test_cases = [
    {
        "premise": "Children smiling and waving at camera",
        "hypothesis": "They are smiling at their parents",
        "expected": "neutral",
        "expected_idx": 1
    },
    {
        "premise": "Children smiling and waving at camera",
        "hypothesis": "There are children present",
        "expected": "entailment",
        "expected_idx": 0
    },
    {
        "premise": "Children smiling and waving at camera",
        "hypothesis": "The kids are frowning",
        "expected": "contradiction",
        "expected_idx": 2
    }
]

# Run predictions
print("\n===== TESTING SENTENCE PAIRS =====")
for i, test in enumerate(test_cases):
    print(f"\nTest {i+1}:")
    print(f"Premise: \"{test['premise']}\"")
    print(f"Hypothesis: \"{test['hypothesis']}\"")
    print(f"Expected: {test['expected']} (Label {test['expected_idx']})")
    
    pred_idx, probs = predict(model, classifier, test['premise'], test['hypothesis'])
    pred_label = label_map.get(pred_idx, "unknown")
    
    print(f"Prediction: {pred_label} (Label {pred_idx})")
    print(f"Probabilities: entailment: {probs[0]:.4f}, neutral: {probs[1]:.4f}, contradiction: {probs[2]:.4f}")
    
    if pred_idx == test['expected_idx']:
        print("✅ CORRECT")
    else:
        print("❌ INCORRECT")
