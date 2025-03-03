import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers.readers import InputExample

# ✅ Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ✅ Hyperparameters
EPOCHS = 3
LEARNING_RATE = 3e-5
BATCH_SIZE = 3  # Adjust batch size for GPU memory
MAX_TOKENS = 128
WARMUP_RATIO = 0.1

# ✅ Load Pre-trained SentenceTransformer Model
model = SentenceTransformer("bert-base-uncased")
model.to(device)

# ✅ Load dataset correctly
train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train").select(range(50))
eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev").select(range(10))

# ✅ Convert dataset to `InputExample` format
def convert_to_input_examples(dataset):
    examples = []
    for row in dataset:
        examples.append(InputExample(texts=[row["premise"], row["hypothesis"]], label=int(row["label"])))
    return examples

train_examples = convert_to_input_examples(train_dataset)
eval_examples = convert_to_input_examples(eval_dataset)

# ✅ Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(eval_examples, shuffle=False, batch_size=BATCH_SIZE)

# ✅ Define SoftmaxLoss with the model
train_loss = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=3  # Number of classes (e.g., Entailment, Contradiction, Neutral)
)

# ✅ Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=int(WARMUP_RATIO * len(train_dataloader)),
    optimizer_params={'lr': LEARNING_RATE}
)

OUTPUT_DIR = "with-class-header"

# ✅ Save the classifier from the loss object
torch.save(train_loss.classifier.state_dict(), f"{OUTPUT_DIR}/final_model/classifier_head.pth")

# ✅ Save the SentenceTransformer model
model.save(f"{OUTPUT_DIR}/final_model")
print("✅ Model and classifier saved separately!")
