from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn

# ✅ Set hyperparameters
EPOCHS = 5  # Increase epochs for better learning
LEARNING_RATE = 2e-5  # Fine-tuning learning rate

# ✅ Load training and validation data
df_train = pd.read_csv("train_balanced.csv")
df_val = pd.read_csv("validation_balanced.csv")

# ✅ Prepare training examples
train_examples = [
    InputExample(texts=[row["sentence1"], row["sentence2"]], label=int(row["label"]))
    for _, row in df_train.iterrows()
]

val_examples = [
    InputExample(texts=[row["sentence1"], row["sentence2"]], label=int(row["label"]))
    for _, row in df_val.iterrows()
]

# ✅ Create full model pipeline
model = SentenceTransformer("bert-base-uncased")

train_loss = losses.ContrastiveLoss(model)

# ✅ DataLoader for training & validation
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)  # Increase batch size for better stability
val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=4)

# ✅ Define Evaluator for Validation
evaluator = evaluation.BinaryClassificationEvaluator(
    [ex.texts[0] for ex in val_examples],
    [ex.texts[1] for ex in val_examples],
    [ex.label for ex in val_examples]
)

# ✅ Train the model with validation evaluation
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,  # Validate after every epoch
    epochs=EPOCHS,  
    warmup_steps=100,
    evaluation_steps=100,
    optimizer_params={'lr': LEARNING_RATE},  # Set custom learning rate
    output_path="st-pair-class-own"
)

# ✅ Save the fine-tuned model
model.save("st-pair-class-own")
