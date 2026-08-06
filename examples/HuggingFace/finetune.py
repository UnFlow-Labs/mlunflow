import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def finetune(model_name:str, training_args:TrainingArguments):
    # Load the "sms_spam" dataset.
    sms_dataset = load_dataset("ucirvine/sms_spam")

    # Split train/test by an 8/2 ratio.
    sms_train_test = sms_dataset["train"].train_test_split(test_size=0.2)
    train_dataset = sms_train_test["train"]
    test_dataset = sms_train_test["test"]

    # Load the tokenizer for "distilbert-base-uncased" model.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


    def tokenize_function(examples):
    # Pad/truncate each text to 512 tokens. Enforcing the same shape
    # could make the training faster.
        return tokenizer(
            examples["sms"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )


    seed = 22

    # Tokenize the train and test datasets
    train_tokenized = train_dataset.map(tokenize_function)
    train_tokenized = train_tokenized.remove_columns(["sms"]).shuffle(seed=seed)

    test_tokenized = test_dataset.map(tokenize_function)
    test_tokenized = test_tokenized.remove_columns(["sms"]).shuffle(seed=seed)
    id2label = {0: "ham", 1: "spam"}
    label2id = {"ham": 0, "spam": 1}

    # Acquire the model from the Hugging Face Hub, providing label and id mappings so that both we and the model can 'speak' the same language.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,num_labels=2,label2id=label2id,id2label=id2label,)
    metric = evaluate.load("accuracy")
    # Define a function for calculating our defined target optimization metric during training
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    # Instantiate a `Trainer` instance that will be used to initiate a training run.
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
    )
    output = trainer.train()
    return {"loss": output.training_loss, "metrics": trainer.evaluate()}

    
training_args = TrainingArguments(
    output_dir="distilbert-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,

)
# finetune.clear_graph()
# output = finetune("distilbert-base-uncased", training_args)
# print(f"Finetuning completed. Loss: {output['loss']}, Metrics: {output['metrics']}")
# find the transformations that changed the learning rate
outcomes = finetune.get_outcomes()
transformations = finetune.query_transformations(
    predicate=lambda transformation: (
        "learning_rate" in transformation.args_changes
        and transformation.args_changes["learning_rate"]["to_value"]
        != transformation.args_changes["learning_rate"]["from_value"]
    )
)
print(f"Found {len(transformations)} transformations that changed the learning rate:")
for transformation in transformations:
    print(f"- {transformation.state1.name} -> {transformation.state2.name}")
    # what else has changed in the transformation?
    for arg, change in transformation.args_changes.items():
        from_value = change.get("from_value")
        to_value = change.get("to_value")
        from_type = change.get("from_type")
        to_type = change.get("to_type")

        if from_value is not None or to_value is not None:
            print(f"  - {arg}: from {from_value} to {to_value}")
        elif from_type is not None or to_type is not None:
            print(f"  - {arg}: from type {from_type} to type {to_type}")
        else:
            print(f"  - {arg}: change details not available")
    output1 = outcomes.get(transformation.state1.name).outputs
    output2 = outcomes.get(transformation.state2.name).outputs
    if output1 is not None and output2 is not None:
        loss_change = output2["loss"] - output1["loss"]
        accuracy_change = output2["metrics"]["eval_accuracy"] - output1["metrics"]["eval_accuracy"]
        print(f"  - Loss change: {loss_change:.4f}")
        print(f"  - Accuracy change: {accuracy_change:.4f}")

    