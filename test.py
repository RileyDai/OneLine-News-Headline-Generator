
# import transformers
# print("Transformers version:", transformers.__version__)
# print("Transformers path:", transformers.__file__)


# from transformers import TrainingArguments
# print("TrainingArguments source:", TrainingArguments.__module__)

from transformers import TrainingArguments

try:
    args = TrainingArguments(
        output_dir="./check_args",
        evaluation_strategy="epoch"
    )
    print("✅ TrainingArguments accepted evaluation_strategy")
except TypeError as e:
    print("❌ TypeError:", e)
