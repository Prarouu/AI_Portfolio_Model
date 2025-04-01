import json
import spacy
import random
from spacy.tokens import DocBin

# Load the dataset
json_file_path = "training_data.json"  # Update with actual path
with open(json_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

data = []
for line in lines:
    try:
        parsed_line = json.loads(line)
        if parsed_line and "content" in parsed_line and "annotation" in parsed_line:
            data.append(parsed_line)
    except json.JSONDecodeError:
        print("Skipping invalid JSON entry")

# Shuffle and split the dataset (80% train, 20% dev)
random.shuffle(data)
split_index = int(0.8 * len(data))
train_data = data[:split_index]
dev_data = data[split_index:]

# Debugging: Check the size of train_data and dev_data
print(f"Train data size: {len(train_data)}")
print(f"Dev data size: {len(dev_data)}")

nlp = spacy.blank("en")
all_labels = set()

def create_spacy_docs(data, filename):
    print(f"Processing {filename} with {len(data)} entries")  # Debugging
    doc_bin = DocBin()
    for entry in data:
        text = entry["content"]
        annotations = {"entities": []}

        if entry["annotation"]:
            for label_data in entry["annotation"]:
                label = label_data.get("label")
                for point in label_data.get("points", []):
                    try:
                        start = int(point.get("start"))
                        end = int(point.get("end"))
                        if start < end and isinstance(label, str):
                            annotations["entities"].append((start, end, label))
                            all_labels.add(label)  # Collect labels
                    except (TypeError, ValueError):
                        print(f"Skipping invalid annotation: {point}")

        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)

    doc_bin.to_disk(filename)
    print(f"Dataset successfully converted: {filename}")

# Create train.spacy and dev.spacy
create_spacy_docs(train_data, "train.spacy")
create_spacy_docs(dev_data, "dev.spacy")

# Add labels to spaCy vocab
for label in all_labels:
    nlp.vocab.strings.add(label)

print("Train and Dev datasets successfully created!")
