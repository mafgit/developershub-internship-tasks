import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

labels = ['World', 'Sports', 'Business', 'Sci/Tech']

def infer(title, description):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "mafgit/news-classifier"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()

    inputs = tokenizer(
        title,
        description,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    max_idx = outputs.logits.argmax(dim=1).item()
    output = labels[max_idx]
    return output


ui = gr.Interface(
    fn=infer,
    inputs=["text", "text"],
    outputs=["text"],
    title="News Classifier",
    description="Classify news articles into World, Sports, Business, or Sci/Tech categories."
)

ui.launch()