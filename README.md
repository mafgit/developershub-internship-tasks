# developershub-internship-tasks

> I completed the mandatory as well as optional tasks in the AI/ML Internship I am currently in: DevelopersHub Corporation©. Here's what I built, what challenged me, and what I took away.

---

### 🧠 Task 5 — Mental Health Support Chatbot (Fine-Tuning)

My first time fine-tuning an LLM end-to-end, and it was the most interesting for me, and it took the most time.

Fine-tuned DistilGPT2 on the Estwld/empathetic_dialogues_llm dataset using Hugging Face's Trainer API. The preprocessing required real thought: flattening conversation arrays into a single training string, tokenization, masking of tokens other than the assistant responses, adding special tokens, etc.

Trained with fp16 + gradient accumulation on a Colab T4, 3 epochs, lr=3e-5, AdamW. Pushed model + tokenizer to Hugging Face Hub.

Also built a streamlit app in which the user can test both fine-tuned and original DistilGPT2. Output is streamed via TextIteratorStreamer on a background thread, and a context window usage bar is also visible.

NOTE: DistilGPT2 is a very small model and the dataset wasn't enough, hence outputs are still gibberish, although you can clearly see that it has changed significantly. But I learnt a lot and had to do a lot of debugging and researching during this task.

---

### 🏠 Task 6 — House Price Prediction (the frustrating one)

Initial RMSE and MAE were too high. The culprit: outliers in price and area skewing the regression. Built an IQR-based removal function, applied log1p on the target, re-trained a Gradient Boosting Regressor. Errors came down to reasonable numbers relative to the dataset's price range.

---

### 📈 Task 2 — Stock Price Prediction (TSLA)

yfinance + 2 years of data. Shifted Close by 1 to engineer the next-day target. Trained Linear Regression and Random Forest, compared both against actual prices.

---

### 🤖 Task 4 — Health Query Chatbot

Built a terminal-based medical assistant with history and safety system prompts using two modes: Llama-3.1-8B via HuggingFace Inference API, and Qwen 2.5 1.5B locally.

---

### ❤️ Task 3 — Heart Disease Prediction

UCI dataset. Median/mode imputation, LabelEncoder, StandardScaler. Logistic Regression vs Random Forest — RF won. Plotted confusion matrix + ROC curve.

---

### 📊 Task 1 — Iris EDA

Made scatter plots, histograms, box plots.