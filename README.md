
# 🛡️ Prompt Injection Detection: Evaluating and Enhancing the Security of Large Language Models

Welcome to the official repository for our research on detecting prompt injection attacks on Large Language Models (LLMs). This repository includes the source code, datasets references, and evaluation scripts necessary to replicate the experiments and results discussed in our paper. Our study investigates various detection mechanisms, from simple rule-based approaches to sophisticated model-based detectors, and explores the efficacy of a combined detection strategy.


![detecotr injection](https://github.com/user-attachments/assets/5e498d24-6117-4725-b4ef-9f08b7357880)



## 🧩 Overview

Prompt injection attacks pose a significant threat to the integrity of LLMs by attempting to manipulate the model's behavior through crafted inputs. This repository provides a comprehensive framework for evaluating the effectiveness of different detection methods in identifying such attacks. Our approach integrates multiple detection strategies, including rule-based, keyword matching, perplexity-based, and model-based detectors, to achieve robust and reliable results.

## 🗂️ Datasets

We utilized two primary datasets for our experiments:

- **Gandalf Ignore Instructions Dataset**: Contains nearly 1,000 samples of prompt injection attacks. This dataset was used exclusively for evaluating the performance of the detection mechanisms.
- **QA Chat Prompts Dataset**: Consists of 100 clean prompts used to establish a perplexity baseline for the perplexity-based detection method.

### Loading the Datasets:

**Gandalf Ignore Instructions Dataset:**

```python
from datasets import load_dataset
gandalf_dataset = load_dataset("Lakera/gandalf_ignore_instructions")
```

**QA Chat Prompts Dataset:**

```python
from datasets import load_dataset
clean_dataset = load_dataset("nm-testing/qa-chat-prompts", split="train_sft[:100]")
```

## 🔍 Detection Methods

### 1. Rule-Based Detectors
- **Rule-Based Detector**: Identifies prompts containing specific suspicious patterns.
- **Keyword Matching Detector**: Flags prompts with keywords commonly associated with malicious activities.

### 2. Perplexity-Based Detector
- **LLaMa-7B Perplexity-Based Detector**: Utilizes the LLaMa-7B model to calculate the perplexity of a prompt and compare it against a threshold derived from clean samples.

### 3. Model-Based Detectors
- **ShieldGemma-2B Detector**: Designed to filter harmful content and enforce safety policies in LLMs.
- **DeBERTa-184M Injection Detector**: Specifically fine-tuned to detect prompt injections by classifying prompts as either "LEGIT" or "INJECTION."

## 📊 Results

### Detector Performance

- **Combined Detector**: Achieved an accuracy of **99.9%**, demonstrating the effectiveness of integrating multiple detection strategies.
- **LLaMa-7B Perplexity-Based Detector**: Achieved an accuracy of **97.3%**, showing its strength in detecting deviations from typical language patterns.
- **DeBERTa-184M Injection Detector**: Achieved an accuracy of **92.1%**, underscoring its reliability in detecting prompt injections.
- **ShieldGemma-2B Detector**: Achieved an accuracy of **30.3%**, focusing on content filtering.
- **Keyword Matching Detector**: Achieved an accuracy of **11.4%**, limited by its reliance on predefined keywords.
- **Rule-Based Detector**: Achieved an accuracy of **8.4%**, highlighting its limitations against sophisticated attacks.

### Visual Analysis

#### Mean Scores of Detectors
![1](https://github.com/user-attachments/assets/de31f7b8-04c1-4669-a365-b11c4550f97a)


The bar chart above clearly illustrates the superior performance of the perplexity-based and DeBERTa-based detectors compared to the rule-based and keyword matching methods. The combined detector's performance further emphasizes the effectiveness of integrating multiple detection strategies.

#### Accuracy with Number of Samples Tested
![2](https://github.com/user-attachments/assets/d3ce177e-22f4-41eb-831e-775ad8799552)


The line chart shows that the combined detector maintains consistently high accuracy across all samples, confirming its robustness. Advanced methods like Perplexity-Based and DeBERTa detectors stabilize at high accuracy levels early on, whereas simpler methods plateau at much lower accuracy levels.


### Huggingface Models:

- `huggyllama/LLaMa-7b`
- `google/shieldgemma-2b`
- `deepset/deberta-v3-base-injection`


## 📜 Conclusions

The experiment confirmed that a combined detection strategy is the most effective approach to identifying prompt injection attacks on LLMs. The integration of advanced detectors, such as those based on perplexity and the DeBERTa model, significantly enhances overall accuracy, making this approach a robust defense against diverse and sophisticated threats.

Thank you for exploring our work on enhancing the security of large language models. We encourage you to experiment with the code, modify the parameters, and contribute to the ongoing research in this critical area.
