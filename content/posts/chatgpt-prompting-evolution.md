---
title: "ChatGPT's Predecessor: The Prompting Paradigm"
date: 2024-01-30T10:00:00-00:00
draft: false
tags: ["chatgpt", "prompting", "language-models", "pre-training", "instruction-tuning"]
categories: ["natural-language-processing"]
math: true
---

*This post explores the evolution of prompting techniques that led to the development of ChatGPT and modern large language models.*

## Introduction

The idea of Prompting is a key step for language models to achieve true unification. ChatGPT is currently very popular, but many people are unaware of its predecessor — the Prompt paradigm. Through this article, I hope to provide readers with an understanding of the ideas behind the Prompt paradigm. The focus of this article is not on the details but rather on the underlying philosophy and inspiration of the Prompt paradigm.

## 1. Prompting: The Latest Paradigm in NLP

Prompting, also known as in-context learning, refers to the “pre-trained-Prompt” paradigm in NLP, which is also a parameter-efficient (parameter-efficient) learning method. However, limiting the understanding of Prompting solely as a parameter-efficient learning approach is a narrow view. To truly grasp the essence of Prompting, it’s best to approach it from a higher perspective.

Let me begin by stating this:

>Models generated under previous paradigms inherently lack the ability to generalize at the task level, whereas Prompting has the ability to generalize at the task level, which is a dimensionality reduction blow to the previous NLP paradigms.

<img src="/images/posts/chatgpt-prompting/prompting-overview.png" 
     alt="Prompting Paradigm Overview" 
     width="600" 
     style="display: block; margin: 0 auto;">
<p style="text-align: center; font-style: italic;">
Figure 1: The evolution from task-specific models to task-general prompting paradigm
</p>

Returning to the pre-training and fine-tuning paradigm in NLP: Pre-trained models are mainly trained using large-scale, non-task-specific corpora. In the fine-tuning step, domain-specific corpora are used to fine-tune the model’s parameters, allowing the final model to solve problems in a specific domain effectively. A large body of research has shown that the pre-training + fine-tuning paradigm can significantly outperform models trained solely on task-specific data.   

This can be briefly illustrated in one sentence:

>“Use as much training data as possible and extract as many common features as possible, makaing the model’s task-specific learning burden lighter.”

Researchers quickly discovered that the performance of pre-trained models depends on the model’s size; simply put, the larger the model, the better the performance. But this led to a problem: traditional fine-tuning requires updating all of the model’s parameters, which clearly does not yield the best training results. This is because, compared to the scale of the pre-trained model, the task-specific data is often minimal. Additionally, when the fine-tuned model is deployed in production, another issue arises: each task or domain requires a separate fine-tuned model, which is not cost-effective.

So, researchers proposed methods for fine-tuning part of the parameters or freezing certain parts of the model: We can use prior knowledge about the model parameter distribution — certain parts of the model have specific functions (or are biased toward specific functions). For example: In a Transformer model used for machine translation, the Encoder part is responsible for encoding the source language into high-dimensional vectors, and the Decoder part generates corresponding sentences in the target language. With such assumptions, when fine-tuning the model’s parameters, we can update only the parts most relevant to the target task. The benefits of this approach are: (1) Faster, more targeted training, often with better results; (2) During deployment, multiple tasks/domains can share the same pre-trained model, updating only the parameters relevant to each specific task/domain.

Building upon this “partial parameter fine-tuning” approach, researchers proposed the Adapter method. By adding a flexible, plug-and-play, trainable adaptation module (Adapter Module), we can modify the data distribution at certain layers (or a few layers) of the model, enabling it to be applied to a variety of tasks and domains. A typical approach is that each task or domain corresponds to a specific Adapter Module, and during training, only the parameters of the corresponding Adapter are updated. During inference, the corresponding Adapter for a task/domain is activated, and the others are ignored. Compared to the "partial parameter fine-tuning" method, the Adapter method has advantages such as: (1) More lightweight, (2) Better performance with the same proportion of parameters.

These methods are all based on the “pre-trained-fine-tuning” paradigm. No matter how lightweight, each task still has a set of specific parameters (even though most parameters can be shared). At this point, most people have not realized that its fatal flaw is that tasks themselves need to be manually defined. However, in real life, knowledge is not categorized according to tasks. In reality, most tasks involve the ability to handle multiple tasks. For example, a Chinese person reads an English article and then writes a summary in Chinese — this involves both translation and summarization tasks.

Prompting, however, can blur the boundaries between tasks — there is no need to manually define tasks. Instead, task descriptions are directly input into the pre-trained model as part of the input. This is why the Prompting paradigm is “the key step towards true unified language models.”

The following diagram is excerpted from a survey: _Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing_.

### Comparison of two NLP paradigms:

![NLP Paradigms Comparison](/images/posts/chatgpt-prompting/paradigm-comparison.png)
*Figure 2: Comparison between pre-training + fine-tuning (c) vs pre-training + prompting (d) paradigms*

The diagram above compares the "pre-trained-fine-tuning" paradigm, and the diagram below compares the "pre-trained-Prompting" paradigm.

## 2. What is Prompting?

The Prompting method was first introduced in 2020 alongside GPT-3. The Prompt paradigm suggests that pre-trained models can already handle many tasks; they just need to be guided during input (also known as providing context). How to guide the model? It's actually simple: the early version of Prompt only needed to describe the task in natural language, turning the task into a “fill-in-the-blank” (for bidirectional models like BERT) or “generation” (for autoregressive models like GPT) task. Here are a few simple examples:

- **Text Sentiment Classification**: The task is to input a text and predict the corresponding sentiment (positive, negative, neutral). In the traditional pre-trained-fine-tuning paradigm, this is done by adding a classifier module on top of the pre-trained model and fine-tuning it using task-specific data. When the input is "The weather is great today", the model outputs "positive". In the Prompting paradigm, we can simply input: "The weather is great today, my mood is [MASK]" ==> [MASK] prediction is "happy", which is then mapped to "positive".

- **Machine Translation**: The task is to input a piece of text in one language and generate a synonymous sentence in another language. In the Prompting paradigm, we can input: "Translate into English: The weather is great today" ==> the model outputs "This is a good day."

From these examples, we can easily see that the Prompt paradigm has the following characteristics: it does not require task-specific data for training and does not need to adjust model parameters. The success of Prompting proves that, when the model and training data are large enough, the model itself is close to an "encyclopedia", and Prompting is the key to unlocking the knowledge within this "encyclopedia."

In just five years, from GPT-1 to GPT-3.5, the model has grown 3000 times larger, and the upcoming GPT-4 will have 100 trillion parameters.

## 3. Classification of Prompting Methods

Prompting methods can be broadly divided into (1) Manual Prompting and (2) Parameterized Prompting.

- **Manual Prompting**: This method involves describing the task in natural language, as previously mentioned. It is mainly divided into "Prefix Prompt" and "Cloze Prompt." The "Prefix Prompt" is usually used for generative NLP tasks (NLG), while the "Cloze Prompt" is used for understanding NLP tasks (NLU). (If you are unfamiliar with generative vs understanding NLP tasks, it’s strongly recommended to review Pan Xiaoxiao's article on NLP pre-training, specifically Chapter 3: Pre-training for NLU vs NLG tasks.)

An example of a Prefix Prompt would be a template for machine translation tasks. The language indicator (also known as the language ID) widely used in multilingual machine translation can be considered a type of Prefix Prompt.

A Cloze Prompt example would be the template for text sentiment classification tasks.

- **Parameterized Prompting** (also known as "Automatic Prompting"): This includes discrete and continuous prompting. "Discrete" means the candidate prompts are still natural language words; "Continuous" means that the prompts don’t have to be natural language sequences but can be any combination of tokens in the vocabulary, or even new tokens not in the original vocabulary. A famous example is Prefix Tuning: adding a continuous vector prefix (new tokens with independent embeddings) at the input layer, and at each hidden layer, the same length prefix is added (this step doesn’t introduce extra parameters but slightly modifies the model structure). During training, separate parameter updates are made for each task. It’s important to note that during downstream task training, only the embedding parameters for the Prefix are updated.

Prompt Tuning further simplifies Prefix Tuning by eliminating the need for corresponding prefixes at every hidden layer. Thus, the model structure remains unchanged, except for the addition of embedding parameters for the prefix token (which accounts for about 0.1% of the total model parameters). Prompt Tuning is thus more flexible.

## 4. Effectiveness and Summary of Prompting

What are the effects of the Prompt paradigm? What are the key conclusions?

- Pre-trained-Prompting can achieve results comparable to pre-trained-fine-tuning, even with a 1000-fold reduction in trainable parameters.
- Model size is a decisive factor: the larger the model, the better the performance of Prompting.
- It has generalization ability for tasks, such as strong few-shot and zero-shot capabilities.
- For specific effects, refer to papers such as: "MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION", "The Power of Scale of Parameter-Efficient Prompt Tuning", and "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"

## 5. Why Did Prompting Evolve into ChatGPT?

Having discussed all this, it’s clear why the Prompting technique evolved into ChatGPT.

**In one sentence**: The Prompting technique breaks the conventional definition of "tasks," making it possible to implement composite tasks (what researchers call "zero-shot"). Most "tasks" in everyday life are not neatly categorized into independent sub-tasks as in traditional NLP, so Prompting’s ability to ignore the task itself is the fundamental reason it evolved into ChatGPT.

The base model used in ChatGPT is GPT-3.5, which already achieves impressive results. The upcoming GPT-4 has several hundred times more parameters than GPT-3.5.

The birth and success of Prompting and ChatGPT herald the beginning of a new era: the independent research on NLP sub-tasks defined by humans will gradually fade into history.

## References

^ [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)

^ [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)

^ [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)

^ [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

^ [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/)

^ [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)

^ [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
