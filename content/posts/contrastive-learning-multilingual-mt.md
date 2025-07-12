---
title: "Contrastive Learning for Many-to-many Multilingual Neural Machine Translation"
date: 2021-06-01T10:00:00-00:00
draft: true
tags: ["contrastive learning", "multilingual", "machine translation", "neural networks"]
categories: ["nlp", "machine-learning", "research"]
math: true
---

*This post discusses our ACL 2021 paper on contrastive learning for multilingual machine translation.*

## Introduction

Multilingual Neural Machine Translation (MNMT) has become increasingly important as it allows a single model to handle multiple language pairs. However, training effective multilingual models remains challenging due to language interference and the need to balance performance across different language pairs.

## The Challenge

Traditional multilingual models often suffer from:
- **Language interference**: Training multiple languages together can hurt individual language performance
- **Imbalanced performance**: Some language pairs perform much better than others
- **Limited generalization**: Models struggle with zero-shot translation between unseen language pairs

## Our Approach: Contrastive Learning

We propose using contrastive learning to improve multilingual translation by:
1. **Learning language-invariant representations**: Encouraging similar semantic content to have similar representations across languages
2. **Reducing language interference**: Using contrastive loss to separate different languages while preserving semantic similarity
3. **Improving zero-shot performance**: Better generalization to unseen language pairs

## Mathematical Formulation

The contrastive loss function:

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(sim(h_i, h_i^+)/\tau)}{\sum_{j=1}^{N} \exp(sim(h_i, h_j)/\tau)}$$

where:
- $h_i$ is the representation of the source sentence
- $h_i^+$ is the representation of the corresponding target sentence (positive pair)
- $h_j$ are representations of other sentences (negative pairs)
- $\tau$ is the temperature parameter
- $sim(\cdot, \cdot)$ is the cosine similarity function

## Model Architecture

Our mRASP2 model builds upon the Transformer architecture with contrastive learning:

```python
class ContrastiveMNMT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.contrastive_head = ContrastiveHead(config.hidden_size)
        
    def forward(self, src_ids, tgt_ids, src_lang, tgt_lang):
        # Encode source and target
        src_hidden = self.encoder(src_ids)
        tgt_hidden = self.encoder(tgt_ids)
        
        # Contrastive learning
        contrastive_loss = self.contrastive_head(src_hidden, tgt_hidden)
        
        # Translation loss
        translation_loss = self.decoder(tgt_ids, src_hidden)
        
        return translation_loss + contrastive_loss
```

## Results

Our experiments on 100+ language pairs show:
- **Consistent improvements**: Average 2-3 BLEU improvement across all language pairs
- **Better zero-shot performance**: 10+ BLEU improvement for unseen language pairs
- **Reduced interference**: Better balance between high-resource and low-resource languages

## Key Insights

1. **Temperature matters**: Lower temperature (0.1-0.3) works better for contrastive learning in MT
2. **Positive pair selection**: Using parallel sentences as positive pairs is crucial
3. **Negative sampling**: Hard negative mining improves performance significantly

## Conclusion

Contrastive learning provides a powerful way to improve multilingual machine translation by learning better cross-lingual representations. The approach is simple yet effective, making it practical for real-world applications.

For more details, see our paper: [Contrastive Learning for Many-to-many Multilingual Neural Machine Translation](https://arxiv.org/abs/2105.09501) 