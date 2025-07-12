---
title: "Introduction to Knowledge Distillation"
date: 2025-06-30T12:00:00-00:00
draft: true
tags: ["knowledge distillation", "model compression", "teacher-student", "deep learning"]
categories: ["machine-learning", "nlp", "knowledge-distillation"]
math: true
---

*This post is an English translation and summary of my original Chinese article on Knowledge Distillation. [Read the original post (Chinese, Zhihu)](https://zhuanlan.zhihu.com/p/102038521) and [the original Hexo post](https://raw.githubusercontent.com/PANXiao1994/PANXiao1994.github.io/refs/heads/master/2019/12/26/blog_tech/ml/algo/nlp/kd/knowledge_distillation/index.html).*

---

## Background & Motivation

Knowledge Distillation (KD) is a model compression technique based on the teacher-student paradigm, popularized by Hinton et al. in their 2015 paper "Distilling the Knowledge in a Neural Network." The core idea is to transfer the knowledge from a large, complex model (the teacher) to a smaller, simpler model (the student), enabling efficient deployment without significant loss in performance.

### Why Model Compression?
- Large models achieve high accuracy but are hard to deploy due to slow inference and high resource requirements.
- In deployment, we need fast, lightweight models.
- Model compression, including distillation, helps reduce model size while retaining as much knowledge as possible.

---

## The "Container and Water" Analogy

A model is like a container, and the knowledge in the data is like water. If the data contains more knowledge than the model can hold, the model underfits (the container is too small). If the model is much larger than the data, it overfits (the container is too big for the water, and the water sloshes around).

![Model Capacity Curve](https://raw.githubusercontent.com/PANXiao1994/PANXiao1994.github.io/master/2019/12/26/blog_tech/ml/algo/nlp/kd/knowledge_distillation/curve.jpeg)

*The relationship between model size and knowledge capacity: as model size increases, the amount of knowledge it can capture grows, but with diminishing returns.*

---

## The Teacher-Student Framework

Knowledge Distillation uses a Teacher-Student model:
- **Teacher Model (Net-T):** A large, well-trained model (or ensemble) that achieves high accuracy.
- **Student Model (Net-S):** A smaller model trained to mimic the teacher's outputs.

The process:
1. **Train the Teacher:** Use a large model or ensemble to achieve high accuracy.
2. **Train the Student:** Use the teacher's output (soft targets) as supervision for the student, in addition to the true labels.

![Teacher-Student Framework](https://raw.githubusercontent.com/PANXiao1994/PANXiao1994.github.io/master/2019/12/26/blog_tech/ml/algo/nlp/kd/knowledge_distillation/kd-1.jpg)

The student learns not only from the ground-truth labels but also from the "soft targets" (probability distributions) produced by the teacher.

---

## Theoretical Foundation

The main goal of machine learning is to train models with strong generalization ability. By using a teacher with good generalization, the student can learn to generalize better, even with fewer parameters.

### Softmax and Temperature

The teacher's output is softened using a temperature parameter $T$ in the softmax:

$$
p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}, \quad q_i = \frac{e^{v_i/T}}{\sum_j e^{v_j/T}}
$$

- $T=1$ is the standard softmax.
- $T>1$ produces a softer, more uniform distribution, revealing more information about class similarities.

---

## Mathematical Derivation

The distillation loss is:

$$
L_{soft} = -\sum_i q_i \log p_i
$$

The gradient with respect to the student's logits is:

$$
\frac{\partial L_{soft}}{\partial z_i} = \frac{1}{T}(q_i - p_i)
$$

When $T$ is large, this approximates a mean squared error between logits:

$$
\frac{\partial L_{soft}}{\partial z_i} \approx \frac{1}{N T^2}(z_i - v_i)
$$

So, distillation with high temperature is similar to minimizing the squared difference between teacher and student logits.

![Distillation Process](https://raw.githubusercontent.com/PANXiao1994/PANXiao1994.github.io/master/2019/12/26/blog_tech/ml/algo/nlp/kd/knowledge_distillation/kd-2.png)

---

## The Role of Temperature

- The softmax temperature $T$ controls the smoothness of the output probability distribution.
- Higher $T$ allows the student to learn from the teacher's knowledge about less likely classes (the "dark knowledge").
- The choice of $T$ is empirical: higher $T$ helps the student learn more from negative labels, but too high $T$ may introduce noise.
- For smaller student models, a lower $T$ is often sufficient.

![Temperature Effect](https://raw.githubusercontent.com/PANXiao1994/PANXiao1994.github.io/master/2019/12/26/blog_tech/ml/algo/nlp/kd/knowledge_distillation/kd-3.jpg)

**How to choose $T$?**
- Higher $T$ means the student pays more attention to negative labels, but may also learn more noise.
- Lower $T$ means the student focuses more on the main class, ignoring some useful information.
- In practice, $T$ is a hyperparameter to tune, and smaller student models often use lower $T$.

---

## Practical Considerations

The student is trained with a combination of the standard cross-entropy loss (with true labels) and the distillation loss (with teacher outputs):

$$
L = \alpha L_{CE} + (1-\alpha) L_{soft}
$$

where $\alpha$ balances the two losses.

![Distillation in Practice](https://raw.githubusercontent.com/PANXiao1994/PANXiao1994.github.io/master/2019/12/26/blog_tech/ml/algo/nlp/kd/knowledge_distillation/kd.png)

- The method is widely used in industry for deploying efficient models on edge devices and in production systems.

---

## Example: MNIST Distillation

![MNIST Example](https://raw.githubusercontent.com/PANXiao1994/PANXiao1994.github.io/master/2019/12/26/blog_tech/ml/algo/nlp/kd/knowledge_distillation/mnist.jpg)

*Knowledge distillation applied to MNIST digit classification.*

---

## Summary & Tips

- Knowledge Distillation is a powerful and practical approach for model compression, enabling the deployment of efficient models without significant loss in accuracy.
- The temperature parameter plays a key role in controlling the information transferred from teacher to student.
- The "container and water" analogy helps understand underfitting and overfitting in model design.
- Always tune the temperature and loss balance hyperparameters for your specific task and model size.

---

**References:**
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- [Original Zhihu post (Chinese)](https://zhuanlan.zhihu.com/p/102038521)
- [Original Hexo post (Chinese)](https://raw.githubusercontent.com/PANXiao1994/PANXiao1994.github.io/refs/heads/master/2019/12/26/blog_tech/ml/algo/nlp/kd/knowledge_distillation/index.html) 