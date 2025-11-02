# SkelDPO

⚡️ SkelDPO is a novel framework that extends conventional DPO by incorporating skeleton-guided structural preference modeling.
This repository contains code, data, and models related to the ICPC 2026 paper: "SkelDPO: A Skeleton-Guided Direct Preference Optimization Framework for Efficient Code Generation".

![SkelDPO Framework](assets/images/SkelDPO.png)

<details>
  <summary>Contributions</summary>

  - Skeleton-guided preference optimization framework (SkelDPO). We propose SkelDPO, which introduces a skeleton preference signal into the traditional DPO framework. By jointly optimizing code and skeleton preferences, the model retains functional correctness while gaining structural inductive capability, thereby significantly improving code generation efficiency.
  - Fine-grained dataset of efficient and inefficient code with aligned skeletons. We construct a fine-grained dataset by screening and aligning efficient and inefficient implementations across multiple tasks. This dataset provides directly usable sample pairs for skeleton preference optimization and establishes a new foundation for future efficiency oriented research.
  - Extensive experimental validation across multiple models and benchmarks. Comprehensive experiments on Mercury and ENAMEL demonstrate that SkelDPO significantly outperforms existing approaches (such as EffiCoder and CodeDPO) and exhibits superior generalization on complex tasks.

</details>

## 🏋️ Training

First, fine-tune the base model in a SkelDPO framework:
<pre>
python train_SkelDPO.py
</pre>

## ✨ Inference

Generate efficient code for different training models:
<pre>
python generate_Mercury.py
python generate_ENAMEL.py
</pre>

## 📊 Evaluate

You can evaluate the functional correctness and efficiency of the generated code on Mercury and ENAMEL:
<pre>
python evaluator.py
python evaluate.py
</pre>
