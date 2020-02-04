---
title:  "Chapter 2. Interpretability"
permalink: /IML/interpretability/
---

이 챕터에서는 해석가능성(interpretability)에 대한 수학적 정의를 다루지 않습니다. Miller(2017)[^1]의 정의에서는 **해석가능성이란 의사결정의 근거를 사람이 이해할 수 있는 정도**라고 말합니다. 또 다른이는 **해석가능성이란 사람이 모델의 결과를 일관적으로 예측해낼 수 있는 정로**라고 말합니다[^2]. 머신러닝 모델의 해석가능성이 높아질수록 어떤 의사결정이 내려졌는지나 예측값이 발생했는지에 대해 왜 그런지 이해하는 것이 더 쉬워집니다. 하나의 모델이 다른 모델보다 더 해석이 쉽다는 것은 다른 모델보다 해당 모델의 의사결정에 대해 더 이해하기 쉽다는 것입니다. 이 책에서는 해석가능함(Interpretable)과 설명가능함(explainable)을 바꿔가며 사용합니다. Miller의 말처럼 해석가능함과 설명가능함은 서로 구분되어져야 합니다. "설명성(explanation)"은 각 예측값들에 대한 설명을 나타낼때 사용합니다. 사람에게 좋은 설명이란 무엇인가 궁금하다면 [설명에 대한 섹션](https://tootouch.github.io/IML/human_friendly_explanations/)을 참고해주세요.

---

[^1]: iller, Tim. “Explanation in artificial intelligence: Insights from the social sciences.” arXiv Preprint arXiv:1706.07269. (2017).

[^2]: Kim, Been, Rajiv Khanna, and Oluwasanmi O. Koyejo. “Examples are not enough, learn to criticize! Criticism for interpretability.” Advances in Neural Information Processing Systems (2016).