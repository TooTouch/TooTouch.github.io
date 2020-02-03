---
title:  "Introduction"
permalink: /IML/introduction/
---

이 책은 여러분에게 (지도학습 방법의) 기계학습 모델을 해석할 수 있는 방법을 알려주는 책입니다. 이후 소개될 chapter들에서는 몇몇 수학적 수식이 포함되어있지만 수식을 모두 이해하지는 못해도 방법에 대한 아이디어는 이해하실 수 있습니다. 이 책은 기계학습을 밑바닥부터 시작하는 사람들을 위한 책은 아닙니다. 만약 기계학습을 처음 공부한다면 기본기를 알려주는 좋은 책들과 자료들이 많이 있습니다. 개인적으로 추천하기에는 Hastie, Tibashirani 그리고 Friedman의 "The Elements of Statistical Learning"[^1]이 있고 온라인 교육과정인 coursera에서 [Andrew Ng의 "Machine Learning"](https://www.coursera.org/learn/machine-learning) 과정을 들을 수 있습니다. 모두 무료이니 부담없이 이용하실 수 있습니다.

해석가능한 기계학습의 새로운 방법들은 빠르게 생겨나고 있습니다. 새롭게 생긴 방법을 모두 담아내는것은 불가능했기 때문에 이 책에서는 최신 방법들이 없을 수 있지만 기계학습의 해석방법들에 대한 기본이 되는 방법들이나 개념들은 모두 담겨있습니다. 이 기본기들이 기계학습 모델들을 해석하는데 도움이 될 수 있습니다. 이 책을 읽기 시작했다면 기본 개념들을 잘 익히고 있기 때문에 이후 [arxiv.org](https://arxiv.org/)에 올라온 새로운 해석가능한 방법들에 대한 논문을 5분만에 빠르게 읽고 평가할 수도 있습니다. (저자는 이 말을 통해 더 많은 책이 팔리길 원하는 것 같다.)

이 책은 몇개의 스토리와 함께 시작합니다. 꼭 책을 이해하는데 필요한 내용은 아니지만 독자들에게 더 많은 생각과 즐거움을 드릴 수 있길 바랍니다. 이후에는 기계학습의 해석가능성에 대해서 공부합니다. 그리고 해석가능성이 중요한 시점이 언제인지 어떤 종류의 설명들이 있는지 알아봅니다. 이 책에 사용된 용어들은 Terminology chapter에서 확인하실 수 있습니다. 대부분 모델들와 방법들은 Data chapter에서 설명한 실제 데이터를 통해 나타냈습니다. 기계학습을 해석가능하도록 하는 방법 중 하나는 선형 모델이나 의사 결정 나무와 같은 해석가능한 모델을 사용하는 것입니다. 다른 방법은 어떤 지도학습 모델이던 사용 가능하도록 모델을 해석할 수 있는 도구(model-agnostic interpretation tools)를 사용하는 것입니다.  Model-Agnostic Methods chapter에서는 partial dependence plots과 permutation feature importance같은 방법들에 대해 설명합니다. Model-agnostic 방법들은 입력값이 변함에 따라 예측값이 얼마나 변하는지는 측정하는식입니다. 각 관측치를 설명으로 반환하는 model-agnostic 방법들은 Example Based Explanation에서 다룹니다. 모든 model-agnostic 방법들은 모든 데이터에 대해 모델 전체를 설명하는지 각 예측치를 설명하는지에 따라 차이가 있습니다. 모델의 전체를 설명하는 방법들은 Partial Dependence Plots, Accumulated Local Effacts, Feature Interaction, Feature Importance, Global Surrogate Models 그리고 Prototypes and Criticisms이 있습니다. 각 예측치에 대해 설명하기 위해서는 Local Surrogate Models, Shapley Value Explanations, Couterfactual Explanations (Adversarial Examples와 연관이 있음)과 같은 방법들이 있습니다. Individual Conditional Expectation과 Influential Instances와 같이 몇몇 방법들은 두 가지 경우 모두 사용할 수 있습니다. 

이 책의 마지막에는 해석가능한 기계 학습에 대한 낙관적인 전망에 대해 알아봅니다.

만약 관심이 있다면 바로 마지막 장으로 건너뛰어 읽어 볼 수 있습니다.

재밌게 읽어주세요!

---

[^1]: Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. “The elements of statistical learning”. [www.web.stanford.edu/~hastie/ElemStatLearn/](http://www.web.stanford.edu/~hastie/ElemStatLearn/) (2009)