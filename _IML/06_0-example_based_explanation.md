---
title:  "Chapter 6. Example-Based Explanations"
permalink: /IML/example_based_explanations/
---

예제 기반 설명 방법은 데이터 세트의 특정 인스턴스를 선택하여 기계 학습 모델의 동작을 설명하거나 기본 데이터 배포를 설명합니다.

<!-- *Keywords: example-based explanations, case-based reasoning (CBR), solving by analogy* -->

예제 기반 설명은 대부분 모델에 구애받지 않습니다. 기계 학습 모델을 해석할 수 있도록 하기 때문입니다.
모델별 방법의 차이점은 예제 기반 방법이 [기능 중요도](#feature-importance) 또는 [부분 의존도](#pdp)와 같은 기능의 요약을 작성하는 것이 아니라 데이터 집합의 인스턴스를 선택하여 모델을 설명하는 것입니다.
예제 기반 설명은 인간적으로 이해할 수 있는 방식으로 데이터의 인스턴스를 나타낼 수 있는 경우에만 의미가 있습니다.
이것은 우리가 직접 볼 수 있기 때문에 이미지에 적합합니다.
일반적으로 예제 기반 방법은 인스턴스의 피쳐 값이 더 많은 컨텍스트를 포함하는 경우 잘 작동합니다. 즉, 데이터에는 이미지나 텍스트와 같은 구조가 있습니다.
한 인스턴스가 수백 또는 수천 개의 (구조화되지 않은) 기능으로 구성될 수 있기 때문에 표식 데이터를 의미 있는 방식으로 표현하는 것은 더 어렵습니다.
인스턴스를 설명하기 위해 모든 피쳐 값을 나열하는 것은 일반적으로 유용하지 않습니다.
몇 가지 기능만 있거나 인스턴스를 요약할 수 있는 방법이 있는 경우 잘 작동합니다.


예제 기반의 설명은 인간이 기계 학습 모델의 정신적 모델과 기계 학습 모델이 훈련된 데이터를 구성하는 데 도움이 됩니다.
특히 복잡한 데이터 분포를 이해하는 데 도움이 됩니다.
하지만 제가 예시를 바탕으로 설명한다는 것은 무엇을 의미할까요?
우리는 직업과 일상 생활에서 종종 그것들을 사용합니다.
먼저 몇 가지 예를 들어 보겠습니다

의사가 보기 드문 기침과 미열이 있는 환자를 봅니다.
환자의 증상은 몇 년 전에 비슷한 증상을 보였었던 또 다른 환자를 떠올리게 합니다.
그녀는 현재 환자도 같은 병에 걸릴 수 있다고 의심하고 혈액 샘플을 채취하여 이 특정한 질병을 검사합니다.

데이터 과학자가 고객 중 한 명을 위해 새로운 프로젝트를 진행하고 있습니다.
키보드의 프로덕션 머신 고장으로 이어지는 위험 요인을 분석합니다.
데이터 과학자는 자신이 작업한 유사한 프로젝트를 기억하고 있으며 고객이 동일한 분석을 원한다고 생각하기 때문에 이전 프로젝트에서 코드 일부를 재사용합니다.

불타고 사람이 살지 않는 집의 창문 선반 위에 새끼 고양이가 앉아 있습니다.
소방서는 이미 도착했고 소방관들 중 한 명은 그가 새끼 고양이를 구하기 위해 건물 안으로 들어가는 위험을 감수할 수 있을지 잠시 고민하고 있습니다.
그는 소방관으로서 그의 삶에서 비슷한 사건들을 기억합니다.
한동안 천천히 타오르던 오래된 목조 가옥들이 불안정한 경우가 많았고 결국 무너졌습니다.
이 사건의 유사성 때문에, 그는 들어가지 않기로 결심합니다. 왜냐하면 집이 무너질 위험이 너무 크기 때문입니다.
다행히도, 그 고양이는 창문 밖으로 뛰어나와 안전하게 착륙하고 아무도 그 화재로 다치지 않았습니다. 해피엔딩입니다.

이 이야기들은 우리 인간들이 어떻게 생각하는지 예시나 유사성을 보여줍니다.
예제 기반 설명의 Blueprint는 다음과 같습니다.
Ting B는 A와 A가 Y를 일으킨 것과 비슷하기 때문에 B도 Y를 일으킬 것으로 예측합니다.
암시적으로, 일부 기계 학습은 작업 예제를 기반으로 접근합니다.
[Decision Tree](#tree)는 대상을 예측하는 데 중요한 기능의 데이터 포인트의 유사성을 바탕으로 데이터를 노드로 분할합니다.
의사 결정 트리는 유사한 인스턴스(= 같은 터미널 노드)를 찾아 해당 인스턴스의 결과 평균을 예측과 같이 반환하여 새 데이터 인스턴스에 대한 예측을 가져옵니다.
The k-nearest neighbors (knn) method works explicitly with example-based predictions. 
For a new instance, a knn model locates the k-nearest neighbors (e.g. the k=3 closest instances) and returns the average of the outcomes of those neighbors as a prediction.
The prediction of a knn can be explained by returning the k neighbors, which -- again -- is only meaningful if we have a good way to represent a single instance.

The chapters in this part cover the following example-based interpretation methods:

- [Counterfactual explanations](#counterfactual) tell us how an instance has to change to significantly change its prediction. 
By creating counterfactual instances, we learn  about how the model makes its predictions and can explain individual predictions.
- [Adversarial examples](#adversarial) are counterfactuals used to fool machine learning models. 
The emphasis is on flipping the prediction and not explaining it. 
- [Prototypes](#proto) are a selection of representative instances from the data and criticisms are instances that are not well represented by those prototypes. [^critique]
- [Influential instances](#influential) are the training data points that were the most influential for the parameters of a prediction model or the predictions themselves. 
Identifying and analysing influential instances helps to find problems with the data, debug the model and understand the model's behavior better.
- [k-nearest neighbors model](#other-interpretable): An (interpretable) machine learning model  based on examples.

---

[^1]: Aamodt, Agnar, and Enric Plaza. "Case-based reasoning: Foundational issues, methodological variations, and system approaches." AI communications 7.1 (1994): 39-59.