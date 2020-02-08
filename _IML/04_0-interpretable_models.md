---
title:  "Chapter 4. Interpretable Models"
permalink: /IML/interpretable_models/
---

해석가능성을 만드는 가장 쉬운 방법은 해석가능한 모델만 사용하는 것입니다. 선형 회귀, 로지스틱 회귀 그리고 의사결정나무는 모두 해석가능한 모델입니다.

이후 소개될 내용들에서는 앞서 언급한 모델들에 대해 다룹니다. 자세하게 설명하기보다는 이미 너무 많은 책, 강의, 논문 그리고 더 많은 자료들이 있기 때문에 기본적인 내용만 다룹니다. 여기서는 모델을 어떻게 해석하는지에 대해서만 얘기합니다. 주로 여기서는 선형 회귀(linear regression), 로지스틱 회귀(logistic regression), 다른 선형 회귀의 확장모델, 의사결정나무(decision trees), 의사결정규칙(dicision rules) 그리고 RuleFit에 대해 더 자세히 다룹니다. 그외 다른 해석가능한 모델들도 있습니다. 

이 책에서 설명하는 모든 해석가능한 모델들은 모듈 수준에서 해석가능합니다(k-최근접이웃 방법은 제외).아래 표는 해석가능한 모델들의 유형과 속성들에 대한 개요입니다. 특성들과 목표값간의 연관성을 선형적으로 모델링하는 모델은 선형 모델입니다. 단조성 제약이 있는 모델은 특성의 전체 범위에 대해 모두 각 특성과 목표값의 관계가 같은 방향을 나타냅니다. 즉, 특성이 커지면 결과값은 항상 커지거나 작아집니다. 단조성은 관계에 대한 설명하는게 쉽기때문에 모델 해석에 유용합니다. 어떤 모델들은 결과값을 예측하기위한 특성간 상호작용이 포함됩니다. 상호작용하는 특성을 만들게되면 어떤 모델이든 상호작용을 포함하게 됩니다. 상호작용은 모델 성늘을 향상시켜줄 수 있지만 너무 과하면 해석이 어려워집니다. 몇몇 모델들은 회귀만 다루기도하고 또는 분류 그리고 둘 다 포함되기도 합니다.

이 표를 통해 여러분은 각자 문제에 맞게 회귀든 분류든 적절한 해석가능한 모델을 선택하실 수 있습니다.

Algorithm|	Linear|	Monotone|	Interaction	Task
---|---|---|---
Linear regression|	Yes|	Yes|	No|	regr
Logistic regression|	N|o	Yes|	No|	class
Decision trees|	No|	Some|	Yes|	class,regr
RuleFit|	Yes|	No|	Yes|	class,regr
Naive Bayes|	No|	Yes|	No|	class
k-nearest neighbors|	No|	No|	No|	class,regr