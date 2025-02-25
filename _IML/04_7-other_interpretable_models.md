---
title:  "4.7 Other Interpretable Models"
permalink: /IML/other_interpretable_models/
toc: true
---

# 다른 해석 가능한 모델

해석 가능한 모델들은 계속 생겨나고 있으며 크기를 알 수 없습니다.
선형 모형, 의사결정 나무, 나이브 베이즈 등 간단한 모델도 포함하지만, 해석할 수 없는 머신러닝 모델을 조합하거나 수정하여 해석할 수 있는 보다 복잡한 모델도 포함되어 있습니다.
특히 후자의 방법은 현재 굉장히 많이 생겨나고 있는 중이며, 추세를 따라가기가 어렵습니다.
이 장에서는 Naive Bayes Classifier와 K-Nearest Neighbors만 다룹니다.

# Naive Bayes Classifier

Naive Bayes 분류기는 조건 확률에 대한 Bayes 이론을 사용합니다.
각 특성에 대해 특성 값에 따라 클래스의 확률을 계산합니다.
Naive Bayes 분류기는 각 특성에 대한 클래스 확률을 독립적으로 계산합니다. 이는 특성의 독립성에 대한 강력한(= naive) 가정과 같습니다.
Naive Bayes는 조건부 확률 모델이며 다음과 같이 $$C_k$$ 클래스의 확률을 모델링합니다.

$$P(C_k|x)=\frac{1}{Z}P(C_k)\prod_{i=1}^n{}P(x_i|C_k)$$

Z는 모든 클래스에 대한 확률의 합계가 1이 되도록 하는 스케일링 파라미터입니다(그렇지 않으면 확률이 아닐 것입니다).
클래스의 조건부 확률은 클래스에 주어진 각 특성의 확률의 클래스 확률입니다. Z에 의해 정규화됩니다.
이 공식은 베이즈의 정리를 사용하여 도출할 수 있습니다.

Naive Bayes는 독립성 가정 때문에 해석할 수 있는 모델입니다.
모듈식으로 해석할 수 있습니다.
조건부 확률을 해석할 수 있기 때문에 특정 클래스 예측에 얼마나 기여하는지는 각 특성에 대해 매우 명확히 알 수 있습니다.

# K-Nearest Neighbors

k-최근접 이웃 방법은 회귀 및 분류에 사용할 수 있으며 데이터 지점의 가장 가까운 인접 항목을 예측에 사용합니다.
분류의 경우 k-최근접 이웃 방법은 관측치의 최근접 이웃의 가장 일반적인 클래스를 할당합니다.
회귀 분석의 경우 이웃의 결과 평균을 냅니다.
까다로운 부분은 올바른 k를 찾아 관측치 간의 거리를 측정하는 방법을 결정하는 것입니다. 이는 궁극적으로 이웃을 정의합니다.

k-최근접 이웃 모델은 관측치 기반의 학습 알고리즘이기 때문에 이 책에 제시된 다른 해석 가능한 모델과 다릅니다.
최근접 이웃은 어떻게 해석될 수 있을까요?
우선 학습할 파라미터가 없기 때문에 모듈형 레벨에서는 해석성이 없습니다.
더욱이, 모델은 본래 지역적이고 명시적으로 학습된 전역 가중치나 구조가 없기 때문에 글로벌 모델 해석성이 부족합니다.
지역 차원에서 해석이 가능할까요?
예측을 설명하기 위해 예측에 사용된 k 인접 항목을 항상 확인할 수 있습니다.
모델이 해석 가능한지 여부는 데이터 집합에서 단일 관측치를 '해석'할 수 있는지 여부에 따라 다릅니다.
한 예가 수백 또는 수천 개의 특성으로 구성되어 있다면, 저는 그것을 해석할 수 없다고 주장할 것입니다.
그러나 특성이 거의 없거나 관측치를 가장 중요한 특성으로 축소할 수 있는 방법이 있다면 k와 최근접 이웃을 제시하면 좋은 설명을 할 수 있습니다.