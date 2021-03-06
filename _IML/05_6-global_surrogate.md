---
title:  "5.6 Global Surrogate"
permalink: /IML/global_surrogate/
toc: true
---

# Global Surrogate  

글로벌 대리 모델(global surrogate model) 은 블랙박스 모델의 예측에 근사하도록 훈련된 해석 가능한 모델입니다.
대리 모델을 해석함으로써 블랙박스 모델에 대한 결론을 도출할 수 있습니다.
더 많은 머신러닝을 사용하여 머신러닝의 해석성을 해결합니다!

# 이론

대리 모델은 엔지니어링에도 사용됩니다.
관심 결과가 연산량이 많거나, 시간이 많이 걸리거나, 측정하기 어려운 경우(예: 복잡한 컴퓨터 시뮬레이션) 결과의 값싸고 빠른 대리 모델을 대신 사용할 수 있습니다.
엔지니어링에 사용되는 대리 모델과 해석 가능한 머신러닝의 차이점은 기본 모델이 (시뮬레이션이 아닌) 머신러닝 모델이며 대리 모델을 해석할 수 있어야 한다는 것입니다.
(해석 가능한) 대리 모델의 목적은 가능한 한 정확하게 기초 모델의 예측을 근사화하고 동시에 해석할 수 있도록 하는 것입니다.
대리 모델 아이디어는 Approximation model, metamodel, response surface model, emulator 등 다양한 이름으로 확인할 수 있습니다

이론은 다음과 같습니다.
사실 대리 모델을 이해하는데 필요한 이론은 별로 없습니다.
G가 해석 가능한 제약 조건 하에서, 블랙 박스 예측 함수 f를 대리 모델 예측 함수 g와 가능한 가깝게 하고 싶습니다.
함수 g의 경우 [interpretable models 장](https://tootouch.github.io/IML/interpretable_models/)과 같은 해석 가능한 모델을 사용할 수 있습니다.

선형 모델을 예로 들 수 있습니다.

$$g(x)=\beta_0+\beta_1{}x_1{}+\ldots+\beta_p{}x_p$$

또는 의사결정 트리입니다.

$$g(x)=\sum_{m=1}^Mc_m{}I\{x\in{}R_m\}$$

대리 모델을 학습하는 것은 model-agnostic 방법입니다. 블랙박스 모델의 내부 작업에 대한 정보는 필요하지 않으며, 데이터와 예측 함수만 필요합니다.
기본 학습 모델을 다른 것으로 교체한 경우에도 대리 방법을 사용할 수 있습니다.
블랙 박스 모델 유형과 대리 모델 유형의 선택은 무엇이든 상관 없습니다.

다음 단계를 수행하여 대리 모델을 학습합니다.

1. 데이터 집합 X를 선택합니다. 이는 블랙박스 모델 학습에 사용된 데이터 세트 또는 동일한 분포의 새 데이터 세트와 동일할 수 있습니다. 애플리케이션에 따라 데이터의 하위 집합 또는 점 그리드를 선택할 수도 있습니다.
2. 선택한 데이터 집합 X의 경우 블랙 박스 모델의 예측값을 얻습니다.
3. 해석 가능한 모델 유형(선형 모델, 의사결정 트리 등)을 선택합니다.
4. 데이터 세트 X와 그 예측에 대한 해석 가능한 모델을 학습합니다.
5. 축하해요! 이제 대리 모델을 갖게 되었습니다.
6. 블랙 박스 모델의 예측을 얼마나 잘 반영하는지 측정합니다.
7. 대리 모델을 해석합니다.

일부 추가 단계가 있거나 약간 다른 대리 모델에 대한 접근 방법을 찾을 수 있지만 일반적으로 여기서 설명한 대로입니다.

대리 모델이 블랙 박스 모델을 얼마나 잘 반영하는지 측정하는 한 가지 방법은 R-제곱을 계산하는 것입니다.

$$R^2=1-\frac{SSE}{SST}=1-\frac{\sum_{i=1}^n(\hat{y}_*^{(i)}-\hat{y}^{(i)})^2}{\sum_{i=1}^n(\hat{y}^{(i)}-\bar{\hat{y}})^2}$$

여기서 $$\hat{y}_*^{(i)}$$는 대리 모델의 i번째 관측치에 대한 예측값입니다. $$\hat{y}^{(i)}$$ 블랙박스 모델의 예측값과 $$\bar{\hat{y}$$ 블랙박스 모델 예측값의 평균입니다.
SSE는 제곱합 오차, SST는 제곱합 합계를 나타냅니다.
R 제곱 측정은 대리 모델에 대한 분산 백분율로 해석할 수 있습니다.
R-제곱이 1(= 낮은 SSE)에 가까우면 해석 가능한 모델은 블랙 박스 모델의 동작에 매우 근사하게 됩니다.
해석 가능한 모델이 매우 가까운 경우 복합 모델을 해석 가능한 모델로 교체해야 할 수 있습니다.
R-제곱이 0(= 하이 SSE)에 가까우면 해석 가능한 모델이 블랙 박스 모델을 설명하지 못합니다.

기본 블랙박스 모델의 모델 성능, 즉 실제 결과를 예측하는 데 얼마나 좋은지 나쁜지에 대해서는 언급하지 않았습니다.
블랙박스 모델의 성능은 대리 모델을 학습하는 데 영향을 미치지 않습니다.
대리 모델에 대한 해석은 실제 세상에 대한 것이 아니라 모델에 대한 것만 다루기 때문에 문제는 없습니다.
하지만 물론 블랙박스 모델이 나쁘면 대리 모델의 해석은 의미가 없게 됩니다. 블랙박스 모델 자체가 의미가 없기 때문입니다.


또한 원본 데이터의 하위 집합에 기반한 대리 모델을 구축하거나 관측치의 가중치를 재조정할 수도 있습니다.
이런 식으로, 우리는 대리 모델의 입력의 분포를 바꾸는데, 이것은 해석의 초점을 바꿉니다(이렇게되면 더 이상 글로벌하다고 할 수 없습니다).
특정 데이터 관측치(선택한 관측치에 가까울수록 가중치)를 기준으로 데이터를 로컬로 가중치를 부여하면 해당 관측치의 개별 예측을 설명할 수 있는 로컬 대리 모델이 생성됩니다.
[다음 장]([#lime](https://tootouch.github.io/IML/local_surrogate/))에서 로컬 모델에 대한 자세한 내용을 읽습니다.

# 예시

대리 모델을 구현하기 위해 회귀 분석 및 분류 예를 고려합니다.

첫째, 날씨 및 캘린더 정보를 고려하여 [일별 대여 자전거 수](https://tootouch.github.io/IML/bike_rentals/)를 예측하는 서포트 벡터 머신을 학습합니다.
서포트 벡터 머신은 해석할 수 없으므로, CART 의사결정 트리를 사용하여 서포트 벡터 머신의 성능에 근접한 해석 가능한 모델로 대체 모델을 학습합니다.

<p align='center'>
    <img src='https://christophm.github.io/interpretable-ml-book/images/surrogate-bike-1.png'><br>
    <i>그림 5.31: 자전거 대여 데이터셋에서 훈련된 서포트 벡터 머신의 예측에 근접한 대리 트리의 터미널 노드입니다. 노드의 분포에 따르면 대리 트리는 온도가 섭씨 13도 이상일 때와 2년 후(435일)에 렌트된 자전거의 수가 더 많을 것으로 예측하고 있습니다.</i>
</p>

대리 모델에는 0.77의 R-quared(변동성 설명력)가 있는데, 이는 블랙 박스 모델의 성능에 상당히 가깝지만 완벽하지는 않다는 것을 의미합니다.
만약 적합성이 완벽하다면, 우리는 서포트 벡터 머신를 버리고 대신 트리를 사용할 수 있습니다.

두 번째 예에서, 우리는 랜덤 포레스트와 함께 [자궁경부암](https://tootouch.github.io/IML/cervical_cancer/)의 확률을 예측합니다.
다시 원래의 데이터 세트를 사용하여 의사결정 트리를 학습하지만, 데이터의 실제 클래스(건강 vs 암) 대신 랜덤 포리스트의 예측을 통해 결과를 예측합니다.

<p align='center'>
    <img src='https://christophm.github.io/interpretable-ml-book/images/surrogate-cervical-1.png'><br>
    <i>그림 5.32: 자궁경부암 데이터 집합에서 훈련된 랜덤 포레스트의 예측에 근접한 대리 모델의 중간 노드입니다. 노드의 개수는 노드에서 블랙 박스 모델 분류의 빈도를 나타냅니다.</i>
</p>

대리 모델에는 0.19의 R-squared(변동 설명)가 있는데, 이는 랜덤 포레스트에 가까운 것이 아니며 복잡한 모델에 대한 결론을 도출할 때 트리를 지나치게 해석해서는 안 된다는 의미입니다.

# 장점 

대리 모델 방법은 **유연성이 있습니다.**
[해석 가능한 모델 장](https://tootouch.github.io/IML/interpretable_models/)의 모든 모델을 사용할 수 있습니다.
이는 해석 가능한 모델뿐만 아니라 기본 블랙박스 모델도 바꿀 수 있음을 의미합니다.
복잡한 모델을 만들어 회사 내의 다른 팀에게 설명한다고 가정합니다.
한 팀은 선형 모델에 익숙하고 다른 팀은 의사결정 트리에 익숙할 수 있습니다.
원래 블랙박스 모델에 대해 두 가지 대리 모델(선형 모델 및 의사결정 트리)을 학습하고 두 가지 종류의 설명을 제공할 수 있습니다.
더 나은 성능을 제공하는 블랙박스 모델을 찾는 경우 동일한 클래스의 대리 모델을 사용할 수 있으므로 해석 방법을 변경할 필요가 없습니다.

저는 이 접근법이 매우 직관적이고 간단하다고 생각합니다.
구현이 쉬울뿐만 아니라 데이터 과학이나 머신러닝에 익숙하지 않은 사람들에게 설명하기도 쉽다는 것을 의미합니다.

**R 제곱 측정값**을 사용하면 대리 모델이 블랙박스 예측치에 근접한 성능을 가지는지 쉽게 측정할 수 있습니다.


# 단점

대리 모델은 실제 결과를 보지 못하므로 **모델에 대한 결론을 도출해야 합니다**.

**R-squared에 가장 적합한 기준**이 무엇인지 명확하지 않습니다. 이는 대리 모델이 블랙박스 모델과 충분히 가깝다는 것을 확신하기 위한 것입니다. 80% 정도가 좋을까요? 50%? 99%?

대리 모델이 블랙박스 모델과 얼마나 가까운지 측정할 수 있습니다.
우리가 아주 가깝지는 않지만, 충분히 가깝다고 가정해 보죠.
해석 가능한 모델은 데이터 집합의 **한 부분집합에 대해 매우 근접하지만 다른 부분집합에는 크게 다를 수 있습니다.**
이 경우 단순 모델에 대한 해석은 모든 데이터 포인트에 대해 동일하지 않습니다.

대리로 선택한 해석 가능한 모델은  **모든 장단점을 그대로 가지고 있습니다.**.

어떤 사람들은 **본질적으로 해석할 수 있는 모델이 없으며**(선형 모델과 의사결정 트리도 포함), 해석 가능성에 대한 환상을 갖는 것은 위험할 수도 있다고 주장합니다.
만약 당신이 이 의견을 공감한다면, 물론 이 방법은 당신을 위한 것이 아닙니다.

# 소프트웨어

예시에는 'iml' R 패키지를 사용했습니다.
만약 여러분이 머신러닝 모델을 훈련시킬 수 있다면, 여러분은 스스로 대리 모델을 구현할 수 있어야 합니다.
블랙 박스 모델의 예측을 예측하기 위해 해석 가능한 모델을 학습하기만 하면 됩니다.