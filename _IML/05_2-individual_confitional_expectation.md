---
title:  "5.2 Individual Conditional Expectation (ICE)"
permalink: /IML/individual_conditional_expectation/
toc: true
---

# Individual Conditional Expectation (ICE) 

개별 조건부 기대(ICE) 그림은 특성이 변경될 때 관측치의 예측이 어떻게 변경되는지 보여주는 관측치당 한 줄씩 표시합니다.

특성의 평균 효과에 대한 부분 의존도는 특정 관측치에 초점을 맞추지 않고 전체 평균에 초점을 맞추기 때문에 전역 방법입니다.
개별 데이터 관측치에 대한 PDP와 동등한 것을 개별 조건부 기대(ICE) 그림이라고 합니다(Goldstein et al. 2017[^1]).
ICE 그림은 *각* 관측치에 대한 특성에 대한 예측의 의존도를 개별적으로 시각화하여, 부분 의존도 그림의 한 선과 비교하여 관측치당 한 줄이 생성됩니다.
PDP는 ICE 그림의 선 평균입니다.
선 및 관측치의 값은 다른 모든 특성을 동일하게 유지하여 계산할 수 있으며, 특성 값을 그리드의 값으로 바꾸고 새로 생성된 관측치의 블랙박스 모델로 예측하여 이 관측치의 변형을 만들 수 있습니다.
그 결과 그리드 및 각 예측의 특성 값이 있는 관측치의 포인트 집합이 됩니다.

부분 의존성 대신 개별적인 기대치를 살펴보는 것은 무슨 의미가 있을까요?
부분 의존도 그림은 상호 작용에 의해 생성된 이질적인 관계를 흐리게 할 수 있습니다.
PDP는 특성와 예측 간의 평균 관계를 보여 줍니다.
이는 PDP가 계산된 특성와 다른 특성 간의 상호 작용이 약한 경우에만 잘 작동합니다.
교호작용의 경우 ICE 그림은 훨씬 더 많은 통찰력을 제공합니다.

보다 공식적인 정의는 다음과 같습니다.
ICE 그림에서 $$\{(x_{S}^{(i)},x_{C}^{(i)})\}_{i=1}^N$$의 각 관측치에 대해 $$\hat{f}_S^{(i)}$$ 곡선이 $$x^{(i)}_{S}$$로 표시된 반면 $$x^{(i)}_{C}$$는 고정되어 있습니다.

# Examples

[자궁경부암 데이터](#cervical)로 돌아가서 각 관측치의 예측이 "Age" 특성과 어떻게 관련되어 있는지 알아보겠습니다.
여성 암 발생 확률을 예측하는 랜덤 포레스트로 분석해 보겠습니다.
[부분 의존도](#pdp)에서는 50세 전후에 암 발생 확률이 증가하는 것을 보아왔지만, 데이터 세트에 있는 모든 여성에게 해당합니까?
ICE의 그림은 대부분의 여성들에게 연령 효과는 50세의 평균 증가 패턴을 따른다는 것을 보여주지만, 몇 가지 예외가 있습니다.
어린 나이에 높은 예측 확률을 가진 소수의 여성들에게, 예측된 암 확률은 나이가 들수록 크게 변하지 않습니다.

<p align='center'>
    <img src='https://christophm.github.io/interpretable-ml-book/images/ice-cervical-1.png'><br>
    <i>그림 5.6: ICE는 연령별 자궁경부암 확률도입니다. 각 선은 한 명의 여자를 나타냅니다. 대부분의 여성들에게 있어 나이와 함께 예측된 암 발생 확률은 증가합니다. 암 발생률이 0.4를 초과하는 일부 여성에게는 이 예측이 더 높은 나이에도 크게 변하지 않습니다.</i>
</p>

다음 그림은 [bike 대여 예측](#bike-data)에 대한 ICE 그림을 보여줍니다.
기본 예측 모델은 랜덤 포리스트입니다.

<p align='center'>
    <img src='https://christophm.github.io/interpretable-ml-book/images/ice-bike-1.png'><br>
    <i>그림 5.7: ICE는 날씨 조건별 예상 자전거 대여량을 나타냅니다. 부분 의존도 그림에서와 동일한 효과를 관찰할 수 있습니다.</i>
</p>

모든 곡선은 같은 코스를 따르는 것 같으므로 명확한 상호 작용이 없습니다.
이는 PDP가 이미 표시된 특징과 예상 자전거 수 사이의 관계를 잘 요약했다는 것을 의미합니다

# Centered ICE Plot

ICE 그림에는 문제가 있습니다.
때로는 ICE 곡선이 서로 다른 예측에서 시작되기 때문에 개인 간에 다른지 구별하기가 어려울 수 있습니다.
간단한 해결 방법은 특성의 특정 점에 원곡선을 가운데 두고 이 점에 대한 예측의 차이만 표시하는 것입니다.
결과 그림을 centered ICE 그림(c-ICE)이라고 합니다.
특성의 하단에 곡선을 고정하는 것이 좋습니다.
새 곡선은 다음과 같이 정의됩니다

$$\hat{f}_{cent}^{(i)}=\hat{f}^{(i)}-\mathbf{1}\hat{f}(x^{a},x^{(i)}_{C})$$

여기서 $\mathbf{1}$는 적절한 치수 수(일반적으로 1 또는 2), $\hat{f}$는 적합 모델이고 x^a^는 앵커 포인트입니다.

## 예시

예를 들어, 나이에 대한 자궁경부암 ICE 그림에 가장 어린 나이에 선을 중심에 둡니다.

<p align='center'>
    <img src='https://christophm.github.io/interpretable-ml-book/images/ice-cervical-centered-1.png'><br>
    <i>그림 5.8: 연령별 암 발생 확률을 예측하는 ICE 중심 그림입니다. 선은 14세에 0으로 고정됩니다. 14세에 비해, 대부분의 여성에 대한 예측은 예측 확률이 증가하는 45세까지 변함이 없습니다.</i>
</p>

중앙 ICE 그림을 사용하면 개별 관측치의 곡선을 더 쉽게 비교할 수 있습니다.
이 특성은 예측 값의 절대 변경이 아니라 특성 범위의 고정점과 비교하여 예측의 차이를 확인하는 데 유용할 수 있습니다.

자전거 대여 예측을 위한 ICE 중심 그림을 살펴보겠습니다.

<p align='center'>
    <img src='https://christophm.github.io/interpretable-ml-book/images/ice-bike-centered-1.png'><br>
    <i>그림 5.9: 중심 ICE 그림에서 날씨 조건별 예상 자전거 수입니다. 선은 관측된 최소치에서 각 형상값과 예측 대비 예측의 차이를 보여줍니다.</i>
</p>

# Derivative ICE Plot

시각적으로 이질성을 더 쉽게 발견할 수 있도록 하는 또 다른 방법은 특성에 관한 예측 함수의 개별 파생(derivative)을 살펴보는 것입니다.
결과 그림을 파생 ICE 그림(d-ICE)이라고 합니다.
함수(또는 곡선)의 파생 모델은 변경이 발생하는지 여부와 변경이 발생하는 방향을 알려줍니다.
파생 ICE 그림을 사용하면 (최소한 일부) 관측치에 대해 블랙박스 예측이 변경되는 특성 값의 범위를 쉽게 찾을 수 있습니다.
분석한 특성 $$x_S$$와 다른 특성 $$x_C$$ 간에 상호 작용이 없는 경우 예측 함수는 다음과 같이 나타낼 수 있습니다.

$$\hat{f}(x)=\hat{f}(x_S,x_C)=g(x_S)+h(x_C),\quad\text{with}\quad\frac{\delta\hat{f}(x)}{\delta{}x_S}=g'(x_S)$$

상호작용이 없으면 개별 부분파생상품은 모든 경우에 동일해야 합니다.
서로 다른 경우 이는 상호 작용 때문이며 d-ICE 그림에서 볼 수 있습니다.
S의 특징과 관련하여 예측 함수의 파생 모델에 대한 개별 곡선을 표시하는 것 외에도, 파생 모델의 표준 편차를 보여주는 것은 추정 파생 모델의 이질성을 가진 S의 특징 영역을 강조하는 데 도움이 됩니다.
파생형 ICE 그림은 계산하는 데 오랜 시간이 걸리며 다소 비현실적입니다.


# 장점

개별 조건부 기대 곡선은 부분 의존도 그림보다 **이해하기 훨씬 직관적입니다.**
한 행은 관심 특성을 변경하는 경우 한 관측치에 대한 예측을 나타냅니다.

부분 의존도 그림과는 달리 ICE 곡선은 **이질적인 관계를 다룰 수 있습니다.**

# 단점 

ICE 곡선 **하나의 특성만 의미 있게 표시할 수 있습니다.** 두 특성에는 수많은 선들이 그려져 겹겹이 쌓일 것이고 그림에서 아무것도 확인하지 못할 수도 있습니다.

ICE 곡선은 PDP와 동일한 문제를 겪습니다.
관심 특성이 다른 특성와 상관되는 경우 **몇몇 특성 분포에 따라 선의 일부 포인트가 잘못된 데이터 포인트**일 수 있습니다.

많은 ICE 곡선이 그려지면 **그림은 과밀해질 수 있으며**는 어떤것도 볼 수 없을 것입니다.
해결책은 다음과 같습니다. 선에 일부 투명도를 추가하거나 선 표본만 그립니다.

ICE 그림에서는 **평균**을 보는 것이 쉽지 않을 수 있습니다.
다음과 같은 간단한 해결 방법이 있습니다.
개별 조건부 기대 곡선과 부분 의존도를 같이 그립니다.

# 소프트웨어 및 대안책

ICE 그림은 R 패키지의 `iml`(이번 예제들에 사용됨), `ICEbox`[^2] 및 `pdp`에서 구현됐습니다.
ICE와 비슷한 특성을 하는 또 다른 R 패키지는 `condvis`입니다.

---

[^1]: Goldstein, Alex, et al. "Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation." Journal of Computational and Graphical Statistics 24.1 (2015): 44-65.

[^2]: Goldstein, Alex, et al. "Package ‘ICEbox’." (2017).

