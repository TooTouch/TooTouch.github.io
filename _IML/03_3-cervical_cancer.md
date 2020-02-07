---
title:  "3.3 Risk Factors for Cervical Cancer (Classification)"
permalink: /IML/cervical_cancer/
---

자궁경부암 데이터는 한 여성이 이 병에 걸리게될지를 예측하기위한 위험 요소들이 포함되어 있습니다. 이 특성들은 인구통계학적 정보(나이와 같은), 생활습관 그리고 의료기록도 포함됩니다. 데이터는 UCI Machine Learning repository에서 다운받을 수 있고 Fernandes, Cardoso 그리고 Fernandes(2017[^1])에 의해 작성됐습니다.

이 책의 예제로 사용된 데이터의 특성들 중 일부입니다.

- 나이
- 성경험 횟수
- 첫 성행위 (나이)
- 임신 횟수
- 흡연 여부 (yes or no)
- 흡연 년수
- 호르몬 피임약 경험 (yes or no)
- 호르몬 피임약 년수
- 자궁내 피임 장치(IUD) 경험 (yes or no)
- 자궁내 피임 장치(IUD) 경험 횟수 
- 성병(STD)에 걸린적 있는지 (yes or no)
- 성병(STD) 진단 횟수
- 첫 STD 진단 후 경과 시간
- 마지막 STD 진단 후 경과 시간
- 생검 결과 ("건강" 또는 "암"). 목표 변수
  
생검은 자궁경부암 진단에 있어서 가장 기본이 되는 진단 방법입니다. 이 책의 예제에서는 생검 결과가 목표 변수입니다. 각 특성들에 대한 결측값은 최빈값으로 대채되었습니다. 이 방법은 실제 대답이 값이 누락된 이유와 관련이 있을 수 있기 때문에 좋은 방법이 아닐 수도 있습니다. 이 질문들은 굉장히 사적인 질문들이기 때문에 편차가 있을 수 있습니다. 그러나 이 책은 결측값 처리에 대한 책이 아닙니다. 그렇기 때문에 최빈값 대체가 예제들에 영향을 미치진 않습니다.

이 데이터로 책의 예제를 직접해보고 싶으시다면 이 책의 Github repository에서 [RData 파일](https://github.com/christophM/interpretable-ml-book/blob/master/R/get-cervical-cancer-dataset.R)과 [R-script](https://github.com/christophM/interpretable-ml-book/blob/master/data/cervical.RData)를 확인해보실 수 있습니다.

---

[^1]: Fernandes, Kelwin, Jaime S Cardoso, and Jessica Fernandes. “Transfer learning with partial observability applied to cervical cancer screening.” In Iberian Conference on Pattern Recognition and Image Analysis, 243–50. Springer. (2017).