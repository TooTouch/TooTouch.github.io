---
title:  "3.1 Bike Rentals (Regression)"
permalink: /IML/bike_rentals/
---

이 데이터는 워싱턴 D.C.에 있는 Capital-Bikeshare이라는 자전거 대여 회사의 일별 자전거 대여수에 대한 데이터입니다. 날씨와 계절에 대한 정보도 함께 있습니다. 데이터는 Capital-Bikeshare에서 무료로 공개되어 있습니다. Fanaee-T와 Gama(2013[^1])가 여기에 날씨 데이터와 계절 정보를 추가했습니다. 목표는 날씨와 일자에 따라 자전거 대여가 얼마나 발생할지를 예측하는 것입니다. 데이터는 [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)에서 다운받으실 수 있습니다.

새로운 특성들이 데이터에 추가되었고 이 책에서는 예제로 모든 특성을 사용하지 않습니다. 아래는 각 특성에 대한 목록입니다.

- 일반 그리고 회원 사용자를 모두 포함만 자전거 렌탈 수. 이 값은 회귀문제의 목표값에 해당합니다.
- 사계절을 나타내는 계절에 대한 값
- 공휴일인지 아닌지를 나타내는 값
- 2011년인지 2012년인지에 대한 값
- 2011.01.01을 기준으로 몇일이 지났는지 나타내는 일자 수. 이 특성은 시간에 따른 트렌드를 고려할 수 있습니다.
- 주중인지 주말인지는 나타내는 값
- 날씨에 대한 상황
  - 맑음(clear), 약간 흐림(few clouds), 거의 흐림(partly cloudy), 흐림(cloudy)
  - 안개 + 흐림(mist + clouds), 안개 + 조금 흐림(mist + broken clouds), 안개 + 약간 흐림(mist + few clouds), 안개(mist)
  - 약한 눈(light snow), 약한 비 + 태풍 + 흐림(light rain + thunderstorm + scattered clouds), 약한 비 + 흐림(light rain + scatterd clouds)
  - 많은 비 + 우박 + 태풍 + 안개(heavy rain + ice pallets + thunderstorm + mist), 눈 + 안개(snow + mist)
- 섭씨 기준 온도
- 상대 습도 (0 ~ 100%)
- 시간 당 풍속 (km/h)
  
이 책의 예제들에서 데이터는 약간 가공되어 사용됩니다. 처리과정에 대한 R-script는 책의 [Github repository](https://github.com/christophM/interpretable-ml-book/blob/master/R/get-bike-sharing-dataset.R)에서 [최종 RData 파일](https://github.com/christophM/interpretable-ml-book/blob/master/data/bike.RData)과 함께 확인하실 수 있습니다.

---

[^1]: Fanaee-T, Hadi, and Joao Gama. “Event labeling combining ensemble detectors and background knowledge.” Progress in Artificial Intelligence. Springer Berlin Heidelberg, 1–15. doi:10.1007/s13748-013-0040-3. (2013).