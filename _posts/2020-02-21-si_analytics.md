---
title: SI Analytics 견학 후기
categories: 
    - Activity
toc: true
---

이번에 모두의 연구소에서 참여하고 있는 AI College에 연구하고 있는 Explainable AI 분야의 협력기업인 SI Analytics (SIA)에 다녀왔다. 원석님이 연락해주신 덕분에 약속을 잡을 수 있게 되었고 다녀올 수 있었다. 감사합니다! :)

<p align='center'>
    <img src="https://drive.google.com/uc?id=1rjbz8b0-61EeQqXDoeLxAsCfx9N4hryE", width='500'>
</p>

# Setrec Initiative (SI)

[Homepage](https://www.satreci.com/)

SIA에 대해 얘기하기 전에 SI에 대해 먼저 알아야한다. 쎄트렉아이(SI)는 현재 대전 전민동에 있고, 1999년에 설립되어 인공위성연구센터 출신 연구원들이 창업한 회사로서 상용 위성을 개발하고 있다. 국내 우주 분야 수출액의 90%가 모두 SI의 성과일 정도로 독점 기업이라해도 무방하다. 

<p align='center'>
    <img src="https://drive.google.com/uc?id=1aSS34Pbo5yqxZz2-AvszPXVEhc4IK_c1"><br>
    <i>출처: <a href='https://www.slideshare.net/TaegyunJeon1/tensorflow-kr-2nd-meetup-lightening-talk-satrec-initiative'>https://www.slideshare.net/TaegyunJeon1/tensorflow-kr-2nd-meetup-lightening-talk-satrec-initiative</a></i>
</p>

SI는 발사를 제외하고 나머지 모든 부분을 만들고 판매한다. 위성을 만드는 것부터 시작해서 위성에서 찍은 이미지를 활용하기까지 각 분야별 자회사를 설립해서 운영하고 있다. 자회사는 SI Imaging Service (위성 영상 판매), SI Detection (환경방사선 감지), **SI Analytics (인공지능)** 이 있다. 

# SI Analytics (SIA)

SIA는 Setrec Initiative의 자회사이고 인공위성에서 찍힌 이미지를 인공지능을 활용하여 여러 솔루션을 제공하고 있다. 대전 전민동에 있는 쎄트렉아이와 멀지않은 곳에 있고 전태균 박사님께서 대표로 계신 곳이다. 

**SI Analytics에 가기전 기대했던 점들**

- DGX-2 영접
- 위성 실물 영접
- XAI 연구원들과 미팅

## **DGX-2 영접**

우선 DGX-2는 볼 수 없었다.. 가장 먼저 도착하자마자 질문한 것이었고 DGX-2는 서울에 있다는 말과 함께 아쉬움을 감출 수 없었다. SIA 마크는 NVIDIA에서 따로 신경써서 제작해주었다고 한다. 눈이 부셔서 보기가 힘들다.

<p align='center'>
    <img src="https://drive.google.com/uc?id=1wf43rBJKDEtO-ecV4KRyRaCtDeabFVzk" width='400'><br>
    <i>출처: 전태균 대표님 페이스북</i>
</p>

V100 32GB 16대,,, 실제로 돌아가는걸 보고싶다. 이전 병원에 있을때 DGX-1이 있어서 사용 해봤었는데 CT나 MRI 이미지 학습할때 굉장히 유용하게 썼었다. 근데 16대라니,,, ImageNet이 내 로컬에서 MNIST 학습하듯 학습될거같다. 

<p align='center'>
    <img src="https://drive.google.com/uc?id=1yyn_UKaCfFAznwN9dpnQv2-CeW-N_1zn"><br>
    <i>출처: 전태균 대표님 페이스북</i>
</p>

## 위성 실물 영접

살면서 위성을 실물로 볼일이 없을거같았다. 보안 때문에 사진은 찍을 수 없었지만 실제로 위성을 본 후기는 마냥 신기하다였다. 카메라 렌즈가 나보다 컸었고 각종 비싸보이는 부품들부터 위성의 모든 것을 한 회사에서 제작하기 때문에 각 파트별로 대표님께서 같이 인솔해주시며 소개해주셨다.  

회사의 특별한 점이라면 위성을 전부 한 곳에서 만들기 때문에 협업이 굉장히 중요하다는 점이고 일정에 따라 정해진 부분을 순차적으로 잘 완수해야 막대한 비용의 손해가 발생하지 않기 때문에 회사의 모토는 **'정직'**이라고 한다. 위성 한 대를 만드는데 많게는 2000억까지 든다고하니 작은 부분 하나라도 실수가 있으면 2000억 안녕... 그래도 보험이 있다고 한다. 또한 이렇게 정말 많은 분야의 전문가들이 다양하게 있는 회사는 많지 않다. 

<p align='center'>
    <img src="https://drive.google.com/uc?id=1tq6R-Tbu70WJ3g3c5IAYYGdtdgHR_KS-" width='500'><br>
    <i>위성을 직접 찍을 수는 없었지만 위성으로 찍은 사진은 찍을 수 있었다.</i>
</p>

위성사진은 30,000x30,000 정도 크기로 촬영가능하고 한번에 반경 10km까지 촬영할 수 있다고한다. 위성은 하루에도 지구를 몇 바퀴씩 계속 돌고있고 계속해서 움직이기 때문에 같은 장소를 찍기위해서는 방향을 바꿔가며 촬영된다고 한다. 또한 사전에 찍을 장소를 정해두고 시간에 맞춰 촬영할 수도 있다. 

<p align='center'>
    <img src="https://drive.google.com/uc?id=1i2JQyH8skOOaV-tXlTy4tlDZdCfPGFIC"><br>
    <i>출처: <a href='https://www.kari.re.kr/cmm/fms/FILE_000000000004803/getImage.do;jsessionid=735F3A7D755EE3E8F765172001200F0F?atchFileId=FILE_000000000004803&fileSn=4&kind=500'>한국항공우주연구원</a></i>
</p>

<p align='center'>
    <img src="https://drive.google.com/uc?id=1KwGkr0CRPZdzPNcgaVUoL5xmHc5Zp4-F"><br>
    <i>출처: <a href='https://www.kari.re.kr/prog/stmaplace/list.do'>한국항공우주연구원</a></i>
</p>


향후에는 조금 더 작은 위성을 더 많이 쏘아올려 더 많은 이미지를 얻고 일반인들도 지금보다 비교적 더 쉽게 위성 사진을 접해서 여러가지 활용할 수 있는 날이 올거 같다.

## XAI 연구원들과 미팅

전태균 대표님께서 SIA에서 연구하고있는 연구원 분들과 얘기할 수 있도록 시간을 마련해주셨다. 이번 AIC에서 앞으로 써야할 논문에 대해 어떻게 아이디어를 정리해볼 수 있을까 계속 고민이였는데 좋은 시간이 되었다. 

가면서 계속 고민하고 있던것은 어떤 질문을 해야 좋을까였다. 그런데 오히려 먼저 저희에 대해 물어봐주셔서 맘이 편했다. 주로 나눴던 얘기는 간단하게 어떤일을 해왔는지와 왜 XAI에 대해 연구를 하게됐는지를 물어보셨고 답변하는 과정에서 내가 왜 이 연구를 하고 있는지 그리고 어떤 방향이 좋은지 스스로도 정리가 되었다. 그리고나서는 기존 가지고 있던 아이디어에 대해서 제대로 정리되지 않은 점을 잘 답변해 주셨다. 

내가 하고 싶은 방향성은 크게 두 가지였다. 

1. Visual Reasoning
2. Attribution Method

영상을 통해 학습된 모델의 궁긍적인 목표는 이미지에 있는 어떤 관계를 추론해낼 수 있는 모델이다. 단순히 인식과 검출을 통해 물체를 찾는게 아닌 사람이 생각하는 관점과 비슷하게 과연 인공지능이 영상을 통해서 스스로 추론을 해내고 어떤 추론을 했는지 표현을 할 수 있는가이다. 그러나 현재 AIC에서 지원을 받을 수 있는 기간은 제한돼있기 때문에 추후 이 분야에 대해서 계속 공부를 해보려한다. 현재로써는 목표는 새로운 Attribution Methods에 대해서 연구해보려고 한다. 지금까지 AIC에서 진행했던 과제들을 통해 여러 attirubution methods와 attention methods 그리고 quantity, quality evaluation까지 구현하며 공부했기 때문에 논문을 쓴다면 보다 나은 attribution method를 제안하는 방향이 맞지 않을까 싶다.  

이전에 지수님이 얘기했었던 GradCAM은 색상을 반영하지 않는 다는 점과 평가 자체에서 색상별로 해볼 수 있지 않을까라는 생각해봤었고, 현재 현업에서 그나마 많이 사용되는 방법이라면 Grad-CAM이다. 대충 설명하고 넘어가기 좋고 아무래도 다른 방법에 비해 간단하고 시각적으로 표현하기 좋기 때문이다. 때문에 더 나은 방법이 있지 않을까싶고 색상에 관련해서 어떤식으로 아이디어를 정해볼지 고민해보고 있다고 말씀드렸었고 진지하게 답변을 해주셔서 감사했었다.  

# 맺음말

정말 오랜만에 대전도 다녀왔고 평생 언제 볼 수 있을지 모르는 위성도 실제로 보고왔다. 회사를 단순히 설명회가 아니라 이렇게 견학을 통해 소개받는건 처음이라 좋은 경험이었다. 

최근에 Youtube에서 제작한 [Age of AI](https://www.youtube.com/playlist?list=PLjq6DwYksrzz_fsWIpPcf6V7p2RNAneKc)를 재밌게 봤는데 (강추!) 인공지능에 발전과 사례들에 대해 시리즈로 제작된 다큐이다. AI를 통한 치유, 화성의 우주 건축가들, 한 번에 알고리즘 하나로 세상을 구하기 등 인공지능을 어떻게 활용할 수 있는지에 대한 내용들이다. 인공지능에 대해 공부를 하고 있지만 어떤 분야에서 어떻게 활용하면 좋을지 계속 고민하고있던 중 참고할만한 좋은 영상들이었다. 

최근 국내에는 인공지능이라는 이름으로 크게 정해진 목표나 분야가 없이 돈 되는 사업은 다 하려고하는 회사들이 많다. 때문에 SIA는 목표나 가치가 정말 확고하고 앞으로가 더 기대되는 회사였다.  

앞으로는 인공지능을 공부하는 것도 좋지만 인공지능을 어떻게 활용하면 좋을지에 대해 더 많이 공부하려고한다. 사회에 좀 더 귀기울이고 뉴스도 잘 보고 무엇보다 밖을 더 많이 다녀야하는데... 코로나 때문에 언제쯤 맘편하게 다닐 수 있을지 모르겠다. 

최근에는 Climate Change 관련해서 관심이 생겨서 보고있는데 아래 웹페이지에 정말 많은 분야의 전문가들이 각 산업별 기후변화에 대해 취하고 있는 행동들과 연구사례들을 잘 서술해주었고 각 분야별 데이터도 공개되어 있어서 공부하기 좋다. 

[Climate Change AI](https://www.climatechange.ai/)

조금 늦은감이 없지않아 있지만 남은 기간 빠르게 아이디어 정리 끝내고 실험해보면서 얼른 써야겠다!!!!