---
title: "AI against COVID-19"
categories: 
    - Research
toc: true
---

COVID-19 백신을 찾기 위해 많은 연구소가 고군분투하며 힘쓰고 있는 현실이다. 코로나 때문에 현재 수많은 인구 피해와 더불어 경제까지 무너져가기 시작하고 사람들 간의 차별도 격화되어 가는 실정에 하루빨리 백신이 시급해지고 있다. 

# 코로나바이러스

코로나바이러스는 이전에도 흔하게 있었던 바이러스이다. 포유류나 새 뿐만 아니라 사람에게도 감염될 수 있는 **호흡기 바이러스(respiratory virus)** 중 하나이다[^1]. 

코로나바이러스라는 이름은 그림 1[^3]과 같이 표면을 둘러싸는 크라운 모양의 스파이크 형태 때문에 지어졌다고 한다. 코로나(corona)는 라틴어로 왕관이라는 뜻이다.

<p align='center'>
    <img width='500' src='https://phil.cdc.gov//PHIL_Images/23312/23312_lores.jpg'><br>그림 1. 코로나바이러스 이미지. 
</p>

사람에게 감염되는 코로나바이러스(human coronaviruses)는 1960년대 처음으로 확인되었고 대표적으로 크게 두 가지 인간 코로나바이러스로 나뉘고 총 **7가지**의 타입이 있다.

**일반 인간 바이러스 (common human coronaviruses)**

1. 229E (alpha coronavirus)
2. NL63 (alpha coronavirus)
3. OC43 (beta coronavirus)
4. HKU1 (beta coronavirus)

**그외 인간 바이러스 (other human coronaviruses)**

5. MERS-CoV (the beta coronavirus that causes Middle East Respiratory Syndrome, or MERS)
6. SARS-CoV (the beta coronavirus that causes severe acute respiratory syndrome, or SARS)
7. **SARS-CoV-2 (the novel coronavirus that causes coronavirus disease 2019, or COVID-19)**

인간 코로나바이러스는 진화를 거듭하며 변화되기도 하는 게 이러한 경우가 5,6,7번과 같은 경우이다. 이 중 7번째 타입이 2019년 새롭게 발견된 코로나바이러스, 즉 **COVID-19** 이다. 

COVID-19는 점액 친화성을 특징으로 하는 Orthocoronavirinae 에 속하며, 인체에 감염을 일으키는 **RNA 바이러스 중 크기가 가장 크다**. 또한, 사람 세포막에서 만들어지는 인지질 이중막 껍데기를 가지고 있다. 인지질막 때문에 바이러스의 외부 단백질의 변이가 발생할 가능성이 높아지고, 그 결과 변이가 발생한 바이러스가 사람이 원래 가지고 있던 면역을 회피할 수 있는 확률이 증가한다.

인지질막에 촘촘히 붙어 있는 것이 바로 **스파이크 단백질(spike protein)** 이고 S 단백질이라고도 불리는 이 스파이크 단백질로 인해 호흡기 점막 친화성을 갖게 된다. 즉, 코로나바이러스가 인후부의 상부 호흡기와 기관지 이하의 하부호흡기 점막에 잘 부착할 수 있고 세포 안으로 잘 침투하여 쉽게 증식할 수 있다[^1].

# RNA란?

앞서 COVID-19가 RNA 바이러스 중 하나라고 얘기하였는데 RNA는 무엇일까? DNA랑 이름도 비슷해서 둘의 역할이 비슷하지 않을까 짐작해볼 수 있다. 

**DNA**와 **RNA**는 그림 2[^6]과 같이 모두 핵산의 한 종류이다. **DNA**(DeoxyriboNucleic Acid, 데옥시리보핵산, 디옥시리보 핵산)는 뉴클레오타이드의 중합체인 두 개의 긴 가닥이 서로 꼬여있는 이중나선 구조로 되어있는 고분자화합물이다[^5]. DNA는 스스로를 복제하고 유전정보를 통해 유전자 발현이 일어나게 한다. 유전자 발현이란 DNA를 구성하는 유전정보, 즉 유전자에 의해 생물을 구성하는 다양한 단백질이 형성되는 과정을 말한다. 

**RNA**(RiboNucleic Acid, 리보핵산)는 오탄당의 일종인 리보스를 기반으로 뉴클레오타이드를 이루는 핵산의 한 종류이다. 하나의 나선이 길게 꼬여 있는 구조를 지니며 DNA 일부가 전사되어 만들어진다.

<p align='center'>
    <img width='500' src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Difference_DNA_RNA-EN.svg/1920px-Difference_DNA_RNA-EN.svg.png'><br>그림 2. RNA(좌) 그리고 DNA(우)의 구조 차이
</p>


## DNA와 RNA의 관계

위에서 보면 RNA는 DNA에 의해 생성되므로 DNA의 전사과정에서 발생한 결과로 생겨난 것으로 생각될 수 있다. 생물학의 중심이론인 '**센트럴 도그마(Central dogma)**'에 따르면 DNA가 자기복제를 하고 전사를 통해 RNA를 만들면 RNA가 단백질을 만들어낸다고 할 수 있다. 그러나 이러한 사실은 DNA에서 자기복제를 통해 유전자 발현을 실행하기 위해서 단백질이 필요한데 이 단백질은 어디서부터 왔는지에 대한 의문이 생긴다. 이 때문에 '닭이 먼저냐 달걀이 먼저냐' 말과 같이 생명과학에서는 '**DNA가 먼저냐 단백질이 먼저냐**'라는 말이 있다. 


이 의문점을 해결하는 것이 바로 RNA이다. 1982년 미국의 분자생물학자 Thomas Robert Cech가 RNA에 존재하는 리보자임(Ribozyme)을 발견하였고, 1989년 노벨 화학상을 수상했다. 리보자임의 발견을 통해 RNA가 스스로 단백질을 형성할 수 있는다는 사실이 발견되었고 '**생명의 기원은 RNA로부터 시작되었을 것이다**' 라는 얘기가 생기기 시작했다.

<p align='center'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Thomas_Robert_Cech.jpg/200px-Thomas_Robert_Cech.jpg'><br>그림 3. Thomas Robert Cech
</p>

영국의 맨체스터 대학의 존 서덜랜드 (John Sutherland) 교수팀은 이러한 가설을 통해 RNA의 기본구성토대인 리보뉴클레오티드 (ribonucleotide)를 초기 지구에 존재했을 것으로 생각되는 조건에서 간단한 화합물로부터 만드는 데 성공했고, 2009년 논문이 Nature에 게재되었다[^7].


## RNA 바이러스란?

지금까지 인류에 큰 피해를 줬던 전염병들 대부분이 RNA 바이러스라고 볼 수 있다. 10대 전염병은 에이즈, 스페인독감, 아시아독감, 홍콩독감, 콜레라, A형 신종 인플루엔자, 에볼라, 홍역, 뇌수막염, 사스 등 있고, 이 가운데 콜레라와 뇌수막염을 제외한 8종이 모두 RNA 바이러스로 분류되고 있다.

바이러스 입자들은 그림 4[^8]과 같이 DNA나 RNA로 만들어진 유전 물질을 보호하는 두 개 또는 세 개의 부분으로 구성되어 있다. 대부분 바이러스는 RNA를 가지고 있고 RNA는 DNA처럼 유전물질로서 역할을 할 수 있다.

<p align='center'>
    <img width='500' src='https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Basic_Scheme_of_Virus_en.svg/2560px-Basic_Scheme_of_Virus_en.svg.png'><br>그림 4. 바이러스 구조
</p>

바이러스는 한 마디로 '단백질로 둘러싸인 핵산'이다. 핵산의 종류에 따라 'DNA 바이러스'와 'RNA 바이러스'로 나뉘는데, 그중에서도 RNA 바이러스들은 유난히 말썽을 피우는 악동들이다. 코로나바이러스 역시 RNA를 유전체로 이용하는 RNA 바이러스 일종이다. RNA 바이러스는 증식 과정에서 돌연변이를 자주 일으킨다. 치료제 내성이 잘 생기고, 백신도 종종 무용지물이 된다. 게다가 돌연변이를 거쳐 숙주를 바꿀 수 있으므로 동물의 바이러스라도 종간 장벽을 넘어 인간에게 넘어올 수 있다[^4]. 

IBS라는 기초과학연구원에서는 SARS-CoV-2(COVID-19)의 유전체인 RNA가 어떻게 복제되고 증폭되는지에 대한 보고서를 게재하였다. 그 과정은 그림 5과 같이 설명되었으며 이 이상의 설명은 내 허용 범위가 아니기에 [보고서](https://www.ibs.re.kr/cop/bbs/BBSMSTR_000000000971/selectBoardArticle.do?nttId=18475)를 참고하기를 바란다. 

<p align='center'>
    <img width='700' src='https://www.ibs.re.kr/dext5data/2020/04/20200429_163315052_17751.jpg'><br>그림 5. COVID-19의 복제 및 전사 과정
</p>

# mRNA Vaccine

최근 COVID-19에 대한 백신으로 가장 유력한 후보가 바로 mRNA 백신이다. 앞서 RNA에 대해 알아봤지만 **mRNA는 무엇**이고 **어떻게 백신으로 활용되는지**를 알아보았다.

1. mRNA란?
2. mRNA가 왜 유력한 백신 후보인가?


## mRNA란?

RNA는 분자구조와 생물학적 기능에 따라 다음과 같이 9가지로 나뉜다[^9].

- **rRNA**(리보솜 RNA ribosomal RNA): 리보솜을 구성하는 RNA이다.
- **mRNA**(전령 RNA messenger RNA): DNA의 유전 정보를 옮겨적은 일종의 청사진 역할을 한다. 이를 기본으로 하여 리보솜에서 단백질을 합성하게 된다.
- **tRNA**(운반 RNA transfer RNA): mRNA의 코돈에 대응하는 안티코돈을 가지고 있다. 꼬리쪽에는 tRNA의 안티코돈과 대응하는 아미노산을 연결해 주는 효소가 있다. 따라서, tRNA의 안티코돈에 대응하는 아미노산을 달고 있다.
- **miRNA**(마이크로 RNA micro RNA): 생물의 유전자 발현을 제어하는 역할을 하는 작은 RNA로, mRNA와 상보적으로 결합해 세포 내 유전자 발현과정에서 중추적인 조절인자로 작용한다.
- **snRNA**(소형 핵 RNA small nuclear RNA): 핵 안에서 RNA를 스플라이싱 하는 기능이 있다.
- **snoRNA**(소형 인 RNA small nucleolar RNA): 핵 안에서 RNA의 변형을 일으킨다.
- **aRNA**(안티센스 RNA antisense RNA): RNA에서 리보솜으로의 번역을 조절하는 역할을 담당한다.
- **siRNA**(소형 방해 RNA small interfering RNA): RNA 방해를 유발한다. 특정 단백질의 생산을 억제함으로써 유전자 발현을 방해한다.
- **piRNA** 

이 중 살펴보려고 하는 것은 바로 mRNA 이다. mRNA는 DNA의 유전정보를 암호화(코돈)하여 리보솜에서 단백질을 형성하기위해 운반하는 역할을 한다. 


## mRNA가 왜 유력한 백신 후보인가?

SARS-CoV-2 바이러스 출현 후 염기서열이 규명되었고, 이 중 S 단백질(Spike protein)이 백신의 표적이 될 수 있다고 밝혀졌다. 바이러스의 S 단백질은 그림 6[^11]과 같이 숙주 세포의 angiotensin converting enzyme (ACE) II receptor에 결합하여 감염력을 가지는데, 항체를 통해 결합 과정을 차단하는 것이 작용기전이다[^12].

<p align='center'>
    <img width='500' src='http://medicine.snu.ac.kr/sites/medicine.snu.ac.kr/files/1_10.jpg'><br>그림 6. SARS-CoV-2의 결합 과정
</p>

현재로서 COVID-19 백신으로 가장 유력한 후보가 바로 미국 **Moderna** 사에서 연구하고 있는 **'mRNA-1273'** 이다. 이 백신은 바이러스와 숙주 세포막이 융합되기 전 상태(prefusion conformation)의 S단백질을 발현시키는 mRNA를 사람에게 주입하는 것이 주요 기전이다[^10].


mRNA 백신이 COVID-19의 유망한 백신 후보로 추앙받고 있는 이유는 바로 백신 개발에 필요한 비용과 활용 가치 때문이다. mRNA 기술은 기존 백신 개발에 비해 훨씬 짦은 시간과 비용이 들고 더 다양한 백신을 만들 수 있다[^13]. 그 외 mRNA 백신 개발의 장래성에 대한 내용은 올해 2월에 Nature에 게재된 [The promise of mRNA vaccines: a biotech and industrial perspective](https://www.nature.com/articles/s41541-020-0159-8) 을 통해 확인 할 수 있다.

최근 7월 소식에 따르면 지난 7월 27일 3만명을 대상으로 임상 3상까지 집입했다는 기사를 보았다. 7월 28일 NEJM에 기재된 mRNA-1273 백신 실험에 따르면 영장류를 대상으로 백신 10μg 그리고 100μg을 투약하여 비교 실험 하였다. 결과는 100μg을 주입한 동물들에게 더 효과가 있음을 실험하여 유의성을 검정하였고, 해당 동물들 모두 백신 주입 2일 후 코에서는 어떠한 바이러스 복제도 발견되지 않았다. 그러나 폐에서는 제한적으로나마 일부 검출되었다[^14].

상황이 급박한 만큼 전세계적으로 백신이 절실한 상황에서 4월에 2상 임상시험을 준비하고 있다는 소식을 들었는데 벌써 3상 진입이라니..! 하루 빨리 좋은 소식이 들리길 바란다. 

# [OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction](https://www.kaggle.com/c/stanford-covid-vaccine/overview)

mRNA가 가진 장점에도 불구하고 단점이 있기 마련이다. 지금부터 소개할 내용 또한 이러한 문제점을 해결하기 위해 많은 전문가들이 참여하도록 장려하기 위함이다. 

mRNA가 가진 문제점을 헤결하고자 스탠포드 대학에서는 캐글(kaggle)의 수많은 데이터 사이언티스트들과 개발자들에게 도움을 요청하였다. 그렇다면 그 문제점이 무엇인지 대회 개요를 통해 알아보았다.

> **캐글이란?**
> 캐글은 데이터 사이언티스트들을 위한 경쟁 플랫폼이다. 수많은 기업에서는 데이터는 있지만 이 데이터를 활용할 사람이 부족하고 수많은 분석가와 개발자는 데이터가 다룰 수 있는 데이터가 부족하다. 그리하여 탄생한 것이 바로 캐글이라는 플랫폼이다.

<p align='center'>
    <img src='https://storage.googleapis.com/kaggle-competitions/kaggle/22111/logos/header.png'>
</p> 

대회 설명을 간략하게 보자면 다음과 같다. 

## 대회 개요

COVID-19 백신이 시급한 지금 여러 한계점들이 있다. 그러나 이러한 한계점을 캐글이라는 플랫폼에서 집단 지성을 이용하여 해결책을 제공받도록 하는 것이 목적이다. 

현재 가장 유력한 백신 후보로는 mRNA 백신이 있지만 큰 문제점이 있다. 기존 독감 백신같은 경우는 일회용 주사기와 같은 곳에 동봉되어 얼린 후 전세계 곳곳으로 배송이 가능하지만 mRNA에는 이러한 방법이 적용되지 않다는 것이다.

연구자들에 말에 따르면 mRNA는 스스로 분해(degredation)되는 경향이 있다고 한다. 그리고 그림 7과 같이 주어진 RNA의 구조에서 어느 부분이 이러한 영향을 미치는지 명확히 알려진 바가 없다고 한다. 

<p align='center'>
    <img width='500' src='https://storage.googleapis.com/kaggle-media/competitions/Stanford/banner%20(2).png'><br>그림 7. mRNA 구조에서는 잘라내야 하는 부분이 있고 안전한 부분이 있지만 어디가 그러한지는 아직 정확히 모른다.
</p>

스탠포드 의대 Rhiju Das 교수가 지도하고 있는 Eterna[^15] 이라는 커뮤니티에서는 게임을 통해 새로운 해결책을 찾고 신약을 개발하기 위한 수많은 과학자와 개발자들이 있다. 이 플랫폼은 공개적으로 열려있기 때문에 누구나 들어가서 게임을 해보고 공부할 수 있다. 게임해보고 싶으신 분은 [여기](https://eternagame.org/)!

이 대회 목적은 위와 같은 이유로 각 RNA가 분해되는 속도를 예측하는 모델을 만드는 것이고 데이터는 Eterna에서 제공하는 3,000개의 RNA 분자데이터를 사용한다. 평가는 학습 데이터를 통해 훈련된 모델로 Eterna 플레이어들이 만든 2세대 RNA 시퀀스로 평가된다고 한다. 최종 평가는 스탠포드 대학에서 캐글을 통해 선정된 우수한 모델과 함께 실험된다고 하고 네이처(Nature)에서 모델에 점수를 매긴다고 한다.(..!)

하루빨리 백신을 위해 몇 년 몇 달이 아닌 몇 주 안으로 좋은 해결책이 나오길 바라며 mRNA 백신 연구를 가속하고 SARS-CoV-2 (COVID-19 바이러스 중 하나)에 맞서 냉장 상태로 보관될 수 있는 백신을 전달하기를 바란다고 한다.

## 데이터

캐글에서 제공하는 데이터는 크게 학습 데이터(train.json), 평가 데이터(test.json), 제출 데이터(submission.csv) 세 가지로 구성 되어있다.

보통은 학습 및 평가 데이터는 csv 파일로 제공되기도 하지만 이번 대회에는 json 파일 형식으로 제공되었다.

데이터 구조를 살펴 보자면 아래와 같이 구성되어있다. 예측을 해야하는 변수는 
'**reactivity**', '**deg_Mg_pH10**', '**deg_pH10**', '**deg_Mg_50C**', 그리고 '**deg_50C**'으로 총 **5개** 이다. 

학습 데이터와 평가 데이터에 공통적으로 들어 있는 변수는 총 **7개**이고 다음과 같다. 

Feature | Description
---|---
**index**  | 말그대로 해당 데이터에 대한 인덱스
**id**  | 해당 RNA에 대한 고유 ID
**sequence**  | A, G, U, 그리고 C로 구성
**structure**  | **(**, **)**, 그리고 **.** 세 가지 구성으로 구성. 예를 들어, (....)는 0번째와 5번째는 서로 짝을 이루고 있고 나머지 1-4는 서로 짝을 이루도 있지 않다는 것을 의미한다. 
**predicted_loop_type**  | Vienna RNAfold 2 structure을 통해 bpRNA에 따라 정의된 Loop type이라고 한다. bpRNA documentation에 의하면 각 타입은 다음과 같이 정의된다.  **S**: paired "Stem" **M**: Multiloop **I**: Internal loop **B**: Bulge **H**: Hairpin loop **E**: dangling End **X**: eXternal loop
**seq_length** | 해당 RNA 시퀀스의 길이
**seq_scored** | 해당 RNA에 대한 목표값을 가지고 있는 위치(position)의 개수. 아래 예시에서는 68

- **predicted_loop_type**에 대해 부연 설명을 하자면 RNAfold는 RNA의 시퀀스로부터 secondary structure를 예측해주는 모델(?)인거 같다. 18년 겨울 나왔던 alphafold가 이러한 모델이었다. RNAfold는 따로 [web server](http://rna.tbi.univie.ac.at/cgi-bin/RNAWebSuite/RNAfold.cgi)로 서비스 되고 있다[^16]. 또한 bpRNA는 RNA secondary structure에 대한 automated annotation이다[^17]. 

**train.json example**

```python
"root" : {19 items
    "index" : int 0
    "id" : string "id_001f94081"
    "sequence" : string "GGAAAAGCUCUAAUAACAGGAGACUAGGACUACGUAUUUCUAGGUAACUGGAAUAACCCAUACCAGCAGUUAGAGUUCGCUCUAACAAAAGAAACAACAACAACAAC"
    "structure" : string ".....((((((.......)))).)).((.....((..((((((....))))))..)).....))....(((((((....)))))))....................."
    "predicted_loop_type" : string "EEEEESSSSSSHHHHHHHSSSSBSSXSSIIIIISSIISSSSSSHHHHSSSSSSIISSIIIIISSXXXXSSSSSSSHHHHSSSSSSSEEEEEEEEEEEEEEEEEEEEE"
    "signal_to_noise" : float 6.894
    "SN_filter" : int 1
    "seq_length" : int 107
    "seq_scored" : int 68
    "reactivity_error" : [...] 68 items
    "deg_error_Mg_pH10" : [...] 68 items
    "deg_error_pH10" : [...] 68 items
    "deg_error_Mg_50C" : [...] 68 items
    "deg_error_50C" : [...] 68 items
    "reactivity" : [...] 68 items
    "deg_Mg_pH10" : [...] 68 items
    "deg_pH10" : [...] 68 items
    "deg_Mg_50C" : [...] 68 items
    "deg_50C" : [...] 68 items
}
```

**test.json example**

```python
"root" : { 7 items
    "index" : int 0
    "id" : string "id_00073f8be"
    "sequence" : string "GGAAAAGUACGACUUGAGUACGGAAAACGUACCAACUCGAUUAAAAUGGUCAAAGAGGUCGAAAUACAGAUGACCUUCGGGUUAUCAAAAGAAACAACAACAACAAC"
    "structure" : string "......((((((((((.(((((.....))))))))((((((((...)))))...)))))))...))).(((((((....)))))))....................."
    "predicted_loop_type" : string "EEEEEESSSSSSSSSSBSSSSSHHHHHSSSSSSSSSSSSSSSSHHHSSSSSBBBSSSSSSSBBBSSSXSSSSSSSHHHHSSSSSSSEEEEEEEEEEEEEEEEEEEEE"
    "seq_length" : int 107
    "seq_scored" : int 68
}
```

## 평가 방법

평가 수식은 다음과 같이 MCRMSE (mean columnwise root mean squared error)를 사용한다고 한다. 뭔가 기존에 알고있는 MSE보다 복잡해 보이겠지만 생각해보면 단순하니 겁먹지 말자.

$$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$$
 
$$N_i$$는 목표 변수 5개를 말한다. 즉, 각 예측을 목표로 하는 5개의 변수들의 MSE를 계산하고 평균을 내는 값이다.

따라서 submission file도 아래와 같다. 아래 id_00073f8be_ 하고는 숫자가 여럿 붙어있는건 시퀀스 인덱스를 말한다. 즉, 위의 test.json 예시에서 id_00073f8be는 107개의 시퀀스가 있었으므로 id_00073f8be_0 부터 id_00073f8be_106 까지 나타낼 수 있다.

```python
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C    
id_00073f8be_0,0.1,0.3,0.2,0.5,0.4
id_00073f8be_1,0.3,0.2,0.5,0.4,0.2
id_00073f8be_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## 대회 기간

대회 기간은 **10월 2일**까지 참여이고 **10월 5일**까지 제출 마감이다. 백신 개발이 시급한 지금 조금이나마 도움이 되는 모델을 제공할 수 있는 기회가 생기길 바란다. 


# 맺음말

현재 동메달을 달리구 있다. 쿠쿡... 사실 누가 올려준 노트북 돌려만 봤다. [OpenVaccine - GRU + LSTM](https://www.kaggle.com/tuckerarrants/openvaccine-gru-lstm) 다들 얼른 돌려보고 잠시나마 동메달권에 머무르시길! 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/93015354-bc1b8680-f5f3-11ea-8762-36740f26875d.png'>
</p>

위에서 언급한 커널 말고도 다른 하나의 커널을 비교해 봤다. 하나는 lightGBM을 베이스로 하는 모델이고 하나는 GRU + LSTM을 사용하는 앙상블 모델이었다. 각각 public score는 **0.31323** 그리고 **0.28685** 이다. 

- [OpenVaccine - GRU + LSTM](https://www.kaggle.com/tuckerarrants/openvaccine-gru-lstm)
- [OpenVaccine-simple-lgb-baseline](https://www.kaggle.com/t88take/openvaccine-simple-lgb-baseline)

LightGBM을 모델로 사용하는 커널은 전처리를 각 sequence, structure, 그리고 predicted_loop_type에 대해 앞뒤 5개씩 윈도우 형식으로 반영하였고 학습하였다. 반면 GRU + LSTM 모델은 위의 세 변수를 딱히 전처리하지 않고 통째로 넣었다. 결과적으로는 후자가 더 좋은 성능을 내었다. 역시 딥러닝..? 그건 아직 잘 모르겠지만 결과는 좋았다.

오히려 짦은 시간이기 때문에 큰 고민없이 도전해볼만한 대회인거 같다. 최근에 ETRI, 빅콘테스트, 그리고 Korea Health Datathon 등 왜이리 재밌어보이는 대회가 많은지 모르겠다.. 나 이번달에 GRE 시험있는데!! 영어 공부를 하지 말라는 것인가... 심히 고민과 아쉬움이 큰 한 해 이다...


# Reference

[^1]: [코로나바이러스는 무엇인가요?, 대한감염학회 코로나19](http://www.ksid.or.kr/rang_board/list.html?num=3414&code=ncov_faq)

[^2]: [Human Coronavirus Types, CDC](https://www.cdc.gov/coronavirus/types.html)

[^3]: [Public Health Image Library (PHIL) : Coronavirus, CDC](https://phil.cdc.gov/Details.aspx?pid=23312)

[^4]: [IBS가 밝혀낸 코로나 19 유전자 지도의 의미](https://www.ibs.re.kr/cop/bbs/BBSMSTR_000000000971/selectBoardArticle.do?nttId=18475)

[^5]: Malacinski, George M. (2004). 《분자생물학》. 번역 심웅섭 외. 월드사이언스. ISBN 89-5881-047-5.

[^6]: [Nucleic acid, wikipedia](https://en.wikipedia.org/wiki/Nucleic_acid)

[^7]: Powner, M. W., Gerland, B. & Sutherland, J. D. ‘[synthesis of activated pyrimidine ribonucleotides in prebiotically plausible conditions](https://www.nature.com/articles/nature08013)’, Nature 459, 239-242 2009

[^8]: [Introduction to viruses, wikipedia](https://en.wikipedia.org/wiki/Introduction_to_viruses)

[^9]: 박상대, 《분자세포생물학》, 아카데미서적, 2006

[^10]: [COVID-19 Vaccine 개발 현황 리뷰, 서울의대 임상약리학교실](https://medicine.snu.ac.kr/en/board/Vaccine/view/17303)

[^11]: “ACE-2: The Receptor for SARS-CoV-2,” www.rndsystems.com, accessed April 18, 2020, https://www.rndsystems.com/resources/articles/ace-2-sars-receptor-identified.

[^12]: Cynthia Liu et al., “Research and Development on Therapeutic Agents and Vaccines for COVID-19 and Related Human Coronavirus Diseases,” ACS Central Science 6, no. 3 (March 25, 2020): 315–31, https://doi.org/10.1021/acscentsci.0c00272.

[^13]: Jackson, N. A., Kester, K. E., Casimiro, D., Gurunathan, S., & DeRosa, F. (2020). [The promise of mRNA vaccines: A biotech and industrial perspective. npj Vaccines](https://www.nature.com/articles/s41541-020-0159-8), 5(1), 1-6.

[^14]: Corbett, K. S., Flynn, B., Foulds, K. E., Francica, J. R., Boyoglu-Barnum, S., Werner, A. P., ... & Nagata, B. M. (2020). [Evaluation of the mRNA-1273 vaccine against SARS-CoV-2 in nonhuman primates](https://www.nejm.org/doi/full/10.1056/NEJMoa2024671?query=featured_home). New England Journal of Medicine.

[^15]: [EteRNA Game](https://eternagame.org/)

[^16]: Hofacker, I. L. (2003). [Vienna RNA secondary structure server](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC169005/). Nucleic acids research, 31(13), 3429-3431.

[^17]: Danaee, P., Rouches, M., Wiley, M., Deng, D., Huang, L., & Hendrix, D. (2018). [bpRNA: large-scale automated annotation and analysis of RNA secondary structure](https://pdfs.semanticscholar.org/f238/ab63e46e331249b8db14c8e83b4cc1d98ad9.pdf?_ga=2.212347412.18823493.1599989364-1197764443.1599989364). Nucleic acids research, 46(11), 5381-5394.