# Modeling
![image](https://user-images.githubusercontent.com/105157967/173301240-8b9481d6-de81-40b4-bb70-978a8b04d971.png)


위와 같은 로직으로 모델을 구현할 것이며, 두가지의 전처리 방법과 5가지의 모델을 이용하여 각각의 성능을 비교해 포드 엔진 불량 탐지를 어떤 모델이 가장 잘 하는지 판단한다. 

# data load
http://www.timeseriesclassification.com/description.php?Dataset=FordA 
위의 URL 에서 데이터를 다운받은 후, 코드에서 저장경로를 변경하여 test와 train 데이터를 각각 불러온다.

# data 처리
Logistic Regression이나 xgboost의 경우 2차원 데이터를 활용하지만, cnn,lstm,cnn-lstm 모델 input은 3차원이므로 확장이 필요하다.

# data preprocessing
본 프로젝트에서는 이상치에 민감하게 반응하지 않기 위해 Minmaxscaler외의 robust, standard scaler방법을 각각 이용했다.

프로젝트 진행 시, robust로 데이터를 처리한 후 모델링을 수행하고, standard로 데이터를 처리한 후 모델링을 수행해 결과 값을 비교한다. 

# 성능 지표
혼동행렬, roc커브, acc, loss그래프를 활용했으며, 
공통으로 성능 확인은 score값을 이용해 예측한 값과 실제 test 데이터셋을 비교한 정확도로 진행했다.

# 결과

![image](https://user-images.githubusercontent.com/105157967/173301838-298bea0e-c66b-4dfe-867a-d79fe2f56d59.png)
