
# #### 저장 경로 

save_path ='C:/Users/cjstk/OneDrive/바탕 화면'  # 저장 경로
file_path = 'C:/Users/cjstk/OneDrive/바탕 화면/02. Dataset_FordEngine/dataset/'  # 사용자 Local 환경 내의 다운로드 받은 데이터 파일이 위치한 경로
train_fn="FordA_TRAIN.arff"  # Train 데이터 파일명
test_fn="FordA_TEST.arff"  # Test 데이터 파일명

import itertools
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, RobustScaler
import tensorflow as tf


#혼동 행렬(Confusion Matrix)

from sklearn.metrics import classification_report, confusion_matrix

def draw_confusion_matrix(model, xt, yt, model_name):
    Y_pred = model.predict(xt)
    if model_name in ["cnn", "rnn"]:
        y_pred = np.argmax(Y_pred, axis=1)
    else: y_pred = Y_pred
    plt.figure(figsize=(3,3))
    cm = confusion_matrix(yt, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['False', 'True'], rotation=45)
    plt.yticks(tick_marks, ['False', 'True'])
    thresh = cm.max()/1.2
    normalize = False
    fmt = '.2f' if normalize else 'd'
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j], fmt), 
                 horizontalalignment="center", 
                 color="white" if cm[i,j] > thresh else "black", 
                 fontsize=12)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path + '{}_cm.png'.format(model_name), dpi=100, bbox_inches='tight')  # 그림 저장
    plt.show()
    print(classification_report(yt, y_pred))


# ROC Curve

from sklearn.metrics import roc_curve, auc

def draw_roc(model,xt, yt, model_name):
    Y_pred = model.predict(xt)
    if model_name in ["cnn", "rnn"]:
        y_pred = np.argmax(Y_pred, axis=1)
    else: y_pred = Y_pred
    fpr, tpr, thr = roc_curve(yt, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic {};'.format(model_name))
    plt.legend(loc="lower right")
    plt.ion()
    plt.tight_layout()
    plt.savefig(save_path + '{}_roc.png'.format(model_name), dpi=100, bbox_inches='tight')  # 그림 저장
    plt.show()


# 손실(Loss) 그래프

def plot_loss_graph(history, pic_name):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history["val_loss"])
    plt.title("Training & Validation Loss")
    plt.ylabel("loss", fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "validation"], loc="best")
    plt.tight_layout()
    plt.savefig(save_path + '{}.png'.format(pic_name), dpi=100, bbox_inches='tight')  # 그림 저장
    plt.show()
    plt.close()


# 정확도(Prediction Rate) 그래프

def plot_prediction_graph(history, pic_name):
    plt.figure()
    plt.plot(history.history["sparse_categorical_accuracy"])
    plt.plot(history.history["val_" + "sparse_categorical_accuracy"])
    plt.title("model " + "Prediction Accuracy")
    plt.ylabel("sparse_categorical_accuracy", fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "validation"], loc="best")
    plt.tight_layout()
    plt.savefig(save_path + '{}.png'.format(pic_name), dpi=100, bbox_inches='tight')  # 그림 저장
    plt.show()
    plt.close()

#file 불러오기

def read_ariff(path):
    raw_data, meta = loadarff(path)
    cols = [x for x in meta]
    data2d = np.zeros([raw_data.shape[0],len(cols)])
    for i,col in zip(range(len(cols)),cols):
        data2d[:,i]=raw_data[col]
    return data2d

train = read_ariff(file_path + train_fn)
test = read_ariff(file_path + test_fn)

print("train_set.shape:", train.shape)
print("test_set.shape:", test.shape)


x_train_temp = train[:,:-1]
y_train_temp = train[:, -1]  
x_test = test[:, :-1]
y_test = test[:, -1] 

normal_x = x_train_temp[y_train_temp==1] 
abnormal_x = x_train_temp[y_train_temp==-1] 
normal_y = y_train_temp[y_train_temp==1]  
abnormal_y = y_train_temp[y_train_temp==-1]  

ind_x_normal = int(normal_x.shape[0]*0.8)  
ind_y_normal = int(normal_y.shape[0]*0.8)  
ind_x_abnormal = int(abnormal_x.shape[0]*0.8) 
ind_y_abnormal = int(abnormal_y.shape[0]*0.8)  


x_train = np.concatenate((normal_x[:ind_x_normal], abnormal_x[:ind_x_abnormal]), axis=0)
x_valid = np.concatenate((normal_x[ind_x_normal:], abnormal_x[ind_x_abnormal:]), axis=0)
y_train = np.concatenate((normal_y[:ind_y_normal], abnormal_y[:ind_y_abnormal]), axis=0)
y_valid = np.concatenate((normal_y[ind_y_normal:], abnormal_y[ind_y_abnormal:]), axis=0)


print("x_train.shape:", x_train.shape)
print("x_valid.shape:", x_valid.shape)
print("y_train.shape:", y_train.shape)
print("y_valid.shape:", y_valid.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)


#Data Imbalance

classes = np.unique(np.concatenate((y_train, y_test), axis=0))  

x = np.arange(len(classes)) 
labels = ["Abnormal", "Normal"]   

values_train = [(y_train == i).sum() for i in classes] 
values_valid = [(y_valid == i).sum() for i in classes]  
values_test = [(y_test == i).sum() for i in classes]  

plt.figure(figsize=(8,4))  

plt.subplot(1,3,1)  
plt.title("Training Data")  
plt.bar(x, values_train, width=0.6, color=["red", "blue"]) 
plt.ylim([0, 1500])
plt.xticks(x, labels)

plt.subplot(1,3,2) 
plt.title("Validation Data")
plt.bar(x, values_valid, width=0.6, color=["red", "blue"]) 
plt.ylim([0, 1500])
plt.xticks(x, labels)  

plt.subplot(1,3,3)  
plt.title("Test Data")
plt.bar(x, values_test, width=0.6, color=["red", "blue"])  
plt.ylim([0, 1500])
plt.xticks(x, labels)

plt.tight_layout()  
plt.savefig(save_path + 'data_imbalance.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()  # 그림 출력


# #특정 시간에서의 시계열 샘플
import random

labels = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure(figsize = (10, 4))
for c in labels:
    c_x_train = x_train[y_train == c]
    if c == -1: c = c + 1 
    time_t = random.randint(0, c_x_train.shape[0])
    plt.scatter(range(0, 500), c_x_train[time_t], label="class = " + str(int(c)), marker='o', s=5)
    
plt.legend(loc="lower right")
plt.xlabel("Sensor", fontsize=15)
plt.ylabel("Sensor Value", fontsize=15)
plt.savefig(save_path + 'ford_data_ts_sample1.png', dpi=100, bbox_inches='tight')
plt.show()
plt.close()


# 특정 시간에서의 시계열 샘플을 플롯 (정상/비정상 샘플 각각 출력)


def get_scatter_plot(c):
    time_t = random.randint(0, c_x_train.shape[0]) 
    plt.scatter(range(0, c_x_train.shape[1]), c_x_train[time_t], 
                marker='o', s=5, c="r" if c == -1  else "b")
    plt.title("at time: t_{}".format(time_t), fontsize=20)
    plt.xlabel("Sensor", fontsize=14)
    plt.ylabel("Sensor Value", fontsize=14)
    plt.savefig(save_path + '{state}.png'.format(state="abnormal" if c == -1 else "normal"), 
                dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

labels = np.unique(np.concatenate((y_train, y_test), axis=0))

for c in labels:
    c_x_train = x_train[y_train == c]
    if c == -1:
        print("비정상 Label 데이터 수: ", len(c_x_train))
        get_scatter_plot(c)
    else:
        print("정상 Label 데이터 수: ", len(c_x_train))
        get_scatter_plot(c)


# 1개의 임의의 센서 값의 시계열 Plot

sensor_number = random.randint(0, 500)  

plt.figure(figsize = (13, 4))
plt.title("sensor_number: {}".format(sensor_number), fontsize=20)
plt.plot(x_train[:, sensor_number])
plt.xlabel("Time", fontsize=15)
plt.ylabel("Sensor Value", fontsize=15)
plt.savefig(save_path + 'ford_a_sensor.png', dpi=100, bbox_inches='tight')
plt.show()
plt.close()


# ### 상관관계
import matplotlib.cm as cm
from matplotlib.collections import EllipseCollection

df = pd.DataFrame(data = x_train, 
                  columns= ["sensor_{}".format(label+1) for label in range(x_train.shape[1])])

data = df.corr()

def plot_corr_ellipses(data, ax =None, **kwargs):

    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec


fig, ax = plt.subplots(1, 1, figsize=(20, 20))
cmap = cm.get_cmap('jet', 31)
m = plot_corr_ellipses(data, ax=ax, cmap=cmap)
cb = fig.colorbar(m)
cb.set_label('Correlation coefficient')
# ax.margins(0.1)

plt.title('Correlation between Feature')
# labels = label
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.tight_layout()
plt.savefig(save_path + 'corr.png', dpi=100, bbox_inches='tight')  # 그림 저장

plt.show()


#데이터 전처리-정규화


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

# Standard Scaler
stder = StandardScaler()
stder.fit(x_train)
x_train = stder.transform(x_train)
x_valid = stder.transform(x_valid)


# Robust Scaler
rscaler = RobustScaler() 
rscaler.fit(x_train)
x_train = rscaler.transform(x_train)
x_valid = rscaler.transform(x_valid)

#Min-Max sclaer
mscaler = MinMaxScaler()
mscaler.fit(x_train)
x_train = mscaler.transform(x_train)
x_valid = mscaler.transform(x_valid)



x_train_exp = np.expand_dims(x_train, -1)  
x_valid_exp = np.expand_dims(x_valid, -1) 
x_test_exp = np.expand_dims(x_test, -1)  

print("x_train_exp의 형태:", x_train_exp.shape)
print("x_valid_exp의 형태:", x_valid_exp.shape)
print("x_test_exp의 형태:", x_test_exp.shape)


y_train[y_train == -1] = 0
y_valid[y_valid == -1] = 0
y_test[y_test == -1] = 0

num_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
num_classes


# Model

# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

clf_lr_1 = LogisticRegression(penalty='l2',
                         tol=0.01, 
                         C=1, 
                         fit_intercept=False, 
                         intercept_scaling=1, 
                         random_state=42, 
                         solver='lbfgs', 
                         max_iter=100,
                         multi_class='auto',
                         verbose=0)


x_train_lr = np.concatenate((x_train, x_valid), axis=0)
y_train_lr = np.concatenate((y_train, y_valid), axis=0)  


history_lr = clf_lr_1.fit(x_train_lr, y_train_lr)


y_pred = clf_lr_1.predict(x_test)
y_pred_proba = clf_lr_1.predict_proba(x_test)


score = clf_lr_1.score(x_test, y_test)
print("%s: %.2f%%" % ("Logistic Regression Prediction Rate", score*100))


# 결과 분석 및 해석

draw_confusion_matrix(clf_lr_1, x_test, y_test, "logistic_regression_sklearn")

draw_roc(clf_lr_1, x_test, y_test, "logistic_regression")

plt.figure()
plt.plot(history_lr)
plt.title("Training Loss")
plt.ylabel("loss", fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.tight_layout()
plt.savefig(save_path + 'lr_learning_curve.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()
plt.close()



#  XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(
    learning_rate=0.1, 
    n_estimators=500, 
    max_depth=5,
    min_child_weight=3, 
    gamma=0.2, 
    subsample=0.6, 
    colsample_bytree=1.0,
    objective='binary:logistic', 
    nthread=4, 
    scale_pos_weight=1, 
    seed=27)

print('Start Training')
xgb.fit(
    x_train, 
    y_train, 
    eval_metric= ['auc', 'error'],
    eval_set=[(x_train, y_train), (x_valid, y_valid)], 
    verbose=True
)

# XGBoost 평가

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

y_pred = xgb.predict(x_test)
y_pred_proba = xgb.predict_proba(x_test)[:, 1]

print("\nAbout xgboost model")
print("Accuracy : %.4g" % accuracy_score(y_test, y_pred))
print("AUC Score (training set): %f" % roc_auc_score(y_test, y_pred_proba))
print("F1 Score (training set): %f" % f1_score(y_test, y_pred))


#그래프
draw_confusion_matrix(xgb, x_test, y_test, "xgboost")

draw_roc(xgb, x_test, y_test, "xgboost")

xgb_results = xgb.evals_result()  # xgboost 모델의 평가 결과 불러오기
epochs = len(xgb_results['validation_0']['error'])    # iteration 수
x_axis = range(0, epochs)  # x축(epoch) 범위 설정


# plot classification error
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x_axis, xgb_results['validation_0']['error'], label='Train')
ax.plot(x_axis, xgb_results['validation_1']['error'], label='Validation')
ax.legend()

plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.tight_layout()
plt.savefig(save_path + 'learning_curve_error_xgb.png', dpi=100, bbox_inches='tight')
plt.show()


# plot auc
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x_axis, xgb_results['validation_0']['auc'], label='Train')
ax.plot(x_axis, xgb_results['validation_1']['auc'], label='Validation')
ax.legend()

plt.ylabel('AUC')
plt.title('XGBoost AUC')
plt.tight_layout()
plt.savefig(save_path + 'learning_curve_auc_xgb.png', dpi=100, bbox_inches='tight')
plt.show()


##Feature Importance


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

feat_imp = xgb.feature_importances_
idx = np.where(feat_imp > 0.003)
feat_imp_important = feat_imp[idx]
feat = ["sensor_{}".format(i+1) for i in idx[0]]
# clf.best_estimator_.booster().get_fscore()
res_df = pd.DataFrame({'Features': feat, 'Importance': feat_imp_important}).sort_values(by='Importance', ascending=False)
res_df.plot('Features', 'Importance', kind='bar', title='Feature Importances', figsize = (7, 5) )
plt.ylabel('Feature Importance Score')
plt.tight_layout()
plt.savefig(save_path + 'xgb_feature_importance.png', dpi=100, bbox_inches='tight')  # 그림 저장
plt.show()
print(res_df)
print(res_df["Features"].tolist())


x_train_feature = x_train[:,[260,318,485,255,12,411,341]]
y_train_feature = y_train[:]


x_train_exp = np.expand_dims(x_train, -1)  # 채널 축 1개 차원을 확장 시킨(Expand) X_train
x_valid_exp = np.expand_dims(x_valid, -1)  # 채널 축 1개 차원을 확장 시킨(Expand) X_vaild 
x_test_exp = np.expand_dims(x_test, -1)  # 채널 축 1개 차원을 확장 시킨(Expand) X_test


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense




#CNN(Convolitional Neural Network)
from tensorflow.keras.layers import LSTM, Flatten, Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv1D, ReLU, GlobalAveragePooling1D, Dense

def make_cnn_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3,padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(2, activation="softmax"))
    return model

cnn_model = make_cnn_model()


from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

epochs = 300
batch_size = 64

callbacks = [
    ModelCheckpoint(
        save_path + "cnn_best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history_cnn = cnn_model.fit(
    x_train_exp,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(x_valid_exp, y_valid),
    verbose=1,
)

#  CNN 평가
cnn_model.summary()
cnn_model = tf.keras.models.load_model(save_path + "cnn_best_model.h5")
scores = cnn_model.evaluate(x_test_exp, y_test)

print("\n""Test accuracy", scores[1])
print("\n""Test loss", scores[0])
print("%s: %.2f%%" % (cnn_model.metrics_names[1], scores[1]*100))


draw_confusion_matrix(cnn_model, x_test_exp, y_test, "cnn")
draw_roc(cnn_model, x_test_exp, y_test, "cnn")



plot_loss_graph(history_cnn, "cnn")
plot_prediction_graph(history_cnn, "cnn")


# lstm 

def make_lstm_model():
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

lstm_model = make_lstm_model()
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
lstm_model.summary()
epochs= 7
batch_size = 64

lstm_model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer='adam', 
                  metrics=["sparse_categorical_accuracy"]
                 )

callbacks = [ModelCheckpoint(file_path + 'lstm_best_model.h5', 
                             monitor='val_loss',
                             save_best_only=True),
             ReduceLROnPlateau(
                 monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
                 ),
             EarlyStopping(monitor="val_loss", patience=10, verbose=1)
             ]

history_lstm = lstm_model.fit(
    x_train_exp,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(x_valid_exp, y_valid),
    verbose=1
)

from tensorflow.keras.models import load_model

lstm_model = tf.keras.models.load_model(save_path + "lstm_best_model.h5")
scores = lstm_model.evaluate(x_test_exp, y_test)

print("\n""Test accuracy", scores[1])
print("\n""Test loss", scores[0])
print("%s: %.2f%%" % (lstm_model.metrics_names[1], scores[1]*100))

plot_loss_graph(history_lstm, "lstm")
plot_prediction_graph(history_lstm, "lstm")


# plot classification error
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x_axis, xgb_results['validation_0']['error'], label='Train')
ax.plot(x_axis, xgb_results['validation_1']['error'], label='Validation')
ax.legend()

plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.tight_layout()
plt.savefig(save_path + 'learning_curve_error_xgb.png', dpi=100, bbox_inches='tight')
plt.show()


# plot auc
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x_axis, xgb_results['validation_0']['auc'], label='Train')
ax.plot(x_axis, xgb_results['validation_1']['auc'], label='Validation')
ax.legend()

plt.ylabel('AUC')
plt.title('XGBoost AUC')
plt.tight_layout()
plt.savefig(save_path + 'learning_curve_auc_xgb.png', dpi=100, bbox_inches='tight')
plt.show()


#CNN-LSTM Model


from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def make_cnn_lstm_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(filters=64, kernel_size=3,padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(LSTM(units=256, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

cnn_lstm_model = make_cnn_lstm_model()



epochs= 200
batch_size = 64

cnn_lstm_model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer='adam', 
                  metrics=["sparse_categorical_accuracy"]
                 )

callbacks = [ModelCheckpoint(file_path + 'cnn_lstm_best_model.h5', 
                             monitor='val_loss',
                             save_best_only=True),
             ReduceLROnPlateau(
                 monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
                 ),
             EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=5, verbose=1,mode = "max")
             ]

history_cnn_lstm = cnn_lstm_model.fit(
    x_train_exp,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(x_valid_exp, y_valid),
    verbose=1
)

from tensorflow.keras.models import load_model

cnn_lstm_model = tf.keras.models.load_model(save_path + "cnn_lstm_best_model.h5")
scores = cnn_lstm_model.evaluate(x_test_exp, y_test)

print("\n""Test accuracy", scores[1])
print("\n""Test loss", scores[0])
print("%s: %.2f%%" % (cnn_lstm_model.metrics_names[1], scores[1]*100))

draw_confusion_matrix(cnn_lstm_model, x_test_exp, y_test, "cnn_lstm")

y_test = y_test.flatten()
draw_roc(cnn_lstm_model, x_test_exp, y_test, "cnn_lstm")

plot_loss_graph(history_cnn_lstm, "cnn_lstm")

plot_prediction_graph(history_cnn_lstm, "cnn_lstm")
