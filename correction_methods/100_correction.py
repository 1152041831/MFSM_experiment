import math
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, curve_fit
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import warnings
warnings.filterwarnings('ignore')
# 创建高斯过程回归模型对象
kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1))

ADD_NOISE_MSE = []
MUL_NOISE_MSE = []
MIX_NOISE_MSE = []

parameter = []
seed = 654  # 设置种子为654
result = []
# 重复运行100次生成随机数组的操作
for i in range(100):
    print("#################################### Group ",i+1," ####################################","seed: ",seed)
    random.seed(seed)  # 设置随机数种子
    parameter = [random.randint(-5, 5) for i in range(10)]

    def y_true(x):
        A,B,C,D,E,F,G,H,I,J = parameter
        return A * math.sin(B * x ** 2 + C) * (D * x ** 2 + E) + F * (G * x ** 3 + H) + I * x + J

    X = np.arange(0,1,0.001)
    X_test = np.arange(0.0498,1,0.0498) # 20
    y_test_true = [y_true(e) for e in X_test]

    Xh1 = np.arange(0,1,0.05) # 20
    Xl1 = np.arange(0,1,0.005) # 200

    # 构造多保真度数据
    Yh1 = [y_true(e)+0.4*random.random()*random.choice([-1, 1]) for e in Xh1]
    Yl1 = [y_true(e)+0.8*random.random()*random.choice([-1, 1]) for e in Xl1]
    # 将Xh和Yh打包成元组列表
    data_Xh1 = list(zip(Xh1, Yh1))
    data_Xl1 = list(zip(Xl1, Yl1))
    # 随机打乱元组列表
    random.shuffle(data_Xh1)
    random.shuffle(data_Xl1)
    # 将元组列表解包为打乱后的Xh和Yh
    Xh_train1, yh_train1 = zip(*data_Xh1)
    Xl_train1, yl_train1 = zip(*data_Xl1)
    # 防止精度误差
    Xh_train1 = [round(num, 3) for num in Xh_train1]
    Xl_train1 = [round(num, 3) for num in Xl_train1]
    yh_train1 = [num for num in yh_train1]
    yl_train1 = [num for num in yl_train1]
    delta1_1 = []
    delta1_2 = []
    Xh_yl_1 = [] # 20个高保真度数据输入值对应的低保真度输出
    for i in range(len(Xh_train1)):
        for j in range(len(Xl_train1)):
            if Xh_train1[i] == Xl_train1[j]:
                delta1_1.append(yh_train1[i] - yl_train1[j])
                delta1_2.append(yh_train1[i] / yl_train1[j])
                Xh_yl_1.append(yl_train1[j])
                break
    # 拟合低保真度数据
    #################################### Zlo(x) ####################################
    gpr0_1 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr0_1.fit(np.array(Xl_train1).reshape(-1,1), np.array(yl_train1).reshape(-1,1))
    pre0_1, cov0_1 = gpr0_1.predict(np.array(Xl_train1).reshape(-1,1),return_std=True)
    Zl1 = pre0_1.ravel()

    # 拟合差值
    gpr1_1 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr1_1.fit(np.array(Xh_train1).reshape(-1,1), np.array(delta1_1).reshape(-1,1))
    #################################### Zd(x) ####################################
    pre1_1, cov1_1 = gpr1_1.predict(np.array(Xl_train1).reshape(-1,1),return_std=True)
    Zd1 = pre1_1.ravel()

    # 拟合比值
    gpr1_2 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr1_2.fit(np.array(Xh_train1).reshape(-1,1), np.array(delta1_2).reshape(-1,1))
    pre1_2, cov1_2 = gpr1_2.predict(np.array(Xl_train1).reshape(-1,1),return_std=True)
    pre_delta1_2 = pre1_2.ravel()

    # 高保真度拟合模型
    gpr1_3 = GaussianProcessRegressor(kernel=kernel, alpha=0.3, normalize_y=True)
    gpr1_3.fit(np.array(Xh_train1).reshape(-1, 1), np.array(yh_train1).reshape(-1, 1))
    mu1_3, cov1_3= gpr1_3.predict(np.array(Xl1).reshape(-1,1),return_std=True)
    pre_HFDM_1 = mu1_3.ravel()
    # test
    mu11_1, cov11_1= gpr1_3.predict(np.array(X_test).reshape(-1,1),return_std=True)
    HFDM_test_1 = mu11_1.ravel()

    # 拟合所有的高保真度数据
    # 在Xl_train中排除Xh_train中的数值
    index_Xl_without_Xh_1 = [i for i in range(len(Xl_train1)) if Xl_train1[i] not in Xh_train1]
    new_Xl_train_1 = [Xl_train1[i] for i in index_Xl_without_Xh_1] #180
    new_yl_train_1 = [yl_train1[i] for i in index_Xl_without_Xh_1] #180
    new_Xh_1 = Xh_train1.copy()
    new_yh_11 = yh_train1.copy()
    new_yh_12 = yh_train1.copy()
    new_Xh_1.extend(new_Xl_train_1)
    # 拟合180个低保真度数据对应高保真度度数据的差值
    pre1_4, cov1_4 = gpr1_1.predict(np.array(new_Xl_train_1).reshape(-1, 1),return_std=True)
    delta_180_11 = pre1_4.ravel()
    new_add_yh_11 = [delta_180_11[i]+new_yl_train_1[i] for i in range(len(new_yl_train_1))]
    new_yh_11.extend(new_add_yh_11)
    # 拟合180个低保真度数据对应高保真度数据的比值
    pre1_5, cov1_5 = gpr1_2.predict(np.array(new_Xl_train_1).reshape(-1, 1),return_std=True)
    delta_180_12 = pre1_5.ravel()
    new_add_yh_12 = [delta_180_12[i]*new_yl_train_1[i] for i in range(len(new_yl_train_1))]
    new_yh_12.extend(new_add_yh_12)

    # AC用原本的高保真度数据和新生成的高保真度数据，拟合
    gpr1_6 = GaussianProcessRegressor(kernel=kernel, alpha=0.45, normalize_y=True)
    gpr1_6.fit(np.array(new_Xh_1).reshape(-1, 1), np.array(new_yh_11).reshape(-1, 1))
    mu1_6, cov1_6 = gpr1_6.predict(np.array(Xl1).reshape(-1,1),return_std=True)
    AC_1 = mu1_6.ravel()
    # test
    mu11_6, cov11_6 = gpr1_6.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_AC_1 = mu11_6.ravel()

    # MC用原本的高保真度数据和新生成的高保真度数据，拟合
    gpr1_7 = GaussianProcessRegressor(kernel=kernel, alpha=0.4, normalize_y=True)
    gpr1_7.fit(np.array(new_Xh_1).reshape(-1, 1), np.array(new_yh_12).reshape(-1, 1))
    mu1_7, cov1_7 = gpr1_7.predict(np.array(Xl1).reshape(-1,1),return_std=True)
    MC_1 = mu1_7.ravel()
    # test
    mu11_7, cov11_7 = gpr1_7.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_MC_1 = mu11_7.ravel()

    # CC_1
    # get p   // Zd(x) = yh - p * Zlo(x)
    # minimize || Zd(x) - (yh - p*Zlo(x)) ||^2
    # p = ( Zlo(x)^T · (yh - Zd(x)) ) / ( Zlo(x)^T · Zlo(x) )
    list_p1 = [(yh_train1[i] - delta1_1[i]) / Xh_yl_1[i] for i in range(len(Xh_train1))]
    p1 = np.dot(np.array(Xh_yl_1).reshape(-1,1).T, (np.array(yh_train1).reshape(-1,1) - np.array(delta1_1).reshape(-1,1))) / np.dot(np.array(Xh_yl_1).reshape(-1,1).T, np.array(Xh_yl_1).reshape(-1,1))

    #################################### Zh(x) ####################################
    gpr1_8 = GaussianProcessRegressor(kernel=kernel, alpha=0.4, normalize_y=True)
    gpr1_8.fit(np.array(Xl_train1).reshape(-1, 1), (p1*np.array(Zl1).reshape(-1,1)+np.array(Zd1).reshape(-1,1)))
    mu1_8, cov1_8 = gpr1_8.predict(np.array(Xl1).reshape(-1,1),return_std=True)
    CC_1 = mu1_8.ravel()
    #test
    mu11_8, cov11_8 = gpr1_8.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_CC_1 = mu11_8.ravel()


    # MSE :
    HFDM_with_truth = np.mean((np.array(y_test_true) - np.array(HFDM_test_1))**2)
    AC_with_truth_1 = np.mean((np.array(y_test_true) - np.array(test_AC_1))**2)
    MC_with_truth_1 = np.mean((np.array(y_test_true) - np.array(test_MC_1))**2)
    CC_with_truth_1 = np.mean((np.array(y_test_true) - np.array(test_CC_1))**2)

    ADD_NOISE_MSE.append([HFDM_with_truth,AC_with_truth_1,MC_with_truth_1,CC_with_truth_1])

    ####################################################################################################################
    Xh2 = np.arange(0,1,0.05) # 20个
    Xl2 = np.arange(0,1,0.005) # 200个
    # 构造多保真度数据
    Yh2 = [random.uniform(0.9, 1.1)*y_true(e) for e in Xh2]
    Yl2 = [random.uniform(0.8, 1.3)*y_true(e) for e in Xl2]
    # 将Xh和Yh打包成元组列表
    data_Xh2 = list(zip(Xh2, Yh2))
    data_Xl2 = list(zip(Xl2, Yl2))
    # 随机打乱元组列表
    random.shuffle(data_Xh2)
    random.shuffle(data_Xl2)
    # 将元组列表解包为打乱后的Xh和Yh
    Xh_train2, yh_train2 = zip(*data_Xh2)
    Xl_train2, yl_train2 = zip(*data_Xl2)
    # 防止精度误差
    Xh_train2 = [round(num, 3) for num in Xh_train2]
    Xl_train2 = [round(num, 3) for num in Xl_train2]
    yh_train2 = [num for num in yh_train2]
    yl_train2 = [num for num in yl_train2]
    delta2_1 = []
    delta2_2 = []
    Xh_yl_2 = [] # 20个高保真度数据输入值对应的低保真度输出
    for i in range(len(Xh_train2)):
        for j in range(len(Xl_train2)):
            if Xh_train2[i] == Xl_train2[j]:
                delta2_1.append(yh_train2[i] - yl_train2[j])
                delta2_2.append(yh_train2[i] / yl_train2[j])
                Xh_yl_2.append(yl_train2[j])
                break
    # 拟合低保真度数据
    #################################### Zlo(x) ####################################
    gpr0_2 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr0_2.fit(np.array(Xl_train2).reshape(-1,1), np.array(yl_train2).reshape(-1,1))
    pre0_2, cov0_2 = gpr0_2.predict(np.array(Xl_train2).reshape(-1,1),return_std=True)
    Zl2 = pre0_2.ravel()

    # 拟合差值
    gpr2_1 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr2_1.fit(np.array(Xh_train2).reshape(-1,1), np.array(delta2_1).reshape(-1,1))
    #################################### Zd(x) ####################################
    pre2_1, cov2_1 = gpr2_1.predict(np.array(Xl_train2).reshape(-1,1),return_std=True)
    Zd2 = pre2_1.ravel()

    # 拟合比值
    gpr2_2 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr2_2.fit(np.array(Xh_train2).reshape(-1,1), np.array(delta2_2).reshape(-1,1))
    pre2_2, cov2_2 = gpr2_2.predict(np.array(Xl_train2).reshape(-1,1),return_std=True)
    pre_delta2_2 = pre2_2.ravel()

    # 高保真度拟合模型
    gpr2_3 = GaussianProcessRegressor(kernel=kernel, alpha=0.3, normalize_y=True)
    gpr2_3.fit(np.array(Xh_train2).reshape(-1, 1), np.array(yh_train2).reshape(-1, 1))
    mu2_3, cov2_3= gpr2_3.predict(np.array(Xl2).reshape(-1,1),return_std=True)
    pre_HFDM_2 = mu2_3.ravel()
    # test
    mu22_1, cov22_1= gpr2_3.predict(np.array(X_test).reshape(-1,1),return_std=True)
    HFDM_test_2 = mu22_1.ravel()

    # 拟合所有的高保真度数据
    # 在Xl_train中排除Xh_train中的数值
    index_Xl_without_Xh_2 = [i for i in range(len(Xl_train2)) if Xl_train2[i] not in Xh_train2]
    new_Xl_train_2 = [Xl_train2[i] for i in index_Xl_without_Xh_2] #180
    new_yl_train_2 = [yl_train2[i] for i in index_Xl_without_Xh_2] #180
    new_Xh_2 = Xh_train1.copy()
    new_yh_21 = yh_train2.copy()
    new_yh_22 = yh_train2.copy()
    new_Xh_2.extend(new_Xl_train_2)
    # 拟合180个低保真度数据对应高保真度度数据的差值
    pre2_4, cov2_4 = gpr2_1.predict(np.array(new_Xl_train_2).reshape(-1, 1),return_std=True)
    delta_180_21 = pre2_4.ravel()
    new_add_yh_21 = [delta_180_21[i]+new_yl_train_2[i] for i in range(len(new_yl_train_2))]
    new_yh_21.extend(new_add_yh_21)
    # 拟合180个低保真度数据对应高保真度数据的比值
    pre2_5, cov2_5 = gpr2_2.predict(np.array(new_Xl_train_2).reshape(-1, 1),return_std=True)
    delta_180_22 = pre2_5.ravel()
    new_add_yh_22 = [delta_180_22[i]*new_yl_train_2[i] for i in range(len(new_yl_train_2))]
    new_yh_22.extend(new_add_yh_22)

    # AC用原本的高保真度数据和新生成的高保真度数据，拟合
    gpr2_6 = GaussianProcessRegressor(kernel=kernel, alpha=0.4, normalize_y=True)
    gpr2_6.fit(np.array(new_Xh_2).reshape(-1, 1), np.array(new_yh_21).reshape(-1, 1))
    mu2_6, cov2_6 = gpr2_6.predict(np.array(Xl2).reshape(-1,1),return_std=True)
    AC_2 = mu2_6.ravel()
    # test
    mu22_6, cov22_6 = gpr2_6.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_AC_2 = mu22_6.ravel()

    # MC用原本的高保真度数据和新生成的高保真度数据，拟合
    gpr2_7 = GaussianProcessRegressor(kernel=kernel, alpha=0.4, normalize_y=True)
    gpr2_7.fit(np.array(new_Xh_2).reshape(-1, 1), np.array(new_yh_22).reshape(-1, 1))
    mu2_7, cov2_7 = gpr2_7.predict(np.array(Xl2).reshape(-1,1),return_std=True)
    MC_2 = mu2_7.ravel()
    # test
    mu22_7, cov22_7 = gpr2_7.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_MC_2 = mu22_7.ravel()

    # CC_2
    # get p   // Zd(x) = yh - p * Zlo(x)
    # minimize || Zd(x) - (yh - p*Zlo(x)) ||^2
    # p = ( Zlo(x)^T · (yh - Zd(x)) ) / ( Zlo(x)^T · Zlo(x) )
    list_p2 = [(yh_train2[i] - delta2_1[i]) / Xh_yl_2[i] for i in range(len(Xh_train2))]
    p2 = np.dot(np.array(Xh_yl_2).reshape(-1,1).T, (np.array(yh_train2).reshape(-1,1) - np.array(delta2_1).reshape(-1,1))) / np.dot(np.array(Xh_yl_2).reshape(-1,1).T, np.array(Xh_yl_2).reshape(-1,1))

    #################################### Zh(x) ####################################
    gpr2_8 = GaussianProcessRegressor(kernel=kernel, alpha=0.3, normalize_y=True)
    gpr2_8.fit(np.array(Xl_train2).reshape(-1, 1), (p2*np.array(Zl2).reshape(-1,1)+np.array(Zd2).reshape(-1,1)))
    mu2_8, cov2_8 = gpr2_8.predict(np.array(Xl2).reshape(-1,1),return_std=True)
    CC_2 = mu2_8.ravel()
    #test
    mu22_8, cov22_8 = gpr2_8.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_CC_2 = mu22_8.ravel()


    # MSE :
    HFDM_with_truth = np.mean((np.array(y_test_true) - np.array(HFDM_test_2))**2)
    AC_with_truth_2 = np.mean((np.array(y_test_true) - np.array(test_AC_2))**2)
    MC_with_truth_2 = np.mean((np.array(y_test_true) - np.array(test_MC_2))**2)
    CC_with_truth_2 = np.mean((np.array(y_test_true) - np.array(test_CC_2))**2)
    MUL_NOISE_MSE.append([HFDM_with_truth,AC_with_truth_2,MC_with_truth_2,CC_with_truth_2])

    Xh3 = np.arange(0,1,0.05) # 20个
    Xl3 = np.arange(0,1,0.005) # 200个
    # 构造多保真度数据
    Yh3 = [random.uniform(0.9, 1.1)*y_true(e)+0.4*random.random()*random.choice([-1, 1]) for e in Xh3]
    Yl3 = [random.uniform(0.8, 1.3)*y_true(e)+0.8*random.random()*random.choice([-1, 1]) for e in Xl3]
    # 将Xh和Yh打包成元组列表
    data_Xh3 = list(zip(Xh3, Yh3))
    data_Xl3 = list(zip(Xl3, Yl3))
    # 随机打乱元组列表
    random.shuffle(data_Xh3)
    random.shuffle(data_Xl3)
    # 将元组列表解包为打乱后的Xh和Yh
    Xh_train3, yh_train3 = zip(*data_Xh3)
    Xl_train3, yl_train3 = zip(*data_Xl3)
    # 防止精度误差
    Xh_train3 = [round(num, 3) for num in Xh_train3]
    Xl_train3 = [round(num, 3) for num in Xl_train3]
    yh_train3 = [num for num in yh_train3]
    yl_train3 = [num for num in yl_train3]

    delta3_1 = []
    delta3_2 = []
    Xh_yl_3 = [] # 20个高保真度数据输入值对应的低保真度输出
    for i in range(len(Xh_train3)):
        for j in range(len(Xl_train3)):
            if Xh_train3[i] == Xl_train3[j]:
                delta3_1.append(yh_train3[i] - yl_train3[j])
                delta3_2.append(yh_train3[i] / yl_train3[j])
                Xh_yl_3.append(yl_train3[j])
                break
    # 拟合低保真度数据
    #################################### Zlo(x) ####################################
    gpr0_3 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr0_3.fit(np.array(Xl_train3).reshape(-1,1), np.array(yl_train3).reshape(-1,1))
    pre0_3, cov0_3 = gpr0_3.predict(np.array(Xl_train3).reshape(-1,1),return_std=True)
    Zl3 = pre0_3.ravel()

    # 拟合差值
    gpr3_1 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr3_1.fit(np.array(Xh_train3).reshape(-1,1), np.array(delta3_1).reshape(-1,1))
    #################################### Zd(x) ####################################
    pre3_1, cov3_1 = gpr3_1.predict(np.array(Xl_train3).reshape(-1,1),return_std=True)
    Zd3 = pre3_1.ravel()

    # 拟合比值
    gpr3_2 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gpr3_2.fit(np.array(Xh_train3).reshape(-1,1), np.array(delta3_2).reshape(-1,1))
    pre3_2, cov3_2 = gpr3_2.predict(np.array(Xl_train3).reshape(-1,1),return_std=True)
    pre_delta3_2 = pre3_2.ravel()

    # 高保真度拟合模型
    gpr3_3 = GaussianProcessRegressor(kernel=kernel, alpha=0.3, normalize_y=True)
    gpr3_3.fit(np.array(Xh_train3).reshape(-1, 1), np.array(yh_train3).reshape(-1, 1))
    mu3_3, cov3_3= gpr2_3.predict(np.array(Xl3).reshape(-1,1),return_std=True)
    pre_HFDM_3 = mu3_3.ravel()
    # test
    mu33_1, cov33_1= gpr3_3.predict(np.array(X_test).reshape(-1,1),return_std=True)
    HFDM_test_3 = mu33_1.ravel()

    # 拟合所有的高保真度数据
    # 在Xl_train中排除Xh_train中的数值
    index_Xl_without_Xh_3 = [i for i in range(len(Xl_train3)) if Xl_train3[i] not in Xh_train3]
    new_Xl_train_3 = [Xl_train3[i] for i in index_Xl_without_Xh_3] #180
    new_yl_train_3 = [yl_train3[i] for i in index_Xl_without_Xh_3] #180
    new_Xh_3 = Xh_train1.copy()
    new_yh_31 = yh_train3.copy()
    new_yh_32 = yh_train3.copy()
    new_Xh_3.extend(new_Xl_train_3)
    # 拟合180个低保真度数据对应高保真度度数据的差值
    pre3_4, cov3_4 = gpr3_1.predict(np.array(new_Xl_train_3).reshape(-1, 1),return_std=True)
    delta_180_31 = pre3_4.ravel()
    new_add_yh_31 = [delta_180_31[i]+new_yl_train_3[i] for i in range(len(new_yl_train_3))]
    new_yh_31.extend(new_add_yh_31)
    # 拟合180个低保真度数据对应高保真度数据的比值
    pre3_5, cov3_5 = gpr3_2.predict(np.array(new_Xl_train_3).reshape(-1, 1),return_std=True)
    delta_180_32 = pre3_5.ravel()
    new_add_yh_32 = [delta_180_32[i]*new_yl_train_3[i] for i in range(len(new_yl_train_3))]
    new_yh_32.extend(new_add_yh_32)

    # AC用原本的高保真度数据和新生成的高保真度数据，拟合
    gpr3_6 = GaussianProcessRegressor(kernel=kernel, alpha=0.5, normalize_y=True)
    gpr3_6.fit(np.array(new_Xh_3).reshape(-1, 1), np.array(new_yh_31).reshape(-1, 1))
    mu3_6, cov3_6 = gpr3_6.predict(np.array(Xl3).reshape(-1,1),return_std=True)
    AC_3 = mu3_6.ravel()
    # test
    mu33_6, cov33_6 = gpr3_6.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_AC_3 = mu33_6.ravel()

    # MC用原本的高保真度数据和新生成的高保真度数据，拟合
    gpr3_7 = GaussianProcessRegressor(kernel=kernel, alpha=0.4, normalize_y=True)
    gpr3_7.fit(np.array(new_Xh_3).reshape(-1, 1), np.array(new_yh_32).reshape(-1, 1))
    mu3_7, cov3_7 = gpr3_7.predict(np.array(Xl3).reshape(-1,1),return_std=True)
    MC_3 = mu3_7.ravel()
    # test
    mu33_7, cov33_7 = gpr3_7.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_MC_3 = mu33_7.ravel()

    # CC_2
    # get p   // Zd(x) = yh - p * Zlo(x)
    # minimize || Zd(x) - (yh - p*Zlo(x)) ||^2
    # p = ( Zlo(x)^T · (yh - Zd(x)) ) / ( Zlo(x)^T · Zlo(x) )
    list_p3 = [(yh_train3[i] - delta3_1[i]) / Xh_yl_3[i] for i in range(len(Xh_train3))]
    p3 = np.dot(np.array(Xh_yl_3).reshape(-1,1).T, (np.array(yh_train3).reshape(-1,1) - np.array(delta3_1).reshape(-1,1))) / np.dot(np.array(Xh_yl_3).reshape(-1,1).T, np.array(Xh_yl_3).reshape(-1,1))

    #################################### Zh(x) ####################################
    gpr3_8 = GaussianProcessRegressor(kernel=kernel, alpha=0.5, normalize_y=True)
    gpr3_8.fit(np.array(Xl_train3).reshape(-1, 1), (p3*np.array(Zl3).reshape(-1,1)+np.array(Zd3).reshape(-1,1)))
    mu3_8, cov3_8 = gpr3_8.predict(np.array(Xl3).reshape(-1,1),return_std=True)
    CC_3 = mu3_8.ravel()
    #test
    mu33_8, cov33_8 = gpr3_8.predict(np.array(X_test).reshape(-1,1),return_std=True)
    test_CC_3 = mu33_8.ravel()


    # MSE :
    HFDM_with_truth = np.mean((np.array(y_test_true) - np.array(HFDM_test_3))**2)
    AC_with_truth_3 = np.mean((np.array(y_test_true) - np.array(test_AC_3))**2)
    MC_with_truth_3 = np.mean((np.array(y_test_true) - np.array(test_MC_3))**2)
    CC_with_truth_3 = np.mean((np.array(y_test_true) - np.array(test_CC_3))**2)
    MIX_NOISE_MSE.append([HFDM_with_truth,AC_with_truth_3,MC_with_truth_3,CC_with_truth_3])

    seed += 1  # 更新种子的值




### ADD ###
print("ADD NOISE MSE:")
ADD_HFDM_MSE = []
ADD_AC_MSE = []
ADD_MC_MSE = []
ADD_CC_MSE = []
for i in range(len(ADD_NOISE_MSE)):
    ADD_HFDM_MSE.append(ADD_NOISE_MSE[i][0])
    ADD_AC_MSE.append(ADD_NOISE_MSE[i][1])
    ADD_MC_MSE.append(ADD_NOISE_MSE[i][2])
    ADD_CC_MSE.append(ADD_NOISE_MSE[i][3])

print("HDFM MSE:",np.mean(ADD_HFDM_MSE),"=>",round(np.mean(ADD_HFDM_MSE),3))
print("AC MSE:",np.mean(ADD_AC_MSE),"=>",round(np.mean(ADD_AC_MSE),3))
print("MC MSE:",np.mean(ADD_MC_MSE),"=>",round(np.mean(ADD_MC_MSE),3))
print("CC MSE:",np.mean(ADD_CC_MSE),"=>",round(np.mean(ADD_CC_MSE),3))

### MUL ###
print("MUL NOISE MSE:")
MUL_HFDM_MSE = []
MUL_AC_MSE = []
MUL_MC_MSE = []
MUL_CC_MSE = []
for i in range(len(MUL_NOISE_MSE)):
    MUL_HFDM_MSE.append(MUL_NOISE_MSE[i][0])
    MUL_AC_MSE.append(MUL_NOISE_MSE[i][1])
    MUL_MC_MSE.append(MUL_NOISE_MSE[i][2])
    MUL_CC_MSE.append(MUL_NOISE_MSE[i][3])

print("HDFM MSE:",np.mean(MUL_HFDM_MSE),"=>",round(np.mean(MUL_HFDM_MSE),3))
print("AC MSE:",np.mean(MUL_AC_MSE),"=>",round(np.mean(MUL_AC_MSE),3))
print("MC MSE:",np.mean(MUL_MC_MSE),"=>",round(np.mean(MUL_MC_MSE),3))
print("CC MSE:",np.mean(MUL_CC_MSE),"=>",round(np.mean(MUL_CC_MSE),3))

### MIX ###
print("MIX NOISE MSE:")
MIX_HFDM_MSE = []
MIX_AC_MSE = []
MIX_MC_MSE = []
MIX_CC_MSE = []
for i in range(len(MIX_NOISE_MSE)):
    MIX_HFDM_MSE.append(MIX_NOISE_MSE[i][0])
    MIX_AC_MSE.append(MIX_NOISE_MSE[i][1])
    MIX_MC_MSE.append(MIX_NOISE_MSE[i][2])
    MIX_CC_MSE.append(MIX_NOISE_MSE[i][3])

print("HDFM MSE:",np.mean(MIX_HFDM_MSE),"=>",round(np.mean(MIX_HFDM_MSE),3))
print("AC MSE:",np.mean(MIX_AC_MSE),"=>",round(np.mean(MIX_AC_MSE),3))
print("MC MSE:",np.mean(MIX_MC_MSE),"=>",round(np.mean(MIX_MC_MSE),3))
print("CC MSE:",np.mean(MIX_CC_MSE),"=>",round(np.mean(MIX_CC_MSE),3))

