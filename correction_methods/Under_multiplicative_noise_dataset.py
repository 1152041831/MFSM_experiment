import json
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, curve_fit, minimize
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 在乘性噪声的数据集下寻找最优参数alpha


# 100个真值函数中，每种方法的MSE
mse_HFDM_in_MUL_100 = []
mse_AC_in_MUL_100 = []
mse_MC_in_MUL_100 = []
mse_CC_in_MUL_100 = []
# RMSE
rmse_HFDM_in_MUL_100 = []
rmse_AC_in_MUL_100 = []
rmse_MC_in_MUL_100 = []
rmse_CC_in_MUL_100 = []
# MAE
mae_HFDM_in_MUL_100 = []
mae_AC_in_MUL_100 = []
mae_MC_in_MUL_100 = []
mae_CC_in_MUL_100 = []
# R^2
r_squared_HFDM_in_MUL_100 = []
r_squared_AC_in_MUL_100 = []
r_squared_MC_in_MUL_100 = []
r_squared_CC_in_MUL_100 = []
# ME
me_HFDM_in_MUL_100 = []
me_AC_in_MUL_100 = []
me_MC_in_MUL_100 = []
me_CC_in_MUL_100 = []
# Error standard deviation, ESD
esd_HFDM_in_MUL_100 = []
esd_AC_in_MUL_100 = []
esd_MC_in_MUL_100 = []
esd_CC_in_MUL_100 = []
# 创建高斯过程回归模型对象
# length_scale=0.5, length_scale_bounds=(1e-2, 1)
kernel = RBF()

parameter = []
all_parameters = []
MSE = []
seed = 654  # 设置种子为654
result = []

# 设置数据集
# 训练集
Xh = np.array([round(e,3) for e in np.arange(0,1,0.05)]) # 20
Xl = np.array([round(e,3) for e in np.arange(0,1,0.005)]) # 200
# 测试集
X_test = np.arange(0.0498,1,0.0498) # 20


# 修正方法
# 输入高、低保真度数据
# 返回gpr模型
def AC(Xh, Xl, Yh, Yl, alpha):
    # 对低保真度数据做gpr
    gpr_Xl = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_Xl.fit(np.array(Xl).reshape(-1,1), np.array(Yl).reshape(-1,1))
    # 计算2种保真度之间的差值
    delta = []
    for i in range(len(Xh)):
        xh = Xh[i]
        # 如果xl存在
        if xh in Xl:
            for j in range(len(Xl)):
                xl = Xl[j]
                if xh == xl:
                    delta.append(Yh[i] - Yl[j])
                    break
        else:
            pre_Yl = gpr_Xl.predict(np.array(xh).reshape(-1,1))
            pre_Yl = pre_Yl[0]
            delta.append(Yh[i] - pre_Yl)

    # 对差值delta做gpr
    gpr_delta = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_delta.fit(np.array(Xh).reshape(-1,1), np.array(delta).reshape(-1,1))

    delta_yl = gpr_delta.predict(np.array(Xl).reshape(-1, 1))
    # 修正低保真度数据的y
    corrected_yl = Yl + delta_yl

    # 得到新的模型, 修正过后的低保真度模型
    # 加性0.45 乘性0.4 混合0.5
    gpr_ac = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_ac.fit(np.array(Xl).reshape(-1, 1), np.array(corrected_yl).reshape(-1, 1))

    return gpr_ac

def MC(Xh, Xl, Yh, Yl, alpha):
    # 对低保真度数据做gpr
    gpr_Xl = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_Xl.fit(np.array(Xl).reshape(-1, 1), np.array(Yl).reshape(-1, 1))
    # 计算2种保真度之间的差值
    delta = []
    for i in range(len(Xh)):
        xh = Xh[i]
        # 如果xl存在
        if xh in Xl:
            for j in range(len(Xl)):
                xl = Xl[j]
                if xh == xl:
                    delta.append(Yh[i] / Yl[j])
                    break
        else:
            pre_Yl = gpr_Xl.predict(np.array(xh).reshape(-1, 1))
            pre_Yl = pre_Yl[0]
            delta.append(Yh[i] / pre_Yl)

    # 对差值delta做gpr
    gpr_delta = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_delta.fit(np.array(Xh).reshape(-1, 1), np.array(delta).reshape(-1, 1))

    delta_yl = gpr_delta.predict(np.array(Xl).reshape(-1, 1))
    # 修正低保真度数据的y
    corrected_yl = np.array(Yl) * delta_yl


    # 得到新的模型, 修正过后的低保真度模型
    # 加性0.4 乘性0.4 混合0.4
    gpr_mc = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_mc.fit(np.array(Xl).reshape(-1, 1), np.array(corrected_yl).reshape(-1, 1))

    return gpr_mc

def CC(Xh, Xl, Yh, Yl, alpha):
    # 对低保真度数据做gpr
    gpr_Xl = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_Xl.fit(np.array(Xl).reshape(-1,1), np.array(Yl).reshape(-1,1))
    Yl_Xh = gpr_Xl.predict(np.array(Xh).reshape(-1,1))
    yl = [] # yl(Xh)
    delta = []
    for i in range(len(Xh)):
        xh = Xh[i]
        # 如果xl存在
        if xh in Xl:
            for j in range(len(Xl)):
                xl = Xl[j]
                if xh == xl:
                    delta.append(Yh[i] - Yl[j])
                    yl.append(Yl[j])
                    break
        else:
            pre_Yl = gpr_Xl.predict(np.array(xh).reshape(-1,1))
            pre_Yl = pre_Yl[0]
            delta.append(Yh[i] - pre_Yl)
            yl.append(pre_Yl)

    p = np.dot(np.array(Yl_Xh).reshape(-1, 1).T,
               (np.array(Yh).reshape(-1, 1) - np.array(delta).reshape(-1, 1))) / np.dot(
        np.array(Yl_Xh).reshape(-1, 1).T, np.array(Yl_Xh).reshape(-1, 1))

    # 对差值delta做gpr
    gpr_delta = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_delta.fit(np.array(Xh).reshape(-1, 1), np.array(delta).reshape(-1, 1))
    # 预测低保真度数据对应的delta
    delta_yl = gpr_delta.predict(np.array(Xl).reshape(-1, 1))

    # 修正低保真度数据的y
    corrected_yl = p * np.array(Yl) + np.array(delta_yl)

    # 得到新的模型, 修正过后的低保真度模型
    # 加性0.4 乘性0.3 混合0.5
    gpr_cc = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_cc.fit(np.array(Xl).reshape(-1, 1), np.array(corrected_yl).reshape(-1, 1))

    # ^yh(x) = p*yl(x) + delta(x)
    return gpr_cc

def HFDM(Xh, Yh, alpha):
    # 对高保真度数据做gpr
    gpr_Xh = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_Xh.fit(np.array(Xh).reshape(-1, 1), np.array(Yh).reshape(-1, 1))

    return gpr_Xh


alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for i in range(100):

    AC_best_alpha_in_MUL, MC_best_alpha_in_MUL, CC_best_alpha_in_MUL, HFDM_best_alpha_in_MUL = None, None, None, None
    AC_best_mse_in_MUL, MC_best_mse_in_MUL, CC_best_mse_in_MUL, HFDM_best_mse_in_MUL = float('inf'), float('inf'), float('inf'), float('inf')

    print("#################################### Group ", i + 1, " ####################################", "seed: ", seed)
    random.seed(seed)
    parameter = [random.randint(-5, 5) for i in range(10)]
    all_parameters.append(parameter)
    def y_true(x):
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = parameter
        return p1 * math.sin(p2 * x ** 2 + p3) * (p4 * x ** 2 + p5) + p6 * (p7 * x ** 3 + p8) + p9 * x + p10

    # 当前真值函数下测试集的y
    y_test = [y_true(e) for e in X_test]
    # 获取当前Yh和Yl
    # [random.uniform(0.9, 1.1)*y_true(e) for e in Xh]
    # [random.uniform(0.8, 1.3)*y_true(e) for e in Xl]
    # [random.uniform(0.9, 1.1)*y_true(e)+0.4*random.random()*random.choice([-1, 1]) for e in Xh]
    # [random.uniform(0.8, 1.3)*y_true(e)+0.8*random.random()*random.choice([-1, 1]) for e in Xl]
    Yh = [random.uniform(0.9, 1.1)*y_true(e) for e in Xh]
    Yl = [random.uniform(0.8, 1.3)*y_true(e) for e in Xl]

    # if seed == 724:
    #     with open('Yh_train_add.json', 'w') as f:
    #         json.dump(Yh, f)
    #     with open('Yl_train_add.json', 'w') as f:
    #         json.dump(Yl, f)

    # 将训练集划分为5份用作交叉验证
    kf = KFold(n_splits=5, random_state=654, shuffle=True)
    Xh_train_5fold = []
    Xh_val_5fold = []
    Xl_train_5fold = []
    Xl_val_5fold = []
    Yh_train_5fold = []
    Yh_val_5fold = []
    Yl_train_5fold = []
    Yl_val_5fold = []

    for train_index, val_index in kf.split(Xh):
        X_train, X_val = np.array(Xh)[train_index], np.array(Xh)[val_index]
        Y_train, Y_val = np.array(Yh)[train_index], np.array(Yh)[val_index]
        Xh_train_5fold.append(X_train)
        Xh_val_5fold.append(X_val)
        Yh_train_5fold.append(Y_train)
        Yh_val_5fold.append(Y_val)
    for train_index, val_index in kf.split(Xl):
        X_train, X_val = np.array(Xl)[train_index], np.array(Xl)[val_index]
        Y_train, Y_val = np.array(Yl)[train_index], np.array(Yl)[val_index]
        Xl_train_5fold.append(X_train)
        Xl_val_5fold.append(X_val)
        Yl_train_5fold.append(Y_train)
        Yl_val_5fold.append(Y_val)

    # 通过交叉验证寻找最优参数alpha
    for alpha in alpha_values:

        AC_5fold_in_MUL = []  # 5fold 中每次在加性噪声的数据集下得到的mse
        MC_5fold_in_MUL = []
        CC_5fold_in_MUL = []
        HFDM_5fold_in_MUL = []

        # 5折交叉验证
        for k in range(5):
            # 获取当前的训练集和验证集
            Xh_train = Xh_train_5fold[k]
            Xl_train = Xl_train_5fold[k]
            Xh_val = Xh_val_5fold[k]
            Xl_val = Xl_val_5fold[k]
            X_val = list(Xh_val) + list(Xl_val)

            # 获取加性噪声下的y
            Yh_train_add = Yh_train_5fold[k]
            Yl_train_add = Yl_train_5fold[k]
            Yh_val_add = Yh_val_5fold[k]
            Yl_val_add = Yl_val_5fold[k]
            Y_val_add = list(Yh_val_add) + list(Yl_val_add)

            # 使用各种修正yl后的gpr(除了HFDM)
            AC_in_add = AC(Xh_train, Xl_train, Yh_train_add, Yl_train_add, alpha)
            MC_in_add = MC(Xh_train, Xl_train, Yh_train_add, Yl_train_add, alpha)
            CC_in_add = CC(Xh_train, Xl_train, Yh_train_add, Yl_train_add, alpha)
            HFDM_in_add = HFDM(Xh_train, Yh_train_add, alpha)

            # 预测验证集上的结果
            AC_in_add_pred = AC_in_add.predict(np.array(X_val).reshape(-1, 1))
            MC_in_add_pred = MC_in_add.predict(np.array(X_val).reshape(-1, 1))
            CC_in_add_pred = CC_in_add.predict(np.array(X_val).reshape(-1, 1))
            HFDM_in_add_pred = HFDM_in_add.predict(np.array(X_val).reshape(-1, 1))

            # 预测验证集上的mse
            mse_AC_in_add = mean_squared_error(Y_val_add, AC_in_add_pred)
            AC_5fold_in_MUL.append(mse_AC_in_add)
            mse_MC_in_add = mean_squared_error(Y_val_add, MC_in_add_pred)
            MC_5fold_in_MUL.append(mse_MC_in_add)
            mse_CC_in_add = mean_squared_error(Y_val_add, CC_in_add_pred)
            CC_5fold_in_MUL.append(mse_CC_in_add)
            mse_HFDM_in_add = mean_squared_error(Y_val_add, HFDM_in_add_pred)
            HFDM_5fold_in_MUL.append(mse_HFDM_in_add)


        # 计算在当前参数alpha下的5折平均mse
        AC_val_avg_mse = np.mean(AC_5fold_in_MUL)
        MC_val_avg_mse = np.mean(MC_5fold_in_MUL)
        CC_val_avg_mse = np.mean(CC_5fold_in_MUL)
        HFDM_val_avg_mse = np.mean(HFDM_5fold_in_MUL)

        # print(f"now alpha:{alpha} \nnow AC avg mse:{AC_val_avg_mse}")
        # print(f"now MC avg mse:{MC_val_avg_mse}")
        # print(f"now CC avg mse:{CC_val_avg_mse}")
        # print(f"now HFDM avg mse:{HFDM_val_avg_mse}")

        # 记录最佳参数和mse
        if AC_val_avg_mse < AC_best_mse_in_MUL:
            AC_best_mse_in_MUL = AC_val_avg_mse
            AC_best_alpha_in_MUL = alpha
        if MC_val_avg_mse < MC_best_mse_in_MUL:
            MC_best_mse_in_MUL = MC_val_avg_mse
            MC_best_alpha_in_MUL = alpha
        if CC_val_avg_mse < CC_best_mse_in_MUL:
            CC_best_mse_in_MUL = CC_val_avg_mse
            CC_best_alpha_in_MUL = alpha
        if HFDM_val_avg_mse < HFDM_best_mse_in_MUL:
            HFDM_best_mse_in_MUL = HFDM_val_avg_mse
            HFDM_best_alpha_in_MUL = alpha

    print("AC MC CC HFDM 的最佳alpha超参数:", AC_best_alpha_in_MUL, MC_best_alpha_in_MUL, CC_best_alpha_in_MUL, HFDM_best_alpha_in_MUL)

    # 使用最佳参数alpha去训练模型
    best_AC = AC(Xh, Xl, Yh, Yl, AC_best_alpha_in_MUL)
    best_MC = MC(Xh, Xl, Yh, Yl, MC_best_alpha_in_MUL)
    best_CC = CC(Xh, Xl, Yh, Yl, CC_best_alpha_in_MUL)
    best_HFDM = HFDM(Xh, Yh, HFDM_best_alpha_in_MUL)

    # 使用测试集预测模型结果
    AC_pred = best_AC.predict(np.array(X_test).reshape(-1, 1))
    MC_pred = best_MC.predict(np.array(X_test).reshape(-1, 1))
    CC_pred = best_CC.predict(np.array(X_test).reshape(-1, 1))
    HFDM_pred = best_HFDM.predict(np.array(X_test).reshape(-1, 1))

    # 计算MSE
    mse_AC_in_add = mean_squared_error(y_test, AC_pred)
    mse_MC_in_add = mean_squared_error(y_test, MC_pred)
    mse_CC_in_add = mean_squared_error(y_test, CC_pred)
    mse_HFDM_in_add = mean_squared_error(y_test, HFDM_pred)

    # 计算均方根误差（RMSE）
    rmse_AC_in_add = mean_squared_error(y_test, AC_pred, squared=False)
    rmse_MC_in_add = mean_squared_error(y_test, MC_pred, squared=False)
    rmse_CC_in_add = mean_squared_error(y_test, CC_pred, squared=False)
    rmse_HFDM_in_add = mean_squared_error(y_test, HFDM_pred, squared=False)

    # 计算平均绝对误差（MAE）
    mae_AC_in_add = mean_absolute_error(y_test, AC_pred)
    mae_MC_in_add = mean_absolute_error(y_test, MC_pred)
    mae_CC_in_add = mean_absolute_error(y_test, CC_pred)
    mae_HFDM_in_add = mean_absolute_error(y_test, HFDM_pred)

    # 计算决定系数（R²）
    r_squared_AC_in_add = r2_score(y_test, AC_pred)
    r_squared_MC_in_add = r2_score(y_test, MC_pred)
    r_squared_CC_in_add = r2_score(y_test, CC_pred)
    r_squared_HFDM_in_add = r2_score(y_test, HFDM_pred)

    # 计算平均误差（ME）
    def mean_error(y_true, y_pred):
        return np.mean(y_true - y_pred)
    me_AC_in_add = mean_error(y_test, AC_pred)
    me_MC_in_add = mean_error(y_test, MC_pred)
    me_CC_in_add = mean_error(y_test, CC_pred)
    me_HFDM_in_add = mean_error(y_test, HFDM_pred)

    # 计算误差标准差（Error standard deviation, ESD）
    def error_standard_deviation(y_true, y_pred):
        errors = y_true - y_pred
        me = mean_error(y_true, y_pred)
        squared_errors = (errors - me) ** 2
        mse = np.mean(squared_errors)
        error_std = np.sqrt(mse)
        return error_std
    esd_AC_in_add = error_standard_deviation(y_test, AC_pred)
    esd_MC_in_add = error_standard_deviation(y_test, MC_pred)
    esd_CC_in_add = error_standard_deviation(y_test, CC_pred)
    esd_HFDM_in_add = error_standard_deviation(y_test, HFDM_pred)


    print("AC MC CC HFDM 's MSE:\n",mse_AC_in_add, mse_MC_in_add, mse_CC_in_add, mse_HFDM_in_add)
    print("AC MC CC HFDM 's RMSE:\n",rmse_AC_in_add, rmse_MC_in_add, rmse_CC_in_add, rmse_HFDM_in_add)
    print("AC MC CC HFDM 's MAE:\n",mae_AC_in_add, mae_MC_in_add, mae_CC_in_add, mae_HFDM_in_add)
    print("AC MC CC HFDM 's R²:\n",r_squared_AC_in_add, r_squared_MC_in_add, r_squared_CC_in_add, r_squared_HFDM_in_add)
    print("AC MC CC HFDM 's ME:\n",me_AC_in_add, me_MC_in_add, me_CC_in_add, me_HFDM_in_add)
    print("AC MC CC HFDM 's ESD:\n", esd_AC_in_add, esd_MC_in_add, esd_CC_in_add, esd_HFDM_in_add)


    # 加入列表中
    mse_AC_in_MUL_100.append(mse_AC_in_add)
    mse_MC_in_MUL_100.append(mse_MC_in_add)
    mse_CC_in_MUL_100.append(mse_CC_in_add)
    mse_HFDM_in_MUL_100.append(mse_HFDM_in_add)

    rmse_AC_in_MUL_100.append(rmse_AC_in_add)
    rmse_MC_in_MUL_100.append(rmse_MC_in_add)
    rmse_CC_in_MUL_100.append(rmse_CC_in_add)
    rmse_HFDM_in_MUL_100.append(rmse_HFDM_in_add)

    mae_AC_in_MUL_100.append(mae_AC_in_add)
    mae_MC_in_MUL_100.append(mae_MC_in_add)
    mae_CC_in_MUL_100.append(mae_CC_in_add)
    mae_HFDM_in_MUL_100.append(mae_HFDM_in_add)

    r_squared_AC_in_MUL_100.append(r_squared_AC_in_add)
    r_squared_MC_in_MUL_100.append(r_squared_MC_in_add)
    r_squared_CC_in_MUL_100.append(r_squared_CC_in_add)
    r_squared_HFDM_in_MUL_100.append(r_squared_HFDM_in_add)

    me_AC_in_MUL_100.append(me_AC_in_add)
    me_MC_in_MUL_100.append(me_MC_in_add)
    me_CC_in_MUL_100.append(me_CC_in_add)
    me_HFDM_in_MUL_100.append(me_HFDM_in_add)

    esd_AC_in_MUL_100.append(esd_AC_in_add)
    esd_MC_in_MUL_100.append(esd_MC_in_add)
    esd_CC_in_MUL_100.append(esd_CC_in_add)
    esd_HFDM_in_MUL_100.append(esd_HFDM_in_add)

    seed += 1



print("重复选取100次真值函数后每个方法的平均的mse为:")
print("AC",np.mean(mse_AC_in_MUL_100))
print("MC",np.mean(mse_MC_in_MUL_100))
print("CC",np.mean(mse_CC_in_MUL_100))
print("HFDM",np.mean(mse_HFDM_in_MUL_100))

print("重复选取100次真值函数后每个方法的平均的rmse为:")
print("AC",np.mean(rmse_AC_in_MUL_100))
print("MC",np.mean(rmse_MC_in_MUL_100))
print("CC",np.mean(rmse_CC_in_MUL_100))
print("HFDM",np.mean(rmse_HFDM_in_MUL_100))

print("重复选取100次真值函数后每个方法的平均的mae为:")
print("AC",np.mean(mae_AC_in_MUL_100))
print("MC",np.mean(mae_MC_in_MUL_100))
print("CC",np.mean(mae_CC_in_MUL_100))
print("HFDM",np.mean(mae_HFDM_in_MUL_100))

print("重复选取100次真值函数后每个方法的平均的R²为:")
print("AC",np.mean(r_squared_AC_in_MUL_100))
print("MC",np.mean(r_squared_MC_in_MUL_100))
print("CC",np.mean(r_squared_CC_in_MUL_100))
print("HFDM",np.mean(r_squared_HFDM_in_MUL_100))

print("重复选取100次真值函数后每个方法的平均的me为:")
print("AC",np.mean(me_AC_in_MUL_100))
print("MC",np.mean(me_MC_in_MUL_100))
print("CC",np.mean(me_CC_in_MUL_100))
print("HFDM",np.mean(me_HFDM_in_MUL_100))

print("重复选取100次真值函数后每个方法的平均的误差标准差（Error standard deviation, ESD）为:")
print("AC",np.mean(esd_AC_in_MUL_100))
print("MC",np.mean(esd_MC_in_MUL_100))
print("CC",np.mean(esd_CC_in_MUL_100))
print("HFDM",np.mean(esd_HFDM_in_MUL_100))




