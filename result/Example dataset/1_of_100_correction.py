import json
import math
import random
import numpy as np
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, curve_fit, minimize
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 创建高斯过程回归模型对象
# length_scale=0.5, length_scale_bounds=(1e-2, 1)
kernel = RBF()

parameter = []
all_parameters = []
MSE = []
result = []


# 修正方法
# 输入高、低保真度数据
# 返回gpr模型
def AC(Xh, Xl, Yh, Yl, alpha):
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
                    delta.append(Yh[i] - Yl[j])
                    break
        else:
            pre_Yl = gpr_Xl.predict(np.array(xh).reshape(-1, 1))
            pre_Yl = pre_Yl[0]
            delta.append(Yh[i] - pre_Yl)

    # 对差值delta做gpr
    gpr_delta = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_delta.fit(np.array(Xh).reshape(-1, 1), np.array(delta).reshape(-1, 1))

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
    corrected_yl = Yl * delta_yl

    # 得到新的模型, 修正过后的低保真度模型
    # 加性0.4 乘性0.4 混合0.4
    gpr_mc = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_mc.fit(np.array(Xl).reshape(-1, 1), np.array(corrected_yl).reshape(-1, 1))

    return gpr_mc


def CC(Xh, Xl, Yh, Yl, alpha):
    # 对低保真度数据做gpr
    gpr_Xl = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=alpha)
    gpr_Xl.fit(np.array(Xl).reshape(-1, 1), np.array(Yl).reshape(-1, 1))
    Yl_Xh = gpr_Xl.predict(np.array(Xh).reshape(-1, 1))
    yl = []  # yl(Xh)
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
            pre_Yl = gpr_Xl.predict(np.array(xh).reshape(-1, 1))
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


# 重复运行100次生成随机数组的操作
seed = 724  # 设置种子为654
random.seed(seed)

# [-1, -5, -4, 5, -1, -2, 1, 0, -4, -2]
parameter = [random.randint(-5, 5) for i in range(10)]
print(parameter)
all_parameters.append(parameter)


def y_true(x):
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = parameter
    return p1 * math.sin(p2 * x ** 2 + p3) * (p4 * x ** 2 + p5) + p6 * (p7 * x ** 3 + p8) + p9 * x + p10


# 设置数据集
# 训练集
Xh = np.array([round(e, 3) for e in np.arange(0, 1, 0.05)])  # 20
Xl = np.array([round(e, 3) for e in np.arange(0, 1, 0.005)])  # 200
# 测试集
X_test = np.arange(0.0498, 1, 0.0498)  # 20
y_test = [y_true(e) for e in X_test]

with open('Yh_train_add.json', 'r') as f:
    Yh_train_add = json.load(f)
with open('Yl_train_add.json', 'r') as f:
    Yl_train_add = json.load(f)

with open('Yh_train_mul.json', 'r') as f:
    Yh_train_mul = json.load(f)
with open('Yl_train_mul.json', 'r') as f:
    Yl_train_mul = json.load(f)

with open('Yh_train_mix.json', 'r') as f:
    Yh_train_mix = json.load(f)
with open('Yl_train_mix.json', 'r') as f:
    Yl_train_mix = json.load(f)


# 这里面所用的参数都是从交叉验证过程中得到的最佳参数
# 仅用高保真度数据做gpr
HFDM_in_add = HFDM(Xh, Yh_train_add, 0.35)
HFDM_in_mul = HFDM(Xh, Yh_train_mul, 0.35)
HFDM_in_mix = HFDM(Xh, Yh_train_mix, 0.4)
# 使用加性修正yl后的gpr
AC_in_add = AC(Xh, Xl, Yh_train_add, Yl_train_add, 0.2)
AC_in_mul = AC(Xh, Xl, Yh_train_mul, Yl_train_mul, 0.2)
AC_in_mix = AC(Xh, Xl, Yh_train_mix, Yl_train_mix, 0.2)
# 使用乘性修正yl后的gpr
MC_in_add = MC(Xh, Xl, Yh_train_add, Yl_train_add, 0.1)
MC_in_mul = MC(Xh, Xl, Yh_train_mul, Yl_train_mul, 0.2)
MC_in_mix = MC(Xh, Xl, Yh_train_mix, Yl_train_mix, 0.15)
# 使用综合修正yl后的gpr
CC_in_add = CC(Xh, Xl, Yh_train_add, Yl_train_add, 0.2)
CC_in_mul = CC(Xh, Xl, Yh_train_mul, Yl_train_mul, 0.2)
CC_in_mix = CC(Xh, Xl, Yh_train_mix, Yl_train_mix, 0.2)

# 预测测试集上的结果
HFDM_in_add_pred = HFDM_in_add.predict(np.array(X_test).reshape(-1, 1))
HFDM_in_mul_pred = HFDM_in_mul.predict(np.array(X_test).reshape(-1, 1))
HFDM_in_mix_pred = HFDM_in_mix.predict(np.array(X_test).reshape(-1, 1))

AC_in_add_pred = AC_in_add.predict(np.array(X_test).reshape(-1, 1))
AC_in_mul_pred = AC_in_mul.predict(np.array(X_test).reshape(-1, 1))
AC_in_mix_pred = AC_in_mix.predict(np.array(X_test).reshape(-1, 1))

MC_in_add_pred = MC_in_add.predict(np.array(X_test).reshape(-1, 1))
MC_in_mul_pred = MC_in_mul.predict(np.array(X_test).reshape(-1, 1))
MC_in_mix_pred = MC_in_mix.predict(np.array(X_test).reshape(-1, 1))

CC_in_add_pred = CC_in_add.predict(np.array(X_test).reshape(-1, 1))
CC_in_mul_pred = CC_in_mul.predict(np.array(X_test).reshape(-1, 1))
CC_in_mix_pred = CC_in_mix.predict(np.array(X_test).reshape(-1, 1))

# 预测Xl上的结果
pre_HFDM_1 = HFDM_in_add.predict(np.array(Xl).reshape(-1, 1))
pre_HFDM_2 = HFDM_in_mul.predict(np.array(Xl).reshape(-1, 1))
pre_HFDM_3 = HFDM_in_mix.predict(np.array(Xl).reshape(-1, 1))

AC_1 = AC_in_add.predict(np.array(Xl).reshape(-1, 1))
AC_2 = AC_in_mul.predict(np.array(Xl).reshape(-1, 1))
AC_3 = AC_in_mix.predict(np.array(Xl).reshape(-1, 1))

MC_1 = MC_in_add.predict(np.array(Xl).reshape(-1, 1))
MC_2 = MC_in_mul.predict(np.array(Xl).reshape(-1, 1))
MC_3 = MC_in_mix.predict(np.array(Xl).reshape(-1, 1))

CC_1 = CC_in_add.predict(np.array(Xl).reshape(-1, 1))
CC_2 = CC_in_mul.predict(np.array(Xl).reshape(-1, 1))
CC_3 = CC_in_mix.predict(np.array(Xl).reshape(-1, 1))

#################################################### 画图 ####################################################
fig, axs = plt.subplots(3, 2, sharey="row", sharex="col", gridspec_kw={'hspace': 0.0, 'wspace': 0}, figsize=(10, 9))
fig.subplots_adjust(left=0.11, right=0.77, top=0.9, bottom=0.1)
prop = fm.FontProperties(size=15, family='Times New Roman')
font_path = fm.findfont(fm.FontProperties(family='Times New Roman'))
####################################  Additive noise  #################################################
axs[0, 0].set_xlim(0, 1.0)
axs[0, 1].set_xlim(0, 1.0)
# axs[0,0].set_ylim(0.45, 2.5)
# axs[0,1].set_ylim(0.45, 2.5)
axs[0, 0].set_ylabel('Y (A.U.)', fontsize=18, fontname='Times New Roman')
axs[0, 0].tick_params(labelsize=18)
axs[0, 0].set_yticks([0, -2, -4, -6], ['0.0', '-2.0', '-4.0', '-6.0'], fontname='Times New Roman')
axs[0, 0].scatter(np.array(Xl), np.array(Yl_train_add), label="X$_l$", s=40, alpha=0.5)
axs[0, 0].scatter(np.array(Xh), np.array(Yh_train_add), label="X$_h$", c="red", s=40, marker="^", alpha=0.8)

axs[0, 1].tick_params(labelsize=18)
axs[0, 1].scatter(np.array(Xl), np.array(Yl_train_add), s=40, alpha=0.1)
axs[0, 1].scatter(np.array(Xh), np.array(Yh_train_add), c="red", s=40, marker="^", alpha=0.1)
axs[0, 1].plot(Xl, [y_true(e) for e in Xl], label="True value", linestyle="--", linewidth=2.0)
axs[0, 1].plot(Xl, pre_HFDM_1, label="HFDM", linestyle="-.", linewidth=2.0, c="red")
axs[0, 1].plot(Xl, AC_1, label="AC", linewidth=2.0, c="orange")
axs[0, 1].plot(Xl, MC_1, label="MC", linewidth=2.0, c="lime")
axs[0, 1].plot(Xl, CC_1, label="CC", linewidth=2.0, c="yellow")

# 放大图
axins1 = axs[0, 1].inset_axes([1.1, 0.2, 0.45, 0.45])
axins1.plot(Xl, [y_true(e) for e in Xl], label="True value", linestyle="--", linewidth=2.0)
axins1.plot(Xl, pre_HFDM_1, label="HFDM", linestyle="-.", linewidth=2.0, c="red")
axins1.plot(Xl, AC_1, label='AC', linewidth=2.0, c="orange")
axins1.plot(Xl, MC_1, label='MC', linewidth=2.0, c="lime")
axins1.plot(Xl, CC_1, label='CC', linewidth=2.0, c="yellow")
axins1.set_xlim(0.55, 0.75)
axins1.set_ylim(-5.51, -4.61)
axins1.set_xticks([0.6, 0.7], ['0.6', '0.7'], fontname='Times New Roman')
axins1.set_yticks([-5], ['-5.0'], fontname='Times New Roman')
# axins.spines['right'].set_visible(False)
# axins.spines['top'].set_visible(False)
axins1.tick_params(axis='both', which='both', direction='in', width=1, labelsize=18)
####################################  Multiplicative noise  #################################################
axs[1, 0].set_ylabel('Y (A.U.)', fontsize=18, fontname='Times New Roman')
axs[1, 0].set_yticks([0, -2, -4, -6], ['0.0', '-2.0', '-4.0', '-6.0'], fontname='Times New Roman')
axs[1, 0].tick_params(labelsize=18)
axs[1, 0].set_xlim(0, 1.0)
axs[1, 1].set_xlim(0, 1.0)
# axs[1,0].set_ylim(0.45, 2.5)
# axs[1,1].set_ylim(0.45, 2.5)

axs[1, 0].scatter(np.array(Xl), np.array(Yl_train_mul), s=40, alpha=0.5)
axs[1, 0].scatter(np.array(Xh), np.array(Yh_train_mul), c="red", s=40, marker="^", alpha=0.8)

axs[1, 1].tick_params(labelsize=18)
axs[1, 1].scatter(np.array(Xl), np.array(Yl_train_mul), s=40, alpha=0.1)
axs[1, 1].scatter(np.array(Xh), np.array(Yh_train_mul), c="red", s=40, marker="^", alpha=0.1)
axs[1, 1].plot(Xl, [y_true(e) for e in Xl], linestyle="--", linewidth=2.0)
axs[1, 1].plot(Xl, pre_HFDM_2, linestyle="-.", linewidth=2.0, c="red")
axs[1, 1].plot(Xl, AC_2, label="AC", linewidth=2.0, c="orange")
axs[1, 1].plot(Xl, MC_2, label="MC", linewidth=2.0, c="lime")
axs[1, 1].plot(Xl, CC_2, label="CC", linewidth=2.0, c="yellow")

# 放大图
axins2 = axs[1, 1].inset_axes([1.1, 0.2, 0.45, 0.45])
axins2.plot(Xl, [y_true(e) for e in Xl], label="True value", linestyle="--", linewidth=2.0)
axins2.plot(Xl, pre_HFDM_2, label="HFDM", linestyle="-.", linewidth=2.0, c="red")
axins2.plot(Xl, AC_2, label='AC', linewidth=2.0, c="orange")
axins2.plot(Xl, MC_2, label='MC', linewidth=2.0, c="lime")
axins2.plot(Xl, CC_2, label='CC', linewidth=2.0, c="yellow")
axins2.set_xlim(0.55, 0.75)
axins2.set_ylim(-5.61, -4.41)
# axins2.set_xticks([0.8,0.9],['0.8','0.9'])
axins2.set_xticks([0.6, 0.7], ['0.6', '0.7'], fontname='Times New Roman')
axins2.set_yticks([-5], ['-5.0'], fontname='Times New Roman')

axins2.tick_params(axis='both', which='both', direction='in', width=1, labelsize=18)
####################################  Mixed noise  #################################################
axs[2, 0].set_ylabel('Y (A.U.)', fontsize=18, fontname='Times New Roman')
axs[2, 0].set_xlabel('X (A.U.)', fontsize=18, fontname='Times New Roman')
axs[2, 0].set_yticks([0, -2, -4, -6, -8], ['0.0', '-2.0', '-4.0', '-6.0', '-8.0'], fontname='Times New Roman')

axs[2, 0].tick_params(labelsize=18)

axs[2, 0].scatter(np.array(Xl), np.array(Yl_train_mix), s=40, alpha=0.5)
axs[2, 0].scatter(np.array(Xh), np.array(Yh_train_mix), c="red", s=40, marker="^", alpha=0.8)

axs[2, 1].tick_params(labelsize=18)
axs[2, 1].set_xlabel('X (A.U.)', fontsize=18, fontname='Times New Roman')

axs[2, 1].scatter(np.array(Xl), np.array(Yl_train_mix), s=40, alpha=0.1)
axs[2, 1].scatter(np.array(Xh), np.array(Yh_train_mix), c="red", s=40, marker="^", alpha=0.1)
axs[2, 1].plot(Xl, [y_true(e) for e in Xl], linestyle="--", linewidth=2.0)
axs[2, 1].plot(Xl, pre_HFDM_3, linestyle="-.", linewidth=2.0, c="red")
axs[2, 1].plot(Xl, AC_3, label="AC", linewidth=2.0, c="orange")
axs[2, 1].plot(Xl, MC_3, label="MC", linewidth=2.0, c="lime")
axs[2, 1].plot(Xl, CC_3, label="CC", linewidth=2.0, c="yellow")

axs[2, 0].set_xlim(0, 1.0)
axs[2, 0].set_xticks([0, 0.2, 0.4, 0.6, 0.8], ['0.0', '0.2', '0.4', '0.6', '0.8'], fontname='Times New Roman')
axs[2, 1].set_xlim(0, 1.0)
axs[2, 1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                     fontname='Times New Roman')

# 放大图
axins3 = axs[2, 1].inset_axes([1.1, 0.2, 0.45, 0.45])  # 0.62 0.085
axins3.plot(Xl, [y_true(e) for e in Xl], label="True value", linestyle="--", linewidth=2.0)
axins3.plot(Xl, pre_HFDM_2, label="HFDM", linestyle="-.", linewidth=2.0, c="red")
axins3.plot(Xl, AC_3, label='AC', linewidth=2.0, c="orange")
axins3.plot(Xl, MC_3, label='MC', linewidth=2.0, c="lime")
axins3.plot(Xl, CC_3, label='CC', linewidth=2.0, c="yellow")
axins3.set_xlim(0.55, 0.75)
axins3.set_ylim(-5.62, -4.42)
axins3.set_xticks([0.6, 0.7], ['0.6', '0.7'], fontname='Times New Roman')
axins3.set_yticks([-5], ['-5.0'], fontname='Times New Roman')
axins3.tick_params(axis='both', which='both', direction='in', width=1, labelsize=18)

handles, labels = axs[0, 0].get_legend_handles_labels()
handles0, labels0 = axs[0, 1].get_legend_handles_labels()

axs[0, 1].legend(handles + handles0, labels + labels0, bbox_to_anchor=(1.65, 1.25), fontsize=14, facecolor='none',
                 edgecolor='black', ncol=7, prop=prop)

axs[0, 0].text(0.85, 0.95, '(a)', transform=axs[0, 0].transAxes, fontsize=18, va='top', fontname='Times New Roman')
axs[0, 1].text(0.85, 0.95, '(b)', transform=axs[0, 1].transAxes, fontsize=18, va='top', fontname='Times New Roman')
axs[1, 0].text(0.85, 0.95, '(c)', transform=axs[1, 0].transAxes, fontsize=18, va='top', fontname='Times New Roman')
axs[1, 1].text(0.85, 0.95, '(d)', transform=axs[1, 1].transAxes, fontsize=18, va='top', fontname='Times New Roman')
axs[2, 0].text(0.85, 0.95, '(e)', transform=axs[2, 0].transAxes, fontsize=18, va='top', fontname='Times New Roman')
axs[2, 1].text(0.85, 0.95, '(f)', transform=axs[2, 1].transAxes, fontsize=18, va='top', fontname='Times New Roman')

# 放大区域在原图中框出来
rect1 = plt.Rectangle((0.55, -5.51), 0.2, 0.9, linewidth=1, linestyle='--', edgecolor='black', facecolor='none',
                      zorder=10)
axs[0, 1].add_patch(rect1)
rect2 = plt.Rectangle((0.55, -5.61), 0.2, 1.2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none',
                      zorder=10)
axs[1, 1].add_patch(rect2)
rect3 = plt.Rectangle((0.55, -5.62), 0.2, 1.2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none',
                      zorder=10)
axs[2, 1].add_patch(rect3)

# 创建一个新的y轴在右侧显示
axins1.yaxis.tick_right()
axins1.yaxis.set_label_position("right")
axins2.yaxis.tick_right()
axins2.yaxis.set_label_position("right")
axins3.yaxis.tick_right()
axins3.yaxis.set_label_position("right")

axs[0, 0].set_ylim(-8, 0)
axs[1, 0].set_ylim(-8, 0)
axs[2, 0].set_ylim(-8, 0)

plt.savefig('output1.pdf', format='pdf')
plt.show()
