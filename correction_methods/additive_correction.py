import math
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# 创建高斯过程回归模型对象
kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e1))



def y_true(x):
    return 0.7 * math.sin(4 * x ** 2 + 3) * (3 * x ** 2 - 2) + 0.6

X_test = np.arange(0.0498,1,0.0498) # 20
y_test_true = [y_true(e) for e in X_test]

Xh1 = np.arange(0,1,0.05) # 20
Xl1 = np.arange(0,1,0.005) # 200

random_seed = 2023
np.random.seed(random_seed)
# 构造多保真度数据
Yh1 = [y_true(e)+0.03*np.random.random()*np.random.choice([-1, 1]) for e in Xh1]
Yl1 = [y_true(e)+0.1*np.random.random()*np.random.choice([-1, 1]) for e in Xl1]
random.seed(random_seed)
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
print("Xh_train1:",len(Xh_train1),Xh_train1)
print("yh_train1:",len(yh_train1),yh_train1)
print("Xl_train1:",len(Xl_train1),Xl_train1)
print("yl_train1:",len(yl_train1),yl_train1)

delta1 = []
for i in range(len(Xh_train1)):
    for j in range(len(Xl_train1)):
        if Xh_train1[i] == Xl_train1[j]:
            delta1.append(yh_train1[i] - yl_train1[j])
            break
print("△1 = yh_train1 - yl_train1:",len(delta1),delta1)


# 拟合差值
print("GPR求差值函数")
gpr1_1 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
gpr1_1.fit(np.array(Xh_train1).reshape(-1,1), np.array(delta1).reshape(-1,1))
pre1_1, cov1_1 = gpr1_1.predict(np.array(Xl_train1).reshape(-1,1),return_std=True)
pre_delta_1 = pre1_1.ravel()
print(f"pre_delta len:{len(pre_delta_1)}")


# 高保真度拟合模型
print("HFM")
gpr2_1 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
gpr2_1.fit(np.array(Xh_train1).reshape(-1, 1), np.array(yh_train1).reshape(-1, 1))
mu2_1, cov2_1= gpr2_1.predict(np.array(Xl1).reshape(-1,1),return_std=True)
pre_HFM = mu2_1.ravel()
print(f"pre_HFM len:{len(pre_HFM)}")
# test
mu21_1, cov21_1= gpr2_1.predict(np.array(X_test).reshape(-1,1),return_std=True)
hfm_test_1 = mu21_1.ravel()
print(f"hfm_test_1 len:{len(hfm_test_1)}")

# 拟合所有的高保真度数据
print("MFSM+")
# 在Xl_train中排除Xh_train中的数值
index_Xl_without_Xh_1 = [i for i in range(len(Xl_train1)) if Xl_train1[i] not in Xh_train1]
print("index_Xl_without_Xh:",len(index_Xl_without_Xh_1))
new_Xl_train_1 = [Xl_train1[i] for i in index_Xl_without_Xh_1] #180
new_yl_train_1 = [yl_train1[i] for i in index_Xl_without_Xh_1] #180
print("new_Xl_train_1:",len(new_Xl_train_1))
new_Xh_1 = Xh_train1.copy()
new_yh_1 = yh_train1.copy()
new_Xh_1.extend(new_Xl_train_1)
# 拟合180个低保真度数据集与高保真度度数据之间的差值
mu11_1, cov11_1 = gpr1_1.predict(np.array(new_Xl_train_1).reshape(-1, 1),return_std=True)
delta_180_1 = mu11_1.ravel()
new_add_yh_1 = [delta_180_1[i]+new_yl_train_1[i] for i in range(len(new_yl_train_1))]
new_yh_1.extend(new_add_yh_1)


# 用原本的高保真度数据和新生成的高保真度数据，拟合
print("MFSM+ fuction 2")
gpr4_1 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
gpr4_1.fit(np.array(new_Xh_1).reshape(-1, 1), np.array(new_yh_1).reshape(-1, 1))
mu4_1, cov4 = gpr4_1.predict(np.array(Xl1).reshape(-1,1),return_std=True)
pre_MFSM_1 = mu4_1.ravel()
print(f"pre_MFSM len:{len(pre_MFSM_1)}")
# test
mu41_1, cov41_1 = gpr4_1.predict(np.array(X_test).reshape(-1,1),return_std=True)
test_AC = mu41_1.ravel()


# MSE :
hfm_with_truth = np.mean((np.array(y_test_true) - np.array(hfm_test_1))**2)
print("hfm_with_truth MSE:",hfm_with_truth)
AC_with_truth = np.mean((np.array(y_test_true) - np.array(test_AC))**2)
print("AC_with_truth MSE:",AC_with_truth)


fig, axs = plt.subplots(1, 2, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0},figsize=(10, 5))
fig.subplots_adjust(left=0.1, right=0.9, top=0.88,bottom=0.1)
plt.subplots_adjust(hspace=0.4, bottom=0.2)

axs[0].set_ylabel('Y (A.U.)', fontsize=18)
axs[0].set_xlabel('X (A.U.)\n(a)', fontsize=18)
axs[0].tick_params(labelsize=18)
axs[0].set_xlim(0, 1.0)
axs[0].set_yticks([0.4,0.8,1.2],['0.4','0.8','1.2'])
axs[0].set_xticks([0,0.2,0.4,0.6,0.8],['0.0', '0.2','0.4', '0.6', '0.8'])
axs[0].scatter(np.array(Xl_train1), np.array(yl_train1),label="Xl",s=40,alpha=0.5)
axs[0].scatter(np.array(Xh_train1), np.array(yh_train1),label="Xh",c="red",s=40,marker="^",alpha=0.8)

axs[1].tick_params(labelsize=18)
axs[1].set_xlabel('X (A.U.)\n(b)', fontsize=18)
axs[1].set_xlim(0, 1.0)
axs[1].set_xticks([0,0.2,0.4,0.6,0.8,1.0],['0.0', '0.2','0.4', '0.6', '0.8', '1.0'])
axs[1].scatter(np.array(Xl_train1), np.array(yl_train1),s=40,alpha=0.1)
axs[1].scatter(np.array(Xh_train1), np.array(yh_train1),c="red",s=40,marker="^",alpha=0.1)
axs[1].plot(Xl1,[0.7 * math.sin(4 * e ** 2 + 3) * (3 * e ** 2 - 2) + 0.6 for e in Xl1],label="True value",linestyle="--",linewidth =2.0)
axs[1].plot(Xl1,pre_HFM,label="hfm",linestyle="-.",linewidth =2.0,c="red")
axs[1].plot(Xl1,pre_MFSM_1,label="AC",linewidth =2.0,c="orange")


handles0, labels0 = axs[0].get_legend_handles_labels()
handles1, labels1 = axs[1].get_legend_handles_labels()
axs[0].legend(handles0, labels0, loc='upper center', ncol=len(labels0), bbox_to_anchor=(0.5, 1.15), fontsize=14, facecolor='none', edgecolor='black')
axs[1].legend(handles1, labels1, loc='upper center', ncol=len(labels1), bbox_to_anchor=(0.5, 1.15), fontsize=14, facecolor='none', edgecolor='black')

plt.show()
