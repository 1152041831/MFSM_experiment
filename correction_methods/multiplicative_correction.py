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

Xh = np.arange(0,1,0.05) # 20个    train:16      test:4
Xl = np.arange(0,1,0.005) # 200个   train:196   test:4

random_seed = 654
np.random.seed(random_seed)
# 构造多保真度数据
Yh = [np.random.uniform(0.9, 1.1)*y_true(e) for e in Xh]
Yl = [np.random.uniform(0.8, 1.3)*y_true(e) for e in Xl]
# 将数据集X和响应变量Y进行随机分割
Xh_train, Xh_test, yh_train, yh_test = train_test_split(Xh, Yh, test_size=0.2, random_state=random_seed)
# y_train和y_test按照相同的索引切割
yh_train = yh_train[:len(Xh_train)]
yh_test = yh_test[:len(Xh_test)]
print("Xh_test:",len(Xh_test),Xh_test)
print("yh_test:",len(yh_test),yh_test)
# 找到Xl中对应Xh的测试集索引
index = [j for i in range(len(Xh_test)) for j in range(len(Xl)) if Xl[j] == Xh_test[i] ]
print(len(index),index)
# 低保真度训练集 & 测试集
Xl_train = [Xl[i] for i in range(len(Xl)) if i not in index]
Xl_test = Xh_test.copy()
yl_train = [Yl[i] for i in range(len(Yl)) if i not in index]

print("Xl_test:",Xl_test)
# 防止精度误差
Xl_train = [round(num, 3) for num in Xl_train]
Xh_train = [round(num, 3) for num in Xh_train]
Xl_test = [round(num, 3) for num in Xl_test]
Xh_test= [round(num, 3) for num in Xh_test]
y_t = [y_true(e) for e in Xh_test]
print("y_t :",len(y_t),y_t)
print("Xl_test length:",len(Xl_test),Xl_test)

print("Xh_test length:",len(Xh_test),Xh_test)
print("yh_test length:",len(yh_test),yh_test)

delta = []
for i in range(len(Xh_train)):
    for j in range(len(Xl_train)):
        if Xh_train[i] == Xl_train[j]:
            delta.append(yh_train[i] / yl_train[j])
            break
print("△ = yh_train / yl_train:",len(delta))
print(delta)


# 拟合比值
print("GPR求比值函数")
gpr1 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
gpr1.fit(np.array(Xh_train).reshape(-1,1), np.array(delta).reshape(-1,1))
pre1, cov1 = gpr1.predict(np.array(Xl_train).reshape(-1,1),return_std=True)
pre_delta = pre1.ravel()
print(f"pre_delta len:{len(pre_delta)}")


# 高保真度拟合
print("HFM")
gpr2 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
gpr2.fit(np.array(Xh_train).reshape(-1, 1), np.array(yh_train).reshape(-1, 1))
mu2, cov2= gpr2.predict(np.array(Xl_train).reshape(-1,1),return_std=True)
pre_HFM = mu2.ravel()
print(f"pre_HFM len:{len(pre_HFM)}")

# 低保真度拟合
print("LFM")
gpr3 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
gpr3.fit(np.array(Xl_train).reshape(-1, 1), np.array(yl_train).reshape(-1, 1))
mu3, cov3 = gpr3.predict(np.array(Xl_train).reshape(-1,1),return_std=True)
pre_LFM = mu3.ravel()
print(f"pre_LFM len:{len(pre_LFM)}")


# 拟合所有的高保真度数据
print("MFSM+")
# 在Xl_train中排除Xh_train中的数值
index_Xl_without_Xh = [i for i in range(len(Xl_train)) if Xl_train[i] not in Xh_train]
print("index_Xl_without_Xh:",len(index_Xl_without_Xh))
new_Xl_train = [Xl_train[i] for i in index_Xl_without_Xh] #180
new_yl_train = [yl_train[i] for i in index_Xl_without_Xh] #180
print("new_Xl_train:",len(new_Xl_train))
new_Xh = Xh_train.copy()
new_yh = yh_train.copy()
new_Xh.extend(new_Xl_train)
# 拟合180个低保真度数据集与高保真度度数据之间的差值
mu11, cov11 = gpr1.predict(np.array(new_Xl_train).reshape(-1, 1),return_std=True)
delta_180 = mu11.ravel()
new_add_yh = [delta_180[i]*new_yl_train[i] for i in range(len(new_yl_train))]
new_yh.extend(new_add_yh)


# 用原本的高保真度数据和新生成的高保真度数据，拟合
print("MFSM+ fuction 2")
gpr4 = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
gpr4.fit(np.array(new_Xh).reshape(-1, 1), np.array(new_yh).reshape(-1, 1))
mu4, cov4 = gpr4.predict(np.array(Xl_train).reshape(-1,1),return_std=True)
pre_MFSM = mu4.ravel()
print(f"pre_MFSM len:{len(pre_MFSM)}")
# test
mu41, cov41 = gpr4.predict(np.array(Xl_test).reshape(-1,1),return_std=True)
test_MC = mu41.ravel()


#MSE
hfm_with_truth = np.mean((np.array(y_t) - np.array(yh_test))**2)
print("hfm_with_truth MSE:",hfm_with_truth)
MC_with_truth = np.mean((np.array(y_t) - np.array(test_MC))**2)
print("MC_with_truth MSE:",MC_with_truth)


fig, axs = plt.subplots(1, 2, sharey=True,gridspec_kw={'hspace': 0, 'wspace': 0},figsize=(10, 5))
fig.subplots_adjust(left=0.1, right=0.9, top=0.88,bottom=0.1)
plt.subplots_adjust(hspace=0.4, bottom=0.2)

axs[0].set_ylabel('Y (A.U.)', fontsize=18)
axs[0].set_yticks([0.5,1.0,1.5],['0.5','1.0','1.5'])
axs[0].set_xlabel('X (A.U.)\n(a)', fontsize=18)
axs[0].tick_params(labelsize=18)
axs[0].set_xlim(0, 1.0)
axs[0].set_xticks([0,0.2,0.4,0.6,0.8],['0.0', '0.2','0.4', '0.6', '0.8'])
axs[0].scatter(np.array(Xl_train), np.array(yl_train),label="Xl",s=40,alpha=0.5)
axs[0].scatter(np.array(Xh_train), np.array(yh_train),label="Xh",c="red",s=40,marker="^",alpha=0.8)

axs[1].tick_params(labelsize=18)
axs[1].set_xlabel('X (A.U.)\n(b)', fontsize=18)
axs[1].set_xlim(0, 1.0)
axs[1].set_xticks([0,0.2,0.4,0.6,0.8,1.0],['0.0', '0.2','0.4', '0.6', '0.8', '1.0'])
axs[1].scatter(np.array(Xl_train), np.array(yl_train),s=40,alpha=0.1)
axs[1].scatter(np.array(Xh_train), np.array(yh_train),c="red",s=40,marker="^",alpha=0.1)
axs[1].plot(Xl_train,[0.7 * math.sin(4 * e ** 2 + 3) * (3 * e ** 2 - 2) + 0.6 for e in Xl_train],label="True value",linestyle="--",linewidth =2.0)
axs[1].plot(Xl_train,pre_HFM,label="hfm",linestyle="-.",linewidth =2.0,c="red")
axs[1].plot(Xl_train,pre_MFSM,label="MC",linewidth =2.0,c="orange")


handles0, labels0 = axs[0].get_legend_handles_labels()
handles1, labels1 = axs[1].get_legend_handles_labels()
axs[0].legend(handles0, labels0, loc='upper center', ncol=len(labels0), bbox_to_anchor=(0.5, 1.15), fontsize=14, facecolor='none', edgecolor='black')
axs[1].legend(handles1, labels1, loc='upper center', ncol=len(labels1), bbox_to_anchor=(0.5, 1.15), fontsize=14, facecolor='none', edgecolor='black')

plt.show()