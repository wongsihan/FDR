import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set()
#构建数据
# np.random.seed(0)
# x = np.random.randn(100)
y1 = [1,2,3,4,5,6,7]
x1 = [-4,-2,0,2,4,6,8]

# 使用pandas来设置x 轴标签 和y 轴标签
x = pd.Series(x1,name="x variable")
y = pd.Series(y1,name="y variable")
"""
案例2：绘制直方图和核函数密度估计图
"""
sns.distplot(x,y)
plt.show()
