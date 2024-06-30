# def colorbar(ax):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
    
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)    
    
#     return cax

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import numpy as np
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm
# from copy import copy
# fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'parojection': '3d'})
# norm = mcolors.Normalize(vmin=0, vmax=400)
# cmap = copy(cm.viridis)
# cax = colorbar(ax)

# im = cm.ScalarMappable(norm=norm, cmap=cmap)

# # im = ax.imshow(np.arange(400).reshape((20, 20)))

# fig.colorbar(im, cax=cax, ax=ax, shrink=0.8)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 创建 Figure 和 3D 轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 生成数据
x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)
c = x + y  # 用于颜色映射的数据

# 绘制3D散点图
norm = Normalize(vmin=min(c), vmax=max(c))
sc = ax.scatter(x, y, z, c=c, cmap='viridis', norm=norm)

# 获取主轴的位置
ax_pos = ax.get_position()

# 计算Colorbar的位置和尺寸
cb_width = 0.03  # Colorbar的宽度
cb_pad = 0.02    # Colorbar与主轴之间的间距
cb_ax_pos = [ax_pos.x1 + cb_pad, ax_pos.y0, cb_width, ax_pos.height]

# 创建Colorbar轴
cb_ax = fig.add_axes(cb_ax_pos)

# 创建 ScalarMappable

sm = ScalarMappable(norm=norm, cmap='viridis')
sm.set_array([])  # 设置空数组，只用于Colorbar的显示

# 显示 Colorbar
cbar = fig.colorbar(sm, cax=cb_ax)
cbar.set_label('Value')

# 显示图形
plt.show()