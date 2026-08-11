"""
YOLOv8 插件包

子包：
- attention: 注意力机制
- conv: 改进卷积模块
- transformer: Transformer 模块
- neck: 改进特征融合（Neck）
- spp: 改进空间金字塔池化
- head: 改进检测头
"""

# 导入所有插件以触发注册
from . import attention
from . import conv
from . import transformer
from . import neck
from . import spp
from . import head
