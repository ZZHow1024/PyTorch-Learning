# PyTorch深度学习入门笔记P1-6

@ZZHow(ZZHow1024)

参考课程：

【**PyTorch深度学习快速入门教程【小土堆】**】

[https://www.bilibili.com/video/BV1hE411t7RN]

# P1. PyTorch环境的配置及安装（Configuration and Installation of PyTorch）

- Anaconda
    
    [Advance AI with Open Source | Anaconda](https://www.anaconda.com/)
    
- NVIDIA 官方驱动
    
    [下载最新版官方 GeForce 驱动程序](https://www.nvidia.cn/geforce/drivers/)
    
- PyTorch
    
    [PyTorch Foundation](https://pytorch.org/)
    

# P2. Python编辑器的选择、安装及配置（PyCharm、Jupyter安装）

- PyCharm
    
    [PyCharm: The only Python IDE you need](https://www.jetbrains.com/pycharm/)
    
- Jupyter
    
    [Project Jupyter](https://jupyter.org/)
    

# P3. Python学习中的两大法宝函数（当然也可以用在PyTorch）

- **`dir()` 函数：列出对象的属性和方法**
    - **功能**：用于返回指定对象的所有**属性和方法列表**（包括内置的属性和方法）。如果不传入参数，则返回当前作用域中的所有变量、模块、函数等名称。
    - **语法**
        
        ```python
        dir([object])
        ```
        
        - **参数**：`object` 是可选的，可以是任何Python对象（如模块、类、实例、函数等）。如果省略参数，则返回当前作用域的名称列表。
    - **返回值**：返回一个**按字母顺序排序的字符串列表**，包含对象的所有属性和方法名。
- **`help()` 函数：获取对象的详细文档**
    - **功能**：`help()` 用于返回指定对象的**详细文档字符串（docstring）**，包括函数/类的功能描述、参数说明、返回值、示例等。它是快速查阅Python内置功能或第三方库文档的首选工具。
    - **语法**
        
        ```python
        help([object])
        ```
        
        - **参数**：`object` 是可选的，可以是模块、类、函数、方法等。如果省略参数，会进入交互式帮助模式（输入对象名后回车查看文档）。
    - **返回值**：直接在控制台打印对象的**文档字符串**（如果没有文档字符串，则可能显示简略信息）。

# P4. PyCharm及Jupyter使用及对比

| **特性** | **Python文件（.py）** | **Python控制台** | **Jupyter Notebook** |
| --- | --- | --- | --- |
| **代码组织形式** | 文件形式，支持多模块 | 交互式逐行输入 | 单元格分块，混合代码与文本 |
| **交互性** | 低（需手动运行） | 高（实时反馈） | 高（分块执行，变量保留） |
| **适用场景** | 完整项目开发 | 快速测试小段代码 | 数据分析、教学文档 |
| **调试支持** | 强（断点、变量监视） | 弱（简单断点） | 中等（依赖单元格执行） |
| **结果可视化** | 需手动输出或保存图表 | 直接打印结果 | 内置图表渲染（Matplotlib等） |
| **持久化与复用性** | 高（代码可保存、版本控制） | 低（会话结束即丢失） | 中等（需保存.ipynb文件） |
| **学习/教学友好性** | 一般（需理解文件结构） | 一般（适合简单逻辑） | 高（适合分步讲解） |

# P5. PyTorch加载数据初认识

- 数据
- Dataset：提供一种方式去获取数据及其 label
- Dataloader：为后面的网络提供不同的数据形式

# P6. Dataset类代码实战

```python
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir: str = root_dir
        self.label_dir: str = label_dir
        self.path: str = os.path.join(self.root_dir, self.label_dir)
        self.image_path_list = os.listdir(str(self.path))

    def __getitem__(self, index):
        image_name: str = self.image_path_list[index]
        image_item_path: str = os.path.join(self.path, image_name)
        image = Image.open(image_item_path)
        label = self.label_dir

        return image, label

    def __len__(self):
        return len(self.image_path_list)
```

- 案例演示：[p6_dataset.py](https://github.com/ZZHow1024/PyTorch-Learning/blob/main/P1-6/p6_dataset.py)