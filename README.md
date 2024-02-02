# InfiniCanvas构图API

本项目为机器学习模型推理构图前端，提供类PyTorch形式的动态模型搭建API，服务于 [InfiniTensor（重构图）](https://github.com/InfiniTensor/RefactorGraph)后端推理框架，支持导出静态化的onnx计算图。

## 安装

1. 安装 [InfiniTensor（重构图）](https://github.com/InfiniTensor/RefactorGraph)后端推理框架
2. 使用 `make` 安装本项目

## 使用说明

### 模型搭建

#### 1. InfiniTensorModel

本项目中的模型均继承自 `InfiniTensorModel` 基类（类似PyTorch中的 `nn.Module`）。

#### 2. 模型构造

在 `__init__()` 函数中定义该模型的各类参数，以及子模块。其中常量使用 `self.constant` 创建；模型参数（可导入导出）使用 `self.parameter` 创建。使用 `self.make_submodel(ModelClassName, ...)` 可在模型中创建子模型（同为`InfiniTensorModel` 的子类）。需要注意的是，子模型创建必须在 `__init__()` 函数中且必须使用 `self.make_submodel` 函数创建，这个函数会隐式调用子模型类的构造函数。这是因为模型与子模型需要共享一套命名空间，且在构造外围模型时子模型必须先完成构建。定义 `__init__()` 函数时，必须传入 `**kwargs` 参数，必须调用 `super().__init__(**kwargs)`。

#### 3. 计算图构造

在 `__call__()` 函数中定义该模型推理时的计算图。 `InfiniTensorModel` 基类提供了一系列算子API，如 `self.add` 等。也可调用子模型的  `__call__()` 函数，代表子模块的计算。与PyTorch的 `forward()` 函数不同的是，这里的 `__call__()` 函数并不会发生任何计算，而只是用来构建计算图，因此传进来的参数是代表张量的张量名称（字符串）。一般来说，传进来的输入以及模型的输出应该分别被添加至 `self.inputs` 和 `self.outputs` 中。原则上，一个模型包括其子模型的 `__call__()` 函数只可被调用一次。

以下为一个语言模型常用的FeedFoward层的定义。

```python
class Linear(InfiniTensorModel):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DTYPE = DTYPE.F32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        shape = (out_features, in_features)
        # 创建参数张量
        self.weight = self.parameter(
            (np.random.random(shape)).astype(dtype.np_type()), "weight"
        )
        self.use_bias = bias
        if self.use_bias:
            self.bias = self.parameter(
                np.random.random(out_features).astype(dtype.np_type()), "bias"
            )

    def __call__(self, input):
        super().__call__([input])
        output = self.matmul(input, self.weight, transB=1)
        if self.use_bias:
            output = self.add(output, self.bias)
        self.outputs.append(output)
        return output

class FeedForward(InfiniTensorModel):
    def __init__(self, hidden_size, intermediate_size, dtype=DTYPE.F32, **kwargs):
        # 调用基类的构造函数
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        # 创建一个Linear层（子模型）
        self.gate_proj = self.make_submodel(
            Linear, self.hidden_size, self.intermediate_size, False, dtype, model_name = "gate_proj"
        )
        self.up_proj = self.make_submodel(
            Linear, self.hidden_size, self.intermediate_size, False, dtype, model_name = "up_proj"
        )
        self.down_proj = self.make_submodel(
            Linear, self.intermediate_size, self.hidden_size, False, dtype, model_name = "down_proj"
        )
        self.act_fn = self.silu

    def __call__(self, x):
        super().__call__([x])
        output = self.down_proj(
            self.mul(self.act_fn(self.gate_proj(x)), self.up_proj(x))
        )
        self.outputs = [output]
        return output
```

### 推理

在进行实际计算之前，应该首先调用模型的 `__call__()` 函数构建计算图。之后，调用 `run()` 函数使用后端推理框架进行推理。需要注意的是，如果在模型中使用了 `self.dynamic_tensor` 创建的动态张量，则推理时必须传入相应的变量表对变量进行转换。如果进行多轮推理且张量形状保持不变，则只需第一轮推理时传入变量表，或者可以将重编译选项关闭。

```python
# 创建模型
model = MyModel(...)
# 将模型运行时设备设置为GPU
model.to("cuda", 0)
# 导入参数
model.load_params(...)
# 调用__call__()函数构图
model(["input1", "input2"]) 
# 输入
inputs = {
    "input1": np.array(...),
    "input2": np.array(...),
}
# 变量表
variable_map = {
    "batchsize": 32
}
# 推理
outputs = model.run(inputs, variable_map)
```
