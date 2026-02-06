# OCL + CLIP Zero-shot Evaluation (Top-K / Attribute mAP / Affordance mAP)

本实验使用 **OpenAI CLIP (ViT-B/32)** 在 **OCL 测试集**上进行 zero-shot 推理评估，计算三类指标：

- **Object Classification：Top-K Accuracy**
- **Attribute Prediction：mAP (114 attributes)**
- **Affordance Prediction：mAP (170 affordances)**

核心思路：将类别/属性/可供性转成文本 prompt，CLIP 分别编码图像与文本，计算余弦相似度排序，再据此计算 Top-K 或 mAP


## 0. 配置环境
首先[安装miniforge](https://blog.csdn.net/lhyyds/article/details/139448689)

接着检查`cuda`版本
```bash
nvcc -V
```
如果版本不是12.1那就把以下两行贴到你的`.bashrc`最后面
```bash
# >>> CUDA 12.1 >>>
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
# <<< CUDA 12.1 <<<
```

其次创建环境
```bash
conda create -n myenv python=3.12
```

又次安装`torch` `torchvision` `torchaudio`以及`torchmetrics`
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```
和
```bash
pip install torchmetrics
```

## 1. 数据与文件

代码默认使用服务器 OCL 数据路径：

- 图片根目录：`/data/DATA/OCL_DATA/OCL_data/data/`
- 测试标注：`/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_annot_test.pkl`
- 类别列表：`/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_object.json`
- 属性列表：`/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_attribute.json`
- 可供性词表：`/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_affordance_word.txt`

---

## 2. 预处理与 bbox 裁剪

从 `OCL_annot_test.pkl` 中读取 `objects[0]['box']`，对图像先做 **bbox crop** 得到目标物体区域，再进行后续 transform 与特征提取。

三套 Dataset（`topkDataset / attrDataset / affDataset`）均采用相同裁剪逻辑：

```python
image = Image.open(path).convert("RGB")
image = image.crop(box_from_pkl)
```
- `convert("RGB")`：避免部分灰度图导致通道数不一致。
## 3. 文本 prompt 设计

本实验中，所有任务均采用 **CLIP 的文本–图像对齐机制**，将类别、属性和可供性统一建模为文本 prompt，并与图像特征计算余弦相似度进行评估。

---

### 3.1 类别 prompts（Top-K）

在物体类别 Top-K 分类任务中，类别名称来自  
`OCL_class_object.json`。

对于每个类别名 `w`，构造如下 prompt：

- `a photo of {w}`

对应实现位于 `dataset.py`：

```python
def to_prompt(self, words):
    return ["a photo of " + w for w in words]
```
这些 prompts 会被统一 tokenize 后，与图像特征进行相似度计算，用于 Top-K Accuracy 评估。

### 3.2 Attribute prompts（114 个）
Attribute 任务使用 dataset_attr.py 中人工设计的 prompt 列表，共 114 条，每条描述一种物体属性，例如：
- `a furry object`
- `a striped object`
- `a quadruped animal`
- `a cooked food`
- `a broken object`  

在评估阶段：
- 每个 attribute prompt 被**单独视为一个查询**
- 对测试集中所有图像计算与该 prompt 的相似度
- 根据相似度排序并结合 ground truth 计算该属性的 AP
- 对全部 114 个属性的 AP 取平均，得到最终**mAP(attribute)**

### 3.3 Affordance prompts（170 个）
Affordance prompts 来源于`OCL_class_affordance_word.txt`，文件中每一行对应一个 affordance 词语，共 **170 个**。
在 `dataset_aff.py` 中通过逐行读取构建 affordance 列表：
```python
affordances = [line.strip() for line in f]
```
评估方式与 attribute 相同：  
- 每个 affordance word 单独作为查询 prompt
- 对全体测试图像计算相似度并排序
- 基于排序结果计算 AP
- 对 170 个 affordance 的 AP 取平均，得到 mAP(affordance)

### 3.4 设计说明
- 所有 prompt 均采用 简洁、描述性自然语言
- 未使用额外的 prompt tuning

## 4. 图像 transform 与模型
### 4.1 图像 transform（test）
在 `train.py` 中，测试 `transform `为：
- `CenterCrop(224)`
- `ToTensor()`
- `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
### 4-2 模型
使用 CLIP 预训练模型：
- `clip.load("ViT-B/32")`
- `推理模式，不更新参数（冻结 requires_grad=False）`

## 5. 指标计算方法
### 5.1 Top-K Accuracy
对每张图像：
1. 得到 image feature 与所有类别文本 feature
2. L2 normalize 后计算相似度（等价余弦相似度）
    ```python
    similarity = image_features @ text_features.T
    ```
3. 取 Top-K 类别索引集合
4. 若 GT 类别在集合内则计为正确
5. 最终根据如下公式：  
    $Accuracy@K = \frac{\#\{\mathrm{GT}\  \\in \ \mathrm{TopK}\}}{N}$
### 5.2 AP / mAP（Attribute & Affordance）
对某个 attribute / affordance prompt：
1. 构造 rank list：`[(similarity, gt_label), ...]`
2. 按 similarity 降序排序
3. 遍历排序列表，遇到正样本累加该位置 precision（`tp / i`）
4. AP = 累加和 / 正样本数
5. 对所有属性 / 可供性平均得到 mAP  

对应实现为 `compute_AP(rank)`。**Note: rank is a list**
## 6. 如何运行
在项目根目录执行：
```bash
python train.py
```
`train.py` 的 `main()` 中包含三类评估调用：
- `test_model(...)`（Top-K Accuracy）
- `test_attr_model(...)`（Attribute mAP）
- `test_aff_model(...)`（Affordance mAP）
可通过注释 / 取消注释选择运行内容。

## 7. 最终结果汇总（来自 output.txt）
### 7.1 Object Classification Top-K Accuracy（bbox-cropped）
| K   | Accuracy |
|-----|----------|
| 5   | 0.8113394132512911 |
| 10  | 0.8683661136138886 |
| 100 | 0.9748379298978135 |
### 7.2 Attribute mAP（114 attributes）
- **mAP(attribute)** = `0.1541992137853377`
### 7.3 Affordance mAP（170 affordances）
- 状态：仍在运行

## Reference
**夯爆了 建议该list的每篇都要看（for 未来要做这个入学测试的学弟妹）**
1. [CLIP怎么用，不过我感觉这篇最重要的是告诉你余弦相似度怎么算](https://blog.csdn.net/python_plus/article/details/139519831)
2. [理解什么是mean Avearage Precision](https://blog.csdn.net/qq_41427568/article/details/105733838)
3. [理解什么是mAP，不过我觉得没上面那篇写的好](https://blog.csdn.net/leviopku/article/details/80835929)
4. [理解什么是mAP，这篇必须夯爆，夯中之夯](https://blog.csdn.net/hsqyc/article/details/81702437)

## 如果你遇到这些bug...
- [`torch.__version__`的`torch`版本和`conda list`的`torch`版本不一致](https://blog.csdn.net/qq_40349484/article/details/143893420)
-> 直接创建一个新环境然后把需要的包再下一次比较快。。。
