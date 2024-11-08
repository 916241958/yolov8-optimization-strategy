## YOLOv8 with WIOU

### 修改说明
本项目对原始的 YOLOv8 模型进行了改动，将损失函数从 CIOU（Complete Intersection Over Union）替换为 WIOU（Weighted Intersection Over Union）。这种改动旨在提高检测精度，特别是在复杂场景下的表现。

### 使用方法
与原版 YOLOv8 的使用方法相同，只需注意以下改动：
- 训练过程中使用的损失函数为 WIOU。

### 注意事项
- 本项目修改了损失函数，并在cfg中增加了我所优化的部分模型，其他部分与原版 YOLOv8 相同。
- 如果在使用过程中遇到问题，请参考原版 YOLOv8 的文档或提出 issue。

感谢大家的支持和理解！😊
