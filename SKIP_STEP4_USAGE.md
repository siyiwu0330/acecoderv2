# Skip Step4 功能使用说明

## 功能概述

为了解决大量问题（problem）时在第二轮结尾卡住的问题，我们添加了 `skip_step4` 参数来控制是否跳过第四步（交叉轮次评估）。

## 使用方法

### 1. 命令行使用

```bash
# 跳过第四步（推荐用于大数据集）
python main.py --output_dir outputs/test --rounds 2 --max_samples 100 --skip_step4 True

# 不跳过第四步（默认行为）
python main.py --output_dir outputs/test --rounds 2 --max_samples 100 --skip_step4 False

# 或者不指定（默认为 False）
python main.py --output_dir outputs/test --rounds 2 --max_samples 100
```

### 2. Gradio 界面使用

1. 启动 Gradio 界面：
   ```bash
   python integrated_gradio_app.py
   ```

2. 在 "Pipeline Control" 标签页中：
   - 找到 "⏭Skip Step 4 (Cross-Round Evaluation)" 复选框
   - 勾选此选项以跳过第四步
   - 点击 "Start Pipeline" 开始运行
