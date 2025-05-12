# Google Colab运行指南 - 多模态情感检测系统

本指南将帮助您在Google Colab中设置和运行多模态情感检测系统。

## 1. 克隆仓库并设置环境

在Colab中创建一个新的笔记本，然后执行以下代码：

```python
# 克隆GitHub仓库
!git clone https://github.com/dwcqwcqw/speech-emotion-detection.git
%cd speech-emotion-detection

# 安装必要的依赖包
!pip install -r requirements.txt

# 安装额外需要的依赖
!pip install streamlit pydub librosa torch transformers sklearn nltk scikit-learn matplotlib seaborn tqdm
!pip install imblearn

# 下载NLTK资源
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## 2. 修复Colab特定的路径问题

由于Colab的运行环境与本地环境不同，需要创建一个Colab特定的配置文件：

```python
%%writefile colab_config.py
import os
import sys

# 确保运行时当前目录是项目根目录
project_root = os.getcwd()
sys.path.append(project_root)

# 创建必要的目录
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/audio', exist_ok=True)
os.makedirs('data/evaluation', exist_ok=True)
os.makedirs('data/sarcasm', exist_ok=True)
```

## 3. 数据集准备

下载并准备RAVDESS数据集和讽刺数据集：

```python
# 运行数据集下载脚本
!python download_sarcasm_dataset.py

# 查看数据集目录结构
!ls -la data/
!ls -la data/sarcasm/
```

## 4. 训练模型

使用以下代码训练情感检测模型：

```python
# 运行训练脚本
!python app/train_model.py
```

## 5. 评估模型

训练完成后，评估模型的性能：

```python
# 运行评估脚本
!python app/evaluate_model.py

# 查看评估结果
!ls -la data/evaluation/
```

## 6. 在Colab中运行Streamlit应用（可选）

如果您想在Colab中运行Streamlit应用进行演示，可以使用以下方法：

```python
# 安装和配置ngrok（用于暴露本地服务器）
!pip install pyngrok
from pyngrok import ngrok

# 启动Streamlit应用
!streamlit run app/app.py &

# 创建公共URL
public_url = ngrok.connect(port=8501)
print(f"Streamlit应用可通过以下链接访问: {public_url}")
```

## 7. 使用MultimodalAnalyzer进行自定义推理

您可以使用以下代码片段在Colab中测试系统的推理能力：

```python
# 导入所需的库
import sys
import os
sys.path.append(os.getcwd())

from app.utils.audio_processor import AudioProcessor
from app.utils.speech_to_text_simple import SimpleSpeechToText 
from app.utils.text_analyzer_extended import TextAnalyzerExtended
from app.utils.emotion_classifier_extended import EmotionClassifierExtended
from app.utils.multimodal_analyzer import MultimodalAnalyzer

# 初始化组件
audio_processor = AudioProcessor()
speech_to_text = SimpleSpeechToText()
text_analyzer = TextAnalyzerExtended()

# 加载训练好的模型
model_path = 'data/models/emotion_classifier_extended.pkl'
emotion_classifier = EmotionClassifierExtended(model_path=model_path)

# 初始化多模态分析器
multimodal_analyzer = MultimodalAnalyzer(
    audio_processor=audio_processor,
    speech_to_text=speech_to_text,
    text_analyzer=text_analyzer,
    emotion_classifier=emotion_classifier,
    weights_path='data/models/multimodal_weights.json' if os.path.exists('data/models/multimodal_weights.json') else None
)

# 选择一个音频文件进行分析
from app.utils.dataset_handler import DatasetHandler
dataset_handler = DatasetHandler()
_, _, test_df = dataset_handler.split_dataset()

# 分析第一个测试样本
if len(test_df) > 0:
    sample = test_df.iloc[0]
    print(f"分析样本: {sample['filename']} (真实情感: {sample['emotion']})")
    
    result = multimodal_analyzer.analyze(sample['path'])
    
    print(f"预测情感: {result['emotion']}")
    print(f"转录文本: {result['transcription']}")
    print(f"模态一致性得分: {result['agreement_score']:.2f}")
    print(f"模态权重: 音频={result['modality_weights']['audio']:.2f}, 文本={result['modality_weights']['text']:.2f}")
    
    # 打印置信度得分
    print("\n情感置信度得分:")
    for emotion, score in sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {score:.4f}")
```

## 故障排除

如果遇到以下问题，可以尝试相应的解决方案：

### 数据集下载问题

如果RAVDESS数据集下载或提取出现问题，可以手动下载并上传到Colab：

```python
# 手动下载RAVDESS数据集
!wget -O data/ravdess.zip https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

# 解压数据集
!unzip -o data/ravdess.zip -d data/
!mkdir -p data/audio

# 将所有WAV文件移动到data/audio目录
!find data/Audio_Speech_Actors_01-24 -name "*.wav" -exec cp {} data/audio/ \;
```

### 路径错误

如果出现路径相关错误，请确保当前工作目录是项目根目录：

```python
import os
print(f"当前工作目录: {os.getcwd()}")
# 如果不是项目根目录，请使用以下命令切换
%cd speech-emotion-detection  # 替换为实际路径
```

### 内存错误

如果遇到内存不足错误，可以尝试以下方法：

1. 在Colab中切换到提供更多RAM的运行时（运行时 > 更改运行时类型 > 选择高RAM）
2. 减少批处理大小或使用较小的数据子集进行训练：

```python
# 使用较小的训练集进行测试
!python app/train_model.py --sample-size 100  # 如果脚本支持此参数
```

### 依赖包冲突

如果遇到依赖包冲突问题，可以尝试在隔离环境中安装：

```python
!pip install -r requirements.txt --no-deps
```

然后手动安装关键依赖：

```python
!pip install torch==1.10.0 transformers==4.11.3 librosa==0.8.1
``` 