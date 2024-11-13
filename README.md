# YouTube Video Analyzer Pro 🚀

[![FastAPI](https://img.shields.io/badge/FastAPI-0.108.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Whisper](https://img.shields.io/badge/Whisper-AI-FF6B6B.svg?style=flat&logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![Llama](https://img.shields.io/badge/Llama_3.2-AI-FF9A00.svg?style=flat&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/KazKozDev/video-analyser/pulls)
[![YouTube API](https://img.shields.io/badge/YouTube-API-red.svg?style=flat&logo=youtube&logoColor=white)](https://developers.google.com/youtube/v3)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=KazKozDev.video-analyser)](https://github.com/KazKozDev/video-analyser)


## 🌟 Overview

YouTube Video Analyzer Pro is a cutting-edge tool that brings professional-grade video analysis capabilities to content creators and viewers alike. Leveraging advanced AI technologies, it provides deep insights into video content, audience engagement, and market positioning.

## ✨ Key Features

### 🧠 AI-Powered Content Analysis
- **Smart Summarization**
  - Generate concise, intelligent video summaries using Llama 3.2 AI
  - Extract key topics and main points automatically
  - Identify crucial timestamps and segments

### 🎯 Advanced Transcription System
- **Multi-Layer Transcription**
  - Primary: YouTube's native transcription API
  - Backup: Whisper AI for videos without existing transcripts
  - Smart timestamps with topic segmentation
  - Custom processing for cleaner output

### 🌍 Professional Translation Hub
- **Enterprise-Grade Translation Support**
  ```
  🇬🇧 English  →  Source Language
  🇪🇸 Spanish  →  Native Quality
  🇫🇷 French   →  Professional Grade
  🇩🇪 German   →  High Accuracy
  🇮🇹 Italian  →  Natural Flow
  🇷🇺 Russian  →  Precise Translation
  ```

### 💬 Comment Intelligence
- **Sentiment Analysis**
  - Emotional tone detection in comments
  - Key themes and patterns identification
  - Audience sentiment tracking
  - Controversy detection
- **Engagement Metrics**
  - Comment frequency analysis
  - User interaction patterns
  - Community feedback categorization
- **Strategic Insights**
  - Viewer pain points identification
  - Content improvement suggestions
  - Community engagement opportunities

### 🔍 Content Discovery Engine
- **Channel Analysis**
  - Similar video recommendations
  - Content pattern detection
  - Performance metric tracking
- **Competitor Intelligence**
  - Related content from other creators
  - Niche trend analysis
  - Market positioning insights

## 🛠️ Technology Stack

### Backend Infrastructure
```mermaid
graph TD
    A[FastAPI] --> B[Python 3.9+]
    B --> C[YouTube Data API]
    B --> D[AI Models]
    D --> E[Whisper AI]
    D --> F[Llama 3.2]
```

### AI Processing Pipeline
- **Speech Recognition**: Whisper AI
- **Content Analysis**: Llama 3.2
- **Translation Engine**: Custom AI Pipeline
- **Comment Analysis**: Advanced NLP Models

### Frontend Architecture
- **Framework**: React
- **UI Components**: Modern Material Design
- **Data Visualization**: Dynamic Charts
- **Responsive Design**: Mobile-First Approach

## ⚡ Performance & Scalability

### Free Tier Benefits
- Full access to all features
- Unlimited video analysis
- Complete comment processing
- All languages supported

### ⚡ Processing Times

**🔍 Basic Analysis**
- Short Videos (<10min): ~30 seconds
- Long Videos (>10min): 1-2 minutes

**📝 Transcription**
- Short Videos (<10min): 1-2 minutes
- Long Videos (>10min): 3-5 minutes

**🌍 Translation**
- Short Videos (<10min): 2-3 minutes
- Long Videos (>10min): 5-10 minutes

**💬 Comment Analysis**
- Short Videos (<10min): 1-2 minutes
- Long Videos (>10min): 2-3 minutes

Note: Processing times may vary depending on server load and video complexity. All operations run locally for maximum privacy and security.

## 🚀 Getting Started

### System Requirements
- Python 3.9+
- 8GB RAM minimum
- GPU recommended for faster processing
- YouTube API Key

### Quick Start
1. **Clone & Setup**
   ```bash
   git clone https://github.com/KazKozDev/video-analyser.git
   cd video-analyser
   ```

2. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configuration**
   ```bash
   cp .env.example .env
   # Add your YouTube API key to .env
   ```

4. **Launch**
   ```bash
   uvicorn server:app --reload
   ```

Visit `http://localhost:8000` and start analyzing! 🎉

## 📊 Use Cases

### Content Creators
- **Channel Optimization**
  - Content performance analysis
  - Audience retention insights
  - Engagement optimization
  - Competitor research
  - Multilingual reach expansion

### Marketing Teams
- **Market Research**
  - Trend analysis
  - Competitor tracking
  - Audience insights
  - Content strategy development

### Researchers & Analysts
- **Data Collection**
  - Transcript extraction
  - Comment analysis
  - Engagement metrics
  - Cross-channel comparisons

## 🤝 Contributing

We welcome contributions! Check our [Contributing Guide](https://github.com/KazKozDev/video-analyser/blob/main/CONTRIBUTING.md) for more information.

1. Fork the repo
2. Create a feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit changes
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See [LICENSE](https://github.com/KazKozDev/video-analyser/blob/main/LICENSE) for more information.

## 🙏 Acknowledgments

- OpenAI for the incredible Whisper AI
- Meta for Llama 3.2
- YouTube API Team
- All contributors and supporters

## ⭐ Support the Project

If you found this tool useful, consider:
- Giving it a star ⭐
- Sharing it with others
- Contributing to its development
- Reporting issues or suggesting features

## 📧 Contact

KazKozDev - [@KazKozDev](https://github.com/KazKozDev)

Project Link: [https://github.com/KazKozDev/video-analyser](https://github.com/KazKozDev/video-analyser)

---

<div align="center">
  Made with ❤️ for the YouTube community
  <br>
  <a href="https://github.com/KazKozDev/video-analyser/stargazers">⭐ Star us on GitHub!</a>
</div>