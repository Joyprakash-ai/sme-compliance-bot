# 🤖 SME Compliance Bot - EU Regulation Assistant

An intelligent conversational AI agent that provides real-time answers to questions about EU regulations for small and medium enterprises. Built with RAG (Retrieval-Augmented Generation) and fine-tuned for legal accuracy.

**✨ Features**
- **Real-Time Regulation Lookup:** Instant answers about GDPR, VAT directives, employment law, and more.
- **Citation & Source Tracking:** Every answer includes references to specific EU directives and articles.
- **Multi-Format Support:** Ask via web interface, API, or Discord/Telegram bot.
- **Context-Aware Responses:** Maintains conversation context for follow-up questions.
- **Custom Knowledge Base:** Easily add your company's internal policies alongside EU regulations.

**🎯 Use Case & Impact**
> **Problem:** An Italian e-commerce SME needs to know VAT rules for selling to Germany and France.
> **Solution:** Instead of hours of research, ask the bot: "What VAT rules apply when selling digital goods from Italy to Germany and France?"
> **Result:** Immediate, accurate answer with directive references, saving 4+ hours of legal research.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key (or local LLM setup)
- PostgreSQL (optional, for conversation history)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/joyprakash-ai/sme-compliance-bot.git
cd sme-compliance-bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Initialize knowledge base
python knowledge_base.py --init

# 5. Launch web interface
python app.py
