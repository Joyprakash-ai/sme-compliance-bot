# Contributing to SME Compliance Bot

Thank you for your interest in making EU compliance more accessible through AI! This project welcomes contributions from developers, legal professionals, and subject matter experts.

## Contribution Areas

### 1. Regulation Coverage
- Add new EU directives and regulations
- Update existing regulations with amendments
- Add country-specific implementations
- Translate regulations to other EU languages

### 2. Feature Development
- Improve RAG (Retrieval-Augmented Generation) pipeline
- Add multi-language support
- Implement better citation mechanisms
- Develop industry-specific modules
- Create API integrations with legal databases

### 3. User Experience
- Improve web interface
- Add mobile app
- Create browser extensions
- Develop chat integrations (Slack, Teams, etc.)

### 4. Testing & Quality
- Add more test cases
- Improve answer accuracy
- Develop benchmark datasets
- Create compliance verification tools

## Development Setup

### Prerequisites
- Python 3.10+
- OpenAI API key
- PostgreSQL (for conversation history)

### Installation
```bash
# Clone repository
git clone https://github.com/joyprakash-ai/sme-compliance-bot.git
cd sme-compliance-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize knowledge base
python knowledge_base.py --init
