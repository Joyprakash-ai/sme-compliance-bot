#!/usr/bin/env python3
"""
Streamlit web interface for SME Compliance Bot.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import ComplianceBot, BotConfig

# Page configuration
st.set_page_config(
    page_title="SME Compliance Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
    }
    .answer-box {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .source-box {
        background-color: #EFF6FF;
        border-left: 4px solid #1E3A8A;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_bot():
    """Initialize the Compliance Bot with session state."""
    if 'bot' not in st.session_state:
        config = BotConfig(
            model_name=st.session_state.get('model', 'gpt-4-turbo-preview'),
            temperature=0.1
        )
        st.session_state.bot = ComplianceBot(config)
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 SME Compliance Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI assistant for EU regulations and compliance questions</p>', unsafe_allow_html=True)
    
    # Initialize bot
    initialize_bot()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/197/197575.png", width=100)
        st.title("Settings")
        
        # Model selection
        model_option = st.selectbox(
            "AI Model",
            ["gpt-4-turbo-preview", "gpt-3.5-turbo"],
            index=0,
            help="Select the AI model to use"
        )
        
        # Country selection
        eu_countries = [
            "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
            "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
            "NL", "PL", "PT", "RO", "SE", "SI", "SK"
        ]
        country = st.selectbox(
            "Jurisdiction",
            ["EU General"] + eu_countries,
            index=0,
            help="Select country for jurisdiction-specific answers"
        )
        
        # Topic filter
        topics = ["All", "Taxation/VAT", "Data Privacy", "Employment Law", 
                 "Consumer Protection", "Environmental Law", "Digital Services"]
        topic = st.selectbox("Topic Filter", topics, index=0)
        
        # Clear history button
        if st.button("Clear Conversation History"):
            st.session_state.conversation_history = []
            st.session_state.bot.clear_history()
            st.success("History cleared!")
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions Asked", len(st.session_state.conversation_history))
        with col2:
            st.metric("EU Directives", "50+")
        
        st.markdown("---")
        
        # Quick questions
        st.subheader("💡 Quick Questions")
        quick_questions = [
            "What is the VAT threshold for small businesses?",
            "Do I need to register for VAT in another EU country?",
            "What are the GDPR requirements for my website?",
            "What employment laws apply to remote workers in the EU?",
            "What consumer rights apply to online sales?"
        ]
        
        for q in quick_questions:
            if st.button(q, key=f"quick_{hash(q)}"):
                st.session_state.question_input = q
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Question input
        question = st.text_area(
            "Ask your EU compliance question:",
            value=st.session_state.get('question_input', ''),
            height=100,
            placeholder="E.g., 'What VAT rules apply when selling digital goods from Italy to Germany?'",
            key="question_input"
        )
        
        # Ask button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            ask_button = st.button("Ask 🤖", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.question_input = ""
            st.rerun()
        
        # Process question
        if ask_button and question:
            with st.spinner("Analyzing regulations..."):
                # Get country code (remove "EU General")
                country_code = None if country == "EU General" else country
                
                # Ask the bot
                response = st.session_state.bot.ask(question, country_code)
                
                # Store in history
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "question": question,
                    "country": country,
                    "answer_preview": response["sections"].get("Answer", "")[:100] + "..."
                })
                
                # Display answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.subheader("📝 Answer")
                st.write(response["sections"].get("Answer", response["answer"]))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display directives
                if response["sections"].get("Applicable Directives"):
                    st.subheader("📚 Applicable Directives")
                    st.write(response["sections"]["Applicable Directives"])
                
                # Display next steps
                if response["sections"].get("Next Steps"):
                    st.subheader("🚀 Next Steps")
                    st.write(response["sections"]["Next Steps"])
                
                # Display sources
                if response["sources"]:
                    st.subheader(f"🔍 Sources ({len(response['sources'])} found)")
                    
                    for i, source in enumerate(response["sources"], 1):
                        with st.expander(f"Source {i}: {source.get('directive', 'Unknown')}"):
                            st.write(f"**Directive:** {source.get('directive', 'N/A')}")
                            st.write(f"**Article:** {source.get('article', 'N/A')}")
                            st.write(f"**Topic:** {source.get('metadata', {}).get('topic', 'Unknown')}")
                            st.write(f"**Content:** {source.get('content', '')}")
                
                # Display disclaimer
                if response["sections"].get("Disclaimer"):
                    st.info(response["sections"]["Disclaimer"])
                else:
                    st.info("⚠️ **Disclaimer:** This information is for guidance only. Always consult a qualified legal professional for compliance matters.")
                
                # Download option
                json_str = json.dumps(response, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download Answer (JSON)",
                    data=json_str,
                    file_name=f"compliance_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col2:
        # Conversation history
        st.subheader("💬 History")
        
        if st.session_state.conversation_history:
            for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Q: {entry['question'][:50]}...", expanded=False):
                    st.write(f"**Time:** {entry['timestamp']}")
                    st.write(f"**Country:** {entry['country']}")
                    st.write(f"**Answer Preview:** {entry['answer_preview']}")
                    
                    if st.button(f"Reuse", key=f"reuse_{i}"):
                        st.session_state.question_input = entry['question']
                        st.rerun()
        else:
            st.write("No questions yet. Ask something!")
        
        st.markdown("---")
        
        # Recent directives
        st.subheader("📜 Recent Directives")
        directives = [
            "VAT Directive 2006/112/EC",
            "GDPR 2016/679",
            "Consumer Rights Directive 2011/83/EU",
            "Digital Services Act",
            "Corporate Sustainability Reporting"
        ]
        
        for directive in directives:
            st.write(f"• {directive}")
        
        st.markdown("---")
        
        # Links
        st.subheader("🔗 Official Sources")
        st.markdown("""
        - [EUR-Lex](https://eur-lex.europa.eu/)
        - [EU VAT Information](https://ec.europa.eu/taxation_customs/business/vat_en)
        - [GDPR Portal](https://gdpr.eu/)
        - [EU Commission](https://ec.europa.eu/)
        """)

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OPENAI_API_KEY not set. Please set it in your environment variables.")
        st.info("You can still explore the interface, but AI features won't work.")
    
    main()
