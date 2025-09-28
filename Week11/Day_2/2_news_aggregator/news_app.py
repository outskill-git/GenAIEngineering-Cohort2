"""
Streamlit UI for News Aggregator API
Connects to FastAPI backend on port 9321
"""

import streamlit as st
import requests
from datetime import datetime
import json
import time
from typing import Dict, List, Optional, Any
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:9321")

# Page Configuration
st.set_page_config(
    page_title="AI News Aggregator",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* News card container */
    .news-container {
        display: grid;
        gap: 1.5rem;
        margin-top: 2rem;
    }

    /* Individual news card */
    .news-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .news-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }

    /* News title */
    .news-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }

    /* Metadata row */
    .news-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        color: #6b7280;
    }

    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    /* Summary text */
    .news-summary {
        color: #4b5563;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* Badges */
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    .badge-high {
        background-color: #10b981;
        color: white;
    }

    .badge-medium {
        background-color: #f59e0b;
        color: white;
    }

    .badge-low {
        background-color: #ef4444;
        color: white;
    }

    .badge-category {
        background-color: #6366f1;
        color: white;
    }

    /* Metrics styling */
    .metric-card {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Search box styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Status indicators */
    .status-online {
        color: #10b981;
        font-weight: 600;
    }

    .status-offline {
        color: #ef4444;
        font-weight: 600;
    }

    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)


class NewsAggregatorUI:
    """Main UI class for the News Aggregator"""

    def __init__(self):
        self.api_url = API_BASE_URL
        self._init_session_state()

    def _init_session_state(self):
        """Initialize session state variables"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'current_results' not in st.session_state:
            st.session_state.current_results = None
        if 'last_topic' not in st.session_state:
            st.session_state.last_topic = ""

    def check_api_status(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=300)
            if response.status_code == 200:
                return {"online": True, "data": response.json()}
            return {"online": False, "data": None}
        except:
            return {"online": False, "data": None}

    def fetch_news(self, topic: str, use_mock: bool = False) -> Optional[Dict]:
        """Fetch news from the API"""
        try:
            response = requests.get(
                f"{self.api_url}/api/news/{topic}",
                params={"mock": use_mock},
                timeout=3000
            )
            response.raise_for_status()
            data = response.json()

            # Update session state
            if data.get('success'):
                st.session_state.current_results = data
                st.session_state.last_topic = topic
                if topic not in st.session_state.search_history:
                    st.session_state.search_history.insert(0, topic)
                    st.session_state.search_history = st.session_state.search_history[:10]

            return data
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API server. Please ensure it's running on port 9321.")
            return None
        except requests.exceptions.Timeout:
            st.error("⏰ Request timed out. Please try again.")
            return None
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return None

    def fetch_trending_topics(self) -> List[str]:
        """Fetch trending topics"""
        try:
            response = requests.get(f"{self.api_url}/api/topics/trending", timeout=5)
            return response.json().get("topics", [])
        except:
            return []

    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5rem;">📰 AI News Aggregator</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Powered by CrewAI Agents | Real-time News Curation
            </p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render sidebar with settings and info"""
        with st.sidebar:
            st.header("⚙️ Settings & Info")

            # API Status
            status = self.check_api_status()
            st.subheader("🔌 API Connection")
            if status["online"]:
                st.markdown('<p class="status-online">✅ Online</p>', unsafe_allow_html=True)
                if status["data"]:
                    with st.expander("API Details"):
                        st.json(status["data"])
            else:
                st.markdown('<p class="status-offline">❌ Offline</p>', unsafe_allow_html=True)
                st.code(f"API URL: {self.api_url}")

            # Data Mode
            st.divider()
            use_mock = st.toggle(
                "Use Mock Data",
                value=False,
                help="Enable to use simulated data without API keys"
            )

            # Trending Topics
            st.divider()
            st.subheader("🔥 Trending Topics")
            trending = self.fetch_trending_topics()
            if trending:
                selected = st.selectbox(
                    "Quick select:",
                    [""] + trending,
                    format_func=lambda x: "Choose a topic..." if x == "" else x
                )
            else:
                selected = ""
                st.info("Unable to fetch trending topics")

            # Search History
            if st.session_state.search_history:
                st.divider()
                st.subheader("🕐 Recent Searches")
                for topic in st.session_state.search_history[:5]:
                    if st.button(f"🔍 {topic}", key=f"hist_{topic}", use_container_width=True):
                        return topic, use_mock

            # Info
            st.divider()
            with st.expander("ℹ️ About"):
                st.markdown("""
                This app uses AI agents to:
                - 🔍 Search for relevant news
                - 📝 Curate and summarize content
                - ✅ Verify credibility
                - 📊 Provide analytics

                **API Endpoints:**
                - `/api/news/{topic}` - Get news
                - `/api/topics/trending` - Trending topics
                - `/api/topics/recent` - Recent searches
                """)

            return selected, use_mock

    def render_news_card(self, item: Dict, index: int):
        """Render a single news card"""
        credibility_badge = {
            "High": "badge-high",
            "Medium": "badge-medium",
            "Low": "badge-low"
        }.get(item.get('credibility', 'Medium'), 'badge-medium')

        card_html = f"""
        <div class="news-card">
            <h3 class="news-title">{item.get('title', 'Untitled')}</h3>
            <div class="news-meta">
                <span class="meta-item">📅 {item.get('date', 'N/A')}</span>
                <span class="meta-item">📰 {item.get('source', 'Unknown')}</span>
                <span class="badge badge-category">{item.get('category', 'General')}</span>
                <span class="badge {credibility_badge}">{item.get('credibility', 'Medium')} Credibility</span>
            </div>
            <p class="news-summary">{item.get('summary', 'No summary available.')}</p>
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)

        # Key points expander
        key_points = item.get('key_points', [])
        if key_points:
            with st.expander("📌 Key Points"):
                for point in key_points:
                    st.write(f"• {point}")

        # Article link
        if item.get('url'):
            st.markdown(f"[🔗 Read full article]({item['url']})")

        st.divider()

    def render_analytics(self, news_items: List[Dict]):
        """Render analytics dashboard"""
        total = len(news_items)
        if total == 0:
            st.info("No data to analyze")
            return

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics
        high_cred = sum(1 for item in news_items if item.get('credibility') == 'High')
        sources = set(item.get('source', 'Unknown') for item in news_items)
        categories = set(item.get('category', 'General') for item in news_items)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Articles</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{high_cred}</div>
                <div class="metric-label">High Credibility</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(sources)}</div>
                <div class="metric-label">News Sources</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(categories)}</div>
                <div class="metric-label">Categories</div>
            </div>
            """, unsafe_allow_html=True)

        # Charts
        st.markdown("### 📊 Detailed Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Source distribution
            st.subheader("Sources Distribution")
            source_counts = {}
            for item in news_items:
                source = item.get('source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            st.bar_chart(source_counts)

        with col2:
            # Category distribution
            st.subheader("Categories Distribution")
            cat_counts = {}
            for item in news_items:
                cat = item.get('category', 'General')
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            st.bar_chart(cat_counts)

        # Credibility breakdown
        st.subheader("Credibility Analysis")
        cred_data = {"High": 0, "Medium": 0, "Low": 0}
        for item in news_items:
            cred = item.get('credibility', 'Medium')
            if cred in cred_data:
                cred_data[cred] += 1

        cols = st.columns(3)
        for i, (level, count) in enumerate(cred_data.items()):
            with cols[i]:
                percentage = (count / total * 100) if total > 0 else 0
                color = {"High": "#10b981", "Medium": "#f59e0b", "Low": "#ef4444"}[level]
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: {color}20; border-radius: 8px;">
                    <h2 style="color: {color}; margin: 0;">{count}</h2>
                    <p style="margin: 0;">{level} ({percentage:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

    def run(self):
        """Main application loop"""
        # Render header
        self.render_header()

        # Get sidebar inputs
        trending_topic, use_mock = self.render_sidebar()

        # Search interface
        col1, col2 = st.columns([5, 1])

        with col1:
            default_topic = trending_topic or st.session_state.last_topic
            print(f"default_topic: {default_topic}")
            topic = st.text_input(
                "🔍 Search for news about:",
                value=default_topic,
                placeholder="Enter any topic (e.g., AI, Climate Change, Technology...)",
                key="search_input_v2"
            )


            search_clicked=True


        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button(
                "Search",
                type="primary",
                use_container_width=True,
                disabled=not topic.strip()
            )

        # Handle search
        if search_clicked and topic.strip():
            with st.spinner(f"🔍 Searching for news about '{topic}'..."):
                # Progress animation
                progress = st.progress(0)
                for i in range(101):
                    time.sleep(0.01)
                    progress.progress(i)

                # Fetch news
                results = self.fetch_news(topic.strip(), use_mock)
                progress.empty()

                if results and results.get('success'):
                    st.success(f"✅ Found {results['news_count']} articles about '{topic}'!")
                    st.session_state.current_results = results
                elif results:
                    st.error(f"❌ {results.get('message', 'Failed to fetch news')}")

        # Display results
        if st.session_state.current_results:
            results = st.session_state.current_results

            st.divider()

            # Results header
            st.markdown(f"### 📰 Results for: **{results['topic']}**")

            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📄 Articles", "📊 Analytics", "🔧 Raw Data"])

            with tab1:
                if results['news_items']:
                    # Filter options
                    col1, col2, col3 = st.columns([2, 2, 6])
                    with col1:
                        cred_filter = st.selectbox(
                            "Filter by credibility:",
                            ["All", "High", "Medium", "Low"],
                            key="cred_filter"
                        )
                    with col2:
                        sort_by = st.selectbox(
                            "Sort by:",
                            ["Date", "Credibility", "Source"],
                            key="sort_filter"
                        )

                    # Apply filters and sorting
                    items = results['news_items']
                    if cred_filter != "All":
                        items = [i for i in items if i.get('credibility') == cred_filter]

                    # Sort items
                    if sort_by == "Date":
                        items.sort(key=lambda x: x.get('date', ''), reverse=True)
                    elif sort_by == "Credibility":
                        cred_order = {"High": 0, "Medium": 1, "Low": 2}
                        items.sort(key=lambda x: cred_order.get(x.get('credibility', 'Medium'), 1))
                    elif sort_by == "Source":
                        items.sort(key=lambda x: x.get('source', 'Unknown'))

                    # Display items
                    st.markdown(f"<div class='news-container'>", unsafe_allow_html=True)
                    for i, item in enumerate(items):
                        self.render_news_card(item, i)
                    st.markdown("</div>", unsafe_allow_html=True)

                    if not items:
                        st.info(f"No articles found with {cred_filter} credibility.")
                else:
                    st.info("No articles found.")

            with tab2:
                self.render_analytics(results['news_items'])

            with tab3:
                # Raw JSON data
                st.json(results)

                # Export button
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(results, indent=2),
                    file_name=f"news_{results['topic'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        # Footer
        st.divider()
        st.markdown(
            f"""
            <div style='text-align: center; color: #6b7280; padding: 1rem;'>
                <small>
                    AI News Aggregator v1.0 | API: <code>{self.api_url}</code> |
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )


# Run the application
if __name__ == "__main__":
    app = NewsAggregatorUI()
    app.run()