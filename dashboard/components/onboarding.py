"""
User onboarding and help component for the enhanced Forex AI dashboard.
Provides guided tours, tooltips, and contextual help for new users.
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OnboardingManager:
    """Manages user onboarding and help features."""

    def __init__(self):
        self.onboarding_data = self._load_onboarding_data()
        self.user_progress = self._load_user_progress()

    def _load_onboarding_data(self) -> Dict[str, Any]:
        """Load onboarding content and tours."""
        return {
            "tours": {
                "welcome": {
                    "title": "Welcome to Forex AI Dashboard",
                    "steps": [
                        {
                            "title": "Navigation Sidebar",
                            "content": "Use the sidebar to navigate between different sections of the dashboard.",
                            "target": "sidebar",
                            "position": "right"
                        },
                        {
                            "title": "Key Metrics",
                            "content": "Monitor your portfolio performance with these key metrics.",
                            "target": "metrics",
                            "position": "bottom"
                        },
                        {
                            "title": "Interactive Charts",
                            "content": "Explore price movements and predictions with interactive visualizations.",
                            "target": "charts",
                            "position": "top"
                        },
                        {
                            "title": "Settings",
                            "content": "Customize your dashboard experience in the settings panel.",
                            "target": "settings",
                            "position": "left"
                        }
                    ]
                },
                "predictions": {
                    "title": "AI Price Predictions",
                    "steps": [
                        {
                            "title": "Symbol Selection",
                            "content": "Choose which currency pair to analyze.",
                            "target": "symbol_selector",
                            "position": "bottom"
                        },
                        {
                            "title": "Timeframe Control",
                            "content": "Select the timeframe for your analysis.",
                            "target": "timeframe_selector",
                            "position": "bottom"
                        },
                        {
                            "title": "Prediction Chart",
                            "content": "View AI-generated price predictions with confidence intervals.",
                            "target": "prediction_chart",
                            "position": "top"
                        },
                        {
                            "title": "Feature Importance",
                            "content": "Understand which factors influence the predictions most.",
                            "target": "feature_importance",
                            "position": "left"
                        }
                    ]
                },
                "performance": {
                    "title": "Performance Analytics",
                    "steps": [
                        {
                            "title": "Performance Metrics",
                            "content": "Track key performance indicators like Sharpe ratio and drawdowns.",
                            "target": "performance_metrics",
                            "position": "right"
                        },
                        {
                            "title": "Risk Analysis",
                            "content": "Analyze portfolio risk with Value at Risk and stress testing.",
                            "target": "risk_analysis",
                            "position": "top"
                        },
                        {
                            "title": "Backtesting",
                            "content": "Test strategies against historical data.",
                            "target": "backtesting",
                            "position": "bottom"
                        }
                    ]
                }
            },
            "tooltips": {
                "refresh_button": "Refresh all dashboard data",
                "export_button": "Export dashboard data to CSV/JSON",
                "alerts_button": "View system alerts and notifications",
                "settings_button": "Customize dashboard preferences",
                "search_button": "Search across all dashboard content"
            },
            "help_articles": {
                "getting_started": {
                    "title": "Getting Started Guide",
                    "content": """
# Welcome to Forex AI Professional Dashboard

## Quick Start Guide

1. **First Time Setup**
   - Configure your preferred symbols and timeframes
   - Set up notification preferences
   - Choose your theme (Light/Dark/Auto)

2. **Understanding the Interface**
   - **Sidebar**: Navigation and quick actions
   - **Header**: Global controls and status
   - **Main Content**: Current page content
   - **Footer**: System information

3. **Key Features**
   - Real-time price monitoring
   - AI-powered predictions
   - Performance analytics
   - Risk management tools
   - System monitoring

## Navigation Tips

- Use keyboard shortcuts (Ctrl+K for search)
- Hover over elements for tooltips
- Click the help button (?) for context-sensitive help
- Use the sidebar for quick navigation

## Best Practices

- Set up alerts for important price levels
- Regularly review performance metrics
- Monitor system health indicators
- Keep your data refreshed for accuracy
                    """
                },
                "predictions": {
                    "title": "Understanding AI Predictions",
                    "content": """
# AI Price Predictions Guide

## How Predictions Work

Our AI models analyze multiple factors to predict price movements:

- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Volume Analysis**: Trading volume patterns
- **Market Sentiment**: News and social media analysis
- **Historical Patterns**: Similar market conditions

## Confidence Levels

- **High (80-100%)**: Strong prediction signal
- **Medium (60-79%)**: Moderate confidence
- **Low (0-59%)**: Weak signal, use caution

## Using Predictions

1. **Filter by Confidence**: Focus on high-confidence predictions
2. **Consider Timeframes**: Different strategies for different timeframes
3. **Risk Management**: Always use stop-loss orders
4. **Diversification**: Don't rely on single predictions

## Feature Importance

Understanding which factors influence predictions:

- **RSI**: Momentum indicator
- **Volume**: Market participation
- **Moving Averages**: Trend direction
- **Bollinger Bands**: Volatility levels
                    """
                },
                "performance": {
                    "title": "Performance Analytics Guide",
                    "content": """
# Performance Analytics Deep Dive

## Key Metrics Explained

### Sharpe Ratio
- Measures risk-adjusted returns
- Higher is better (>1.0 is good, >2.0 is excellent)
- Formula: (Return - Risk-free rate) / Volatility

### Sortino Ratio
- Similar to Sharpe but only considers downside volatility
- Better for asymmetric return distributions

### Maximum Drawdown
- Largest peak-to-trough decline
- Lower is better (aim for <10%)
- Critical for risk management

### Value at Risk (VaR)
- Estimated maximum loss over a period
- Usually at 95% or 99% confidence
- Helps set position sizes

## Risk Management

### Position Sizing
- Use VaR to determine maximum position size
- Consider correlation between assets
- Implement stop-loss orders

### Diversification
- Spread risk across multiple assets
- Consider different timeframes
- Balance high-risk/high-reward trades

### Stress Testing
- Test strategies under extreme conditions
- Historical crisis periods
- Black swan events

## Backtesting Best Practices

1. **Out-of-Sample Testing**: Test on unseen data
2. **Walk-Forward Analysis**: Simulate real trading
3. **Transaction Costs**: Include spreads and commissions
4. **Slippage**: Account for price impact
5. **Market Impact**: Consider large order effects
                    """
                }
            }
        }

    def _load_user_progress(self) -> Dict[str, Any]:
        """Load user onboarding progress."""
        progress_file = Path("config/onboarding_progress.json")
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "completed_tours": [],
            "viewed_articles": [],
            "last_updated": datetime.now().isoformat()
        }

    def _save_user_progress(self):
        """Save user onboarding progress."""
        progress_file = Path("config/onboarding_progress.json")
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'w') as f:
            json.dump(self.user_progress, f, indent=2)

    def show_welcome_tour(self):
        """Show the welcome tour for new users."""
        if "welcome" not in self.user_progress["completed_tours"]:
            self._run_tour("welcome")

    def show_contextual_help(self, page: str):
        """Show contextual help for specific pages."""
        if page in self.onboarding_data["tours"]:
            help_content = self.onboarding_data["tours"][page]
            with st.expander(f"💡 Help: {help_content['title']}", expanded=False):
                st.markdown("### Quick Tips:")
                for step in help_content["steps"][:3]:  # Show first 3 tips
                    st.markdown(f"• **{step['title']}**: {step['content']}")

                if st.button("Take Guided Tour", key=f"tour_{page}"):
                    self._run_tour(page)

    def _run_tour(self, tour_name: str):
        """Run an interactive guided tour."""
        if tour_name not in self.onboarding_data["tours"]:
            st.error(f"Tour '{tour_name}' not found")
            return

        tour = self.onboarding_data["tours"][tour_name]

        # Create tour modal
        with st.container():
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.markdown(f"## 🎯 {tour['title']}")

                # Progress indicator
                progress = st.progress(0)
                step_placeholder = st.empty()

                # Tour steps
                for i, step in enumerate(tour["steps"]):
                    progress.progress((i + 1) / len(tour["steps"]))

                    with step_placeholder.container():
                        st.markdown(f"### Step {i + 1} of {len(tour['steps'])}")
                        st.markdown(f"**{step['title']}**")
                        st.markdown(step["content"])

                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if i > 0 and st.button("⬅️ Previous", key=f"prev_{i}"):
                                # Go back to previous step
                                st.rerun()
                        with col2:
                            if st.button("Skip Tour", key=f"skip_{i}"):
                                break
                        with col3:
                            if i < len(tour["steps"]) - 1:
                                if st.button("Next ➡️", key=f"next_{i}"):
                                    continue
                            else:
                                if st.button("Finish Tour 🎉", key=f"finish_{i}"):
                                    self._complete_tour(tour_name)
                                    st.success("Tour completed! 🎉")
                                    break

                progress.empty()

    def _complete_tour(self, tour_name: str):
        """Mark a tour as completed."""
        if tour_name not in self.user_progress["completed_tours"]:
            self.user_progress["completed_tours"].append(tour_name)
            self.user_progress["last_updated"] = datetime.now().isoformat()
            self._save_user_progress()

    def show_help_article(self, article_name: str):
        """Show a help article."""
        if article_name in self.onboarding_data["help_articles"]:
            article = self.onboarding_data["help_articles"][article_name]

            with st.expander(f"📚 {article['title']}", expanded=False):
                st.markdown(article["content"])

                # Mark as viewed
                if article_name not in self.user_progress["viewed_articles"]:
                    self.user_progress["viewed_articles"].append(article_name)
                    self._save_user_progress()

    def get_tooltip(self, element_name: str) -> Optional[str]:
        """Get tooltip text for UI elements."""
        return self.onboarding_data["tooltips"].get(element_name)

    def show_progress_summary(self):
        """Show user progress summary."""
        completed_tours = len(self.user_progress["completed_tours"])
        viewed_articles = len(self.user_progress["viewed_articles"])
        total_tours = len(self.onboarding_data["tours"])
        total_articles = len(self.onboarding_data["help_articles"])

        with st.expander("📊 Learning Progress", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Tours Completed", f"{completed_tours}/{total_tours}")
                progress = completed_tours / total_tours if total_tours > 0 else 0
                st.progress(progress)

            with col2:
                st.metric("Articles Viewed", f"{viewed_articles}/{total_articles}")
                progress = viewed_articles / total_articles if total_articles > 0 else 0
                st.progress(progress)

            if completed_tours < total_tours:
                st.info("💡 Complete guided tours to learn all dashboard features!")

            if viewed_articles < total_articles:
                st.info("📖 Check out help articles for detailed guides!")

    def show_quick_help(self):
        """Show quick help panel."""
        with st.sidebar.expander("❓ Quick Help", expanded=False):
            st.markdown("### Getting Started")
            st.markdown("• Take the **Welcome Tour** for an overview")
            st.markdown("• Use **Contextual Help** on each page")
            st.markdown("• Check **Help Articles** for detailed guides")

            st.markdown("### Keyboard Shortcuts")
            st.markdown("• `Ctrl+K`: Global search")
            st.markdown("• `R`: Refresh data")
            st.markdown("• `H`: Show/hide help")

            st.markdown("### Tips")
            st.markdown("• Hover over elements for tooltips")
            st.markdown("• Use filters to customize views")
            st.markdown("• Set up alerts for important events")

class OnboardingComponent:
    """Streamlit component for user onboarding."""

    def __init__(self):
        self.manager = OnboardingManager()

    def render_welcome_banner(self):
        """Render welcome banner for new users."""
        if not self.manager.user_progress["completed_tours"]:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 1rem;
                margin-bottom: 2rem;
                text-align: center;
            ">
                <h2 style="margin: 0 0 1rem 0;">🎉 Welcome to Forex AI Professional Dashboard!</h2>
                <p style="margin: 0 0 1rem 0; opacity: 0.9;">
                    Get started with our interactive guided tour to explore all features.
                </p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("🚀 Start Welcome Tour", type="primary", use_container_width=True):
                    self.manager.show_welcome_tour()
            with col2:
                if st.button("📚 View Help Articles", use_container_width=True):
                    self.manager.show_help_article("getting_started")
            with col3:
                if st.button("⚙️ Go to Settings", use_container_width=True):
                    st.session_state.current_page = "settings"
                    st.rerun()

    def render_help_button(self, page: str):
        """Render contextual help button."""
        if st.button("❓ Help", key=f"help_{page}", help="Get help for this page"):
            self.manager.show_contextual_help(page)

    def render_progress_indicator(self):
        """Render learning progress indicator."""
        self.manager.show_progress_summary()

    def get_tooltip(self, element_name: str) -> Optional[str]:
        """Get tooltip for UI element."""
        return self.manager.get_tooltip(element_name)