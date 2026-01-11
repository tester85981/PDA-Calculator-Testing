import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Any, Optional

# ==========================================
# PDA TERMINAL - STREAMLIT APPLICATION
# Prime Drive Affiliates - Quantamental Division
# ==========================================

# Page configuration
st.set_page_config(
    page_title="PDA Terminal | Monte Carlo & Bayesian Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    .stMetric {background-color: #1E1E1E; padding: 15px; border-radius: 5px;}
    h1 {color: #00D9FF; border-bottom: 2px solid #00D9FF; padding-bottom: 10px;}
    h2 {color: #4ECDC4;}
    h3 {color: #FF6B6B;}
    </style>
""", unsafe_allow_html=True)


class PDAMonteCarloEngine:
    """Monte Carlo simulation engine for affiliate offer underwriting."""
    
    def __init__(self, config: Dict[str, Any]):
        self.num_simulations = config.get('num_simulations', 100000)
        self.payout = config.get('payout', 280.00)
        self.initial_budget = config.get('test_budget', 10000.00)
        self.roi_threshold = config.get('roi_threshold', 20.0)
        self.scale_multiplier = config.get('scale_multiplier', 5.0)
        self.cpm_params = config.get('cpm_params', {'mean': 55.0, 'std': 10.0})
        self.cvr_params = config.get('cvr_params', {'alpha': 40, 'beta': 960})
        self.ctr_params = config.get('ctr_params', {'left': 0.008, 'mode': 0.012, 'right': 0.02})
        self.reversal_rate = config.get('reversal_rate', 0.12)
        self.results = {}
        
    def generate_distributions(self, n: int, budget: float) -> Tuple[np.ndarray, ...]:
        cpm = np.random.normal(self.cpm_params['mean'], self.cpm_params['std'], n)
        cpm = np.clip(cpm, 1, None)
        ctr = np.random.triangular(self.ctr_params['left'], self.ctr_params['mode'], self.ctr_params['right'], n)
        ctr = np.clip(ctr, 0.001, 1.0)
        cvr = np.random.beta(self.cvr_params['alpha'], self.cvr_params['beta'], n)
        cvr = np.clip(cvr, 0.0001, 1.0)
        cpc = cpm / (1000 * ctr)
        cpa = cpc / cvr
        adjusted_payout = self.payout * (1 - self.reversal_rate)
        impressions = (budget / cpm) * 1000
        clicks = impressions * ctr
        leads = clicks * cvr
        revenue = leads * adjusted_payout
        profit = revenue - budget
        roi = (profit / budget) * 100
        return (cpm, ctr, cvr, cpc, cpa, impressions, clicks, leads, revenue, adjusted_payout, profit, roi)
    
    def run_simulation(self) -> Dict[str, Any]:
        phase1_sims = int(self.num_simulations * 0.1)
        (cpm1, ctr1, cvr1, cpc1, cpa1, imp1, clk1, lds1, rev1, adj_payout1, prf1, roi1) = \
            self.generate_distributions(phase1_sims, self.initial_budget)
        
        phase1_roi = roi1.mean()
        phase2_sims = self.num_simulations - phase1_sims
        
        if phase1_roi >= self.roi_threshold:
            scaled_budget = self.initial_budget * self.scale_multiplier
            (cpm2, ctr2, cvr2, cpc2, cpa2, imp2, clk2, lds2, rev2, adj_payout2, prf2, roi2) = \
                self.generate_distributions(phase2_sims, scaled_budget)
            cpm = np.concatenate([cpm1, cpm2])
            ctr = np.concatenate([ctr1, ctr2])
            cvr = np.concatenate([cvr1, cvr2])
            cpc = np.concatenate([cpc1, cpc2])
            cpa = np.concatenate([cpa1, cpa2])
            profit = np.concatenate([prf1, prf2])
            roi = np.concatenate([roi1, roi2])
            revenue = np.concatenate([rev1, rev2])
            leads = np.concatenate([lds1, lds2])
            scaled = True
        else:
            (cpm2, ctr2, cvr2, cpc2, cpa2, imp2, clk2, lds2, rev2, adj_payout2, prf2, roi2) = \
                self.generate_distributions(phase2_sims, self.initial_budget)
            cpm = np.concatenate([cpm1, cpm2])
            ctr = np.concatenate([ctr1, ctr2])
            cvr = np.concatenate([cvr1, cvr2])
            cpc = np.concatenate([cpc1, cpc2])
            cpa = np.concatenate([cpa1, cpa2])
            profit = np.concatenate([prf1, prf2])
            roi = np.concatenate([roi1, roi2])
            revenue = np.concatenate([rev1, rev2])
            leads = np.concatenate([lds1, lds2])
            scaled = False
        
        prob_profit = (profit > 0).mean() * 100
        expected_value = profit.mean()
        median_profit = np.median(profit)
        ci_95 = (np.percentile(profit, 2.5), np.percentile(profit, 97.5))
        ci_50 = (np.percentile(profit, 25), np.percentile(profit, 75))
        cvar_95 = profit[profit <= np.percentile(profit, 5)].mean()
        var_95 = np.percentile(profit, 5)
        max_drawdown = profit.min()
        
        self.results = {
            'distributions': {'cpm': cpm, 'ctr': ctr, 'cvr': cvr, 'cpc': cpc, 'cpa': cpa, 
                            'profit': profit, 'roi': roi, 'revenue': revenue, 'leads': leads},
            'metrics': {
                'probability_of_profit': prob_profit, 'expected_value': expected_value,
                'median_profit': median_profit, 'mean_roi': roi.mean(), 'median_roi': np.median(roi),
                'mean_cpa': cpa.mean(), 'median_cpa': np.median(cpa), 'mean_cpc': cpc.mean(),
                'adjusted_payout': self.payout * (1 - self.reversal_rate),
                'reversal_rate': self.reversal_rate, 'ci_95': ci_95, 'ci_50': ci_50,
                'cvar_95': cvar_95, 'var_95': var_95, 'max_drawdown': max_drawdown,
                'std_dev': profit.std(), 'scaled': scaled,
                'final_budget': scaled_budget if scaled else self.initial_budget
            }
        }
        return self.results


class BayesianLiveMonitor:
    """Real-time Bayesian Truth Detection system."""
    
    def __init__(self, priors: Dict[str, np.ndarray], config: Dict[str, Any]):
        self.priors = priors
        self.config = config
        self.target_cpa = config.get('target_cpa', 220.00)
        self.payout = config.get('payout', 280.00)
        self.reversal_rate = config.get('reversal_rate', 0.12)
        self.adjusted_payout = self.payout * (1 - self.reversal_rate)
        self.kill_threshold = config.get('kill_threshold', 0.15)
        self.scale_threshold = config.get('scale_threshold', 0.80)
        self.ci_width_threshold = config.get('ci_width_threshold', 0.10)
        
    def update_with_live_data(self, spend: float, clicks: int, leads: int) -> Dict[str, Any]:
        if clicks == 0 or leads == 0:
            return self._insufficient_data_response(spend, clicks, leads)
        
        observed_ctr = clicks / (spend / self.priors['cpm'].mean() * 1000)
        observed_cvr = leads / clicks
        observed_cpc = spend / clicks
        observed_cpa = spend / leads
        
        # Bayesian updates
        implied_cpm = observed_cpc * 1000 * observed_ctr
        prior_cpm_mean = self.priors['cpm'].mean()
        prior_cpm_var = self.priors['cpm'].var()
        n_observations = max(clicks / 100, 1)
        posterior_cpm_precision = 1/prior_cpm_var + n_observations/(implied_cpm**2 * 0.1)
        posterior_cpm_var = 1 / posterior_cpm_precision
        posterior_cpm_mean = (prior_cpm_mean/prior_cpm_var + n_observations*implied_cpm/(implied_cpm**2 * 0.1)) / posterior_cpm_precision
        posterior_cpm = np.random.normal(posterior_cpm_mean, np.sqrt(posterior_cpm_var), len(self.priors['cpm']))
        posterior_cpm = np.clip(posterior_cpm, 1, None)
        
        estimated_impressions = int((spend / implied_cpm) * 1000)
        prior_ctr_alpha, prior_ctr_beta = 12, 988
        posterior_ctr_alpha = prior_ctr_alpha + clicks
        posterior_ctr_beta = prior_ctr_beta + (estimated_impressions - clicks)
        posterior_ctr = np.random.beta(posterior_ctr_alpha, posterior_ctr_beta, len(self.priors['ctr']))
        posterior_ctr = np.clip(posterior_ctr, 0.001, 1.0)
        
        prior_cvr_alpha, prior_cvr_beta = 40, 960
        posterior_cvr_alpha = prior_cvr_alpha + leads
        posterior_cvr_beta = prior_cvr_beta + (clicks - leads)
        posterior_cvr = np.random.beta(posterior_cvr_alpha, posterior_cvr_beta, len(self.priors['cvr']))
        posterior_cvr = np.clip(posterior_cvr, 0.0001, 1.0)
        
        posterior_cpc = posterior_cpm / (1000 * posterior_ctr)
        posterior_cpa = posterior_cpc / posterior_cvr
        
        prob_success = (posterior_cpa < self.target_cpa).mean()
        cpa_ci_95 = (np.percentile(posterior_cpa, 97.5), np.percentile(posterior_cpa, 2.5))
        ci_width = (cpa_ci_95[0] - cpa_ci_95[1]) / posterior_cpa.mean()
        expected_cpa = posterior_cpa.mean()
        median_cpa = np.median(posterior_cpa)
        
        decision = self._generate_decision(prob_success, ci_width, expected_cpa)
        
        posteriors = {'cpm': posterior_cpm, 'ctr': posterior_ctr, 'cvr': posterior_cvr,
                     'cpc': posterior_cpc, 'cpa': posterior_cpa}
        
        return {
            'decision': decision, 'prob_success': prob_success, 'expected_cpa': expected_cpa,
            'median_cpa': median_cpa, 'target_cpa': self.target_cpa, 'ci_width': ci_width,
            'cpa_ci_95': cpa_ci_95,
            'observed_metrics': {'cpc': observed_cpc, 'cpa': observed_cpa, 
                               'ctr': observed_ctr, 'cvr': observed_cvr},
            'posteriors': posteriors,
            'priors_vs_posterior': {
                'cpm_shift': posterior_cpm.mean() - self.priors['cpm'].mean(),
                'ctr_shift': posterior_ctr.mean() - self.priors['ctr'].mean(),
                'cvr_shift': posterior_cvr.mean() - self.priors['cvr'].mean()
            }
        }
    
    def _generate_decision(self, prob_success: float, ci_width: float, expected_cpa: float) -> str:
        if prob_success < self.kill_threshold:
            return "🔴 KILL"
        if prob_success > self.scale_threshold and ci_width < self.ci_width_threshold:
            return "🟢 SCALE"
        if prob_success > self.scale_threshold and ci_width >= self.ci_width_threshold:
            return "🟡 SCALE (Low Confidence)"
        return "🟡 HOLD"
    
    def _insufficient_data_response(self, spend: float, clicks: int, leads: int) -> Dict[str, Any]:
        return {
            'decision': "⚪ INSUFFICIENT DATA", 'prob_success': None, 'expected_cpa': None,
            'median_cpa': None, 'target_cpa': self.target_cpa, 'ci_width': None,
            'cpa_ci_95': (None, None),
            'observed_metrics': {
                'cpc': None if clicks == 0 else spend / clicks,
                'cpa': None if leads == 0 else spend / leads,
                'ctr': None, 'cvr': None if clicks == 0 else leads / clicks
            },
            'posteriors': None, 'priors_vs_posterior': None
        }


# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================

st.title("🎯 PDA Terminal | Monte Carlo & Bayesian Engine")
st.markdown("**Quantamental Division** • Real-Time Offer Underwriting & Live Monitoring")

# ==========================================
# SIDEBAR - CONFIGURATION
# ==========================================

st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Offer Parameters")
payout = st.sidebar.number_input("Payout ($)", min_value=0.0, value=280.0, step=10.0)
reversal_rate = st.sidebar.slider("Reversal Rate (%)", 0, 30, 12) / 100
target_cpa = st.sidebar.number_input("Target CPA ($)", min_value=0.0, value=220.0, step=10.0)

st.sidebar.subheader("Monte Carlo Settings")
num_simulations = st.sidebar.selectbox("Simulations", [10000, 50000, 100000], index=2)
test_budget = st.sidebar.number_input("Test Budget ($)", min_value=0.0, value=10000.0, step=1000.0)
roi_threshold = st.sidebar.slider("ROI Threshold for Scaling (%)", 0, 50, 20)

st.sidebar.subheader("Distribution Parameters")
cpm_mean = st.sidebar.number_input("CPM Mean ($)", min_value=0.0, value=55.0, step=5.0)
cpm_std = st.sidebar.number_input("CPM Std Dev ($)", min_value=0.0, value=10.0, step=1.0)
ctr_min = st.sidebar.slider("CTR Min (%)", 0.1, 2.0, 0.8) / 100
ctr_mode = st.sidebar.slider("CTR Mode (%)", 0.5, 3.0, 1.2) / 100
ctr_max = st.sidebar.slider("CTR Max (%)", 1.0, 5.0, 2.0) / 100
cvr_mean = st.sidebar.slider("CVR Mean (%)", 1.0, 10.0, 4.0) / 100

# Calculate Beta parameters from mean
cvr_alpha = 40
cvr_beta = int((1 - cvr_mean) / cvr_mean * cvr_alpha)

st.sidebar.subheader("Bayesian Thresholds")
kill_threshold = st.sidebar.slider("Kill Threshold (PoS %)", 5, 30, 15) / 100
scale_threshold = st.sidebar.slider("Scale Threshold (PoS %)", 60, 95, 80) / 100
ci_width_threshold = st.sidebar.slider("CI Width Threshold (%)", 5, 20, 10) / 100

# Build configuration
config = {
    'num_simulations': num_simulations,
    'payout': payout,
    'test_budget': test_budget,
    'roi_threshold': roi_threshold,
    'scale_multiplier': 5.0,
    'cpm_params': {'mean': cpm_mean, 'std': cpm_std},
    'cvr_params': {'alpha': cvr_alpha, 'beta': cvr_beta},
    'ctr_params': {'left': ctr_min, 'mode': ctr_mode, 'right': ctr_max},
    'reversal_rate': reversal_rate
}

# Run simulation button
if st.sidebar.button("🚀 Run Monte Carlo Simulation", type="primary"):
    st.session_state['run_simulation'] = True

# ==========================================
# MONTE CARLO SIMULATION
# ==========================================

if st.session_state.get('run_simulation', False):
    with st.spinner("Running Monte Carlo simulation..."):
        engine = PDAMonteCarloEngine(config)
        results = engine.run_simulation()
        st.session_state['results'] = results
        st.session_state['engine'] = engine

if 'results' in st.session_state:
    results = st.session_state['results']
    metrics = results['metrics']
    
    # ==========================================
    # TOP METRICS
    # ==========================================
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Probability of Profit",
            f"{metrics['probability_of_profit']:.2f}%",
            delta="High Confidence" if metrics['probability_of_profit'] > 70 else "Monitor"
        )
    
    with col2:
        st.metric(
            "Expected Value",
            f"${metrics['expected_value']:,.2f}",
            delta=f"ROI: {metrics['mean_roi']:.1f}%"
        )
    
    with col3:
        if 'bayesian_result' in st.session_state:
            bayesian_result = st.session_state['bayesian_result']
            if bayesian_result['prob_success'] is not None:
                st.metric(
                    "Bayesian Confidence",
                    f"{bayesian_result['prob_success']*100:.1f}%",
                    delta=bayesian_result['decision']
                )
            else:
                st.metric("Bayesian Confidence", "No Live Data", delta="⚪ Awaiting")
        else:
            st.metric("Bayesian Confidence", "No Live Data", delta="⚪ Awaiting")
    
    # ==========================================
    # MONTE CARLO VISUALIZATIONS
    # ==========================================
    
    st.markdown("---")
    st.header("📊 Monte Carlo Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "💰 Risk Metrics", "🎯 Cost Structure", "📋 Summary"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit distribution
            fig_profit = go.Figure()
            fig_profit.add_trace(go.Histogram(
                x=results['distributions']['profit'],
                nbinsx=50,
                name='Profit',
                marker_color='green',
                opacity=0.7
            ))
            fig_profit.add_vline(x=0, line_dash="dash", line_color="red", 
                                annotation_text="Break-even", annotation_position="top")
            fig_profit.add_vline(x=metrics['expected_value'], line_color="blue",
                                annotation_text=f"Mean: ${metrics['expected_value']:.0f}")
            fig_profit.update_layout(
                title="Profit Distribution",
                xaxis_title="Net Profit ($)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_profit, use_container_width=True)
        
        with col2:
            # ROI distribution
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Histogram(
                x=results['distributions']['roi'],
                nbinsx=50,
                name='ROI',
                marker_color='purple',
                opacity=0.7
            ))
            fig_roi.add_vline(x=0, line_dash="dash", line_color="red",
                            annotation_text="0% ROI")
            fig_roi.add_vline(x=metrics['mean_roi'], line_color="blue",
                            annotation_text=f"Mean: {metrics['mean_roi']:.1f}%")
            fig_roi.update_layout(
                title="ROI Distribution",
                xaxis_title="ROI (%)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_roi, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Risk metrics table
            risk_df = pd.DataFrame({
                'Metric': ['Probability of Profit', 'Expected Value', 'CVaR (95%)', 
                          'VaR (95%)', 'Max Drawdown', 'Std Deviation'],
                'Value': [
                    f"{metrics['probability_of_profit']:.2f}%",
                    f"${metrics['expected_value']:,.2f}",
                    f"${metrics['cvar_95']:,.2f}",
                    f"${metrics['var_95']:,.2f}",
                    f"${metrics['max_drawdown']:,.2f}",
                    f"${metrics['std_dev']:,.2f}"
                ]
            })
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Cumulative distribution
            sorted_profit = np.sort(results['distributions']['profit'])
            cumulative = np.arange(1, len(sorted_profit) + 1) / len(sorted_profit) * 100
            
            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(
                x=sorted_profit,
                y=cumulative,
                mode='lines',
                name='Cumulative',
                line=dict(color='navy', width=2)
            ))
            fig_cdf.add_vline(x=0, line_dash="dash", line_color="red")
            fig_cdf.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_cdf.update_layout(
                title="Cumulative Profit Distribution",
                xaxis_title="Net Profit ($)",
                yaxis_title="Cumulative Probability (%)",
                height=400
            )
            st.plotly_chart(fig_cdf, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # CPA distribution
            fig_cpa = go.Figure()
            cpa_data = results['distributions']['cpa']
            cpa_trimmed = cpa_data[cpa_data < np.percentile(cpa_data, 99)]
            fig_cpa.add_trace(go.Histogram(
                x=cpa_trimmed,
                nbinsx=50,
                name='CPA',
                marker_color='teal',
                opacity=0.7
            ))
            fig_cpa.add_vline(x=metrics['mean_cpa'], line_color="blue",
                            annotation_text=f"Mean: ${metrics['mean_cpa']:.2f}")
            fig_cpa.add_vline(x=metrics['adjusted_payout'], line_color="red", line_dash="dash",
                            annotation_text=f"Adj Payout: ${metrics['adjusted_payout']:.2f}")
            fig_cpa.update_layout(
                title="CPA Distribution (Derived: CPC/CVR)",
                xaxis_title="CPA ($)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_cpa, use_container_width=True)
        
        with col2:
            # Cost structure table
            cost_df = pd.DataFrame({
                'Metric': ['Mean CPC', 'Mean CPA', 'Median CPA', 'Base Payout', 
                          'Adjusted Payout', 'Reversal Rate', 'Margin (Payout-CPA)'],
                'Value': [
                    f"${metrics['mean_cpc']:.2f}",
                    f"${metrics['mean_cpa']:,.2f}",
                    f"${metrics['median_cpa']:,.2f}",
                    f"${payout:.2f}",
                    f"${metrics['adjusted_payout']:.2f}",
                    f"{metrics['reversal_rate']*100:.1f}%",
                    f"${metrics['adjusted_payout'] - metrics['mean_cpa']:.2f}"
                ]
            })
            st.dataframe(cost_df, use_container_width=True, hide_index=True)
    
    with tab4:
        # Executive summary
        if metrics['probability_of_profit'] >= 70 and metrics['mean_roi'] >= 15:
            decision = "🟢 APPROVED - High confidence offer"
            decision_color = "green"
        elif metrics['probability_of_profit'] >= 50 and metrics['mean_roi'] >= 0:
            decision = "🟡 CONDITIONAL - Proceed with caution"
            decision_color = "orange"
        else:
            decision = "🔴 REJECTED - Risk exceeds tolerance"
            decision_color = "red"
        
        st.markdown(f"### Recommendation: :{decision_color}[{decision}]")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Financial Metrics**")
            st.write(f"- Expected Value: ${metrics['expected_value']:,.2f}")
            st.write(f"- Mean ROI: {metrics['mean_roi']:.2f}%")
            st.write(f"- Probability of Profit: {metrics['probability_of_profit']:.2f}%")
            
        with col2:
            st.markdown("**Confidence Intervals**")
            st.write(f"- 95% CI: ${metrics['ci_95'][0]:,.2f} to ${metrics['ci_95'][1]:,.2f}")
            st.write(f"- 50% CI: ${metrics['ci_50'][0]:,.2f} to ${metrics['ci_50'][1]:,.2f}")
    
    # ==========================================
    # BAYESIAN LIVE MONITORING
    # ==========================================
    
    st.markdown("---")
    st.header("🔬 Bayesian Live Monitoring")
    st.markdown("Input live campaign data to update posterior distributions in real-time")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        live_spend = st.number_input("Live Spend ($)", min_value=0.0, value=0.0, step=100.0, key="live_spend")
    
    with col2:
        live_clicks = st.number_input("Live Clicks", min_value=0, value=0, step=10, key="live_clicks")
    
    with col3:
        live_leads = st.number_input("Live Leads", min_value=0, value=0, step=1, key="live_leads")
    
    if live_spend > 0 and live_clicks > 0 and live_leads > 0:
        bayesian_config = {
            'target_cpa': target_cpa,
            'payout': payout,
            'reversal_rate': reversal_rate,
            'kill_threshold': kill_threshold,
            'scale_threshold': scale_threshold,
            'ci_width_threshold': ci_width_threshold
        }
        
        monitor = BayesianLiveMonitor(results['distributions'], bayesian_config)
        bayesian_result = monitor.update_with_live_data(live_spend, live_clicks, live_leads)
        st.session_state['bayesian_result'] = bayesian_result
        
        # Display Bayesian results
        st.markdown("### 📊 Bayesian Update Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Decision", bayesian_result['decision'])
        
        with col2:
            if bayesian_result['prob_success'] is not None:
                st.metric("Probability of Success", f"{bayesian_result['prob_success']*100:.2f}%")
        
        with col3:
            if bayesian_result['expected_cpa'] is not None:
                st.metric("Expected CPA", f"${bayesian_result['expected_cpa']:.2f}")
        
        with col4:
            if bayesian_result['ci_width'] is not None:
                st.metric("CI Width", f"{bayesian_result['ci_width']*100:.2f}%")
        
        # Bayesian visualization
        if bayesian_result['posteriors'] is not None:
            st.markdown("### 📈 Prior vs Posterior Distributions")
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('CPM', 'CTR', 'CVR', 'CPC (Derived)', 'CPA (Derived)')
            )
            
            metrics_list = ['cpm', 'ctr', 'cvr', 'cpc', 'cpa']
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
            
            for metric, (row, col) in zip(metrics_list, positions):
                # Prior
                fig.add_trace(
                    go.Histogram(
                        x=results['distributions'][metric],
                        name=f'{metric.upper()} Prior',
                        marker_color='blue',
                        opacity=0.5,
                        nbinsx=30,
                        showlegend=(row==1 and col==1)
                    ),
                    row=row, col=col
                )
                
                # Posterior
                fig.add_trace(
                    go.Histogram(
                        x=bayesian_result['posteriors'][metric],
                        name=f'{metric.upper()} Posterior',
                        marker_color='green',
                        opacity=0.5,
                        nbinsx=30,
                        showlegend=(row==1 and col==1)
                    ),
                    row=row, col=col
                )
                
                # Add observed value for CPC and CPA
                if metric in ['cpc', 'cpa']:
                    observed = bayesian_result['observed_metrics'][metric]
                    fig.add_vline(
                        x=observed,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Obs: ${observed:.2f}",
                        annotation_position="top",
                        row=row, col=col
                    )
                
                # Add target for CPA
                if metric == 'cpa':
                    fig.add_vline(
                        x=target_cpa,
                        line_dash="dot",
                        line_color="purple",
                        annotation_text=f"Target: ${target_cpa:.2f}",
                        annotation_position="top right",
                        row=row, col=col
                    )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title_text="Bayesian Update: Prior (Blue) → Posterior (Green)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Shift analysis
            st.markdown("### 🔄 Prior → Posterior Shifts")
            shift_df = pd.DataFrame({
                'Variable': ['CPM', 'CTR', 'CVR'],
                'Prior Mean': [
                    f"${results['distributions']['cpm'].mean():.2f}",
                    f"{results['distributions']['ctr'].mean()*100:.3f}%",
                    f"{results['distributions']['cvr'].mean()*100:.3f}%"
                ],
                'Posterior Mean': [
                    f"${bayesian_result['posteriors']['cpm'].mean():.2f}",
                    f"{bayesian_result['posteriors']['ctr'].mean()*100:.3f}%",
                    f"{bayesian_result['posteriors']['cvr'].mean()*100:.3f}%"
                ],
                'Shift': [
                    f"{bayesian_result['priors_vs_posterior']['cpm_shift']:+.2f}",
                    f"{bayesian_result['priors_vs_posterior']['ctr_shift']*100:+.3f}%",
                    f"{bayesian_result['priors_vs_posterior']['cvr_shift']*100:+.3f}%"
                ]
            })
            st.dataframe(shift_df, use_container_width=True, hide_index=True)
            
            # Observed metrics
            st.markdown("### 📊 Observed Metrics")
            obs_df = pd.DataFrame({
                'Metric': ['CPC', 'CPA', 'CTR', 'CVR'],
                'Observed Value': [
                    f"${bayesian_result['observed_metrics']['cpc']:.2f}",
                    f"${bayesian_result['observed_metrics']['cpa']:.2f}",
                    f"{bayesian_result['observed_metrics']['ctr']*100:.3f}%",
                    f"{bayesian_result['observed_metrics']['cvr']*100:.3f}%"
                ]
            })
            st.dataframe(obs_df, use_container_width=True, hide_index=True)
    
    elif live_spend > 0 or live_clicks > 0 or live_leads > 0:
        st.warning("⚠️ Please enter non-zero values for all three fields (Spend, Clicks, Leads) to perform Bayesian update")
    
else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Monte Carlo Simulation' to begin")
    
    # Show feature overview
    st.markdown("---")
    st.header("🎯 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Monte Carlo Engine
        - **100,000+ simulations** for institutional accuracy
        - **Structural dependencies**: CPC = CPM/(1000×CTR), CPA = CPC/CVR
        - **Risk metrics**: CVaR, VaR, Max Drawdown
        - **Confidence intervals**: 95% and 50% profit ranges
        - **Auto-scaling logic**: Tests budget scaling when ROI > threshold
        """)
    
    with col2:
        st.markdown("""
        ### Bayesian Live Monitor
        - **Real-time updates** using conjugate priors
        - **Probability of Success** calculation
        - **Kill/Hold/Scale signals** based on thresholds
        - **Confidence interval tracking** for decision certainty
        - **Prior → Posterior visualization** shows belief updates
        """)
    
    st.markdown("---")
    st.markdown("""
    ### How to Use
    1. **Configure** offer parameters and distribution assumptions in the sidebar
    2. **Run** the Monte Carlo simulation to generate prior distributions
    3. **Input** live campaign data (Spend, Clicks, Leads) to trigger Bayesian updates
    4. **Monitor** the Probability of Success and decision signals in real-time
    5. **Scale or Kill** based on the algorithmic recommendations
    """)
