import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pypfopt import expected_returns, risk_models, black_litterman
from pypfopt.efficient_frontier import EfficientFrontier
import riskfolio as rp
import requests
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Asset Allocation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', sans-serif !important; }

.stApp { background-color: #080c14; color: #e8eef7; }

[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(100,160,255,0.25) !important;
    border-radius: 6px !important;
}
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stToggle label { color: #dbeafe !important; }
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(100,160,255,0.25) !important;
    border-radius: 6px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 500 !important;
    font-size: 0.9rem !important; width: 100% !important;
    padding: 0.55rem 1.2rem !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(100,160,255,0.1); gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(180,200,240,0.5) !important;
    border-radius: 0 !important; font-size: 0.85rem !important;
    font-weight: 500 !important; padding: 8px 16px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important; color: #93c5fd !important;
    border-bottom: 2px solid #3b82f6 !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(100,160,255,0.1) !important;
    border-radius: 10px !important; padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #000000 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
}

h1 { color: #ffffff !important; font-weight: 600 !important; font-size: 1.6rem !important; }
h2 { color: #e2eaf7 !important; font-weight: 500 !important; font-size: 1.15rem !important; }
h3 { color: #c8d9f5 !important; font-weight: 500 !important; font-size: 0.95rem !important; }
p, span, li { color: rgba(200,220,255,0.7) !important; }
hr { border-color: rgba(100,160,255,0.1) !important; margin: 1.2rem 0 !important; }

[data-testid="stDataFrame"] {
    border: 1px solid rgba(100,160,255,0.1) !important;
    border-radius: 8px !important; overflow: hidden !important;
}

.stSuccess { background: rgba(16,185,129,0.08) !important; border: 1px solid rgba(16,185,129,0.25) !important; border-radius: 8px !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.25) !important; border-radius: 8px !important; }
.stError   { background: rgba(239,68,68,0.08)  !important; border: 1px solid rgba(239,68,68,0.25)  !important; border-radius: 8px !important; }
.stInfo    { background: rgba(59,130,246,0.08)  !important; border: 1px solid rgba(59,130,246,0.25)  !important; border-radius: 8px !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

BG      = "#080c14"
BG_PLOT = "#0c1220"
CINZA   = "#64748b"
AZUL_1  = "#1d4ed8"
AZUL_2  = "#3b82f6"
AZUL_3  = "#93c5fd"
BRANCO  = "#e8eef7"
VERM    = "#f87171"
AMAR    = "#fbbf24"

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "azul", ["#0c1a3a", "#1d4ed8", "#3b82f6", "#93c5fd", "#dbeafe"]
)

def set_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG_PLOT,
        "axes.edgecolor": "#1e293b", "axes.labelcolor": BRANCO,
        "xtick.color": CINZA, "ytick.color": CINZA, "text.color": BRANCO,
        "grid.color": "#111827", "grid.linewidth": 0.5,
        "legend.facecolor": BG_PLOT, "legend.edgecolor": "#1e293b",
        "legend.fontsize": 8.5, "axes.titlesize": 11,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

set_style()

st.markdown("""
<div style="padding:8px 0 24px 0; border-bottom:1px solid rgba(100,160,255,0.1); margin-bottom:24px;">
    <p style="color:rgba(147,197,253,0.6) !important; font-size:0.72rem;
              text-transform:uppercase; letter-spacing:0.12em; margin:0 0 6px 0;">
        QUANTITATIVE FINANCE
    </p>
    <h1 style="margin:0; font-size:1.75rem; color:white !important; font-weight:600;">
        Asset Allocation <span style="color:#3b82f6;">Otimizado</span>
    </h1>
    <p style="margin:6px 0 0 0; color:rgba(180,210,255,0.45) !important; font-size:0.83rem;">
        Markowitz · Black-Litterman · Walk-Forward · Stress Testing
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configurações")
    st.markdown("---")

    ativos_input = st.text_input(
        "Ativos (separados por ;)",
        value="ITUB4.SA; VALE3.SA; WEGE3.SA; PETR4.SA"
    )

    start_date = st.date_input("Data início", value=pd.to_datetime("2020-01-01"))

    st.markdown("**Taxa livre de risco**")
    metodo_rf = st.radio("", ["NTN-B 2035", "CDI via BCB", "Manual"],
                         label_visibility="collapsed")

    if metodo_rf == "Manual":
        risk_free = st.number_input("Taxa anual (%)", value=11.5, step=0.1) / 100
    elif metodo_rf == "NTN-B 2035":
        ca, cb = st.columns(2)
        ipca_exp  = ca.number_input("IPCA (%)", value=4.5, step=0.1) / 100
        taxa_real = cb.number_input("Real (%)", value=7.2, step=0.1) / 100
        risk_free = (1 + taxa_real) * (1 + ipca_exp) - 1
        st.caption(f"Nominal: **{risk_free:.2%}**")
    else:
        risk_free = None

    st.markdown("---")
    st.markdown("**Restrições de peso**")
    ca, cb = st.columns(2)
    min_w = ca.number_input("Mín (%)", value=0, min_value=0, max_value=20) / 100
    max_w = cb.number_input("Máx (%)", value=100, min_value=20, max_value=100) / 100

    st.markdown("**Rebalanceamento**")
    freq_map   = {"Mensal": "ME", "Trimestral": "QE", "Anual": "YE"}
    freq_label = st.selectbox("", list(freq_map.keys()), label_visibility="collapsed")
    rebal_freq = freq_map[freq_label]
    custo_tx   = st.number_input("Custo por rebal. (%)", value=0.10, step=0.05) / 100

    st.markdown("---")
    usar_bl = st.toggle("Black-Litterman", value=False)

    views_sidebar = {}
    confianca     = 0.5
    if usar_bl:
        st.markdown("**Target prices (12m)**")
        ativos_prev = [a.strip().upper() for a in ativos_input.split(";") if a.strip()]
        for ativo in ativos_prev:
            tp = st.number_input(
                ativo, min_value=0.0,
                value=st.session_state.get(f"tp_{ativo}", 0.0),
                step=0.50, format="%.2f", key=f"tp_{ativo}"
            )
            if tp > 0:
                views_sidebar[ativo] = tp
        confianca = st.slider("Confiança", 0.0, 1.0, 0.5)

    st.markdown("---")
    rodar = st.button("Otimizar portfólio →")

if rodar:
    ativos = [a.strip().upper() for a in ativos_input.split(";") if a.strip()]
    end    = pd.Timestamp.today().strftime("%Y-%m-%d")
    start  = start_date.strftime("%Y-%m-%d")
    FREQ   = 252

    with st.spinner("Baixando dados..."):
        precos = pd.DataFrame()
        for ativo in ativos:
            d = yf.download(ativo, start=start, end=end,
                            auto_adjust=True, progress=False)
            if not d.empty:
                precos[ativo] = d["Close"]
        precos = precos.dropna()

        ibov = yf.download("^BVSP", start=start, end=end,
                           auto_adjust=True, progress=False)
        ibov = ibov["Close"].dropna() if not ibov.empty else pd.Series(dtype=float)

        bova11 = yf.download("BOVA11.SA", start=start, end=end,
                              auto_adjust=True, progress=False)
        bova11 = bova11["Close"].dropna() if not bova11.empty else pd.Series(dtype=float)

    if metodo_rf == "CDI via BCB":
        try:
            url = (
                "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4389/dados"
                f"?formato=json"
                f"&dataInicial={pd.Timestamp(start).strftime('%d/%m/%Y')}"
                f"&dataFinal={pd.Timestamp(end).strftime('%d/%m/%Y')}"
            )
            df_cdi = pd.DataFrame(requests.get(url, timeout=10).json())
            df_cdi["valor"] = df_cdi["valor"].str.replace(",", ".").astype(float)
            risk_free = df_cdi["valor"].mean() / 100
        except Exception:
            risk_free = 0.115

    with st.spinner("Calculando..."):
        returns = expected_returns.returns_from_prices(precos)
        mu = expected_returns.mean_historical_return(
            returns, returns_data=True, frequency=FREQ)
        S  = risk_models.CovarianceShrinkage(precos, frequency=FREQ).ledoit_wolf()

    views = {}
    if usar_bl and views_sidebar:
        for ativo, tp in views_sidebar.items():
            if ativo in precos.columns:
                pa = float(precos[ativo].iloc[-1])
                if pa > 0:
                    views[ativo] = (tp / pa) - 1
        try:
            mc   = {a: 1 / len(ativos) for a in ativos}
            delt = black_litterman.market_implied_risk_aversion(precos)
            pri  = black_litterman.market_implied_prior_returns(mc, delt, S)
            blm  = black_litterman.BlackLittermanModel(
                S, pi=pri, absolute_views=views,
                omega="idzorek", view_confidences=[confianca] * len(views)
            )
            mu_final = blm.bl_returns()
            S_final  = blm.bl_cov()
        except Exception:
            mu_final, S_final = mu, S
    else:
        mu_final, S_final = mu, S

    n = len(ativos)
    if min_w * n > 1.0:
        min_w = 0.0

    def otimizar(obj="sharpe"):
        ef = EfficientFrontier(mu_final, S_final)
        if min_w > 0:
            ef.add_constraint(lambda w: w >= min_w)
        if max_w < 1:
            ef.add_constraint(lambda w: w <= max_w)
        try:
            ef.max_sharpe(risk_free_rate=risk_free) if obj == "sharpe" else ef.min_volatility()
            return ef
        except Exception:
            ef2 = EfficientFrontier(mu_final, S_final)
            ef2.max_sharpe(risk_free_rate=risk_free) if obj == "sharpe" else ef2.min_volatility()
            return ef2

    with st.spinner("Otimizando..."):
        ef_sh = otimizar("sharpe")
        w_sh  = ef_sh.clean_weights()
        r_sh, v_sh, s_sh = ef_sh.portfolio_performance(risk_free_rate=risk_free)

        ef_mv = otimizar("minvol")
        w_mv  = ef_mv.clean_weights()
        r_mv, v_mv, s_mv = ef_mv.portfolio_performance(risk_free_rate=risk_free)

        arr_eq = np.array([1 / n] * n)
        w_eq   = {a: 1 / n for a in ativos}
        r_eq   = float(np.dot(arr_eq, mu_final))
        v_eq   = float(np.sqrt(arr_eq @ S_final.values @ arr_eq))
        s_eq   = (r_eq - risk_free) / v_eq

        try:
            prt = rp.Portfolio(returns=returns)
            prt.assets_stats(method_mu="hist", method_cov="ledoit")
            wrpdf = prt.rp_optimization(model="Classic", rm="MV", rf=risk_free, b=None)
            w_rp  = dict(zip(ativos, wrpdf["weights"].values))
            arpr  = np.array([w_rp.get(a, 0) for a in ativos])
            r_rp  = float(np.dot(arpr, mu_final))
            v_rp  = float(np.sqrt(arpr @ S_final.values @ arpr))
            s_rp  = (r_rp - risk_free) / v_rp
        except Exception:
            w_rp, r_rp, v_rp, s_rp = w_eq, r_eq, v_eq, s_eq

        portfolios = {
            "Max Sharpe":       {"pesos": w_sh, "ret": r_sh, "vol": v_sh, "sharpe": s_sh},
            "Min Volatilidade": {"pesos": w_mv, "ret": r_mv, "vol": v_mv, "sharpe": s_mv},
            "Equal Weight":     {"pesos": w_eq, "ret": r_eq, "vol": v_eq, "sharpe": s_eq},
            "Risk Parity":      {"pesos": w_rp, "ret": r_rp, "vol": v_rp, "sharpe": s_rp},
        }

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Alocação", "Correlação", "Fronteira", "Backtest", "Risco", "Walk-Forward"
    ])

    with tab1:
        hhi  = sum(v ** 2 for v in w_sh.values())
        n_ef = 1 / hhi if hhi > 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno Esperado", f"{r_sh:.2%}")
        c2.metric("Volatilidade",     f"{v_sh:.2%}")
        c3.metric("Sharpe",           f"{s_sh:.3f}")
        c4.metric("N Efetivo",        f"{n_ef:.1f} / {n}")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            aloc  = {k: v for k, v in w_sh.items() if v > 0.001}
            nal   = len(aloc)
            cpie  = [CMAP(i / max(nal - 1, 1)) for i in range(nal)]
            fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=BG)
            ax.set_facecolor(BG)
            _, txts, atxts = ax.pie(
                aloc.values(), labels=aloc.keys(), autopct="%1.1f%%",
                colors=cpie, wedgeprops=dict(edgecolor=BG, linewidth=2, width=0.6),
                startangle=90, pctdistance=0.78,
            )
            for t in txts:
                t.set_color(BRANCO); t.set_fontsize(9.5)
            for at in atxts:
                at.set_color("#0c1a3a"); at.set_fontsize(8.5); at.set_fontweight("bold")
            ax.set_title("Composição — Max Sharpe", color=BRANCO, fontweight="500", pad=14, fontsize=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            st.markdown("**Estratégias comparadas**")
            df_comp = pd.DataFrame({
                "Portfólio":    list(portfolios.keys()),
                "Retorno":      [f'{p["ret"]:.2%}'    for p in portfolios.values()],
                "Volatilidade": [f'{p["vol"]:.2%}'    for p in portfolios.values()],
                "Sharpe":       [f'{p["sharpe"]:.3f}' for p in portfolios.values()],
            })
            st.dataframe(df_comp, use_container_width=True, hide_index=True, height=180)
            st.markdown("**Pesos — Max Sharpe**")
            df_p = pd.DataFrame([
                {"Ativo": k, "Peso": f"{v:.1%}"}
                for k, v in sorted(w_sh.items(), key=lambda x: -x[1]) if v > 0.001
            ])
            st.dataframe(df_p, use_container_width=True, hide_index=True, height=200)

        if usar_bl and views:
            st.markdown("---")
            st.markdown("**Views Black-Litterman aplicadas**")
            df_bl = pd.DataFrame([{
                "Ativo": a,
                "Preço Atual": f"R$ {float(precos[a].iloc[-1]):.2f}",
                "Target": f"R$ {views_sidebar[a]:.2f}",
                "Upside": f"{v:.1%}",
                "Retorno BL": f"{float(mu_final[a]):.2%}",
            } for a, v in views.items()])
            st.dataframe(df_bl, use_container_width=True, hide_index=True)

    with tab2:
        corr = returns.corr()
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=BG)
            ax.set_facecolor(BG)
            sns.heatmap(corr, annot=True, cmap=CMAP, fmt=".2f", vmin=-1, vmax=1,
                        ax=ax, linewidths=0.4, linecolor=BG,
                        annot_kws={"color": BRANCO, "fontsize": 8.5})
            ax.set_title("Correlação", color=BRANCO, fontweight="500")
            ax.tick_params(colors=CINZA)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col2:
            dist = 1 - corr.abs()
            np.fill_diagonal(dist.values, 0)
            link = linkage(squareform(dist.values), method="ward")
            ord_ = leaves_list(link)
            co   = corr.iloc[ord_, :].iloc[:, ord_]
            fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=BG)
            ax.set_facecolor(BG)
            sns.heatmap(co, annot=True, cmap=CMAP, fmt=".2f", vmin=-1, vmax=1,
                        ax=ax, linewidths=0.4, linecolor=BG,
                        annot_kws={"color": BRANCO, "fontsize": 8.5})
            ax.set_title("Agrupada por Hierarquia", color=BRANCO, fontweight="500")
            ax.tick_params(colors=CINZA)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        cv   = corr.where(mask).stack()
        c1, c2, c3 = st.columns(3)
        c1.metric("Correlação média", f"{cv.mean():.3f}")
        c2.metric("Mais correlacionado",  str(cv.idxmax()), f"{cv.max():.3f}")
        c3.metric("Menos correlacionado", str(cv.idxmin()), f"{cv.min():.3f}")

    with tab3:
        with st.spinner("Simulando Monte Carlo..."):
            nsim = 3000
            res  = np.zeros((3, nsim))
            for i in range(nsim):
                ww = np.random.dirichlet(np.ones(n))
                r_ = float(np.dot(ww, mu_final))
                v_ = float(np.sqrt(ww @ S_final.values @ ww))
                res[0, i] = r_; res[1, i] = v_
                res[2, i] = (r_ - risk_free) / v_

        fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)
        ax.set_facecolor(BG_PLOT)
        sc = ax.scatter(res[1]*100, res[0]*100, c=res[2], cmap=CMAP, alpha=0.3, s=5)
        cb = plt.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Sharpe", color=CINZA, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=CINZA, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=CINZA)
        mkrs = {"Max Sharpe": ("*",AZUL_3,320), "Min Volatilidade": ("D","#bfdbfe",100),
                "Equal Weight": ("s",BRANCO,100), "Risk Parity": ("^",AMAR,100)}
        for nome, p in portfolios.items():
            m, c, sz = mkrs[nome]
            ax.scatter(p["vol"]*100, p["ret"]*100, marker=m, color=c, s=sz,
                       edgecolors=BG, linewidth=0.8, label=nome, zorder=5)
        ax.set_xlabel("Volatilidade (%)")
        ax.set_ylabel("Retorno Esperado (%)")
        ax.set_title("Fronteira Eficiente + Monte Carlo", color=BRANCO, fontweight="500")
        ax.legend(framealpha=0.6)
        ax.grid(alpha=0.12)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab4:
        with st.spinner("Calculando backtest..."):
            rd    = precos.pct_change().dropna()
            warr  = np.array([w_sh.get(a, 0) for a in rd.columns])
            pstat = (1 + rd @ warr).cumprod()
            drb   = rd.resample(rebal_freq).first().index
            val   = 1.0; pac = warr.copy(); vals = []
            for dt in rd.index:
                if dt in drb:
                    val *= (1 - np.sum(np.abs(pac - warr)) * custo_tx)
                    pac  = warr.copy()
                rd_ = rd.loc[dt].values
                ret_d = float(np.dot(pac, rd_))
                pac = pac * (1 + rd_); pac /= pac.sum()
                val *= (1 + ret_d); vals.append(val)
            prebal = pd.Series(vals, index=rd.index)
            cdid   = (1 + risk_free) ** (1/252) - 1
            cdicm  = (1 + pd.Series(cdid, index=prebal.index)).cumprod()
            ibovc  = (1 + ibov.pct_change().dropna()).cumprod()
            bvavc  = (1 + bova11.pct_change().dropna()).cumprod()

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG,
                                 gridspec_kw={"hspace": 0.4})
        for ax in axes:
            ax.set_facecolor(BG_PLOT)
        axes[0].plot((prebal-1)*100,  color=AZUL_3, lw=1.8, label="Portfólio (rebalanceado)")
        axes[0].plot((pstat-1)*100,   color=AZUL_2, lw=1.2, ls="--", alpha=0.7, label="Buy-and-hold")
        axes[0].plot((ibovc.reindex(prebal.index).ffill()-1)*100, color=CINZA, lw=1, alpha=0.6, label="IBOV")
        axes[0].plot((bvavc.reindex(prebal.index).ffill()-1)*100, color="#94a3b8", lw=1, alpha=0.6, label="BOVA11")
        axes[0].plot((cdicm-1)*100, color=AMAR, lw=1.2, ls=":", alpha=0.8, label="CDI")
        axes[0].set_title("Performance Comparada", color=BRANCO, fontweight="500")
        axes[0].set_ylabel("Retorno Acumulado (%)")
        axes[0].legend(framealpha=0.5, ncol=3)
        axes[0].grid(alpha=0.1)
        rans  = prebal.resample("YE").last().pct_change().dropna() * 100
        cbar  = [AZUL_2 if v >= 0 else VERM for v in rans.values]
        axes[1].bar(rans.index.year, rans.values, color=cbar, edgecolor=BG, alpha=0.85, width=0.6)
        axes[1].axhline(0, color=CINZA, lw=0.7)
        axes[1].set_title("Retorno Anual (%)", color=BRANCO, fontweight="500")
        axes[1].set_ylabel("Retorno (%)")
        axes[1].grid(axis="y", alpha=0.1)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        c1, c2, c3 = st.columns(3)
        c1.metric("Retorno acumulado (rebalanceado)", f"{prebal.iloc[-1]-1:.2%}")
        c2.metric("Retorno acumulado (buy-and-hold)", f"{pstat.iloc[-1]-1:.2%}")
        c3.metric("Impacto do custo", f"{(prebal.iloc[-1]/pstat.iloc[-1])-1:.2%}")

    with tab5:
        prd   = prebal.pct_change().dropna()
        rmax  = prebal.cummax()
        dd    = (prebal / rmax) - 1
        maxdd = dd.min()
        CONF  = 0.95
        varh  = np.percentile(prd, (1-CONF)*100)
        cvarh = prd[prd <= varh].mean()
        varp  = norm.ppf(1-CONF, prd.mean(), prd.std())
        aret  = prebal.iloc[-1] ** (252/len(prd)) - 1
        avol  = prd.std() * np.sqrt(252)
        sharr = (aret - risk_free) / avol
        dside = prd[prd < 0].std() * np.sqrt(252)
        sort  = (aret - risk_free) / dside if dside > 0 else np.nan
        calm  = aret / abs(maxdd) if maxdd != 0 else np.nan

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Sharpe", f"{sharr:.3f}")
        c2.metric("Sortino", f"{sort:.3f}")
        c3.metric("Calmar",  f"{calm:.3f}")
        c4.metric("Max Drawdown", f"{maxdd:.2%}")
        c1,c2,c3 = st.columns(3)
        c1.metric("VaR Histórico 95%",   f"{varh:.2%}")
        c2.metric("VaR Paramétrico 95%", f"{varp:.2%}")
        c3.metric("CVaR (ES) 95%",       f"{cvarh:.2%}")
        st.markdown("---")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG,
                                 gridspec_kw={"wspace": 0.3})
        for ax in axes:
            ax.set_facecolor(BG_PLOT)
        axes[0].fill_between(dd.index, dd*100, 0, color=VERM, alpha=0.35)
        axes[0].plot(dd*100, color=VERM, lw=0.7)
        axes[0].set_title("Drawdown (%)", color=BRANCO, fontweight="500")
        axes[0].grid(alpha=0.1)
        axes[1].hist(prd*100, bins=60, color=AZUL_1, alpha=0.7, edgecolor=BG)
        axes[1].axvline(varh*100,  color=AMAR, lw=1.8, ls="--", label=f"VaR 95%: {varh:.2%}")
        axes[1].axvline(cvarh*100, color=VERM, lw=1.8, ls="--", label=f"CVaR: {cvarh:.2%}")
        axes[1].set_title("Distribuição de Retornos", color=BRANCO, fontweight="500")
        axes[1].legend(framealpha=0.5)
        axes[1].grid(alpha=0.1)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")
        st.markdown("**Stress Testing**")
        cens = {
            "COVID Queda (fev-mar/20)":      ("2020-02-01","2020-03-31"),
            "COVID Recuperação (abr-dez/20)":("2020-04-01","2020-12-31"),
            "Juros/Inflação 2022 (1º sem)":  ("2022-01-01","2022-06-30"),
            "Eleições Brasil (out/22)":       ("2022-10-01","2022-10-31"),
            "Crise Americanas (jan-fev/23)":  ("2023-01-01","2023-02-28"),
            "Rally IA 2023":                  ("2023-01-01","2023-12-31"),
        }
        bvs  = (1 + bova11.pct_change().dropna()).cumprod()
        rows = []; rps = []; rbs = []; nms = []
        for nome, (ini,fim) in cens.items():
            try:
                p = prebal.loc[ini:fim]
                if len(p) < 5: continue
                rp_ = p.iloc[-1]/p.iloc[0]-1
                rc  = (1+cdid)**len(p)-1
                b   = bvs.loc[ini:fim]
                rb  = b.iloc[-1]/b.iloc[0]-1 if len(b)>=2 else float("nan")
                rows.append({"Cenário":nome,"Portfólio":f"{rp_:.2%}",
                              "BOVA11":f"{rb:.2%}" if not np.isnan(rb) else "—",
                              "CDI":f"{rc:.2%}","":("🟢" if rp_>0 else "🔴")})
                rps.append(rp_*100); rbs.append(rb*100 if not np.isnan(rb) else 0)
                nms.append(nome)
            except Exception:
                continue
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(11, max(3.5, len(nms)*0.75)), facecolor=BG)
            ax.set_facecolor(BG_PLOT)
            x = np.arange(len(nms)); w = 0.36
            ax.barh(x+w/2, rps, w,
                    color=[AZUL_3 if v>=0 else VERM for v in rps], alpha=0.85, label="Portfólio")
            ax.barh(x-w/2, rbs, w,
                    color=["#bfdbfe" if v>=0 else "#fca5a5" for v in rbs], alpha=0.55, label="BOVA11")
            ax.set_yticks(x); ax.set_yticklabels(nms, fontsize=8.5)
            ax.axvline(0, color=CINZA, lw=0.7)
            ax.set_xlabel("Retorno (%)")
            ax.set_title("Stress Testing — Portfólio vs BOVA11", color=BRANCO, fontweight="500")
            ax.legend(framealpha=0.5); ax.grid(axis="x", alpha=0.1)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Período insuficiente. Tente iniciar em 2020 ou antes.")

    with tab6:
        st.info("Otimiza em janelas de 2 anos e testa nos 3 meses seguintes — elimina look-ahead bias.")
        JT = 504; JTE = 63
        ard = precos.pct_change().dropna(); ntot = len(ard)
        wfr = []; wfi = []
        with st.spinner("Rodando Walk-Forward..."):
            for si in range(0, ntot-JT-JTE, JTE):
                tr = ard.iloc[si:si+JT]; te = ard.iloc[si+JT:si+JT+JTE]
                if len(tr)<JT or len(te)==0: continue
                try:
                    ptr  = precos.iloc[si:si+JT]
                    mwf  = expected_returns.mean_historical_return(tr, returns_data=True, frequency=252)
                    swf  = risk_models.CovarianceShrinkage(ptr, frequency=252).ledoit_wolf()
                    ewf  = EfficientFrontier(mwf, swf)
                    ewf.max_sharpe(risk_free_rate=risk_free)
                    wwf  = ewf.clean_weights()
                    warw = np.array([wwf.get(a, 0) for a in te.columns])
                    wfr.extend((te @ warw).values); wfi.extend(te.index)
                except Exception:
                    continue
        if wfr:
            wfs  = pd.Series(wfr, index=wfi)
            wfc  = (1+wfs).cumprod()
            stp  = pstat.reindex(wfc.index).ffill(); stp = stp/stp.iloc[0]
            fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=BG)
            ax.set_facecolor(BG_PLOT)
            ax.plot(wfc, color=AZUL_3, lw=1.8, label="Walk-Forward (realista)")
            ax.plot(stp, color=CINZA,  lw=1.3, ls="--", label="Buy-and-Hold")
            ax.fill_between(wfc.index, wfc, stp, where=(wfc>=stp), alpha=0.07, color=AZUL_3)
            ax.fill_between(wfc.index, wfc, stp, where=(wfc<stp),  alpha=0.07, color=VERM)
            ax.set_title("Walk-Forward vs Buy-and-Hold", color=BRANCO, fontweight="500")
            ax.set_ylabel("Retorno Acumulado")
            ax.legend(framealpha=0.5); ax.grid(alpha=0.1)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            wa = wfc.iloc[-1]**(252/len(wfs))-1
            wv = wfs.std()*np.sqrt(252)
            ws = (wa-risk_free)/wv
            c1,c2,c3 = st.columns(3)
            c1.metric("Retorno Anualizado",  f"{wa:.2%}")
            c2.metric("Volatilidade",        f"{wv:.2%}")
            c3.metric("Sharpe Walk-Forward", f"{ws:.3f}")
            st.markdown("---")
            sd = s_sh - ws
            c1, c2 = st.columns(2)
            c1.metric("Sharpe otimizado",    f"{s_sh:.3f}")
            c2.metric("Sharpe walk-forward", f"{ws:.3f}", delta=f"{-sd:.3f}", delta_color="inverse")
            if sd < 0.2:   st.success("Modelo muito robusto — diferença menor que 0.2")
            elif sd < 0.5: st.warning("Nível aceitável — diferença entre 0.2 e 0.5")
            else:          st.error("Backtest inflado — diferença maior que 0.5")
        else:
            st.warning("Dados insuficientes. Tente um período mais longo (mínimo 3 anos).")
