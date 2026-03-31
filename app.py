import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pypfopt import expected_returns, risk_models, black_litterman
from pypfopt.efficient_frontier import EfficientFrontier
import riskfolio as rp
import requests
import warnings
warnings.filterwarnings('ignore')

# ── Configuração da página ───────────────────────────────────
st.set_page_config(
    page_title='Asset Allocation',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Asset Allocation Otimizado')
st.caption('Modelo quantitativo de otimização de portfólio — Mercado Brasileiro')

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Configurações')

    ativos_input = st.text_input(
        'Ativos (separados por ;)',
        value='ITUB4.SA; VALE3.SA; WEGE3.SA; PETR4.SA'
    )

    start = st.date_input('Data início', value=pd.to_datetime('2020-01-01'))

    st.subheader('Taxa Livre de Risco')
    metodo_rf = st.radio(
        'Método',
        ['NTN-B 2035 (recomendado)', 'CDI histórico via BCB', 'Manual']
    )

    if metodo_rf == 'Manual':
        risk_free = st.number_input('Taxa anual (%)', value=11.5) / 100
    elif metodo_rf == 'NTN-B 2035 (recomendado)':
        ipca_exp  = st.number_input('IPCA esperado (%)', value=4.5) / 100
        taxa_real = st.number_input('Taxa real NTN-B (%)', value=7.2) / 100
        risk_free = (1 + taxa_real) * (1 + ipca_exp) - 1
        st.info(f'Risk-free nominal: {risk_free:.2%}')
    else:
        risk_free = None

    st.subheader('Restrições de Peso')
    min_w = st.slider('Peso mínimo por ativo (%)', 0, 20, 0) / 100
    max_w = st.slider('Peso máximo por ativo (%)', 20, 100, 100) / 100

    st.subheader('Rebalanceamento')
    freq_map   = {'Mensal': 'ME', 'Trimestral': 'QE', 'Anual': 'YE'}
    freq_label = st.selectbox('Frequência', list(freq_map.keys()))
    rebal_freq = freq_map[freq_label]
    custo_tx   = st.number_input('Custo por rebalanceamento (%)', value=0.10) / 100

    usar_bl = st.toggle('Usar Black-Litterman')
    rodar   = st.button('🚀 Otimizar Portfólio', type='primary', use_container_width=True)

# ── Execução ─────────────────────────────────────────────────
if rodar:
    ativos = [a.strip().upper() for a in ativos_input.split(';')]
    end    = pd.Timestamp.today().strftime('%Y-%m-%d')
    start  = start.strftime('%Y-%m-%d')
    FREQ   = 252

    with st.spinner('📡 Baixando dados...'):
        precos = pd.DataFrame()
        for ativo in ativos:
            d = yf.download(ativo, start=start, end=end,
                            auto_adjust=True, progress=False)
            if not d.empty:
                precos[ativo] = d['Close']
        precos = precos.dropna()

        ibov = yf.download('^BVSP', start=start, end=end,
                           auto_adjust=True, progress=False)
        ibov = ibov['Close'].dropna() if not ibov.empty else pd.Series(dtype=float)

        bova11 = yf.download('BOVA11.SA', start=start, end=end,
                              auto_adjust=True, progress=False)
        bova11 = bova11['Close'].dropna() if not bova11.empty else pd.Series(dtype=float)

    if metodo_rf == 'CDI histórico via BCB':
        try:
            url = (
                f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4389/dados'
                f'?formato=json'
                f'&dataInicial={pd.Timestamp(start).strftime("%d/%m/%Y")}'
                f'&dataFinal={pd.Timestamp(end).strftime("%d/%m/%Y")}'
            )
            df_cdi = pd.DataFrame(requests.get(url, timeout=10).json())
            df_cdi['valor'] = df_cdi['valor'].str.replace(',', '.').astype(float)
            risk_free = df_cdi['valor'].mean() / 100
            st.sidebar.success(f'CDI médio do período: {risk_free:.2%}')
        except:
            risk_free = 0.115
            st.sidebar.warning('Falha ao buscar CDI. Usando 11.5%')

    with st.spinner('⚙️ Calculando retornos e covariância...'):
        returns = expected_returns.returns_from_prices(precos)
        mu = expected_returns.mean_historical_return(
            returns, returns_data=True, frequency=FREQ)
        S  = risk_models.CovarianceShrinkage(
            precos, frequency=FREQ).ledoit_wolf()

    if usar_bl:
        st.subheader('🔮 Views Black-Litterman')
        st.caption('Informe o target price de cada ativo.')
        views = {}
        cols  = st.columns(len(ativos))
        for i, ativo in enumerate(ativos):
            preco_atual = float(precos[ativo].iloc[-1])
            with cols[i]:
                tp = st.number_input(
                    f'{ativo} — atual: R$ {preco_atual:.2f}',
                    min_value=0.0, value=0.0, step=0.5,
                    key=f'tp_{ativo}'
                )
                if tp > 0:
                    upside = (tp / preco_atual) - 1
                    views[ativo] = upside
                    st.metric('Upside', f'{upside:.1%}')
        confianca = st.slider('Confiança nas views', 0.0, 1.0, 0.5)

        market_caps = {a: 1 / len(ativos) for a in ativos}
        delta    = black_litterman.market_implied_risk_aversion(precos)
        prior    = black_litterman.market_implied_prior_returns(
            market_caps, delta, S)
        bl_model = black_litterman.BlackLittermanModel(
            S, pi=prior, absolute_views=views,
            omega='idzorek',
            view_confidences=[confianca] * len(views)
        )
        mu_final = bl_model.bl_returns()
        S_final  = bl_model.bl_cov()
    else:
        mu_final = mu
        S_final  = S
        views    = {}

    n = len(ativos)
    if min_w * n > 1.0:
        min_w = 0.0
        st.warning('Peso mínimo ajustado para 0% — restrição era inviável.')

    def otimizar(objetivo='sharpe'):
        ef = EfficientFrontier(mu_final, S_final)
        if min_w > 0:
            ef.add_constraint(lambda w: w >= min_w)
        if max_w < 1:
            ef.add_constraint(lambda w: w <= max_w)
        try:
            if objetivo == 'sharpe':
                ef.max_sharpe(risk_free_rate=risk_free)
            else:
                ef.min_volatility()
            return ef
        except:
            ef2 = EfficientFrontier(mu_final, S_final)
            if objetivo == 'sharpe':
                ef2.max_sharpe(risk_free_rate=risk_free)
            else:
                ef2.min_volatility()
            return ef2

    with st.spinner('🔢 Otimizando portfólios...'):
        ef_sh = otimizar('sharpe')
        w_sh  = ef_sh.clean_weights()
        r_sh, v_sh, s_sh = ef_sh.portfolio_performance(risk_free_rate=risk_free)

        ef_mv = otimizar('minvol')
        w_mv  = ef_mv.clean_weights()
        r_mv, v_mv, s_mv = ef_mv.portfolio_performance(risk_free_rate=risk_free)

        arr_eq = np.array([1/n] * n)
        w_eq   = {a: 1/n for a in ativos}
        r_eq   = float(np.dot(arr_eq, mu_final))
        v_eq   = float(np.sqrt(arr_eq @ S_final.values @ arr_eq))
        s_eq   = (r_eq - risk_free) / v_eq

        try:
            port_rp = rp.Portfolio(returns=returns)
            port_rp.assets_stats(method_mu='hist', method_cov='ledoit')
            w_rp_df = port_rp.rp_optimization(
                model='Classic', rm='MV', rf=risk_free, b=None)
            w_rp    = dict(zip(ativos, w_rp_df['weights'].values))
            arr_rp  = np.array([w_rp.get(a, 0) for a in ativos])
            r_rp    = float(np.dot(arr_rp, mu_final))
            v_rp    = float(np.sqrt(arr_rp @ S_final.values @ arr_rp))
            s_rp    = (r_rp - risk_free) / v_rp
        except:
            w_rp, r_rp, v_rp, s_rp = w_eq, r_eq, v_eq, s_eq

        portfolios = {
            'Max Sharpe':       {'pesos': w_sh, 'ret': r_sh, 'vol': v_sh, 'sharpe': s_sh},
            'Min Volatilidade': {'pesos': w_mv, 'ret': r_mv, 'vol': v_mv, 'sharpe': s_mv},
            'Equal Weight':     {'pesos': w_eq, 'ret': r_eq, 'vol': v_eq, 'sharpe': s_eq},
            'Risk Parity':      {'pesos': w_rp, 'ret': r_rp, 'vol': v_rp, 'sharpe': s_rp},
        }
      # ════════════════════════════════════════════════════════
    # ABAS
    # ════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '🏆 Alocação Ideal',
        '📉 Correlação',
        '🌐 Fronteira Eficiente',
        '📈 Backtest',
        '⚠️ Risco',
        '🔁 Walk-Forward',
    ])

    # ── ABA 1: Alocação Ideal ────────────────────────────────
    with tab1:
        st.subheader('Portfólio Ótimo — Max Sharpe')

        c1, c2, c3, c4 = st.columns(4)
        hhi  = sum(v**2 for v in w_sh.values())
        n_ef = 1 / hhi if hhi > 0 else 0
        c1.metric('Retorno Esperado', f'{r_sh:.2%}')
        c2.metric('Volatilidade',     f'{v_sh:.2%}')
        c3.metric('Sharpe Ratio',     f'{s_sh:.3f}')
        c4.metric('N Efetivo',        f'{n_ef:.1f} de {n}')

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            alocacao = {k: v for k, v in w_sh.items() if v > 0.001}
            fig_pie, ax = plt.subplots(figsize=(5, 5))
            cores = plt.cm.Set2(np.linspace(0, 1, len(alocacao)))
            ax.pie(alocacao.values(), labels=alocacao.keys(),
                   autopct='%1.1f%%', colors=cores,
                   wedgeprops=dict(edgecolor='white', linewidth=2))
            ax.set_title('Composição do Portfólio', fontweight='bold')
            st.pyplot(fig_pie)

        with col2:
            df_comp = pd.DataFrame({
                'Portfólio':    list(portfolios.keys()),
                'Retorno':      [f'{p["ret"]:.2%}'   for p in portfolios.values()],
                'Volatilidade': [f'{p["vol"]:.2%}'   for p in portfolios.values()],
                'Sharpe':       [f'{p["sharpe"]:.3f}' for p in portfolios.values()],
            })
            st.markdown('**Comparação de estratégias**')
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown('**Pesos — Max Sharpe**')
            df_pesos = pd.DataFrame([
                {'Ativo': k, 'Peso': f'{v:.1%}'}
                for k, v in sorted(w_sh.items(), key=lambda x: -x[1])
                if v > 0.001
            ])
            st.dataframe(df_pesos, use_container_width=True, hide_index=True)

    # ── ABA 2: Correlação ────────────────────────────────────
    with tab2:
        corr = returns.corr()
        col1, col2 = st.columns(2)

        with col1:
            fig_c, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap='coolwarm',
                        fmt='.2f', vmin=-1, vmax=1,
                        ax=ax, linewidths=0.5)
            ax.set_title('Correlação', fontweight='bold')
            st.pyplot(fig_c)

        with col2:
            dist = 1 - corr.abs()
            np.fill_diagonal(dist.values, 0)
            link  = linkage(squareform(dist.values), method='ward')
            order = leaves_list(link)
            corr_ord = corr.iloc[order, :].iloc[:, order]
            fig_cl, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr_ord, annot=True, cmap='coolwarm',
                        fmt='.2f', vmin=-1, vmax=1,
                        ax=ax2, linewidths=0.5)
            ax2.set_title('Correlação Agrupada por Hierarquia', fontweight='bold')
            st.pyplot(fig_cl)

        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        corr_vals = corr.where(mask).stack()
        col1, col2, col3 = st.columns(3)
        col1.metric('Correlação média', f'{corr_vals.mean():.3f}')
        col2.metric('Par mais correlacionado',
                    str(corr_vals.idxmax()), f'{corr_vals.max():.3f}')
        col3.metric('Par menos correlacionado',
                    str(corr_vals.idxmin()), f'{corr_vals.min():.3f}')

    # ── ABA 3: Fronteira Eficiente ───────────────────────────
    with tab3:
        with st.spinner('Simulando Monte Carlo...'):
            n_sim = 3000
            res   = np.zeros((3, n_sim))
            for i in range(n_sim):
                ww = np.random.dirichlet(np.ones(n))
                r_ = float(np.dot(ww, mu_final))
                v_ = float(np.sqrt(ww @ S_final.values @ ww))
                res[0, i] = r_
                res[1, i] = v_
                res[2, i] = (r_ - risk_free) / v_

        fig_ef, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(res[1]*100, res[0]*100, c=res[2],
                        cmap='viridis', alpha=0.3, s=6)
        plt.colorbar(sc, ax=ax, label='Sharpe')

        mkrs = {
            'Max Sharpe':       ('*', 'gold',   300),
            'Min Volatilidade': ('D', 'cyan',   120),
            'Equal Weight':     ('s', 'white',  120),
            'Risk Parity':      ('^', 'orange', 120),
        }
        for nome, p in portfolios.items():
            m, c, sz = mkrs[nome]
            ax.scatter(p['vol']*100, p['ret']*100, marker=m,
                       color=c, s=sz, edgecolors='black',
                       linewidth=0.8, label=nome, zorder=5)

        ax.set_xlabel('Volatilidade (%)')
        ax.set_ylabel('Retorno Esperado (%)')
        ax.set_title('Fronteira Eficiente + Monte Carlo', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig_ef)

    # ── ABA 4: Backtest ──────────────────────────────────────
    with tab4:
        with st.spinner('Calculando backtest...'):
            ret_d  = precos.pct_change().dropna()
            w_arr  = np.array([w_sh.get(a, 0) for a in ret_d.columns])

            port_static = (1 + ret_d @ w_arr).cumprod()

            datas_rb = ret_d.resample(rebal_freq).first().index
            valor    = 1.0
            p_atual  = w_arr.copy()
            vals     = []

            for data in ret_d.index:
                if data in datas_rb:
                    turnover = np.sum(np.abs(p_atual - w_arr))
                    valor   *= (1 - turnover * custo_tx)
                    p_atual  = w_arr.copy()
                r_d     = ret_d.loc[data].values
                ret_dia = float(np.dot(p_atual, r_d))
                p_atual = p_atual * (1 + r_d)
                p_atual /= p_atual.sum()
                valor   *= (1 + ret_dia)
                vals.append(valor)

            port_rebal = pd.Series(vals, index=ret_d.index)

            cdi_d   = (1 + risk_free) ** (1/252) - 1
            cdi_cum = (1 + pd.Series(cdi_d, index=port_rebal.index)).cumprod()

            ibov_ret = ibov.pct_change().dropna()
            ibov_cum = (1 + ibov_ret).cumprod()

            bova11_ret = bova11.pct_change().dropna()
            bova11_cum = (1 + bova11_ret).cumprod()

        fig_bt, axes = plt.subplots(2, 1, figsize=(12, 9))

        ax = axes[0]
        ax.plot((port_rebal - 1)*100,    label='Portfólio (rebalanceado)', linewidth=2)
        ax.plot((port_static - 1)*100,   label='Portfólio (buy-and-hold)', linestyle='--')
        ax.plot((ibov_cum.reindex(port_rebal.index).ffill() - 1)*100,
                label='IBOV', alpha=0.8)
        ax.plot((bova11_cum.reindex(port_rebal.index).ffill() - 1)*100,
                label='BOVA11', alpha=0.8)
        ax.plot((cdi_cum - 1)*100, label='CDI', linestyle=':', linewidth=1.5)
        ax.set_title('Performance Comparada', fontweight='bold')
        ax.set_ylabel('Retorno Acumulado (%)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        ret_anuais = port_rebal.resample('YE').last().pct_change().dropna() * 100
        cores_bar  = ['#2ecc71' if v >= 0 else '#e74c3c' for v in ret_anuais.values]
        ax2.bar(ret_anuais.index.year, ret_anuais.values,
                color=cores_bar, edgecolor='white')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title('Retorno Anual (%)', fontweight='bold')
        ax2.set_ylabel('Retorno (%)')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_bt)

        c1, c2, c3 = st.columns(3)
        c1.metric('Retorno acumulado (rebalanceado)', f'{port_rebal.iloc[-1]-1:.2%}')
        c2.metric('Retorno acumulado (buy-and-hold)', f'{port_static.iloc[-1]-1:.2%}')
        c3.metric('Impacto do custo de transação',
                  f'{(port_rebal.iloc[-1]/port_static.iloc[-1])-1:.2%}')

    # ── ABA 5: Risco ─────────────────────────────────────────
    with tab5:
        port_ret_d  = port_rebal.pct_change().dropna()
        rolling_max = port_rebal.cummax()
        drawdown    = (port_rebal / rolling_max) - 1
        max_dd      = drawdown.min()
        dias_dd     = (drawdown < 0).sum()

        CONF      = 0.95
        var_hist  = np.percentile(port_ret_d, (1 - CONF) * 100)
        cvar_hist = port_ret_d[port_ret_d <= var_hist].mean()
        var_param = norm.ppf(1 - CONF, port_ret_d.mean(), port_ret_d.std())

        ann_ret  = port_rebal.iloc[-1] ** (252 / len(port_ret_d)) - 1
        ann_vol  = port_ret_d.std() * np.sqrt(252)
        sharpe_r = (ann_ret - risk_free) / ann_vol
        downside = port_ret_d[port_ret_d < 0].std() * np.sqrt(252)
        sortino  = (ann_ret - risk_free) / downside if downside > 0 else np.nan
        calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Sharpe Ratio',  f'{sharpe_r:.3f}')
        col2.metric('Sortino Ratio', f'{sortino:.3f}')
        col3.metric('Calmar Ratio',  f'{calmar:.3f}')
        col4.metric('Max Drawdown',  f'{max_dd:.2%}')

        col1, col2, col3 = st.columns(3)
        col1.metric('VaR Histórico 95%',  f'{var_hist:.2%}')
        col2.metric('VaR Paramétrico 95%', f'{var_param:.2%}')
        col3.metric('CVaR (ES) 95%',       f'{cvar_hist:.2%}')

        st.divider()
        fig_r, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].fill_between(drawdown.index, drawdown*100, 0,
                             color='#e74c3c', alpha=0.6)
        axes[0].plot(drawdown*100, color='#c0392b', linewidth=0.8)
        axes[0].set_title('Drawdown (%)', fontweight='bold')
        axes[0].grid(alpha=0.3)

        axes[1].hist(port_ret_d*100, bins=60, color='#3498db',
                     alpha=0.7, edgecolor='white')
        axes[1].axvline(var_hist*100, color='orange', linewidth=2,
                        linestyle='--', label=f'VaR 95%: {var_hist:.2%}')
        axes[1].axvline(cvar_hist*100, color='red', linewidth=2,
                        linestyle='--', label=f'CVaR: {cvar_hist:.2%}')
        axes[1].set_title('Distribuição de Retornos Diários', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_r)

        # Stress Testing
        st.subheader('🧪 Stress Testing')
        cenarios = {
            'COVID — Queda (fev-mar 2020)':        ('2020-02-01', '2020-03-31'),
            'COVID — Recuperação (abr-dez 2020)':  ('2020-04-01', '2020-12-31'),
            'Juros/Inflação 2022 (1º sem)':        ('2022-01-01', '2022-06-30'),
            'Eleições Brasil out/2022':             ('2022-10-01', '2022-10-31'),
            'Crise Americanas jan/2023':            ('2023-01-01', '2023-02-28'),
            'Rally IA 2023':                        ('2023-01-01', '2023-12-31'),
        }
        rows = []
        for nome, (ini, fim) in cenarios.items():
            try:
                p = port_rebal.loc[ini:fim]
                b = bova11_cum.loc[ini:fim]
                if len(p) < 5:
                    continue
                ret_p   = p.iloc[-1] / p.iloc[0] - 1
                ret_b   = b.iloc[-1] / b.iloc[0] - 1 if len(b) > 1 else np.nan
                ret_cdi = (1 + cdi_d) ** len(p) - 1
                rows.append({
                    'Cenário':   nome,
                    'Portfólio': f'{ret_p:.2%}',
                    'BOVA11':    f'{ret_b:.2%}' if not np.isnan(ret_b) else '—',
                    'CDI':       f'{ret_cdi:.2%}',
                    'Resultado': '🟢' if ret_p > 0 else '🔴'
                })
            except:
                continue
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── ABA 6: Walk-Forward ──────────────────────────────────
    with tab6:
        st.info('Otimiza em janelas de 2 anos e testa nos 3 meses seguintes — elimina look-ahead bias.')

        JANELA_TREINO = 504
        JANELA_TESTE  = 63

        all_ret = precos.pct_change().dropna()
        n_total = len(all_ret)

        wf_returns = []
        wf_index   = []

        with st.spinner('🔁 Rodando Walk-Forward...'):
            for si in range(0, n_total - JANELA_TREINO - JANELA_TESTE, JANELA_TESTE):
                treino = all_ret.iloc[si: si + JANELA_TREINO]
                teste  = all_ret.iloc[si + JANELA_TREINO: si + JANELA_TREINO + JANELA_TESTE]
                if len(treino) < JANELA_TREINO or len(teste) == 0:
                    continue
                try:
                    p_tr  = precos.iloc[si: si + JANELA_TREINO]
                    mu_wf = expected_returns.mean_historical_return(
                        treino, returns_data=True, frequency=252)
                    S_wf  = risk_models.CovarianceShrinkage(
                        p_tr, frequency=252).ledoit_wolf()
                    ef_wf = EfficientFrontier(mu_wf, S_wf)
                    ef_wf.max_sharpe(risk_free_rate=risk_free)
                    w_wf  = ef_wf.clean_weights()
                    w_arr_wf = np.array([w_wf.get(a, 0) for a in teste.columns])
                    wf_returns.extend((teste @ w_arr_wf).values)
                    wf_index.extend(teste.index)
                except:
                    continue

        if wf_returns:
            wf_series = pd.Series(wf_returns, index=wf_index)
            wf_cum    = (1 + wf_series).cumprod()

            static_p = port_static.reindex(wf_cum.index).ffill()
            static_p = static_p / static_p.iloc[0]

            fig_wf, ax = plt.subplots(figsize=(12, 5))
            ax.plot(wf_cum, label='Walk-Forward (realista)',
                    linewidth=2, color='#2ecc71')
            ax.plot(static_p, label='Buy-and-Hold (mesmo período)',
                    linewidth=1.5, linestyle='--', color='#3498db')
            ax.set_title('Walk-Forward Validation vs Buy-and-Hold',
                         fontweight='bold')
            ax.set_ylabel('Retorno Acumulado')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig_wf)

            wf_ann = wf_cum.iloc[-1] ** (252 / len(wf_series)) - 1
            wf_vol = wf_series.std() * np.sqrt(252)
            wf_sh  = (wf_ann - risk_free) / wf_vol

            c1, c2, c3 = st.columns(3)
            c1.metric('Retorno Anualizado', f'{wf_ann:.2%}')
            c2.metric('Volatilidade',       f'{wf_vol:.2%}')
            c3.metric('Sharpe Walk-Forward', f'{wf_sh:.3f}')

            sharpe_delta = s_sh - wf_sh
            st.divider()
            st.markdown('**Índice de otimismo do backtest**')
            col1, col2 = st.columns(2)
            col1.metric('Sharpe otimizado (célula 7)', f'{s_sh:.3f}')
            col2.metric('Sharpe walk-forward (realista)', f'{wf_sh:.3f}',
                        delta=f'{-sharpe_delta:.3f}',
                        delta_color='inverse')

            if sharpe_delta < 0.2:
                st.success('✅ Modelo muito robusto — diferença menor que 0.2')
            elif sharpe_delta < 0.5:
                st.warning('⚠️ Nível aceitável de otimismo — diferença entre 0.2 e 0.5')
            else:
                st.error('🔴 Backtest inflado — diferença maior que 0.5, cuidado com as conclusões')
        else:
            st.warning('Dados insuficientes para o Walk-Forward. Tente um período mais longo.')
