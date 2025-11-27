# streamlit_dashboard_v5.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Airbnb NYC Value Explorer",
    layout="wide",
)

# --------------------------
# Data loader (cached)
# --------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    
    # 1. 计算百分比残差 (Pct Residual)
    if "residual" in df.columns and "price_pred" in df.columns:
        df["pct_residual"] = df["residual"] / df["price_pred"].replace(0, np.nan)
    else:
        df["pct_residual"] = 0.0

    # 2. 优化 EI 分布 (Log Transform + Re-normalization)
    if "EI" in df.columns:
        df["EI_log"] = np.log1p(df["EI"]) 
        min_ei = df["EI_log"].min()
        max_ei = df["EI_log"].max()
        if max_ei > min_ei:
            df["EI"] = (df["EI_log"] - min_ei) / (max_ei - min_ei)
        else:
            df["EI"] = 0.5 

    # 类型转换
    if "neighbourhood_group" in df.columns:
        df["neighbourhood_group"] = df["neighbourhood_group"].astype(str)
    if "room_type" in df.columns:
        df["room_type"] = df["room_type"].astype(str)
        
    return df

DATA_PATH = "rpi_ei_results.csv" 
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}. Please ensure the CSV file is in the same directory.")
    st.stop()

# --------------------------
# Sidebar & Navigation
# --------------------------
st.sidebar.title("Airbnb NYC Dashboard")

page = st.sidebar.radio(
    "Choose page",
    [
        "Overview",
        "Borough Explorer",
        "Listing Search",
        "Methodology & Notes",
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

# --- 改进 1: 动态显示 Borough Filter ---
# 只有在 'Overview' 页面时，才显示这个多选框
selected_boroughs = []
if "neighbourhood_group" in df.columns:
    all_boroughs = sorted(df["neighbourhood_group"].dropna().unique())
    
    if page == "Overview":
        st.sidebar.markdown("**Region Filter (For Overview Page Only)**")
        selected_boroughs = st.sidebar.multiselect(
            "Select Boroughs",
            options=all_boroughs,
            default=all_boroughs
        )
    else:
        # 如果不在 Overview 页面，默认全选（用于后台逻辑），但不显示控件
        selected_boroughs = all_boroughs

# --- Global Filters (Price & Room Type) ---
# 这些 filter 在所有页面常驻，因为它们会影响 Search 和 Explore 的数据池
st.sidebar.markdown("**Global Settings**")

if "room_type" in df.columns:
    room_types = sorted(df["room_type"].dropna().unique())
    selected_room_types = st.multiselect(
        "Room type",
        options=room_types,
        default=room_types
    )
else:
    selected_room_types = []

if "price" in df.columns:
    try:
        min_p, max_p = float(df["price"].min()), float(df["price"].max())
    except:
        min_p, max_p = 0.0, 1000.0
    price_range = st.slider(
        "Price range",
        min_value=min_p,
        max_value=max_p,
        value=(min_p, max_p)
    )
else:
    price_range = (0.0, 1e9)

# --------------------------
# Logic: Construct Filter Masks
# --------------------------
# 1. Common Mask (Price + Room Type)
mask_common = pd.Series(True, index=df.index)
if selected_room_types:
    mask_common &= df["room_type"].isin(selected_room_types)
if "price" in df.columns:
    mask_common &= (df["price"] >= price_range[0]) & (df["price"] <= price_range[1])

# 2. Borough Mask (For Overview Only)
mask_borough = pd.Series(True, index=df.index)
if selected_boroughs:
    mask_borough &= df["neighbourhood_group"].isin(selected_boroughs)

# Create Datasets
df_overview = df[mask_common & mask_borough].copy() # 受所有 Filter 影响
df_explorer_base = df[mask_common].copy() # 只受 Price/Room 影响，忽略 Sidebar Borough

rpi_col = "RPI_by_neigh_group" if "RPI_by_neigh_group" in df.columns else ("RPI" if "RPI" in df.columns else None)
hover_cols = [c for c in ["id", "neighbourhood_group", "room_type", "price", "residual", "pct_residual", "RPI", "EI"] if c in df.columns]

# --------------------------
# Page: Overview
# --------------------------
if page == "Overview":
    st.title("Airbnb NYC Value Explorer — Overview")
    st.markdown("City-level strategic view. Use the sidebar to filter regions, room types, and prices.")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Listings (Filtered)", len(df_overview))
    
    with col2:
        if "residual" in df_overview.columns:
            # Overview 页面保持 MAD (平均绝对偏差)
            mad = df_overview["residual"].abs().mean()
            st.metric("Avg Price Deviation ($)", f"${mad:.2f}", help="Mean Absolute Deviation (Volatility)")
        else:
            st.metric("Avg Price Deviation", "N/A")

    with col3:
        if "EI" in df_overview.columns:
            st.metric("Mean EI (Log-Scaled)", f"{df_overview['EI'].mean():.2f}")
        else:
            st.metric("Mean EI", "N/A")

    # Hidden Gems
    if ("RPI" in df_overview.columns) and ("EI" in df_overview.columns):
        hidden_gems = ((df_overview["RPI"] < 0) & (df_overview["EI"] >= 0.5)).sum()
        total = len(df_overview)
        ratio = hidden_gems/total if total > 0 else 0
        with col4:
            st.metric("Hidden Gems", f"{hidden_gems} ({ratio:.1%})")

    st.markdown("---")

    # RPI vs EI Scatter
    if ("RPI" in df_overview.columns) and ("EI" in df_overview.columns):
        st.subheader("Market Positioning: RPI vs EI")
        fig_q = px.scatter(
            df_overview,
            x="RPI",
            y="EI",
            color="neighbourhood_group" if "neighbourhood_group" in df_overview.columns else None,
            hover_data=hover_cols,
            opacity=0.6,
            height=600,
            title="Strategic Quadrants (EI is Log-Normalized)"
        )
        fig_q.add_vline(x=0, line_dash="dash", line_color="grey")
        fig_q.add_hline(y=0.5, line_dash="dash", line_color="grey")
        fig_q.update_layout(xaxis_title="RPI (Price Premium)", yaxis_title="EI (Exposure - Log Scaled)")
        st.plotly_chart(fig_q, use_container_width=True)

        st.markdown(
            """
            **Quadrant Interpretation:**
            - **Right-Top: Hot Spot** (high RPI, high EI) - High premium, high demand.
            - **Left-Top: Hidden Gem** (low RPI, high EI) - Undervalued, high demand. **Opportunity.**
            - **Right-Bottom: Overhyped** (high RPI, low EI) - High premium, low demand. **Risk.**
            - **Left-Bottom: Cold Zone** (low RPI, low EI) - Low premium, low demand.
            """
        )

    # st.markdown("---")

    # if rpi_col and ("neighbourhood_group" in df_overview.columns):
    #     c1, c2 = st.columns(2)
    #     with c1:
    #         st.subheader(f"RPI Boxplot by Borough")
    #         fig_box = px.box(
    #             df_overview,
    #             x="neighbourhood_group",
    #             y=rpi_col,
    #             points="outliers",
    #             labels={"neighbourhood_group": "Borough", rpi_col: "RPI"},
    #             height=400,
    #         )
    #         st.plotly_chart(fig_box, use_container_width=True)
    #     with c2:
    #         st.subheader("RPI Density Curve")
    #         fig_hist = px.histogram(
    #             df_overview,
    #             x=rpi_col,
    #             color="neighbourhood_group",
    #             marginal="violin",
    #             opacity=0.6,
    #             histnorm="probability density",
    #             barmode="overlay",
    #             height=400,
    #         )
    #         st.plotly_chart(fig_hist, use_container_width=True)
    #         # ... (前面的代码保持不变) ...

    st.markdown("---")

    # 检查是否有必要的数据列
    rpi_col_to_use = "RPI_by_neigh_group" if "RPI_by_neigh_group" in df_overview.columns else ("RPI" if "RPI" in df_overview.columns else None)
    
    if rpi_col_to_use and ("neighbourhood_group" in df_overview.columns):
        
        # --- 1. 动态计算洞察 (Dynamic Insights) ---
        # 按地区分组计算 RPI 的中位数 (溢价水平) 和 标准差 (波动性)
        stats = df_overview.groupby("neighbourhood_group")[rpi_col_to_use].agg(['median', 'std', 'count'])
        
        # 找出溢价最高和最低的区
        if not stats.empty:
            most_expensive = stats['median'].idxmax()
            most_value = stats['median'].idxmin()
            most_volatile = stats['std'].idxmax()
            
            highest_median = stats['median'].max()
            lowest_median = stats['median'].min()
        else:
            most_expensive = most_value = most_volatile = "N/A"
            highest_median = lowest_median = 0

        # --- 2. 展示文字解读 ---
        st.subheader("Market Analysis by Borough")
        
        # 使用 st.info 或 st.markdown 展示结论
        st.info(f"""
        **💡 Key Takeaways:**
        - **Highest Premium:** **{most_expensive}** has the highest median RPI ({highest_median:.2f}), suggesting listings here generally command a price premium.
        - **Best Value Zone:** **{most_value}** has the lowest median RPI ({lowest_median:.2f}), indicating more undervalued listings relative to the model.
        - **Most Volatile:** **{most_volatile}** shows the widest price variation, implying a mix of extreme bargains and overpriced units.
        """)

        # --- 3. 图表展示 ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**1. Pricing Spread (Volatility)**")
            st.caption("A wider box means less consistent pricing. A higher median line means higher premiums.")
            fig_box = px.box(
                df_overview,
                x="neighbourhood_group",
                y=rpi_col_to_use,
                points="outliers", # 展示离群点
                color="neighbourhood_group", # 自动根据颜色区分，更美观
                labels={"neighbourhood_group": "Borough", rpi_col_to_use: "RPI (Z-Score)"},
                height=400,
            )
            # 隐藏图例以节省空间，因为x轴已经有名字了
            fig_box.update_layout(showlegend=False) 
            st.plotly_chart(fig_box, use_container_width=True)
            
        with c2:
            st.markdown("**2. RPI Distribution Shape**")
            st.caption("Peaks to the left of 0 = Market is undervalued. Peaks to the right = Market is overheated.")
            fig_hist = px.histogram(
                df_overview,
                x=rpi_col_to_use,
                color="neighbourhood_group",
                marginal="violin", # 顶部增加小提琴图，增强视觉信息
                opacity=0.6,
                histnorm="probability density",
                barmode="overlay",
                height=400,
            )
            fig_hist.update_layout(
                xaxis_title="RPI (Price Premium)", 
                yaxis_title="Density",
                legend=dict(orientation="h", y=1.1) # 把图例放到上面，防止遮挡
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("RPI or neighbourhood_group missing — chart not available.")


# --------------------------
# Page: Borough Explorer
# --------------------------
elif page == "Borough Explorer":
    st.title("Borough Explorer")
    
    if "neighbourhood_group" in df_explorer_base.columns:
        all_boroughs = sorted(df_explorer_base["neighbourhood_group"].dropna().unique())
        
        # 页面内的选择器
        selected_borough = st.selectbox("Select Borough to Explore", all_boroughs)
        
        # Filter Logic
        df_b = df_explorer_base[df_explorer_base["neighbourhood_group"] == selected_borough].copy()
        
        st.markdown(f"### Deep Dive: {selected_borough}")
        
        # Metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Listings", len(df_b))
        with c2:
            if "residual" in df_b.columns:
                # 依然使用 MAD 来看价格波动的绝对值
                mad_b = df_b['residual'].abs().mean()
                st.metric("Avg Price Deviation ($)", f"${mad_b:.2f}", help="Avg absolute difference between Actual and Predicted Price")
            else:
                st.metric("Avg Price Deviation", "N/A")
        
        # --- 改进 2: 恢复 Avg % Over/Underprice 指标 ---
        with c3:
            if "pct_residual" in df_b.columns:
                # 使用 mean() (带符号)，而不是 abs()，以显示该区域整体是偏贵还是偏便宜
                avg_pct = df_b['pct_residual'].mean()
                st.metric("Avg % Over/Underprice", f"{avg_pct:.1%}", help="Positive = Borough is overpriced on average. Negative = Undervalued.")
            else:
                st.metric("Avg % Over/Underprice", "N/A")

        st.markdown("---")

        tab1, tab2 = st.tabs(["Price Performance", "Top Listings"])
        
        with tab1:
            if "residual" in df_b.columns:
                st.subheader("Residual Distribution")
                fig_res = px.histogram(df_b, x="residual", nbins=50, marginal="box", color_discrete_sequence=['#636EFA'])
                fig_res.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)

            if "price" in df_b.columns and "price_pred" in df_b.columns:
                st.subheader("Predicted vs Actual")
                fig_pvp = px.scatter(
                    df_b, x="price_pred", y="price", color="room_type",
                    hover_data=["residual", "pct_residual"], opacity=0.6
                )
                max_val = max(df_b["price"].max(), df_b["price_pred"].max())
                fig_pvp.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig_pvp, use_container_width=True)

        with tab2:
            st.subheader("Market Inefficiencies")
            n_show = st.number_input("Rows to show", 5, 50, 10)
            
            df_under = df_b.sort_values("residual").head(n_show) 
            df_over = df_b.sort_values("residual", ascending=False).head(n_show) 
            
            col_disp = [c for c in ["id", "room_type", "price", "price_pred", "residual", "pct_residual", "RPI", "EI"] if c in df_b.columns]

            c_left, c_right = st.columns(2)
            with c_left:
                st.write("**Top Undervalued (Good Deals)**")
                st.dataframe(df_under[col_disp].style.format({"price": "${:.0f}", "price_pred": "${:.0f}", "residual": "${:.1f}", "pct_residual": "{:.1%}", "RPI": "{:.2f}"}))
            
            with c_right:
                st.write("**Top Overpriced (Premium)**")
                st.dataframe(df_over[col_disp].style.format({"price": "${:.0f}", "price_pred": "${:.0f}", "residual": "${:.1f}", "pct_residual": "{:.1%}", "RPI": "{:.2f}"}))
                
    else:
        st.warning("Data missing 'neighbourhood_group'.")

# --------------------------
# Page: Listing Search
# --------------------------
elif page == "Listing Search":
    st.title("Listing Search")
    
    input_id = st.text_input("Enter Listing ID", "")

    if input_id:
        match = df[df["id"].astype(str) == input_id.strip()]
        
        if match.empty:
            st.error("Listing not found in source data.")
        else:
            row = match.iloc[0]
            st.subheader(f"Listing: {row['id']} ({row.get('neighbourhood_group', 'Unknown')})")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Actual Price", f"${row['price']:.0f}")
            m2.metric("Predicted Price", f"${row['price_pred']:.0f}")
            delta_color = "inverse" if row['residual'] > 0 else "normal"
            m3.metric("Residual ($)", f"${row['residual']:.1f}", delta_color="off") 

            m4, m5, m6 = st.columns(3)
            m4.metric("RPI (Z-Score)", f"{row.get('RPI', 0):.2f}")
            m5.metric("% Deviation", f"{row.get('pct_residual', 0):.1%}")
            m6.metric("EI (Log-Scaled)", f"{row.get('EI', 0):.2f}")
            
            st.info(f"Listing is **{row.get('pct_residual', 0):.1%}** {'expensive' if row.get('pct_residual', 0)>0 else 'cheaper'} than market model.")

            st.subheader("Quadrant Context")
            df_plot = df_explorer_base.copy() 
            df_plot['Is Target'] = df_plot['id'].astype(str) == input_id.strip()
            df_plot = df_plot.sort_values('Is Target')
            
            if ("RPI" in df_plot.columns) and ("EI" in df_plot.columns):
                fig = px.scatter(
                    df_plot, x="RPI", y="EI", 
                    color="Is Target",
                    color_discrete_map={True: "red", False: "#cccccc"},
                    opacity=0.6,
                    title="Red dot is the selected listing (Grey: Other listings matching Price/Room filters)"
                )
                fig.add_vline(x=0, line_dash="dash")
                fig.add_hline(y=0.5, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Page: Methodology & Notes
# --------------------------
elif page == "Methodology & Notes":
    st.title("Methodology & Notes")
    st.markdown("""
    ## Data & Features

    - Source: NYC Airbnb listings (cleaned).
    - Key variables used:
      - `price`: nightly price
      - `rating`: review_scores_rating
      - `neighbourhood_group`: borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
      - `room_type`: Entire home / Private room / Shared room / Hotel room
      - `EI` (Exposure Index): based on `reviews_per_month` (and optionally `number_of_reviews`)
      - `neighbourhood_value`: (optional) smoothed neighborhood mean price used in model

    ## Regression model (for RPI construction)

    We estimate a city-wide OLS model to predict price:

    `price ~ rating + neighbourhood_value + C(neighbourhood_group) + C(room_type) + bedrooms + beds + baths`

    - `neighbourhood_value` is a smoothed local mean price (if available), capturing micro-location price levels.
    - From the model we compute `price_pred` (predicted price) and `residual = price - price_pred`.

    ## Rating Premium Index (RPI)

    - Global RPI: z-score of residuals across the whole city:

      `RPI = (residual - mean(residual)) / std(residual)`

    - Within-borough RPI: z-score computed within each `neighbourhood_group`:

      `RPI_by_neigh_group = (residual - mean_borough) / std_borough`

    Interpretation:
    - RPI < 0 => listing priced below model prediction (potentially undervalued)
    - RPI > 0 => listing priced above model prediction (potentially overpriced)

    ## Exposure Index (EI)

    - Use `reviews_per_month` as the primary proxy for exposure (optionally combine with `number_of_reviews`).
    - Normalize EI within each borough with min-max scaling so EI ∈ [0, 1]:

      `EI = (x - min_borough) / (max_borough - min_borough)`

    ## Quadrant Framework

    - Left-Top: **Hidden Gem** (low RPI, high EI)
    - Right-Top: **Hot Spot** (high RPI, high EI)
    - Right-Bottom: **Overhyped** (high RPI, low EI)
    - Left-Bottom: **Cold Zone** (low RPI, low EI)

    This dashboard is designed to allow drilling from city-level views to borough-level exploration and single-listing inspection.
    """)