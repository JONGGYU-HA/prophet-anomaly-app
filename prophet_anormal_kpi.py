import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.make_holidays import make_holidays_df

st.title("📈 Prophet 기반 KPI 이상 탐지")

uploaded_file = st.file_uploader("엑셀 또는 CSV 파일 업로드", type=["xlsx", "csv"])

if uploaded_file:

    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "csv":
        with st.spinner("📊 CSV 데이터를 불러오는 중입니다..."):
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ 데이터 로드 완료")
            except Exception as e:
                st.error(f"CSV 로드 실패: {e}")
                st.stop()

    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("시트 선택", xls.sheet_names)

        # 상태 저장
        if "sheet_loaded" not in st.session_state:
            st.session_state.sheet_loaded = False
        if "prev_sheet" not in st.session_state:
            st.session_state.prev_sheet = sheet
        elif st.session_state.prev_sheet != sheet:
            st.session_state.sheet_loaded = False
            st.session_state.prev_sheet = sheet

        df = None

        if sheet and not st.session_state.sheet_loaded:
            with st.spinner(f"📊 '{sheet}' 시트 데이터를 불러오는 중입니다..."):
                try:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    st.session_state.df = df
                    st.session_state.sheet_loaded = True
                    st.success("✅ 데이터 로드 완료")
                except Exception as e:
                    st.error(f"시트 로드 실패: {e}")
                    st.stop()

        elif st.session_state.sheet_loaded:
            df = st.session_state.df

    if df is not None:
        time_col = st.selectbox("시간 컬럼 선택", df.columns, index=0)
        kpi_col = st.selectbox("KPI 컬럼 선택", df.columns, index=1)
        group_col = st.selectbox("그룹핑 컬럼 (예: LNCEL name)", df.columns)

        # ✅ 집계 방식 (MRBTS 등 대단위일 때만)
        group_col_clean = group_col.strip().upper()
        if "MRBTS" in group_col_clean or "LNBTS" in group_col_clean or "NRBTS" in group_col_clean:
            agg_method = st.selectbox("📌 집계 방식 선택", options=["sum", "mean", "median"], index=0)
        else:
            agg_method = None

        if agg_method:
            agg_func = {"sum": np.sum, "mean": np.mean, "median": np.median}[agg_method]
            df_grouped = df.groupby([group_col, time_col])[kpi_col].agg(agg_func).reset_index()
        else:
            df_grouped = df.copy()

        changepoint_str = st.text_input("Change Point 입력 (YYYY-MM-DD HH:MM:SS)", "2025-05-06 12:00:00")
        changepoint = pd.to_datetime(changepoint_str)

        anomaly_threshold = st.slider("이상치 개수 조건 (N개 이상만 표시)", min_value=1, max_value=20, value=3)

        if st.button("실행"):
            df[time_col] = pd.to_datetime(df[time_col])

            anomalous_cells = []
            anomaly_stats = {}

            with st.spinner("분석 중입니다..."):
                for i, group in enumerate(df_grouped[group_col].dropna().unique()):
                    df_loc = df_grouped[df_grouped[group_col] == group]

                    df_prophet = df_loc[[time_col, kpi_col]].rename(columns={time_col: 'ds', kpi_col: 'y'})
                    df_prophet = df_prophet.dropna()

                    if df_prophet.shape[0] < 2:
                        continue

                    if changepoint in df_prophet['ds'].values:
                        holidays_df = make_holidays_df(year_list=[2025], country='KR')
                        model = Prophet(daily_seasonality=True, weekly_seasonality=True,
                                        holidays=holidays_df, changepoints=[changepoint])
                    else:
                        model = Prophet(daily_seasonality=True, weekly_seasonality=True)

                    try:
                        model.fit(df_prophet)
                        future = model.make_future_dataframe(periods=20, freq='H')
                        forecast = model.predict(future)

                        df_merged = pd.merge(df_prophet, forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']],
                                             on='ds', how='left')
                        df_merged['anomaly'] = (df_merged['y'] > df_merged['yhat_upper']) | \
                                               (df_merged['y'] < df_merged['yhat_lower'])

                        df_post_change = df_merged[df_merged['ds'] >= changepoint]
                        num_anomalies = df_post_change['anomaly'].sum()

                        if num_anomalies >= anomaly_threshold:
                            anomalous_cells.append(group)
                            anomaly_stats[group] = int(num_anomalies)

                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
                            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                                            color='lightgray', label='Confidence Interval')
                            ax.plot(df_merged['ds'], df_merged['y'], label='Actual', color='black', alpha=0.6)
                            ax.scatter(df_post_change[df_post_change['anomaly']]['ds'],
                                       df_post_change[df_post_change['anomaly']]['y'],
                                       color='red', label='Anomaly', zorder=5)
                            ax.axvline(changepoint, color='green', linestyle='--', label='Changepoint')
                            ax.set_title(f"[{group}] - Anomaly Detection")
                            ax.set_xlabel("Time")
                            ax.set_ylabel(kpi_col)
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)

                    except Exception as e:
                        st.warning(f"⚠️ {group} 처리 중 오류 발생: {e}")

            st.success("모든 셀 처리 완료 ✅")

            if anomalous_cells:
                st.subheader("📌 이상치가 감지된 셀 목록")
                st.write(pd.DataFrame.from_dict(anomaly_stats, orient='index', columns=['Anomaly Count']))
            else:
                st.info("이상치가 감지된 셀이 없습니다.")
