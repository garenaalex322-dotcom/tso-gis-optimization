import streamlit as st
import pandas as pd
import numpy as np
import pulp
import pydeck as pdk
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="ГИС Оптимизация ТСО v4", layout="wide", page_icon="🌍")


# --- 1. ЗАГРУЗКА ГРАНИЦ ТАТАРСТАНА (БЕЗОПАСНЫЙ ПАТЧ КООРДИНАТ) ---
@st.cache_data
def get_tatarstan_geojson():
    try:
        with open('tatarstan_districts_osm.geojson', 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Умный патч: Татарстан это Широта~55, Долгота~50. Если X > Y, значит это [Lat, Lon] и его надо перевернуть.
        for feature in data.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') == 'Polygon':
                new_coords = [[[pt[1], pt[0]] if pt[0] > pt[1] else pt for pt in ring] for ring in
                              geom.get('coordinates', [])]
                geom['coordinates'] = new_coords
            elif geom.get('type') == 'MultiPolygon':
                new_coords = [[[[pt[1], pt[0]] if pt[0] > pt[1] else pt for pt in ring] for ring in poly] for poly in
                              geom.get('coordinates', [])]
                geom['coordinates'] = new_coords
        return data
    except Exception as e:
        st.sidebar.warning("Файл tatarstan_districts_osm.geojson не найден. Карта будет без подложки.")
        return None


# --- 2. КЭШИРОВАНИЕ И ИНТЕГРАЦИЯ ДАННЫХ ---
@st.cache_data
def load_data():
    try:
        df_matrix = pd.read_excel('СУПЕР_МАТРИЦА_ЭТАЛОН_ОБЪЕДИНЕННАЯ.xlsx')
        df_matrix['lat_cluster'] = df_matrix['latitude'].round(2)
        df_matrix['lon_cluster'] = df_matrix['longitude'].round(2)

        df_zones = df_matrix.groupby(['Район', 'Населенный_пункт', 'lat_cluster', 'lon_cluster']).agg({
            'acq_date': 'max', 'latitude': 'count', 'Население': 'max'
        }).reset_index().rename(columns={'latitude': 'Кол_во_инцидентов'})

        df_fire = pd.read_excel('Риски_РТ_Для_QGIS.xlsx')
        df_zones = pd.merge(df_zones, df_fire[['Район', 'Индекс_Риска_R']], on='Район', how='left')
        df_zones.rename(columns={'Индекс_Риска_R': 'Индекс_Огня'}, inplace=True)

        df_flood = pd.read_excel('ТОП_РИСКИ_ПАВОДКОВ_ФИНАЛ.xlsx')
        df_flood['lat_cluster'] = df_flood['latitude'].round(2)
        df_flood['lon_cluster'] = df_flood['longitude'].round(2)
        df_flood_cluster = df_flood.groupby(['Район', 'Населенный_пункт', 'lat_cluster', 'lon_cluster'])[
            'Индекс_Риска_F'].max().reset_index()
        df_zones = pd.merge(df_zones, df_flood_cluster, on=['Район', 'Населенный_пункт', 'lat_cluster', 'lon_cluster'],
                            how='left')
        df_zones.rename(columns={'Индекс_Риска_F': 'Индекс_Воды'}, inplace=True)

        df_multi = pd.read_excel('МУЛЬТИРИСК_РТ_ФИНАЛ_QGIS.xlsx')
        df_zones = pd.merge(df_zones, df_multi[['Район', 'Мультириск_ФИНАЛ']], on='Район', how='left')

        try:
            df_towers = pd.read_excel('Анализ_Связи_ПФО_ФИНАЛ (2).xlsx')
            df_towers = df_towers[df_towers['Регион'] == 'Республика Татарстан'].drop_duplicates(
                subset=['Населенный_пункт'])
            df_zones = pd.merge(df_zones,
                                df_towers[['Населенный_пункт', 'До_ближайшей_2G_вышки_км', 'До_ближайшей_4G_вышки_км']],
                                on='Населенный_пункт', how='left')
        except:
            pass

        try:
            df_old = pd.read_excel('Справочник_ТСО_ФИНАЛ.xlsx')
            old_lats = np.radians(df_old['latitude'].values)
            old_lons = np.radians(df_old['longitude'].values)

            def check_siren_radius(lat, lon, radius_km=0.600):
                lat1, lon1 = np.radians(lat), np.radians(lon)
                a = np.sin((old_lats - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(old_lats) * np.sin(
                    (old_lons - lon1) / 2) ** 2
                return "ДА" if np.any(6371.0 * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))) <= radius_km) else "НЕТ"

            df_zones['Старая_сирена'] = df_zones.apply(lambda r: check_siren_radius(r['lat_cluster'], r['lon_cluster']),
                                                       axis=1)
        except:
            df_zones['Старая_сирена'] = "НЕТ"

        df_zones = df_zones.fillna({
            'Индекс_Огня': 0.1, 'Индекс_Воды': 0.1, 'Мультириск_ФИНАЛ': 0.0,
            'До_ближайшей_2G_вышки_км': 10.0, 'До_ближайшей_4G_вышки_км': 10.0, 'acq_date': 'Нет данных'
        }).dropna(subset=['Население'])

        return df_zones
    except Exception as e:
        st.error(f"Ошибка загрузки файлов: {e}")
        return None


# --- 3. МАТЕМАТИЧЕСКАЯ МОДЕЛЬ ИЗ 'моделлллльль_ласт вариант.py' ---
def run_optimization(df_zones, w_fire, w_flood, alpha, budget_large, budget_small, q_min,
                     d2_max, d4_max, p_fire_cell, p_water_cell, p_fire_wire, p_water_wire, digital_bonus):
    # Идентичный каталог из скрипта
    catalog = [
        {"name": "Речевые уст.", "ch": "Проводной", "cost": 100, "cov": 2500, "rel": 0.95, "time": 10, "k_act": 0.90},
        {"name": "Речевые уст.", "ch": "Сотовая", "cost": 120, "cov": 2500, "rel": 0.95, "time": 10, "k_act": 0.90},
        {"name": "Речевые уст.", "ch": "IP", "cost": 80, "cov": 2500, "rel": 0.95, "time": 5, "k_act": 0.90},
        {"name": "Мобильные комп.", "ch": "Сотовая", "cost": 200, "cov": 3000, "rel": 0.98, "time": 20, "k_act": 0.85},
        {"name": "Мобильные комп.", "ch": "Спутник", "cost": 1500, "cov": 3000, "rel": 0.98, "time": 25, "k_act": 0.85},
        {"name": "SMS-оповещение", "ch": "Сотовая", "cost": 50, "cov": 10000, "rel": 0.95, "time": 5, "k_act": 0.80},
        {"name": "Моб. приложения", "ch": "Сотовая", "cost": 100, "cov": 10000, "rel": 0.95, "time": 2, "k_act": 0.80},
        {"name": "Моб. приложения", "ch": "IP", "cost": 30, "cov": 10000, "rel": 0.95, "time": 2, "k_act": 0.80},
        {"name": "ТВ-системы", "ch": "Спутник", "cost": 2000, "cov": 10000, "rel": 0.95, "time": 15, "k_act": 0.40},
        {"name": "ТВ-системы", "ch": "ТВ/радио", "cost": 1000, "cov": 10000, "rel": 0.95, "time": 15, "k_act": 0.40},
        {"name": "Радиовещание", "ch": "Радио", "cost": 150, "cov": 8000, "rel": 0.95, "time": 10, "k_act": 0.50},
        {"name": "Радиовещание", "ch": "Спутник", "cost": 1800, "cov": 8000, "rel": 0.95, "time": 15, "k_act": 0.50},
        {"name": "Радиовещание", "ch": "ТВ/радио", "cost": 800, "cov": 8000, "rel": 0.95, "time": 15, "k_act": 0.50},
        {"name": "БАС (Дроны)", "ch": "Сотовая", "cost": 300, "cov": 4000, "rel": 0.95, "time": 15, "k_act": 0.85},
        {"name": "БАС (Дроны)", "ch": "Спутник", "cost": 450, "cov": 4000, "rel": 0.95, "time": 20, "k_act": 0.85}
    ]

    # Разделяем реальную надежность (для графиков) и бусты (для Симплекса)
    def calc_q_real(equip, r_f, r_w, d2, d4, pop):
        if pop < 50: return 0.0
        rel = equip["rel"]
        if "Сотовая" in equip["ch"]:
            if d2 > d2_max: return 0.0
            if "приложения" in equip["name"].lower() and d4 > d4_max: return 0.0
            rel -= (p_fire_cell * max(0, r_f - 0.1))
            if r_w > 0.5: rel -= p_water_cell
        if "Проводной" in equip["ch"] or "IP" in equip["ch"]:
            rel -= (p_water_wire * r_w + p_fire_wire * r_f)
        return max(0.0, rel)

    def calc_q_boosted(equip, r_f, r_w, d2, d4, pop, q_real):
        if q_real == 0.0: return 0.0
        boost = 0.0
        name, ch = equip["name"], equip["ch"]

        # Интеграция элитных ниш и системы гарантированных мест из оригинального скрипта
        if name == "ТВ-системы" and ch == "Спутник" and pop >= 50000:
            boost = 10.0
        elif name == "ТВ-системы" and ch == "ТВ/радио" and 20000 <= pop < 50000:
            boost = 10.0
        elif name == "Речевые уст." and ch == "Проводной" and 10000 <= pop < 20000:
            boost = 10.0
        elif name == "Речевые уст." and ch == "IP" and 5000 <= pop < 10000:
            boost = 10.0
        elif name == "Речевые уст." and ch == "Сотовая" and 3000 <= pop < 5000:
            boost = 10.0
        elif name == "БАС (Дроны)" and ch == "Спутник" and r_f > 0.80 and d2 > 8.0:
            boost = 10.0
        elif name == "БАС (Дроны)" and ch == "Сотовая" and r_f > 0.80 and d2 <= 8.0:
            boost = 10.0
        elif name == "Мобильные комп." and ch == "Спутник" and r_w > 0.80 and d2 > 8.0 and pop > 500:
            boost = 10.0
        elif name == "Мобильные комп." and ch == "Сотовая" and r_w > 0.80 and d2 <= 8.0 and pop > 500:
            boost = 10.0

        if boost == 0.0:
            if "Моб. приложения" in name:
                if ch == "IP" and d4 <= 3.0:
                    boost = 8.0
                elif ch == "Сотовая" and 3.0 < d4 <= 8.0:
                    boost = 8.0
                elif ch == "Сотовая" and d4 > 8.0 and pop % 3 == 0:
                    boost = 8.0
            elif name == "SMS-оповещение" and ch == "Сотовая":
                if d4 > 8.0 and pop % 3 == 1: boost = 8.0
            elif "Радиовещание" in name:
                if d4 > 8.0 and pop % 3 == 2:
                    if ch == "Радио" and pop < 1000:
                        boost = 8.0
                    elif ch == "ТВ/радио" and 1000 <= pop < 3000:
                        boost = 8.0
                    elif ch == "Спутник" and pop >= 3000:
                        boost = 8.0
            elif "Моб. приложения" in name and ch == "Сотовая":
                boost = 2.0

        return q_real + boost

    prob = pulp.LpProblem("Final_Optimization", pulp.LpMinimize)
    vars_dict, obj_terms, init_risk = {}, [], 0

    for j, row in df_zones.iterrows():
        vars_dict[j] = {}
        fire_idx, water_idx = row.get('Индекс_Огня', 0.0), row.get('Индекс_Воды', 0.0)
        R_base = (w_fire * fire_idx) + (w_flood * water_idx)
        init_risk += R_base

        if row['Старая_сирена'] == "ДА":
            for k in range(len(catalog)):
                prob += pulp.LpVariable(f"z_{j}_{k}", cat=pulp.LpBinary) == 0
            continue

        curr_b = budget_small if row['Население'] <= 500 else budget_large
        valid_keys = []

        for k, equip in enumerate(catalog):
            v = pulp.LpVariable(f"z_{j}_{k}", cat=pulp.LpBinary)
            vars_dict[j][k] = v

            Q_real = calc_q_real(equip, fire_idx, water_idx, row['До_ближайшей_2G_вышки_км'],
                                 row['До_ближайшей_4G_вышки_км'], row['Население'])
            Q_opt = calc_q_boosted(equip, fire_idx, water_idx, row['До_ближайшей_2G_вышки_км'],
                                   row['До_ближайшей_4G_вышки_км'], row['Население'], Q_real)

            ui_bonus = digital_bonus if equip["name"] in ["Моб. приложения", "SMS-оповещение"] else 1.0

            if equip["cost"] <= curr_b and Q_real >= q_min:
                valid_keys.append(k)
                O = min(equip["cov"], row['Население']) * equip["k_act"]
                tf = (60 - equip["time"]) / 60.0
                red = (Q_opt * O * tf * ui_bonus) / (row['Население'] * alpha + 1)
                obj_terms.append(-R_base * red * v)
            else:
                prob += v == 0

        if valid_keys:
            prob += pulp.lpSum([vars_dict[j][k] for k in valid_keys]) == 1
            prob += pulp.lpSum([vars_dict[j][k] for k in range(len(catalog))]) == 1
        else:
            for k in range(len(catalog)): prob += vars_dict[j][k] == 0

    prob += pulp.lpSum(obj_terms)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    report = []
    real_final_risk = 0

    for j, row in df_zones.iterrows():
        fire_idx, water_idx = row['Индекс_Огня'], row['Индекс_Воды']
        if fire_idx >= 0.5 and water_idx >= 0.5:
            th = "МУЛЬТИРИСК"; c = [128, 0, 128, 200]
        elif water_idx > fire_idx:
            th = "ПАВОДОК"; c = [50, 100, 255, 200]
        else:
            th = "ПОЖАР"; c = [255, 50, 50, 200]

        pop_int = int(row['Население'])
        R_base = (w_fire * fire_idx) + (w_flood * water_idx)

        if row['Старая_сирена'] == 'ДА':
            report.append({"Район": row['Район'], "Н.П.": row['Населенный_пункт'], "Широта": row['lat_cluster'],
                           "Долгота": row['lon_cluster'], "Население": pop_int, "Тип угрозы": th,
                           "ТСО": "ОБОРУДОВАНО СТАРОЙ СИРЕНОЙ", "Канал": "-", "Охват": pop_int, "Стоимость": 0,
                           "Надежность": 1.0, "color": [100, 100, 100, 150]})
            continue

        winner_found = False
        node_residual = R_base

        for k, equip in enumerate(catalog):
            if pulp.value(vars_dict[j][k]) is not None and pulp.value(vars_dict[j][k]) > 0.5:
                Q_real = calc_q_real(equip, fire_idx, water_idx, row['До_ближайшей_2G_вышки_км'],
                                     row['До_ближайшей_4G_вышки_км'], pop_int)
                O_final = int(min(equip["cov"], pop_int) * equip["k_act"])
                tf = (60 - equip["time"]) / 60.0

                # Честный остаточный риск (Без бустов)
                red = (Q_real * O_final * tf) / (pop_int * alpha + 1)
                node_residual = R_base - (R_base * red)

                report.append({"Район": row['Район'], "Н.П.": row['Населенный_пункт'], "Широта": row['lat_cluster'],
                               "Долгота": row['lon_cluster'], "Население": pop_int, "Тип угрозы": th,
                               "ТСО": equip['name'], "Канал": equip['ch'], "Охват": O_final, "Стоимость": equip['cost'],
                               "Надежность": round(Q_real, 3), "color": c})
                winner_found = True

        real_final_risk += node_residual

        if not winner_found:
            report.append({"Район": row['Район'], "Н.П.": row['Населенный_пункт'], "Широта": row['lat_cluster'],
                           "Долгота": row['lon_cluster'], "Население": pop_int, "Тип угрозы": th, "ТСО": "ОТБРАКОВАНО",
                           "Канал": "-", "Охват": 0, "Стоимость": 0, "Надежность": 0.0, "color": [50, 50, 50, 150]})

    return pd.DataFrame(report), init_risk, real_final_risk


# --- ИНТЕРФЕЙС STREAMLIT ---
boundary_data = get_tatarstan_geojson()
data_result = load_data()

if data_result is not None:
    st.sidebar.header("⚙️ Глобальные параметры ЛП")
    w_fire = st.sidebar.slider("🔥 Вес риска ПОЖАРОВ (Gamma)", 0.0, 1.0, 0.6, 0.05)
    w_flood = st.sidebar.slider("🌊 Вес риска НАВОДНЕНИЙ (1 - Gamma)", 0.0, 1.0, 0.4, 0.05)
    alpha = st.sidebar.slider("Коэффициент запаса (Alpha)", 0.1, 1.5, 0.9, 0.1)
    budget_large = st.sidebar.number_input("Бюджет крупных сел (у.е.)", 1000, 10000, 5000, 500)
    budget_small = st.sidebar.number_input("Бюджет малых деревень (у.е.)", 50, 1000, 250, 50)
    q_min = st.sidebar.slider("Мин. порог надежности ТСО (Q_min)", 0.1, 0.9, 0.60, 0.05)

    st.sidebar.header("📱 Константы сотовой связи")
    d2_max = st.sidebar.slider("Max дистанция до вышки 2G (км)", 5.0, 50.0, 30.0, 1.0)
    d4_max = st.sidebar.slider("Max дистанция до вышки 4G (км)", 5.0, 50.0, 25.0, 1.0)
    p_fire_cell = st.sidebar.slider("Штраф сотовой сети при пожаре", 0.0, 1.0, 0.20, 0.05)
    p_water_cell = st.sidebar.slider("Штраф сотовой сети при паводке", 0.0, 1.0, 0.15, 0.05)

    st.sidebar.header("🔌 Константы проводных/IP сетей")
    p_water_wire = st.sidebar.slider("Штраф кабелей при паводке", 0.0, 1.0, 0.65, 0.05)
    p_fire_wire = st.sidebar.slider("Штраф кабелей при пожаре", 0.0, 1.0, 0.75, 0.05)

    st.sidebar.header("🚀 Коэффициент автоматизации")
    digital_bonus = st.sidebar.slider("Digital Bonus для Приложений и SMS", 1.0, 3.0, 1.5, 0.1)

    if st.sidebar.button("🚀 ЗАПУСТИТЬ СИМПЛЕКС-МЕТОД", type="primary"):
        with st.spinner("Идет расчет глобального оптимума..."):
            df_res, r_in, r_out = run_optimization(data_result, w_fire, w_flood, alpha, budget_large, budget_small,
                                                   q_min,
                                                   d2_max, d4_max, p_fire_cell, p_water_cell, p_water_wire, p_fire_wire,
                                                   digital_bonus)

        df_res['Вероятность_ошибки'] = 1.0 - df_res['Надежность']
        summary_table = df_res.groupby(['ТСО', 'Канал']).agg(
            Количество=('ТСО', 'count'), Общая_стоимость=('Стоимость', 'sum'),
            Общий_охват=('Охват', 'sum'), Ср_надежность=('Надежность', 'mean'), Ср_ошибка=('Вероятность_ошибки', 'mean')
        ).reset_index()
        summary_table['Система_и_Канал'] = summary_table['ТСО'] + " (" + summary_table['Канал'] + ")"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего кластеров", len(df_res))
        col2.metric("Установлено новых ТСО",
                    len(df_res[~df_res['ТСО'].isin(['ОБОРУДОВАНО СТАРОЙ СИРЕНОЙ', 'ОТБРАКОВАНО'])]))
        col3.metric("Общий бюджет (у.е.)", f"{df_res['Стоимость'].sum():,}")
        col4.metric("Снижение общего риска", f"{r_in:.2f} → {r_out:.2f}",
                    f"-{(((r_in - r_out) / r_in * 100) if r_in > 0 else 0):.1f}%")

        # КАРТА И ДАННЫЕ В TOOLTIP
        layers = []
        if boundary_data:
            layers.append(pdk.Layer("GeoJsonLayer", boundary_data, opacity=0.3, stroked=True, filled=True,
                                    get_fill_color=[100, 150, 200, 20], get_line_color=[100, 100, 100, 150],
                                    line_width_min_pixels=1))
        layers.append(pdk.Layer("ScatterplotLayer", data=df_res[df_res['ТСО'] != 'ОБОРУДОВАНО СТАРОЙ СИРЕНОЙ'],
                                get_position=["Долгота", "Широта"], get_color="color", get_radius=400, pickable=True,
                                filled=True))

        df_old = df_res[df_res['ТСО'] == 'ОБОРУДОВАНО СТАРОЙ СИРЕНОЙ'].copy()
        if not df_old.empty:
            df_old['icon'] = '🚨'
            layers.append(pdk.Layer("ScatterplotLayer", data=df_old, get_position=["Долгота", "Широта"],
                                    get_fill_color=[0, 0, 0, 0], get_line_color=[50, 200, 50, 255], get_radius=600,
                                    stroked=True, line_width_min_pixels=3, pickable=True))
            layers.append(
                pdk.Layer("TextLayer", data=df_old, get_position=["Долгота", "Широта"], get_text="icon", get_size=20,
                          get_alignment_baseline="'bottom'"))

        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=df_res['Широта'].mean(), longitude=df_res['Долгота'].mean(),
                                             zoom=6),
            tooltip={
                "html": "<b>{Н.П.}</b> ({Район})<br/><b>Население:</b> {Население} чел.<br/><b>Угроза:</b> {Тип угрозы}<br/><b>Выбрано:</b> {ТСО} ({Канал})<br/><b>Стоимость:</b> {Стоимость} у.е.<br/><b>Охват:</b> {Охват} чел.<br/><b>Надежность:</b> {Надежность}"}
        ))

        st.subheader("📊 Аналитические графики эффективности")
        g1, g2 = st.columns(2)
        with g1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=summary_table.sort_values('Количество', ascending=False), x='Количество',
                        y='Система_и_Канал', palette='viridis', ax=ax1)
            ax1.set_title('Частота распределения ТСО по регионам', fontsize=12, fontweight='bold')
            st.pyplot(fig1)
        with g2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=summary_table.sort_values('Общий_охват', ascending=False), x='Общий_охват',
                        y='Система_и_Канал', palette='magma', ax=ax2)
            ax2.set_title('Гарантированный охват населения по типам ТСО', fontsize=12, fontweight='bold')
            st.pyplot(fig2)

        st.subheader("📋 Детальный реестр кластеров")
        st.dataframe(df_res, use_container_width=True)
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Скачать итоговый реестр (CSV)", data=csv, file_name="TSO_Optimization_Final.csv",
                           mime="text/csv")
else:
    st.info("Ошибка инициализации. Пожалуйста, проверьте наличие всех файлов Excel в рабочей директории проекта.")