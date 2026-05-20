import streamlit as st
import pandas as pd
import numpy as np
import pulp
import pydeck as pdk
import json
import warnings

warnings.filterwarnings('ignore')

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="ГИС Оптимизация ТСО", layout="wide", page_icon="🌍")


# --- 1. ЗАГРУЗКА ГРАНИЦ ТАТАРСТАНА (ЛОКАЛЬНЫЙ GEOJSON) ---
@st.cache_data
def get_tatarstan_geojson():
    try:
        with open('tatarstan_districts_osm.geojson', 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Патч координат: переворачиваем [Lat, Lon] в [Lon, Lat] для PyDeck
        for feature in data.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') == 'Polygon':
                new_coords = [[[pt[1], pt[0]] for pt in ring] for ring in geom.get('coordinates', [])]
                geom['coordinates'] = new_coords
            elif geom.get('type') == 'MultiPolygon':
                new_coords = [[[[pt[1], pt[0]] for pt in ring] for ring in poly] for poly in
                              geom.get('coordinates', [])]
                geom['coordinates'] = new_coords
        return data
    except Exception as e:
        st.warning(f"Ошибка загрузки границ: {e}")
        return None


# --- 2. КЭШИРОВАНИЕ ДАННЫХ ---
@st.cache_data
def load_data():
    try:
        df_matrix = pd.read_excel('СУПЕР_МАТРИЦА_ЭТАЛОН_ОБЪЕДИНЕННАЯ.xlsx')
        df_matrix['lat_cluster'] = df_matrix['latitude'].round(2)
        df_matrix['lon_cluster'] = df_matrix['longitude'].round(2)

        df_zones = df_matrix.groupby(['Район', 'Населенный_пункт', 'lat_cluster', 'lon_cluster']).agg({
            'acq_date': 'max',
            'latitude': 'count',
            'Население': 'max'
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
            df_towers = df_towers[df_towers['Регион'] == 'Республика Татарстан']
            df_towers = df_towers.drop_duplicates(subset=['Населенный_пункт'])
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
                dlat = old_lats - lat1
                dlon = old_lons - lon1
                a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(old_lats) * np.sin(dlon / 2.0) ** 2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                dist = 6371.0 * c
                return "ДА" if np.any(dist <= radius_km) else "НЕТ"

            df_zones['Старая_сирена'] = df_zones.apply(lambda r: check_siren_radius(r['lat_cluster'], r['lon_cluster']),
                                                       axis=1)
        except:
            df_zones['Старая_сирена'] = "НЕТ"

        df_zones = df_zones.fillna(
            {'Индекс_Огня': 0.0, 'Индекс_Воды': 0.0, 'Мультириск_ФИНАЛ': 0.0, 'До_ближайшей_2G_вышки_км': 10.0,
             'До_ближайшей_4G_вышки_км': 10.0})
        df_zones = df_zones.dropna(subset=['Население'])
        return df_zones
    except Exception as e:
        st.error(f"Ошибка загрузки файлов: {e}")
        return None


# --- 3. ЯДРО ОПТИМИЗАЦИИ ---
def run_model(df_zones, w_fire, w_flood, alpha, small_budget, q_min):
    BUDGET = 5000
    catalog = [
        {"name": "Электросирены", "ch": "Проводной", "cost": 100, "cov": 2000, "rel": 0.95, "time": 20, "k_act": 1.0},
        {"name": "Речевые уст.", "ch": "Проводной", "cost": 100, "cov": 2500, "rel": 0.95, "time": 20, "k_act": 1.0},
        {"name": "Речевые уст.", "ch": "Сотовая", "cost": 200, "cov": 2500, "rel": 0.95, "time": 10, "k_act": 1.0},
        {"name": "Речевые уст.", "ch": "IP", "cost": 30, "cov": 2500, "rel": 0.90, "time": 5, "k_act": 1.0},
        {"name": "Мобильные комп.", "ch": "Сотовая", "cost": 200, "cov": 3000, "rel": 0.98, "time": 12, "k_act": 1.0},
        {"name": "Мобильные комп.", "ch": "Спутник", "cost": 2000, "cov": 3000, "rel": 0.98, "time": 30, "k_act": 1.0},
        {"name": "SMS-оповещение", "ch": "Сотовая", "cost": 120, "cov": 10000, "rel": 0.90, "time": 15, "k_act": 0.85},
        {"name": "Моб. приложения", "ch": "Сотовая", "cost": 200, "cov": 10000, "rel": 0.90, "time": 12, "k_act": 0.85},
        {"name": "Моб. приложения", "ch": "IP", "cost": 30, "cov": 10000, "rel": 0.90, "time": 5, "k_act": 0.85},
        {"name": "ТВ-системы", "ch": "Спутник", "cost": 2000, "cov": 10000, "rel": 0.95, "time": 30, "k_act": 0.25},
        {"name": "ТВ-системы", "ch": "ТВ/радио", "cost": 1000, "cov": 10000, "rel": 0.95, "time": 40, "k_act": 0.25},
        {"name": "Радиовещание", "ch": "Радио", "cost": 150, "cov": 8000, "rel": 0.95, "time": 20, "k_act": 0.25},
        {"name": "Радиовещание", "ch": "Спутник", "cost": 2000, "cov": 8000, "rel": 0.95, "time": 30, "k_act": 0.25},
        {"name": "Радиовещание", "ch": "ТВ/радио", "cost": 1000, "cov": 8000, "rel": 0.95, "time": 40, "k_act": 0.25},
        {"name": "БАС (Дроны)", "ch": "Сотовая", "cost": 300, "cov": 4000, "rel": 0.96, "time": 15, "k_act": 0.95},
        {"name": "БАС (Дроны)", "ch": "Спутник", "cost": 3000, "cov": 4000, "rel": 0.96, "time": 25, "k_act": 0.95}
    ]

    def calc_q_dyn(equip, r_f, r_w, d2, d4, pop):
        rel = equip["rel"]
        if pop < 100 and (
                equip["name"] in ["Радиовещание", "ТВ-системы", "SMS-оповещение", "Моб. приложения", "Электросирены"] or
                equip["ch"] in ["Проводной", "IP"]):
            return 0.0
        if "Сотовая" in equip["ch"]:
            if d2 > 15.0: return 0.0
            if "приложения" in equip["name"].lower() and d4 > 10.0: return 0.0
            rel -= (0.6 * max(0, r_f - 0.1))
            if r_w > 0.5: rel -= 0.3
        if "Проводной" in equip["ch"] or "IP" in equip["ch"]:
            rel -= (0.65 * r_w + 0.75 * r_f)
        return max(0.0, rel)

    prob = pulp.LpProblem("WebGIS_Optimization", pulp.LpMinimize)
    vars_dict, obj_terms, init_risk = {}, [], 0
    zones = df_zones.to_dict('records')

    for j, row in enumerate(zones):
        vars_dict[j] = {}
        fire_idx, water_idx = row.get('Индекс_Огня', 0.0), row.get('Индекс_Воды', 0.0)
        R_base = (w_fire * fire_idx) + (w_flood * water_idx)
        init_risk += R_base

        curr_b = small_budget if row.get('Население', 500) <= 500 else BUDGET
        valid_keys = []

        if row.get('Старая_сирена', 'НЕТ') == 'ДА':
            for k in range(len(catalog)):
                v = pulp.LpVariable(f"z_{j}_{k}", cat=pulp.LpBinary)
                vars_dict[j][k] = v
                prob += v == 0
            continue

        for k, equip in enumerate(catalog):
            v = pulp.LpVariable(f"z_{j}_{k}", cat=pulp.LpBinary)
            vars_dict[j][k] = v
            Q = calc_q_dyn(equip, fire_idx, water_idx, row.get('До_ближайшей_2G_вышки_км', 10),
                           row.get('До_ближайшей_4G_вышки_км', 10), row.get('Население', 500))
            if equip["cost"] <= curr_b and Q >= q_min:
                valid_keys.append(k)
                O = min(equip["cov"], row.get('Население', 500)) * equip["k_act"]
                tf = (60 - equip["time"]) / 60.0
                red = (Q * O * tf) / (row.get('Население', 500) * alpha + 1)
                obj_terms.append(-R_base * red * v)
            else:
                prob += v == 0

        prob += pulp.lpSum([vars_dict[j][k] for k in range(len(catalog))]) <= 1
        if valid_keys:
            prob += pulp.lpSum([vars_dict[j][k] for k in valid_keys]) == 1

    prob += pulp.lpSum(obj_terms)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    report = []
    for j, row in enumerate(zones):
        fire_idx, water_idx = row.get('Индекс_Огня', 0.0), row.get('Индекс_Воды', 0.0)
        if fire_idx >= 0.5 and water_idx >= 0.5:
            th = "МУЛЬТИРИСК"; c = [128, 0, 128, 200]
        elif water_idx > fire_idx:
            th = "ПАВОДОК"; c = [50, 100, 255, 200]
        else:
            th = "ПОЖАР"; c = [255, 50, 50, 200]

        pop_int = int(row.get('Население', 0))

        if row.get('Старая_сирена', 'НЕТ') == 'ДА':
            report.append({"Район": row['Район'], "Н.П.": row['Населенный_пункт'], "Широта": row['lat_cluster'],
                           "Долгота": row['lon_cluster'], "Население": pop_int, "Тип угрозы": th,
                           "ТСО": "ОБОРУДОВАНО СТАРОЙ СИРЕНОЙ", "Канал": "-", "Охват": pop_int, "Стоимость": 0,
                           "Надежность": 1.0, "color": [100, 100, 100, 150]})
            continue

        winner_found = False
        for k, equip in enumerate(catalog):
            if pulp.value(vars_dict[j][k]) is not None and pulp.value(vars_dict[j][k]) > 0.5:
                Q = calc_q_dyn(equip, fire_idx, water_idx, row.get('До_ближайшей_2G_вышки_км', 10),
                               row.get('До_ближайшей_4G_вышки_км', 10), pop_int)
                report.append({"Район": row['Район'], "Н.П.": row['Населенный_пункт'], "Широта": row['lat_cluster'],
                               "Долгота": row['lon_cluster'], "Население": pop_int, "Тип угрозы": th,
                               "ТСО": equip['name'], "Канал": equip['ch'],
                               "Охват": int(min(equip["cov"], pop_int) * equip["k_act"]), "Стоимость": equip['cost'],
                               "Надежность": round(Q, 3), "color": c})
                winner_found = True
        if not winner_found:
            report.append({"Район": row['Район'], "Н.П.": row['Населенный_пункт'], "Широта": row['lat_cluster'],
                           "Долгота": row['lon_cluster'], "Население": pop_int, "Тип угрозы": th, "ТСО": "ОТБРАКОВАНО",
                           "Канал": "-", "Охват": 0, "Стоимость": 0, "Надежность": 0.0, "color": [50, 50, 50, 150]})

    return pd.DataFrame(report), init_risk, init_risk + pulp.value(prob.objective)


# --- ИНТЕРФЕЙС ---
st.title("📡 Web-GIS: Интеллектуальная оптимизация")
boundary_data = get_tatarstan_geojson()

data_result = load_data()

if data_result is not None:
    df_z = data_result
    w_fire = st.sidebar.slider("🔥 Вес риска ПОЖАРОВ", 0.0, 1.0, 0.6, 0.05)
    w_flood = st.sidebar.slider("🌊 Вес риска НАВОДНЕНИЙ", 0.0, 1.0, 0.4, 0.05)
    alpha = st.sidebar.slider("Коэффициент запаса (Alpha)", 0.1, 1.5, 0.9, 0.1)
    small_budget = st.sidebar.number_input("Бюджет малых деревень (<500 чел)", 100, 1000, 250, 50)
    q_min = st.sidebar.slider("Мин. порог надежности (Q_min)", 0.1, 0.9, 0.60, 0.05)

    if st.sidebar.button("🚀 ЗАПУСТИТЬ ОПТИМИЗАЦИЮ", type="primary"):
        df_res, r_in, r_out = run_model(df_z, w_fire, w_flood, alpha, small_budget, q_min)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего кластеров", len(df_res))
        col2.metric("Установлено систем",
                    len(df_res[~df_res['ТСО'].isin(['ОБОРУДОВАНО СТАРОЙ СИРЕНОЙ', 'ОТБРАКОВАНО'])]))
        col3.metric("Общий бюджет (у.е.)", f"{df_res['Стоимость'].sum():,}")
        col4.metric("Снижение риска", f"{r_in:.2f} → {r_out:.2f}", f"-{((r_in - r_out) / r_in * 100):.1f}%")

        # СБОРКА СЛОЕВ КАРТЫ
        layers = []

        if boundary_data:
            boundary_layer = pdk.Layer(
                "GeoJsonLayer",
                boundary_data,
                opacity=0.3,
                stroked=True,
                filled=True,
                get_fill_color=[100, 150, 200, 20],
                get_line_color=[100, 100, 100, 150],
                line_width_min_pixels=1
            )
            layers.append(boundary_layer)

        # Новые ТСО и Отбракованные точки
        points_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_res[df_res['ТСО'] != 'ОБОРУДОВАНО СТАРОЙ СИРЕНОЙ'],
            get_position=["Долгота", "Широта"],
            get_color="color",
            get_radius=400,  # УМЕНЬШЕННЫЙ РАДИУС 400м
            pickable=True,
            filled=True,
        )
        layers.append(points_layer)

        # Выделение старых ТСО (Зеленое кольцо + мигалка)
        df_old = df_res[df_res['ТСО'] == 'ОБОРУДОВАНО СТАРОЙ СИРЕНОЙ'].copy()
        if not df_old.empty:
            df_old['icon'] = '🚨'
            # Кольцо радиусом 600м
            coverage_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_old,
                get_position=["Долгота", "Широта"],
                get_fill_color=[0, 0, 0, 0],
                get_line_color=[50, 200, 50, 255],
                get_radius=600,
                stroked=True,
                line_width_min_pixels=3,
                pickable=True
            )
            layers.append(coverage_layer)

            # Эмодзи
            text_layer = pdk.Layer(
                "TextLayer",
                data=df_old,
                get_position=["Долгота", "Широта"],
                get_text="icon",
                get_size=25,
                get_alignment_baseline="'bottom'",
            )
            layers.append(text_layer)

        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=df_res['Широта'].mean(), longitude=df_res['Долгота'].mean(),
                                             zoom=6),
            tooltip={"html": "<b>{Н.П.}</b><br/>{Тип угрозы}<br/>{ТСО}<br/>Охват: {Охват} чел."}
        ))
        st.dataframe(df_res)
else:
    st.info("Пожалуйста, убедитесь, что все файлы Excel находятся в папке с приложением.")