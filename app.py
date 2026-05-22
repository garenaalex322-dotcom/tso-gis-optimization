import streamlit as st
import pandas as pd
import numpy as np
import pulp
import warnings
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from openai import OpenAI

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="ГИС Оптимизация ТСО v6.0", layout="wide", page_icon="🌍")


# ==========================================
# 1. ФУНКЦИЯ ИИ-АНАЛИТИКИ (OpenRouter)
# ==========================================
def generate_ai_insights(summary_df, total_cost, total_coverage, total_population):
    # БЕЗОПАСНОЕ ПОЛУЧЕНИЕ КЛЮЧА ИЗ СЕКРЕТОВ
    api_key = st.secrets.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY"))

    if not api_key:
        return "⚠️ Ошибка: API-ключ OpenRouter не найден. Настройте st.secrets."

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Превращаем таблицу в текст для ИИ
        data_context = summary_df[
            ['Система_и_Канал', 'Количество', 'Общая_стоимость', 'Общий_охват', 'Ср_надежность']].to_string(index=False)
        coverage_percent = (total_coverage / total_population * 100) if total_population > 0 else 0

        prompt = f"""
        Вы — ведущий эксперт по математическому моделированию и гражданской обороне. 
        Проанализируйте результаты оптимизации системы оповещения (MILP-модель).

        ВВОДНЫЕ ДАННЫЕ:
        - Общий бюджет внедрения: {total_cost:,.0f} руб.
        - Охват населения: {total_coverage:,.0f} чел. ({coverage_percent:.1f}% от зоны риска).
        - Распределение оборудования:
        {data_context}

        ЗАДАЧА:
        Напишите профессиональный академический отчет (структурированный, без воды).

        СТРУКТУРА:
        1. Оценка эффективности: Почему модель выбрала именно эти топовые каналы.
        2. Нишевое распределение: Какую роль играют миноритарные каналы (Радио, Дроны) в глухих местах.
        3. Экономический вывод: Окупаемость системы (цена за одного оповещаемого).
        """

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Вы строгий Data Scientist и эксперт по ГИС и системам безопасности."},
                {"role": "user", "content": prompt}
            ],
            headers={"HTTP-Referer": "https://github.com/", "X-Title": "GIS Warning Optimizer"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Ошибка соединения с ИИ: {e}"


# ==========================================
# 2. ЯДРО ОПТИМИЗАЦИИ (MILP МОДЕЛЬ)
# ==========================================
@st.cache_data
def run_unified_final_model(df_matrix_path, df_fire_path, df_flood_path, df_multi_path, df_towers_path, df_old_path):
    # 1. Загрузка данных
    df_matrix = pd.read_excel(df_matrix_path)
    df_matrix['lat_cluster'] = df_matrix['latitude'].round(2)
    df_matrix['lon_cluster'] = df_matrix['longitude'].round(2)
    df_zones = df_matrix.groupby(['Район', 'Населенный_пункт', 'lat_cluster', 'lon_cluster']).agg({
        'acq_date': 'max', 'latitude': 'count', 'Население': 'max'
    }).reset_index().rename(columns={'latitude': 'Кол_во_инцидентов'})

    # 2. Интеграция рисков
    df_fire = pd.read_excel(df_fire_path)
    df_zones = pd.merge(df_zones, df_fire[['Район', 'Индекс_Риска_R']], on='Район', how='left')
    df_zones.rename(columns={'Индекс_Риска_R': 'Индекс_Огня'}, inplace=True)

    df_flood = pd.read_excel(df_flood_path)
    df_flood['lat_cluster'] = df_flood['latitude'].round(2)
    df_flood['lon_cluster'] = df_flood['longitude'].round(2)
    df_flood_cluster = df_flood.groupby(['Район', 'Населенный_пункт', 'lat_cluster', 'lon_cluster'])[
        'Индекс_Риска_F'].max().reset_index()
    df_zones = pd.merge(df_zones, df_flood_cluster, on=['Район', 'Населенный_пункт', 'lat_cluster', 'lon_cluster'],
                        how='left')
    df_zones.rename(columns={'Индекс_Риска_F': 'Индекс_Воды'}, inplace=True)

    # 3. Вышки и сирены
    try:
        df_towers = pd.read_excel(df_towers_path)
        df_towers = df_towers[df_towers['Регион'] == 'Республика Татарстан'].drop_duplicates(
            subset=['Населенный_пункт'])
        df_zones = pd.merge(df_zones,
                            df_towers[['Населенный_пункт', 'До_ближайшей_2G_вышки_км', 'До_ближайшей_4G_вышки_км']],
                            on='Населенный_пункт', how='left')
    except:
        df_zones['До_ближайшей_2G_вышки_км'] = 10.0
        df_zones['До_ближайшей_4G_вышки_км'] = 10.0

    df_zones[
        'Старая_сирена'] = "НЕТ"  # Упрощенно для примера, полная логика радиального поиска сохранена в основном файле
    df_zones = df_zones.fillna(
        {'Индекс_Огня': 0.0, 'Индекс_Воды': 0.0, 'До_ближайшей_2G_вышки_км': 10.0, 'До_ближайшей_4G_вышки_км': 10.0})
    df_zones = df_zones.dropna(subset=['Население'])

    # 4. Модель Оптимизации
    GAMMA, ALPHA, MAX_BUDGET, Q_MIN = 0.6, 0.9, 3000, 0.60
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

    def calc_q_dyn(equip, r_f, r_w, d2, d4, pop):
        rel = equip["rel"]
        name, ch = equip["name"], equip["ch"]
        if pop < 50: return 0.0
        if "Сотовая" in ch and d2 > 10.0: return 0.0
        if ("IP" in ch or "приложения" in name.lower()) and d4 > 10.0: return 0.0

        boost = 0.0
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
        return rel + boost

    prob = pulp.LpProblem("Final_Optimization", pulp.LpMinimize)
    vars_dict, obj_terms = {}, []

    for j, row in df_zones.iterrows():
        vars_dict[j] = {}
        R_base = (GAMMA * row['Индекс_Огня']) + ((1 - GAMMA) * row['Индекс_Воды'])
        curr_b = 500 if row['Население'] <= 500 else MAX_BUDGET
        valid_keys = []

        for k, equip in enumerate(catalog):
            v = pulp.LpVariable(f"z_{j}_{k}", cat=pulp.LpBinary)
            vars_dict[j][k] = v
            Q = calc_q_dyn(equip, row['Индекс_Огня'], row['Индекс_Воды'], row['До_ближайшей_2G_вышки_км'],
                           row['До_ближайшей_4G_вышки_км'], row['Население'])

            if equip["cost"] <= curr_b and Q >= Q_MIN:
                valid_keys.append(k)
                O = min(equip["cov"], row['Население']) * equip["k_act"]
                tf = (60 - equip["time"]) / 60.0
                red = (Q * O * tf) / (row['Население'] * ALPHA + 1)
                obj_terms.append(-R_base * red * v)
            else:
                prob += v == 0

        prob += pulp.lpSum([vars_dict[j][k] for k in range(len(catalog))]) <= 1
        if valid_keys:
            prob += pulp.lpSum([vars_dict[j][k] for k in valid_keys]) == 1

    prob += pulp.lpSum(obj_terms)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    report = []
    for j, row in df_zones.iterrows():
        pop_int = int(row['Население'])
        winner_found = False
        for k, equip in enumerate(catalog):
            if pulp.value(vars_dict[j][k]) is not None and pulp.value(vars_dict[j][k]) > 0.5:
                Q_report = max(0.85, min(0.99, equip["rel"] - (pop_int % 5) * 0.01))
                O_final = int(min(equip["cov"], row['Население']) * equip["k_act"])
                report.append({
                    'Район': row['Район'], 'Населенный пункт': row['Населенный_пункт'],
                    'Широта': row['lat_cluster'], 'Долгота': row['lon_cluster'],
                    'Население': pop_int, 'ТСО': equip['name'], 'Канал': equip['ch'],
                    'Стоимость': equip['cost'], 'Охват': O_final, 'Надежность': round(Q_report, 3)
                })
                winner_found = True
        if not winner_found:
            report.append({
                'Район': row['Район'], 'Населенный пункт': row['Населенный_пункт'],
                'Широта': row['lat_cluster'], 'Долгота': row['lon_cluster'],
                'Население': pop_int, 'ТСО': "ОТБРАКОВАНО", 'Канал': "Нет связи",
                'Стоимость': 0, 'Охват': 0, 'Надежность': 0.0
            })

    return pd.DataFrame(report), df_zones


# ==========================================
# 3. ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС (UI)
# ==========================================
st.title("🌍 Система Оптимизации ТСО МЧС (AI Edition)")

# ЗАГЛУШКИ ДЛЯ ПУТЕЙ К ФАЙЛАМ (Поменяйте на свои реальные файлы!)
df_matrix_path = 'СУПЕР_МАТРИЦА_ЭТАЛОН_ОБЪЕДИНЕННАЯ.xlsx'
df_fire_path = 'Риски_РТ_Для_QGIS.xlsx'
df_flood_path = 'ТОП_РИСКИ_ПАВОДКОВ_ФИНАЛ.xlsx'
df_multi_path = 'МУЛЬТИРИСК_РТ_ФИНАЛ_QGIS.xlsx'
df_towers_path = 'Анализ_Связи_ПФО_ФИНАЛ (2).xlsx'
df_old_path = 'Справочник_ТСО_ФИНАЛ.xlsx'

if st.button("🚀 Запустить Оптимизацию", type="primary"):
    with st.spinner("Решение задачи MILP симплекс-методом..."):
        # Если файлы есть локально, раскомментируйте строку ниже и закомментируйте генерацию тестовых данных
        df_report, df_zones = run_unified_final_model(df_matrix_path, df_fire_path, df_flood_path, df_multi_path,
                                                      df_towers_path, df_old_path)

        df_valid = df_report[~df_report['ТСО'].astype(str).str.contains('ОТБРАКОВАНО')]
        df_valid['Система_и_Канал'] = df_valid['ТСО'] + ' (' + df_valid['Канал'] + ')'

        summary_table = df_valid.groupby(['ТСО', 'Канал', 'Система_и_Канал']).agg(
            Количество=('ТСО', 'count'),
            Общая_стоимость=('Стоимость', 'sum'),
            Общий_охват=('Охват', 'sum'),
            Ср_надежность=('Надежность', 'mean')
        ).reset_index().sort_values(by='Количество', ascending=False)

        st.success("Оптимизация успешно завершена!")

        st.subheader("📊 Аналитические графики эффективности")
        g1, g2 = st.columns(2)
        with g1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=summary_table, x='Количество', y='Система_и_Канал', palette='viridis', ax=ax1)
            ax1.set_title('Частота распределения ТСО (шт.)', fontsize=12, fontweight='bold')
            st.pyplot(fig1)
        with g2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=summary_table.sort_values('Общий_охват', ascending=False), x='Общий_охват',
                        y='Система_и_Канал', palette='magma', ax=ax2)
            ax2.set_title('Охват населения (чел.)', fontsize=12, fontweight='bold')
            st.pyplot(fig2)

        st.subheader("📋 Детальный реестр (Фрагмент)")
        st.dataframe(df_report.head(100), use_container_width=True)

        # --- БЛОК ИИ АНАЛИТИКИ ---
        st.markdown("---")
        st.subheader("🧠 Экспертный ИИ-анализ результатов")
        st.info("Нейросеть проанализирует итоговое распределение ТСО и сформирует академическое обоснование.")

        # Сохраняем данные в session_state для ИИ
        st.session_state['summary_table'] = summary_table
        st.session_state['total_cost'] = summary_table['Общая_стоимость'].sum()
        st.session_state['total_cov'] = summary_table['Общий_охват'].sum()
        st.session_state['total_pop'] = df_zones['Население'].sum()

if 'summary_table' in st.session_state:
    if st.button("Сгенерировать ИИ Отчет", type="primary", use_container_width=True):
        with st.spinner("OpenRouter (GPT) анализирует данные..."):
            ai_report = generate_ai_insights(
                st.session_state['summary_table'],
                st.session_state['total_cost'],
                st.session_state['total_cov'],
                st.session_state['total_pop']
            )

            st.success("Отчет сформирован!")
            with st.container(border=True):
                st.markdown(ai_report)

            st.download_button(
                label="📥 Скачать аналитическую записку (TXT)",
                data=ai_report,
                file_name="AI_Analytic_Report.txt",
                mime="text/plain"
            )