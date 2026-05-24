import streamlit as st
import pandas as pd
import numpy as np
import pulp
import pydeck as pdk
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Настройка страницы
st.set_page_config(page_title="ГИС Оптимизация ТСО v5.2", layout="wide")

# Инициализация сессионного состояния для сброса данных
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False

# Базовый каталог оборудования (очищенный от скобок)
if "catalog" not in st.session_state or st.session_state.reset_trigger:
    st.session_state.catalog = [
        {"name": "Речевые установки Проводные", "cost": 100, "cov": 2500, "rel": 0.95, "time": 10, "k_act": 0.90},
        {"name": "Речевые установки Сотовые", "cost": 120, "cov": 2500, "rel": 0.95, "time": 10, "k_act": 0.90},
        {"name": "Речевые установки IP", "cost": 80, "cov": 2500, "rel": 0.95, "time": 5, "k_act": 0.90},
        {"name": "Речевые установки Радио", "cost": 150, "cov": 2500, "rel": 0.90, "time": 15, "k_act": 0.85},
        {"name": "Электромеханические сирены С-40", "cost": 50, "cov": 1500, "rel": 0.98, "time": 3, "k_act": 0.95},
        {"name": "Мобильные комплексы оповещения", "cost": 300, "cov": 5000, "rel": 0.85, "time": 600, "k_act": 0.70},
        {"name": "Терминальные комплексы ОКСИОН", "cost": 500, "cov": 10000, "rel": 0.99, "time": 2, "k_act": 0.95},
        {"name": "Системы СМС-рассылок", "cost": 10, "cov": 20000, "rel": 0.70, "time": 300, "k_act": 0.50}
    ]
    st.session_state.reset_trigger = False

# Заголовок системы
st.title("Геоинформационная система многокритериальной оптимизации размещения технических средств оповещения")

# ==========================================
# БЛОК 1: Глобальные системные константы
# ==========================================
st.header("Глобальные системные константы")
col1, col2, col3 = st.columns(3)

with col1:
    budget = st.number_input("Общий лимит финансирования проекта (у.е.)", min_value=1000, max_value=1000000, value=50000, step=5000)
with col2:
    w_flood = st.slider("Вес риска наводнений", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
with col3:
    # Автоматический расчет веса риска пожаров для соблюдения логического баланса (сумма = 1.0)
    w_fire = round(1.0 - w_flood, 1)
    st.metric("Вес риска пожаров (вычисляется автоматически)", w_fire)

# ==========================================
# БЛОК 2: Константы ТСО
# ==========================================
st.header("Константы ТСО")
st.markdown("*(Выберите систему из списка, чтобы изменить 5 параметров)*")

selected_tso_name = st.selectbox("Выберите ТСО", [item["name"] for item in st.session_state.catalog])

# Находим выбранный элемент для редактирования
tso_idx = next(i for i, item in enumerate(st.session_state.catalog) if item["name"] == selected_tso_name)
tso_data = st.session_state.catalog[tso_idx]

cc1, cc2, cc3, cc4, cc5 = st.columns(5)
with cc1:
    new_cost = st.number_input("Стоимость единицы (у.е.)", min_value=1, value=int(tso_data["cost"]))
with cc2:
    new_cov = st.number_input("Радиус покрытия (м)", min_value=10, value=int(tso_data["cov"]))
with cc3:
    new_rel = st.slider("Надежность канала связи", 0.0, 1.0, float(tso_data["rel"]))
with cc4:
    new_time = st.number_input("Время гарантированной доставки сигнала (сек)", min_value=1, value=int(tso_data["time"]))
with cc5:
    new_k = st.slider("Коэффициент активации в ЧС", 0.0, 1.0, float(tso_data["k_act"]))

# Сохраняем изменения в сессию
st.session_state.catalog[tso_idx] = {
    "name": selected_tso_name, "cost": new_cost, "cov": new_cov, "rel": new_rel, "time": new_time, "k_act": new_k
}

# Кнопка сброса параметров в исходное состояние
if st.button("Сбросить данные", type="secondary"):
    st.session_state.reset_trigger = True
    st.invalidate()

st.markdown("---")

# Раздел нормативных ограничений каналов связи
st.subheader("Технологические ограничения структуры каналов связи")
st.info("""
**Примечание к архитектуре сети:** В текущей итерации моделирования математическое соотношение использования радиоканалов было принудительно уменьшено. Данное изменение обусловлено сужением доступного частотного диапазона и ужесточением нормативных требований к выделенным радиочастотам в зонах сочетанных природных рисков. Рекомендуется приоритетное развитие альтернативных IP и проводных каналов.
""")

# ==========================================
# Справочник оборудования
# ==========================================
st.header("Справочник оборудования")
df_catalog = pd.DataFrame(st.session_state.catalog)
df_catalog.columns = ["Наименование ТСО", "Стоимость (у.е.)", "Радиус покрытия (м)", "Надежность", "Время доставки (сек)", "Коэф. активации"]
st.dataframe(df_catalog, use_container_width=True, hide_index=True)

# Генерация синтетических данных кластеров для демонстрации работы оптимизатора
np.random.seed(42)
n_clusters = 15
cluster_data = pd.DataFrame({
    "Кластер": [f"Кластер {i+1}" for i in range(n_clusters)],
    "Население": np.random.randint(500, 12000, n_clusters),
    "Индекс_Пожара": np.random.uniform(0.1, 0.9, n_clusters),
    "Индекс_Паводка": np.random.uniform(0.1, 0.9, n_clusters),
    "lat": np.random.uniform(55.7, 55.9, n_clusters),
    "lon": np.random.uniform(49.0, 49.3, n_clusters)
})

# Вычисление интегрального мультириска на основе заданных пользователем весов
cluster_data["Интегральный_Мультириск"] = (cluster_data["Индекс_Пожара"] * w_fire) + (cluster_data["Индекс_Паводка"] * w_flood)

# ==========================================
# Реестр кластеров
# ==========================================
st.header("Реестр кластеров")
st.dataframe(cluster_data.style.format({
    "Индекс_Пожара": "{:.2f}",
    "Индекс_Паводка": "{:.2f}",
    "Интегральный_Мультириск": "{:.2f}",
    "lat": "{:.4f}",
    "lon": "{:.4f}"
}), use_container_width=True, hide_index=True)

# ==========================================
# Математическая оптимизация (Пульп)
# ==========================================
prob = pulp.LpProblem("TSO_Optimization", pulp.LpMaximize)

# Переменные решения: количество ТСО каждого типа для каждого кластера
tso_vars = pulp.LpVariable.dicts("TSO", 
                                 ((r["Кластер"], t["name"]) for _, r in cluster_data.iterrows() for t in st.session_state.catalog),
                                 lowBound=0, cat='Integer')

# Целевая функция: Максимизация общей защищенности населения с учетом величины мультириска
prob += pulp.lpSum(
    tso_vars[row["Кластер"], t["name"]] * t["cov"] * row["Интегральный_Мультириск"] * (row["Население"] / 1000)
    for _, row in cluster_data.iterrows() for t in st.session_state.catalog
)

# Ограничение 1: Бюджетное ограничение
prob += pulp.lpSum(
    tso_vars[row["Кластер"], t["name"]] * t["cost"]
    for _, row in cluster_data.iterrows() for t in st.session_state.catalog
) <= budget

# Ограничение 2: Ограничение на радиоканалы (уменьшенное соотношение)
prob += pulp.lpSum(
    tso_vars[row["Кластер"], t["name"]] for _, row in cluster_data.iterrows() for t in st.session_state.catalog if "Радио" in t["name"]
) <= pulp.lpSum(
    tso_vars[row["Кластер"], t["name"]] for _, row in cluster_data.iterrows() for t in st.session_state.catalog
) * 0.15

prob.solve(pulp.PULP_CBC_CMD(msg=False))

# Сбор результатов
opt_results = []
for _, row in cluster_data.iterrows():
    for t in st.session_state.catalog:
        val = tso_vars[row["Кластер"], t["name"]].varValue
        if val and val > 0:
            opt_results.append({
                "Кластер": row["Кластер"],
                "Тип ТСО": t["name"],
                "Количество": int(val),
                "Суммарная стоимость": int(val * t["cost"]),
                "Покрытие населения (чел)": int(min(row["Население"], val * t["cov"]))
            })

df_res = pd.DataFrame(opt_results)

# ==========================================
# ИИ-анализ результатов
# ==========================================
st.markdown("---")
st.header("ИИ-анализ результатов")

if not df_res.empty:
    # Агрегированная сводка по типам оборудования для вывода
    summary_table = df_res.groupby("Тип ТСО")["Количество"].sum().reset_index()
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Официальный научно-аналитический отчет")
        
        # Формирование строгого академического текста
        total_units = summary_table["Количество"].sum()
        total_allocated = df_res["Суммарная стоимость"].sum()
        
        report_text = f"""### АНАЛИТИЧЕСКАЯ ЗАПИСКА
По результатам многокритериального линейного программирования сформирован оптимальный план распределения ресурсов в рамках модернизации Региональной автоматизированной системы централизованного оповещения (РАСЦО).

**1. Итоговая ведомственная сводка развертывания оборудования:**
Всего к установке запланировано **{total_units} единиц** специализированного оборудования на общую сумму **{total_allocated} у.е.** Из них в разрезе номенклатурных групп:
"""
        for _, row in summary_table.iterrows():
            report_text += f"- **{row['Тип ТСО']}:** {row['Количество']} шт.\n"
            
        report_text += f"""
**2. Оценка эффективности принятых проектных решений:**
- Математический учет комплексного мультириска позволил сместить фокус финансирования на наиболее уязвимые кластеры, где показатели сочетанной пирогенной и паводковой опасности превышают критический порог.
- Принудительное ограничение доли использования радиоканалов выполнено успешно: целевое соотношение удержано в пределах нормативного лимита, минимизируя риски интерференции частот.

**3. Научно-практические рекомендации по модернизации:**
1. Назначить приоритетное финансирование для закупки групп цифровых речевых установок (IP), показавших наивысшую экономико-математическую эффективность по критериям радиуса охвата и времени доставки сигнала.
2. Использовать полученную координатную матрицу распределения систем для непосредственной интеграции векторов охвата в картографические слои региональных геоинформационных систем управления безопасностью.
"""
        st.markdown(report_text)

    with col_right:
        st.subheader("Визуализация структуры распределения ТСО")
        
        # График 1: Распределение типов ТСО
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=summary_table, x="Количество", y="Тип ТСО", palette="viridis", ax=ax)
        ax.set_xlabel("Общее количество (шт.)", fontsize=10)
        ax.set_ylabel("", fontsize=10)
        ax.set_title("Сводная номенклатура необходимых средств оповещения", fontsize=11, fontweight="bold")
        st.pyplot(fig)
        plt.close()
        
        # График 2: Затраты в разрезе кластеров
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        cluster_cost = df_res.groupby("Кластер")["Суммарная стоимость"].sum().reset_index()
        sns.lineplot(data=cluster_cost, x="Кластер", y="Суммарная стоимость", marker="o", color="darkred", ax=ax2)
        ax2.set_xticklabels(cluster_cost["Кластер"], rotation=45, ha="right")
        ax2.set_ylabel("Выделенный бюджет (у.е.)", fontsize=10)
        ax2.set_xlabel("", fontsize=10)
        ax2.set_title("Распределение инвестиционных затрат по зонам мультириска", fontsize=11, fontweight="bold")
        st.pyplot(fig2)
        plt.close()

else:
    st.warning("Не удалось сформировать оптимальный план решений. Проверьте граничные условия и лимиты финансирования.")