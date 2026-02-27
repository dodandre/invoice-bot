import streamlit as st
import json
from langchain_openai import ChatOpenAI
from functions import (
    pick_tables, load_column_mappings, generate_sql_json, fix_date_filters,
    json_to_sql, run_sql, get_dynamic_analysis_plan, perform_analysis_from_plan,
    stream_query_to_redpanda, decide_query_action, get_langchain_response,
    split_comparison_query, compare_dataframes, generate_comparison_summary
)
from config import OPENAI_API_KEY

# ✅ Initialize LangChain LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)

# ✅ Initialize session state
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}  # {sql_query: df}
if "query_to_sql_map" not in st.session_state:
    st.session_state.query_to_sql_map = {}  # {user_query: sql_query}

st.set_page_config(page_title="Memory-Powered SQL Assistant", layout="wide")
st.title("💬 Memory-Powered SQL Assistant")

# ✅ Function to handle comparison queries
def handle_compare(user_query, llm):
    st.info("🔍 Detected a comparison query — splitting into sub-queries...")

    sub_queries = split_comparison_query(user_query, llm)
    if not sub_queries:
        st.warning("⚠️ Could not split the query. Treating as normal query.")
        return None, None

    dfs = []
    sql_queries = []

    for idx, sq in enumerate(sub_queries, 1):
        st.write(f"### 🔸 Sub-query {idx}: `{sq}`")

        sql_query = st.session_state.query_to_sql_map.get(sq)
        if not sql_query:
            pick_res = pick_tables(sq)
            if not pick_res or "selected_tables" not in pick_res:
                st.error(f"❌ Could not identify tables for: {sq}")
                continue

            selected = [t["name"] for t in pick_res["selected_tables"]]
            column_map = load_column_mappings(selected)
            spec = generate_sql_json(sq, selected, column_map)
            if not spec:
                st.error(f"❌ Failed to generate SQL for: {sq}")
                continue

            spec = fix_date_filters(spec)
            sql_query = json_to_sql(spec)

        try:
            if sql_query in st.session_state.query_cache:
                df = st.session_state.query_cache[sql_query]
                st.info("✅ Loaded from cache.")
            else:
                df = run_sql(sql_query)
                st.session_state.query_cache[sql_query] = df
                if not df.empty:
                    st.session_state.query_to_sql_map[sq] = sql_query

            if df.empty:
                st.warning(f"⚠️ No results for: {sq}")
            else:
                st.dataframe(df.head(10), use_container_width=True, height=250)
                dfs.append(df)
                sql_queries.append(sql_query)

        except Exception as e:
            st.error(f"❌ SQL Execution Failed for: {sq}\n\n{e}")

    if len(dfs) < 2:
        st.warning("⚠️ Need at least 2 datasets to compare.")
        return None, None

    # ✅ Compare
    st.write("### ⚖️ Comparing datasets...")
    combined, summary_df, error = compare_dataframes(dfs)
    if error:
        st.error(error)
        return dfs, sql_queries

    st.dataframe(summary_df, use_container_width=True)

    # ✅ AI Summary
    st.write("### 🧠 AI-Generated Comparison Summary")
    summary_text = generate_comparison_summary(summary_df, llm)
    st.markdown(summary_text)

    return dfs, sql_queries

# ✅ Input
user_query = st.text_input("Ask your question:", key="user_query")

if st.button("Generate Answer"):
    if not user_query.strip():
        st.warning("Please enter a query.")
        st.stop()

    decision = decide_query_action(user_query, st.session_state.last_sql)
    st.write(f"🧠 **LangChain Decision:** `{decision['action']}` — {decision['reason']}")

    if decision["action"] == "casual":
        st.info("💬 Responding casually...")
        response = get_langchain_response(
            user_query,
            st.session_state.query_cache.get(st.session_state.last_sql)
        )
        st.markdown(f"**Response:** {response}")
        st.stop()

    elif decision["action"] == "reuse":
        sql_query = st.session_state.last_sql
        st.success("♻️ Reusing last SQL...")
        try:
            if sql_query in st.session_state.query_cache:
                df = st.session_state.query_cache[sql_query]
                st.info("✅ Loaded from cache.")
            else:
                df = run_sql(sql_query)
                st.session_state.query_cache[sql_query] = df
        except Exception as e:
            st.error(f"❌ SQL Error:\n\n{e}")
            st.stop()

    elif decision["action"] == "new":
        try:
            pick_res = pick_tables(user_query)
            if not pick_res or "selected_tables" not in pick_res:
                st.error("❌ Could not identify tables.")
                st.stop()

            selected = [t["name"] for t in pick_res["selected_tables"]]
            st.write(f"📋 Selected Tables: `{selected}`")

            column_map = load_column_mappings(selected)
            spec = generate_sql_json(user_query, selected, column_map)
            if not spec:
                st.error("❌ Failed to generate SQL spec.")
                st.stop()

            spec = fix_date_filters(spec)
            sql_query = json_to_sql(spec)

            st.session_state.last_sql = sql_query
            df = run_sql(sql_query)
            st.session_state.query_cache[sql_query] = df

            if not df.empty:
                st.session_state.query_to_sql_map[user_query] = sql_query

        except Exception as e:
            st.error(f"❌ SQL Execution Failed:\n\n{e}")
            st.stop()

    elif decision["action"] == "compare":
        dfs, sql_queries = handle_compare(user_query, llm)
        if dfs is None:
            st.stop()

    else:
        st.error("❌ Unknown decision type.")
        st.stop()

    # ✅ Display results
    if decision["action"] in ["new", "reuse"]:
        if df.empty:
            st.warning("⚠️ No results returned.")
        else:
            st.write("### 🔍 Sample Results")
            st.dataframe(df.head(50), use_container_width=True, height=500)

            st.write("### 📈 AI-Generated Data Analysis")
            plan = get_dynamic_analysis_plan(user_query, df)
            perform_analysis_from_plan(df, plan)

# ✅ Sidebar: Query ↔ SQL Mapping
with st.sidebar.expander("🧠 Query → SQL Mapping", expanded=False):
    if st.session_state.query_to_sql_map:
        for user_q, sql_q in st.session_state.query_to_sql_map.items():
            st.markdown(f"**User Query:** {user_q}")
            st.code(sql_q, language="sql")
    else:
        st.info("No mappings yet.")

# ✅ Debug Session
with st.sidebar.expander("🛠 Raw Session Memory", expanded=False):
    st.write(st.session_state.query_to_sql_map)
