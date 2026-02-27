import streamlit as st
from functions import (
    pick_tables, load_column_mappings, generate_sql_json, fix_date_filters,
    json_to_sql, run_sql, get_dynamic_analysis_plan, perform_analysis_from_plan,
    stream_query_to_redpanda, decide_query_action, get_langchain_response
)

# 🧠 Initialize session state variables
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""

if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}  # {sql_query: df}

if "query_to_sql_map" not in st.session_state:
    st.session_state.query_to_sql_map = {}  # {user_query: sql_query}

# 📄 Page config
st.set_page_config(page_title="Memory-Powered SQL Assistant", layout="wide")
st.title("💬 Memory-Powered SQL Assistant")

# 🧾 User input
user_query = st.text_input("Ask your question:", key="user_query")

if st.button("Generate Answer"):
    if not user_query.strip():
        st.warning("Please enter a query.")
        st.stop()

    # 🔍 Decide what kind of action to take
    decision = decide_query_action(user_query, st.session_state.last_sql)
    st.write(f"🧠 **LangChain Decision:** `{decision['action']}` — {decision['reason']}")

    # 💬 Handle casual (non-SQL) queries
    if decision["action"] == "casual":
        st.info("💬 Responding casually based on previous context...")
        response = get_langchain_response(
            user_query,
            st.session_state.query_cache.get(st.session_state.last_sql, None)
        )
        st.markdown(f"**Response:** {response}")
        st.stop()

    # ♻️ Reuse previous SQL and DataFrame
    elif decision["action"] == "reuse":
        sql_query = st.session_state.last_sql
        st.success("♻️ Reusing last SQL query.")

        # Try to retrieve DataFrame from cache
        if sql_query in st.session_state.query_cache:
            df = st.session_state.query_cache[sql_query]
            st.info("✅ Loaded data from cache.")
        else:
            try:
                st.warning("⚠️ Data not cached — re-running SQL.")
                df = run_sql(sql_query)
                st.session_state.query_cache[sql_query] = df
            except Exception as e:
                st.error(f"❌ Error while re-running SQL:\n\n{e}")
                df = None
                st.stop()

    # 🆕 New SQL Generation Flow
    elif decision["action"] == "new":
        try:
            pick_res = pick_tables(user_query)
            if not pick_res or "selected_tables" not in pick_res:
                st.error("❌ Could not identify relevant tables.")
                st.stop()

            selected = [t["name"] for t in pick_res["selected_tables"]]
            st.write(f"📋 Selected Tables: `{selected}`")

            column_map = load_column_mappings(selected)
            spec = generate_sql_json(user_query, selected, column_map)

            if not spec:
                st.error("❌ Failed to generate SQL specification.")
                st.stop()

            spec = fix_date_filters(spec)
            sql_query = json_to_sql(spec)

            st.session_state.last_sql = sql_query

            # 🔐 Safe SQL execution
            df = run_sql(sql_query)
            st.session_state.query_cache[sql_query] = df

            # ✅ Log user_query → sql_query only if df is not empty
            if not df.empty:
                st.session_state.query_to_sql_map[user_query] = sql_query

            # Optional: Stream metadata to Redpanda
            # stream_query_to_redpanda(user_query, sql_query, df, {})

        except Exception as e:
            st.error(f"❌ SQL Generation or Execution Failed:\n\n{e}")
            df = None
            st.stop()

    else:
        st.error("❌ Unknown action type.")
        st.stop()

    # 📊 Display results
    if df is not None:
        if df.empty:
            st.warning("⚠️ No results returned from SQL query.")
        else:
            st.write("### 🔍 Sample Results")
            st.dataframe(df.head(50), use_container_width=True, height=500)

            # 📈 AI Analysis
            st.write("### 📈 AI-Generated Data Analysis")
            plan = get_dynamic_analysis_plan(user_query, df)
            perform_analysis_from_plan(df, plan)

# 🧠 Sidebar: Show memory of user query → SQL mapping
with st.sidebar.expander("🧠 Query → SQL Mapping", expanded=False):
    if st.session_state.query_to_sql_map:
        for user_q, sql_q in st.session_state.query_to_sql_map.items():
            st.markdown(f"**User Query:** {user_q}")
            st.code(sql_q, language="sql")
            st.markdown("---")
    else:
        st.info("No query mappings yet.")

# 🛠️ Optional: Raw session state view (for deep debugging)
with st.sidebar.expander("🛠 Raw Session Memory", expanded=False):
    st.write(st.session_state.query_to_sql_map)
