import streamlit as st
from functions import (
    pick_tables, load_column_mappings, generate_sql_json, fix_date_filters,
    json_to_sql, run_sql, strip_where_from_sql, get_core_billing_sql, trace_sales_order_number,
    trace_document_number,
    get_document_flow_for_order,
    get_dynamic_analysis_plan, perform_analysis_from_plan,
    stream_query_to_redpanda, decide_query_action, get_langchain_response,
    load_pdf_text, get_insights_from_all_providers, pick_best_analysis,
    transcribe_audio,
)

st.set_page_config(page_title="Memory-Powered SQL Assistant", layout="wide")

# Initialize session state variables
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""

if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}  # {sql_query: df}

if "last_voice_audio" not in st.session_state:
    st.session_state.last_voice_audio = None

# "Try without filters" clicked after an empty result: run sample SQL and show
if st.session_state.get("try_without_filters") and st.session_state.get("last_sql"):
    st.session_state["try_without_filters"] = False
    sample_sql = strip_where_from_sql(st.session_state.last_sql)
    if sample_sql != st.session_state.last_sql:
        st.title("💬 Memory-Powered SQL Assistant")
        st.info("📋 Sample data (same query **without** WHERE clause):")
        st.code(sample_sql, language="sql")
        df_sample = run_sql(sample_sql)
        if not df_sample.empty:
            st.success(f"Found **{len(df_sample)}** row(s). Showing up to 50.")
            st.dataframe(df_sample.head(50), use_container_width=True, height=400)
            st.caption("So data exists; your filter (e.g. customer or date) may have no matches. Try another value or broaden the question.")
        else:
            st.warning("Still no rows — tables may be empty or the join returns no matches.")
    st.stop()

# "Try core query" clicked when join returned no rows: run minimal billing + sales order + PO query
if st.session_state.get("try_core_billing"):
    st.session_state["try_core_billing"] = False
    st.title("💬 Memory-Powered SQL Assistant")
    core_sql = get_core_billing_sql(100)
    st.info("📋 Core query (billing + sales order + purchase order + customer only):")
    st.code(core_sql, language="sql")
    df_core = run_sql(core_sql)
    if not df_core.empty:
        st.success(f"Found **{len(df_core)}** row(s). Core tables have data; the earlier join may have included tables with no matches (e.g. e-invoice).")
        st.dataframe(df_core.head(100), use_container_width=True, height=450)
    else:
        st.warning("Core query also returned no rows — VBRK, VBRP, VBAK, or KNA1 may be empty or the link (VBRP.AUBEL = VBAK.VBELN, etc.) has no matches.")
    st.stop()

st.title("💬 Memory-Powered SQL Assistant")

with st.sidebar:
    st.subheader("🧠 Insights")
    st.caption("Best analysis is chosen automatically from all configured sources (ChatGPT, Claude, Gemini, Perplexity). Next-best alternatives are shown below.")
    st.subheader("📄 Optional PDF context")
    pdf_uploads = st.file_uploader("Upload PDF(s) for context in insights", type=["pdf"], accept_multiple_files=True, key="pdf_uploads")
    with st.expander("🔍 Trace document flow"):
        st.markdown("**Document flow across the full cycle** — Same as SAP Document Flow: **Standard Order** → **Delivery** → **Invoice** → **Accounting document**. When you trace a document, its flow is shown and **highlighted**.")
        # Flowchart: full cycle (matches SAP Document Flow screen)
        flow_dot = """
        digraph {
            rankdir=LR;
            node [shape=box];
            A [label="Order number\\n(VBAK / VBAP)\\nVBELN = order"];
            B [label="Delivery number\\n(LIKP / LIPS)\\nVBELN = delivery"];
            C [label="Invoice number\\n(VBRK / VBRP)\\nVBELN = billing"];
            D [label="Accounting document number\\n(BSAD.BELNR)\\nLink: BSAD.VBELN = VBRK.VBELN"];
            A -> B -> C -> D;
            S [label="Document number"];
            S -> A [label="VBAK.VBELN, VBRP.AUBEL"];
            S -> B [label="LIKP.VBELN"];
            S -> C [label="VBRK.VBELN"];
            S -> D [label="BSAD.BELNR"];
        }
        """
        try:
            st.graphviz_chart(flow_dot, use_container_width=True)
        except Exception:
            st.markdown("""
            **Document flow (as in SAP):** Order number (VBAK/VBAP) → Delivery number (LIKP/LIPS) → Invoice number (VBRK/VBRP) → Accounting document number (BSAD.BELNR).

            **Where document number appears:** VBAK.VBELN (order), VBRP.AUBEL (order on invoice), LIKP.VBELN (delivery), VBRK.VBELN (invoice), BSAD.BELNR (accounting doc); link from invoice to accounting: BSAD.VBELN = VBRK.VBELN.
            """)
        st.caption("Enter a document number below and click **Trace** to see **Order number**, **Delivery number**, **Invoice number**, and **Accounting document number** with item-level breakdown (VBAP, LIPS, VBRP) and links (VBFA, BSAD).")
        trace_so = st.text_input("Document number", key="trace_so_num", placeholder="e.g. 41 or 0090001234")
        if st.button("Trace", key="trace_so_btn") and trace_so:
            norm, trace_results = trace_document_number(trace_so)
            if norm is None:
                st.warning("Enter a document number.")
            else:
                st.write(f"**Normalized (10-char):** `{norm}`")
                found_any = False
                labels = {
                    "VBRK.VBELN": "VBRK (Billing header) — VBELN",
                    "VBRP.VBELN": "VBRP (Billing item) — VBELN",
                    "VBRP.AUBEL": "VBRP (Billing item) — AUBEL (reference sales order)",
                    "VBRP.AUPOS": "VBRP (Billing item) — AUPOS (position)",
                    "PSIF_INV_HDR.VBELN": "/PSIF/INV_HDR (e-invoice header) — VBELN",
                    "PSIF_INV_ITEM.VBELN": "/PSIF/INV_ITEM (e-invoice item) — VBELN",
                    "LIKP.VBELN": "LIKP (Delivery header) — VBELN",
                    "LIPS.VBELN": "LIPS (Delivery item) — VBELN",
                    "LIPS.VGBEL": "LIPS (Delivery item) — VGBEL (reference sales order)",
                    "VBAK.VBELN": "VBAK (Sales order header) — VBELN",
                    "VBAP.VBELN": "VBAP (Sales order item) — VBELN",
                    "VBAP.POSNR": "VBAP (Sales order item) — POSNR (position)",
                    "VBFA.VBELV": "VBFA (Document flow) — VBELV (preceding document)",
                    "VBFA.VBELN": "VBFA (Document flow) — VBELN (subsequent document)",
                    "BSAD.VBELN": "BSAD (Accounting, cleared) — VBELN (billing doc → BELNR = accounting doc)",
                    "BSAD.BELNR": "BSAD (Accounting, cleared) — BELNR (accounting document number)",
                    "BSID.VBELN": "BSID (Accounting, open items) — VBELN (billing doc → BELNR = accounting doc)",
                    "BSID.BELNR": "BSID (Accounting, open items) — BELNR (accounting document number)",
                }
                # ——— 1. Order number (explicit section + item breakdown) ———
                st.markdown("#### 📌 Order number")
                st.caption("Sources: VBAK.VBELN (header), VBAP.VBELN/POSNR (items). Links: VBRP.AUBEL = order; LIPS.VGBEL = order.")
                order_keys = ["VBAK.VBELN", "VBAP.VBELN", "VBAP.POSNR", "VBRP.AUBEL", "LIPS.VGBEL"]
                for key in order_keys:
                    df = trace_results.get(key)
                    if df is not None and not df.empty:
                        found_any = True
                        label = labels.get(key, key)
                        st.success(f"**{label}** — {len(df)} row(s).")
                        st.dataframe(df.head(20), use_container_width=True, height=180)
                # ——— 2. Delivery number (explicit section + item breakdown) ———
                st.markdown("#### 📌 Delivery number")
                st.caption("Sources: LIKP.VBELN (header), LIPS.VBELN (items). Links: VBFA (VBELV/VBELN links order ↔ delivery ↔ invoice).")
                deliv_keys = ["LIKP.VBELN", "LIPS.VBELN", "VBFA.VBELV", "VBFA.VBELN"]
                for key in deliv_keys:
                    df = trace_results.get(key)
                    if df is not None and not df.empty:
                        found_any = True
                        label = labels.get(key, key)
                        st.success(f"**{label}** — {len(df)} row(s).")
                        st.dataframe(df.head(20), use_container_width=True, height=180)
                # ——— 3. Invoice number (explicit section + item breakdown) ———
                st.markdown("#### 📌 Invoice number")
                st.caption("Sources: VBRK.VBELN (header), VBRP.VBELN (items). Links: VBRP.AUBEL = order; VBFA links invoice to delivery.")
                inv_keys = ["VBRK.VBELN", "VBRP.VBELN", "VBRP.AUPOS", "PSIF_INV_HDR.VBELN", "PSIF_INV_ITEM.VBELN"]
                for key in inv_keys:
                    df = trace_results.get(key)
                    if df is not None and not df.empty:
                        found_any = True
                        label = labels.get(key, key)
                        st.success(f"**{label}** — {len(df)} row(s).")
                        st.dataframe(df.head(20), use_container_width=True, height=180)
                # ——— 4. Accounting document number (explicit section) ———
                st.markdown("#### 📌 Accounting document number")
                st.caption("Sources: BSAD.BELNR / BSID.BELNR (accounting doc); BSAD.VBELN = BSID.VBELN = VBRK.VBELN (billing). BSAD = cleared items, BSID = open items.")
                acct_keys = ["BSAD.VBELN", "BSAD.BELNR", "BSID.VBELN", "BSID.BELNR"]
                for key in acct_keys:
                    df = trace_results.get(key)
                    if df is not None and not df.empty:
                        found_any = True
                        label = labels.get(key, key)
                        st.success(f"**{label}** — {len(df)} row(s).")
                        st.dataframe(df.head(20), use_container_width=True, height=180)
                if not found_any:
                    st.info("Not found in VBRK, VBRP, LIKP, LIPS, VBAK, VBAP, VBFA, BSAD, or BSID. Try a 10-character document number (e.g. 0080003409 or 80003409).")
                # Document flow highlight when found as order (VBAK.VBELN or VBRP.AUBEL)
                in_order = (trace_results.get("VBAK.VBELN") is not None and not trace_results["VBAK.VBELN"].empty) or (trace_results.get("VBRP.AUBEL") is not None and not trace_results["VBRP.AUBEL"].empty)
                if in_order:
                    flow = get_document_flow_for_order(norm)
                    if flow and (flow["deliveries"] or flow["billings"]):
                        st.markdown("---")
                        st.markdown("**Document flow for this trace** — Order number → Delivery number → Invoice number → Accounting document number:")
                        path = [("Order number", flow["order"])]
                        if flow["deliveries"]:
                            path.append(("Delivery number", flow["deliveries"][0]))
                        if flow["billings"]:
                            path.append(("Invoice number", flow["billings"][0]))
                        if flow.get("accounting_docs"):
                            path.append(("Accounting document number", flow["accounting_docs"][0]))
                        nodes_dot = []
                        edges_dot = []
                        for i, (stage, doc_num) in enumerate(path):
                            nid = f"N{i}"
                            safe_doc = str(doc_num).replace("\\", "\\\\").replace('"', '\\"')
                            label = f"{stage}\\n{safe_doc}"
                            nodes_dot.append(f'{nid} [label="{label}", style=filled, fillcolor=lightyellow, penwidth=2];')
                            if i > 0:
                                edges_dot.append(f"N{i-1} -> {nid};")
                        flow_highlight_dot = "digraph { rankdir=LR; node [shape=box]; " + " ".join(nodes_dot) + " " + " ".join(edges_dot) + " }"
                        try:
                            st.graphviz_chart(flow_highlight_dot, use_container_width=True)
                        except Exception:
                            st.markdown(" → ".join([f"**{s}** `{d}`" for s, d in path]))
                        extra = []
                        if len(flow["deliveries"]) > 1:
                            extra.append(f"{len(flow['deliveries'])} delivery(ies)")
                        if len(flow["billings"]) > 1:
                            extra.append(f"{len(flow['billings'])} invoice(s)")
                        if flow.get("accounting_docs") and len(flow["accounting_docs"]) > 1:
                            extra.append(f"{len(flow['accounting_docs'])} accounting doc(s)")
                        if extra:
                            st.caption("Showing one path. Full set: " + ", ".join(extra) + ".")
                        else:
                            st.caption("Order number → Delivery number → Invoice number → Accounting document number (links: VBFA, VBRP.AUBEL, BSAD.VBELN).")
                    else:
                        st.caption("No delivery or invoice links found in VBFA/VBRP for this order; flow may be incomplete or not yet delivered/billed.")

# Ask your question — speak OR write (both options)
st.subheader("Ask your question")
st.caption("Choose either **speak** (voice) or **write** (type); then click **Generate Answer**.")
speak_col, write_col = st.columns(2)

with speak_col:
    st.markdown("**🎤 Speak**")
    audio_data = None
    audio_input_fn = getattr(st, "audio_input", None)
    if callable(audio_input_fn):
        audio_data = audio_input_fn("Record with microphone", key="voice_input")
    else:
        st.caption("*Upgrade to Streamlit 1.37+ for microphone recording.*")
    st.caption("Or upload audio (WAV, MP3, WebM, OGG, M4A):")
    audio_upload = st.file_uploader("Upload audio", type=["wav", "mp3", "webm", "ogg", "m4a"], key="voice_upload", label_visibility="collapsed")
    if audio_upload is not None:
        audio_data = audio_upload

with write_col:
    st.markdown("**✏️ Write**")
    user_query = st.text_input(
        "Type your question",
        key="user_query",
        placeholder="e.g. Compare invoices paid on time vs delayed...",
        label_visibility="collapsed",
    )

if audio_data is not None:
    raw = audio_data.read()
    if raw and raw != st.session_state.get("last_voice_audio"):
        st.session_state.last_voice_audio = raw
        ext = getattr(audio_data, "name", "") or "audio.wav"
        if not ext.lower().endswith((".wav", ".mp3", ".webm", ".ogg", ".m4a")):
            ext = "audio.wav"
        transcript = transcribe_audio(raw, ext)
        if transcript:
            st.session_state.user_query = transcript
            st.session_state.run_after_voice = True
            st.rerun()

# Single button works for both speak and write
run_pipeline = st.button("Generate Answer") or st.session_state.pop("run_after_voice", False)
if run_pipeline:
    if not user_query.strip():
        st.warning("Please enter a query.")
        st.stop()

    # 🔍 Decide what kind of action to take
    decision = decide_query_action(user_query, st.session_state.last_sql)
    st.write(f"🧠 **LangChain Decision:** `{decision['action']}` — {decision['reason']}")

    # 💡 Handle casual (non-SQL) queries
    if decision["action"] == "casual":
        st.info("💬 Responding casually based on previous context...")
        response = get_langchain_response(user_query, st.session_state.query_cache.get(st.session_state.last_sql, None))
        st.markdown(f"**Response:** {response}")
        st.stop()

    # 🔁 Reuse previous SQL and DataFrame
    elif decision["action"] == "reuse":
        sql_query = st.session_state.last_sql
        st.success("♻️ Reusing last SQL query.")

        # Try to retrieve DataFrame from cache
        if sql_query in st.session_state.query_cache:
            df = st.session_state.query_cache[sql_query]
            st.info("✅ Loaded data from cache.")
        else:
            st.warning("⚠️ Data not cached — re-running SQL.")
            df = run_sql(sql_query)
            st.session_state.query_cache[sql_query] = df

    # 🆕 New SQL Generation Flow
    elif decision["action"] == "new":
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
        df = run_sql(sql_query)
        st.session_state.query_cache[sql_query] = df

        # Optional: Stream metadata to Redpanda
        # stream_query_to_redpanda(user_query, sql_query, df, {})

    else:
        # Fallback: treat unknown action (e.g. "compare") as new query
        st.info("🆕 Treating as new query.")
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
        df = run_sql(sql_query)
        st.session_state.query_cache[sql_query] = df

    # 📊 Display results — always show a clear "Query results" section
    st.write("---")
    st.subheader("📋 Query results")
    if df.empty:
        st.warning("⚠️ No rows returned from the SQL query.")
        if st.session_state.last_sql:
            st.caption("SQL that was run:")
            st.code(st.session_state.last_sql, language="sql")
        st.caption("Try a different filter (e.g. customer or date) or rephrase the question. Use **Trace document flow** in the sidebar to find document numbers that exist in VBRK, VBRP, LIKP, LIPS, etc.")
        # Auto-run without WHERE to show if any data exists (no click needed)
        if st.session_state.last_sql and "WHERE" in st.session_state.last_sql.upper():
            sample_sql = strip_where_from_sql(st.session_state.last_sql)
            if sample_sql != st.session_state.last_sql:
                with st.expander("📋 Show sample data (same query without filters)", expanded=True):
                    st.caption("Query used:")
                    st.code(sample_sql, language="sql")
                    df_sample = run_sql(sample_sql)
                    if not df_sample.empty:
                        st.success(f"Found **{len(df_sample)}** row(s). Your filter may have no matches — try another value.")
                        st.dataframe(df_sample.head(50), use_container_width=True, height=350)
                    else:
                        st.info("No rows in tables for this join — data may be empty or the link between tables returns nothing.")
                        st.caption("You can try a **core query** (billing + sales order + purchase order + customer only, no e-invoice tables) to see if the main tables have data.")
                        if st.button("Try core query (billing + sales order + PO only)", key="try_core_billing"):
                            st.session_state.try_core_billing = True
                            st.rerun()
            if st.button("Run again without filters (full screen)", key="try_no_filter"):
                st.session_state.try_without_filters = True
                st.rerun()
    else:
        st.success(f"**{len(df)}** row(s) returned.")
        st.dataframe(df.head(50), use_container_width=True, height=500)

        # 🧠 Dynamic analysis (charts, graphs, calculations)
        st.write("### 📈 Charts, graphs & analysis")
        plan = get_dynamic_analysis_plan(user_query, df)
        perform_analysis_from_plan(df, plan)

        # 🤖 Auto: run all providers, pick best, show next-best alternatives
        st.write("### 🧠 Insights (industry trends, customer revenue, products)")
        pdf_text = ""
        if pdf_uploads:
            parts = []
            for f in pdf_uploads:
                t = load_pdf_text(f)
                if t:
                    parts.append(f"[{f.name}]\n{t}")
            pdf_text = "\n\n".join(parts)
        with st.spinner("Running analysis from all configured sources and selecting best..."):
            all_responses = get_insights_from_all_providers(user_query, df, pdf_text=pdf_text)
            best_provider, best_text, alternatives = pick_best_analysis(user_query, all_responses)
        st.success(f"**Best analysis (from {best_provider})**")
        st.markdown(best_text)
        if alternatives:
            for i, (alt_provider, alt_text) in enumerate(alternatives, 1):
                with st.expander(f"Next best alternative {i}: {alt_provider}"):
                    st.markdown(alt_text)
