import streamlit as st
from functions import (
    pick_tables, load_column_mappings, generate_sql_json, fix_date_filters, inject_product_name_filter_if_needed, inject_makt_single_language_if_needed, inject_material_number_filter_if_needed,
    filter_dataframe_by_product_name_if_requested,
    json_to_sql, run_sql, strip_where_from_sql, get_core_billing_sql, get_product_performance_fallback_sql, trace_sales_order_number,
    trace_document_number,
    get_document_flow_for_order,
    get_dynamic_analysis_plan, perform_analysis_from_plan, _deduplicate_material_price_rows, _deduplicate_supplier_per_part_rows,
    is_sales_by_customer_query, _aggregate_by_customer_sales,
    apply_procurement_type_display, apply_industry_display,
    COGS_CALCULATION_EXPLANATION, get_cogs_calculation_answer_if_asked, show_single_material_cost_summary,
    stream_query_to_redpanda, decide_query_action, get_langchain_response,
    is_from_list_below_procurement_query, get_material_numbers_from_dataframe, query_procurement_type_for_materials,
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
            df_sample, _ = _deduplicate_material_price_rows(df_sample, list(df_sample.columns))
            st.success(f"Found **{len(df_sample)}** row(s). Showing up to 50.")
            st.caption("One row per material (duplicate languages and repeated prices removed for clarity).")
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
        df = None
        # "From the list below which are procured internally/externally" → use prior result's materials only
        if (
            is_from_list_below_procurement_query(user_query)
            and st.session_state.get("last_sql")
            and st.session_state.last_sql in st.session_state.get("query_cache", {})
        ):
            prior_df = st.session_state.query_cache[st.session_state.last_sql]
            matnrs = get_material_numbers_from_dataframe(prior_df)
            if matnrs:
                df, procurement_list_sql = query_procurement_type_for_materials(matnrs)
                st.session_state._last_procurement_list_sql = procurement_list_sql  # store so we can show it if empty
                st.session_state._result_from_procurement_list = True  # skip product-name filter for this result
                if not df.empty:
                    st.info("Using the product list from your **previous query** — showing procurement type (MARC.BESKZ: **E** = In-house, **F** = External, **X** = Both) for those materials only.")
                    st.session_state.last_sql = "(procurement type for materials from previous result)"
                    st.session_state.query_cache[st.session_state.last_sql] = df
                else:
                    st.warning("No procurement data (MARC) found for the materials from your previous result.")
            else:
                st.warning("No product list in the previous result (no material numbers found). Run a query that lists products first (e.g. Harley products or motorcycle components), then ask which are procured internally/externally.")
                st.session_state._result_from_procurement_list = False
        if df is None:
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
            spec = inject_product_name_filter_if_needed(user_query, spec)
            spec = inject_material_number_filter_if_needed(user_query, spec)
            spec = inject_makt_single_language_if_needed(spec)
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
        spec = inject_product_name_filter_if_needed(user_query, spec)
        spec = inject_material_number_filter_if_needed(user_query, spec)
        spec = inject_makt_single_language_if_needed(spec)
        sql_query = json_to_sql(spec)
        st.session_state.last_sql = sql_query
        df = run_sql(sql_query)
        st.session_state.query_cache[sql_query] = df

    # If product performance query returned no rows, try known-good fallback SQL (VBRK, VBRP, MAKT + date range)
    if df.empty:
        fallback_sql = get_product_performance_fallback_sql(user_query)
        if fallback_sql:
            df = run_sql(fallback_sql)
            if not df.empty:
                st.session_state.last_sql = fallback_sql
                st.session_state.query_cache[fallback_sql] = df
                st.session_state._used_product_performance_fallback = True
            else:
                # Try without date filter in case there is no data in the requested year range
                fallback_all_sql = get_product_performance_fallback_sql(user_query, with_date_filter=False)
                if fallback_all_sql and fallback_all_sql != fallback_sql:
                    df = run_sql(fallback_all_sql)
                    if not df.empty:
                        st.session_state.last_sql = fallback_all_sql
                        st.session_state.query_cache[fallback_all_sql] = df
                        st.session_state._used_product_performance_fallback = True
                        st.session_state._product_performance_all_time_fallback = True

    # One row per material when result has material + price/description (no repeated material, no multi-language duplicates)
    from_procurement_list = st.session_state.pop("_result_from_procurement_list", False)
    if not df.empty:
        df, _ = _deduplicate_material_price_rows(df, list(df.columns))
        # One row per (part, supplier) when result has material + vendor columns — avoid repeated vendor entries per part
        df, supplier_deduped = _deduplicate_supplier_per_part_rows(df, list(df.columns))
        if supplier_deduped:
            st.session_state._supplier_per_part_deduped = True
        # One row per customer with total value when user asks for sales/revenue by customer (no repetitive same customer)
        if is_sales_by_customer_query(user_query):
            df, customer_agg = _aggregate_by_customer_sales(df, list(df.columns))
            if customer_agg:
                st.session_state._customer_sales_aggregated = True
        # When user asked for a product by name (e.g. Harley, or "compete against products like harley jackets"), keep only rows for that product so results and calculations stay on-subject
        if not from_procurement_list:
            df = filter_dataframe_by_product_name_if_requested(user_query, df)

    # 📊 Display results — always show a clear "Query results" section
    st.write("---")
    st.subheader("📋 Query results")
    if df.empty:
        st.warning("⚠️ No rows returned from the SQL query.")
        # Show the SQL that was actually run (procurement path may have run different SQL than last_sql)
        actual_sql_run = st.session_state.pop("_last_procurement_list_sql", None)
        last_sql_val = st.session_state.get("last_sql") or ""
        procurement_placeholder = "(procurement type for materials from previous result)"
        sql_to_show = actual_sql_run or (last_sql_val if last_sql_val != procurement_placeholder else None)
        if sql_to_show:
            with st.expander("🔍 SQL that was run", expanded=True):
                st.code(sql_to_show, language="sql")
        if actual_sql_run:
            st.caption("**Query:** Procurement type (MARC.BESKZ) for the materials from your **previous result** (MARA + MAKT + MARC). No rows: those materials may have no plant data in **MARC**, or the material numbers may not exist in MARC for any plant.")
            st.caption("Try running a query that lists products with **plant** or **valuation** data first, or ask **\"Which products are procured internally and externally?\"** without \"from the list below\" to get all materials with procurement type from master data.")
        else:
            last_sql_upper = (last_sql_val or "").upper()
            if last_sql_upper != procurement_placeholder and "MAKT" in last_sql_upper and ("MBEW" in last_sql_upper or "KEKO" in last_sql_upper):
                st.caption("For **material cost** queries: the material may not exist in MAKT with that description, or it may have no cost data in MBEW/KEKO. Try **\"Show materials with standard price\"** or **\"List cost estimates\"** to see what data exists; then filter by material name.")
                with st.expander("📘 How is cost of goods calculated for the product?", expanded=True):
                    st.markdown(COGS_CALCULATION_EXPLANATION)
            with st.expander("💡 What to try", expanded=True):
                st.markdown("""
                - **Different filters:** Use another year (e.g. 1999, 2001), customer, or material name.
                - **Date range:** For \"year 2000\" the app uses full year (Jan–Dec); if still no rows, that year may have no data in the tables.
                - **Rephrase:** e.g. \"Show materials with standard price\", \"List customers\", \"Best sales by customer\".
                - **Trace document flow** in the sidebar to find document numbers that exist in VBRK, VBRP, LIKP, LIPS.
                """)
        # Auto-run without WHERE to show if any data exists (no click needed)
        if st.session_state.last_sql and "WHERE" in (st.session_state.last_sql or "").upper() and st.session_state.last_sql != procurement_placeholder:
            sample_sql = strip_where_from_sql(st.session_state.last_sql)
            if sample_sql != st.session_state.last_sql:
                with st.expander("📋 Show sample data (same query without filters)", expanded=True):
                    st.caption("Query used:")
                    st.code(sample_sql, language="sql")
                    df_sample = run_sql(sample_sql)
                    if not df_sample.empty:
                        df_sample, _ = _deduplicate_material_price_rows(df_sample, list(df_sample.columns))
                        st.success(f"Found **{len(df_sample)}** row(s). Your filter may have no matches — try another value.")
                        st.caption("One row per material (duplicate languages and repeated prices removed for clarity).")
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
        if st.session_state.pop("_used_product_performance_fallback", False):
            if st.session_state.pop("_product_performance_all_time_fallback", False):
                st.caption("No rows in the requested year range; showing **all-time** product performance (VBRK, VBRP, MAKT) so you still get data and insights.")
            else:
                st.caption("Used a fallback product-performance query (VBRK, VBRP, MAKT with date range) because the first query returned no rows.")
        df_display = apply_procurement_type_display(df)
        df_display = apply_industry_display(df_display)
        st.dataframe(df_display.head(50), use_container_width=True, height=500)
        if st.session_state.pop("_supplier_per_part_deduped", False):
            st.caption("**Suppliers per part:** Duplicate rows (same part and same vendor from multiple orders) have been removed so each part–supplier pair appears once.")
        if st.session_state.pop("_customer_sales_aggregated", False):
            st.caption("**Sales by customer:** Rows are aggregated so each customer appears once with their **total** sales value (repetitive line-level entries removed).")
        if "procurement_type" in df.columns or any("procurement" in str(c).lower() and "type" in str(c).lower() for c in df.columns):
            st.caption("**Procurement type (MARC.BESKZ):** **E** = In-house produced, **F** = Externally procured, **X** = Both.")
        if any("industry" in str(c).lower() or (c or "").upper() in ("BRSCH", "INDUSTRY_KEY") for c in df.columns):
            st.caption("**Industry:** Single-letter codes (e.g. M, C) are shown as full sector names (e.g. Mechanical engineering, Chemical).")
        # Revenue is restricted to billing category types A, B, C, D, E, I, L, W (VBRK.FKTYP)
        if "VBRK" in (st.session_state.get("last_sql") or ""):
            st.caption("ℹ️ Revenue/value in this result is shown only for billing category types **A, B, C, D, E, I, L, W**.")

        # Direct answer when user asks "how is cost of goods calculated for the product" (so they get the explanation, not "see screen")
        cogs_answer = get_cogs_calculation_answer_if_asked(user_query, df)
        if cogs_answer:
            st.write("### 📘 Answer: How cost of goods is calculated for the product")
            st.markdown(cogs_answer)

        # Detailed cost summary when user asked for a specific product number (e.g. H10500) and result has that single material
        show_single_material_cost_summary(user_query, df_display)

        # Prominent notice: result analysis is validated by AI for plausibility and official recognition
        st.info("**Result analysis:** The charts, calculations, and insights below use the query result plus AI platforms (e.g. ChatGPT, Claude, Gemini, Perplexity) to add intelligence and plausibility. This step is **required** so the analysis holds official recognition — the raw query result alone is not sufficient.")

        # 🧠 Dynamic analysis (charts, graphs, calculations) — use df_display so charts/tables show same labels as results (industry + procurement descriptions)
        st.write("### 📈 Charts, graphs & analysis")
        plan = get_dynamic_analysis_plan(user_query, df_display)
        perform_analysis_from_plan(df_display, plan, user_query)

        # 🤖 Insights required: query result + AI platforms (ChatGPT, etc.) for plausibility and official recognition
        st.write("### 🧠 Insights (industry trends, customer revenue, products, market & competitive)")
        st.info("**Required for plausibility:** Insights are validated and enriched by AI platforms (e.g. ChatGPT, Claude, Gemini, Perplexity) using the query result, SQL context, and any uploaded PDFs. This step is **required** so analysis is plausible and holds official recognition — the raw query result alone is not sufficient.")
        pdf_text = ""
        if pdf_uploads:
            parts = []
            for f in pdf_uploads:
                t = load_pdf_text(f)
                if t:
                    parts.append(f"[{f.name}]\n{t}")
            pdf_text = "\n\n".join(parts)
        last_sql = st.session_state.get("last_sql") or ""
        with st.spinner("Running analysis from all configured sources and selecting best..."):
            all_responses = get_insights_from_all_providers(user_query, df_display, pdf_text=pdf_text, sql_query=last_sql)
            best_provider, best_text, alternatives = pick_best_analysis(user_query, all_responses)
        st.success(f"**Best analysis (from {best_provider})**")
        st.markdown(best_text)
        st.caption("Insights use **query result** + **SQL context** + **uploaded PDF(s)**. AI platforms (ChatGPT, Claude, Gemini, Perplexity) add intelligence and plausibility so the analysis holds official recognition. Compare alternatives above for different perspectives.")
        if alternatives:
            for i, (alt_provider, alt_text) in enumerate(alternatives, 1):
                with st.expander(f"Next best alternative {i}: {alt_provider}"):
                    st.markdown(alt_text)
