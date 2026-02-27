import json
import re
import os
import io
import openai
import pyodbc
import pandas as pd
from datetime import datetime
import streamlit as st
import altair as alt
from config import (
    OPENAI_API_KEY, CONN_STR, TABLE_DESCRIPTIONS, DATE_COLUMNS,
    MAPPINGS_FOLDER, conversation, memory, MAX_DOC_ROWS, llm,
)
try:
    from config import INCLUDE_FSCM_CREDIT_TABLE
except ImportError:
    INCLUDE_FSCM_CREDIT_TABLE = None
try:
    from config import ANTHROPIC_API_KEY, GOOGLE_GEMINI_API_KEY, PERPLEXITY_API_KEY
except (ImportError, AttributeError):
    ANTHROPIC_API_KEY = ""
    GOOGLE_GEMINI_API_KEY = ""
    PERPLEXITY_API_KEY = ""
#KAFKA_TOPICS, producer
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document



#test 
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.chains import LLMChain
#test




from openai import OpenAI
# Global memory object
openai.api_key = OPENAI_API_KEY
#client = openai.OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)
chat_history = []


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Convert speech to text using OpenAI Whisper. Returns transcript or empty string on error."""
    if not OPENAI_API_KEY or not audio_bytes:
        return ""
    try:
        file_like = io.BytesIO(audio_bytes)
        file_like.name = filename
        resp = client.audio.transcriptions.create(model="whisper-1", file=file_like)
        return (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        st.warning(f"Voice transcription failed: {e}")
        return ""


def pick_tables(user_query: str):
    prompt = f"""
User query: "{user_query}"

Available tables:
{json.dumps(TABLE_DESCRIPTIONS, indent=2)}

Task:
- Identify tables relevant to answer the query.
- If the query involves customer number or customer name, always include KNA1 (customer number = KUNNR, customer name = NAME1).
- For value determination, revenue, or "best products by value": prefer VBRK (header) and VBRP (item: NETWR, MATNR); join on VBELN. These are the best tables for value.
- For "best products" or "products by value and industry": use VBRK and VBRP for value; add KNA1 for industry (BRSCH) if needed; join VBRK.VBELN = VBRP.VBELN. Do not add any WHERE filter so all data is returned.
- For industry trends, industry analysis, or showing industry by graphs/charts/diagrams: use VBRK and VBRP for value and KNA1 for industry (BRSCH); join VBRK.KUNAG = KNA1.KUNNR and VBRK.VBELN = VBRP.VBELN so the result has industry and value for charts and insights.
- Sales order data: use VBAK (header) and VBAP (item); eOrder / electronic sales order: use PSIF_SLS_HDR and PSIF_SLS_ITEM (tables /PSIF/SLS_HDR and /PSIF/SLS_ITEM). Sales and purchase orders are stored in VBAK, VBAP, VBEP, VBFA, VBPA, VBUK, VBUP — include these tables when the user asks about sales orders, purchase orders, order status, schedule lines, document flow, or partners. VBFA shows the flow distinctively (preceding/subsequent document links).
- When showing billing documents or invoices: always include VBRK and VBRP and join to VBAK (VBRP.AUBEL = VBAK.VBELN) so results can show billing document number (VBRK.VBELN), sales order number (VBRP.AUBEL only — never AUPOS; AUPOS is position number only), purchase order number (VBAK.BSTNK), and for process flow include delivery number (LIKP.VBELN) by joining VBFA (VBFA shows the flow distinctively: VBRP.VBELN = VBFA.VBELN, VBRP.POSNR = VBFA.POSNN) and LIKP (VBFA.VBELV = LIKP.VBELN). Include VBAK whenever VBRK or VBRP are selected; include VBFA and LIKP when the user needs process flow or delivery number.
- Delivery-specific data: when the user asks about delivery, deliveries, delivery number, or process flow (order → delivery → invoice): always include LIKP (delivery header; VBELN = delivery number), LIPS (delivery item; VBELN = delivery, VGBEL = sales order number), and VBFA (the link: invoice VBRP.VBELN = VBFA.VBELN, delivery LIKP.VBELN = VBFA.VBELV). Delivery links to sales order via LIPS.VGBEL = VBAK.VBELN; delivery links to invoice via VBFA. So include VBRP, VBAK, VBFA, LIKP, and LIPS together whenever the query involves invoice, order, or delivery.
- Comparing sales orders and invoices (process flow): use VBAK/VBAP (sales order), VBRK/VBRP (invoice), LIKP and LIPS (delivery), and VBFA (link). Always include in results for process flow: billing document number (VBRK.VBELN), sales order number (VBRP.AUBEL or VBAK.VBELN only — never AUPOS or POSNR; AUPOS is position number only), delivery number (LIKP.VBELN), and purchase order number (VBAK.BSTNK). Join VBRP to VBAK (VBRP.AUBEL = VBAK.VBELN); join VBRP to VBFA (VBRP.VBELN = VBFA.VBELN AND VBRP.POSNR = VBFA.POSNN) and VBFA to LIKP (VBFA.VBELV = LIKP.VBELN); join LIKP to LIPS (LIKP.VBELN = LIPS.VBELN); join LIPS to VBAK (LIPS.VGBEL = VBAK.VBELN). The purchase order number (BSTNK) links order, delivery, and invoice.
- Invoice value per customer (name or number): use e-invoicing tables PSIF_INV_HDR and PSIF_INV_ITEM with KNA1 for customer name and number; join PSIF_INV_HDR.KUNNR = KNA1.KUNNR.
- Discrepancies (payment terms, revenues, delay reason): compare invoices created (VBRK, VBRP) to invoices sent and when sent (PSIF_INV_HDR, PSIF_ACK). Include VBRK, VBRP, PSIF_INV_HDR, and PSIF_ACK when the user asks about discrepancies, payment terms, revenue impact, or reason for delay (e.g. supplier delay or other); link on VBELN. Use PSIF_ACK.ACK_TEXT or ACK_ID to attribute supplier delay vs other reason.
- Invoices created but not sent (by customer): use VBRK (created), PSIF_INV_HDR or PSIF_ACK (sent), and KNA1 (customer name/number). You must use a LEFT JOIN from VBRK to the sent table and filter WHERE sent.VBELN IS NULL so only created-not-sent rows are returned. Include VBRK, PSIF_INV_HDR (or PSIF_ACK), and KNA1.
- Paid on time vs delayed and which customers caused delay: use VBRK (invoice, FKDAT, ZTERM) and BSAD (AUGDT = clearing/payment date, VBELN, KUNNR) with KNA1 (customer name, number). Join VBRK.VBELN = BSAD.VBELN, BSAD.KUNNR = KNA1.KUNNR. Compare paid on time vs delayed and include customer name and number for visibility of which customers caused payment delay.
- Credit risk and customer credit: use KNKK (credit master data: credit limit KLIMK, credit exposure SKFOR = total value of open orders/deliveries, risk category per credit control area), KNKA (central data: risk class, credit group, overall limits KLIMG/KLIME), and T691A (risk categories used for credit checks); join KNKK and KNKA to KNA1 on KUNNR; join KNKK to T691A on KNKK.CTLPC = T691A.CTLPC AND KNKK.KKBER = T691A.KKBER for risk category definitions. When the user asks about credit risk for a specific invoice number: also include VBRK (billing/invoice) and join VBRK.KUNAG = KNA1.KUNNR so the query can filter by VBRK.VBELN = that invoice and show credit data for that customer.
""" + (
    " Also include UKM_BP_CMS_SGM (S/4HANA FSCM: credit limit, block reason, CREDIT_LIMIT) and join to KNA1 on UKM_BP_CMS_SGM.PARTNER = KNA1.KUNNR so both ERP and FSCM credit data are available."
    if fscm_credit_table_available() else
    " Do not include table UKM_BP_CMS_SGM — it is not available in this system (ERP only)."
) + """
- For other comparisons by industry, customer, value, or products: include the tables that hold that data (e.g. KNA1, VBRK, VBRP, VBAK, VBAP, VBEP, VBFA, VBPA, VBUK, VBUP, KNKK, KNKA, T691A, PSIF_SLS_HDR, PSIF_SLS_ITEM). Do not restrict to country-specific data unless the user explicitly asks for a specific country.
- Only return JSON in this format:

{{
  "query": "...",
  "selected_tables": [
    {{ "name": "TABLE_NAME", "description": "..." }}
  ]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content =  response.choices[0].message.content
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            out = json.loads(m.group(0))
        else:
            return None
    # Ensure delivery chain (VBFA, LIKP, LIPS) is present when invoice/order tables are selected — delivery links to sales order and invoice
    if out and "selected_tables" in out:
        names = {t["name"] for t in out["selected_tables"]}
        process_flow_tables = {"VBRK", "VBRP", "VBAK"}
        delivery_chain = [("VBFA", "Document flow: links invoice (VBRP) to delivery (LIKP)"), ("LIKP", "Delivery header; VBELN = delivery number"), ("LIPS", "Delivery item; VBELN = delivery, VGBEL = sales order")]
        if names & process_flow_tables:
            for tbl, desc in delivery_chain:
                if tbl not in names and tbl in TABLE_DESCRIPTIONS:
                    out["selected_tables"].append({"name": tbl, "description": desc})
    return out

def load_column_mappings(selected_tables: list):
    mappings = {}
    for tbl in selected_tables:
        path = os.path.join(MAPPINGS_FOLDER, f"{tbl}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    mappings[tbl] = json.load(f)
            except json.JSONDecodeError:
                st.warning(f"⚠️ JSON decode error in mapping file {path}")
    return mappings

def generate_sql_json(user_query: str, selected_tables: list, column_mappings: dict):
    from config import CUSTOMER_NUMBER_COLUMN, CUSTOMER_NAME_COLUMN
    cust_tbl, cust_num_col = CUSTOMER_NUMBER_COLUMN
    cust_name_col = CUSTOMER_NAME_COLUMN[1]
    prompt = f"""
User query: "{user_query}"

Tables available:
{json.dumps({tbl: TABLE_DESCRIPTIONS[tbl] for tbl in selected_tables}, indent=2)}

Column mappings:
{json.dumps(column_mappings, indent=2)}

**Customer number and name (always use KNA1):**
- Customer number: use {cust_tbl}.{cust_num_col} only.
- Customer name: use {cust_tbl}.{cust_name_col} only.
- When the query asks for customer, customer ID, customer number, or customer name, include {cust_tbl} and select {cust_num_col} and/or {cust_name_col} as appropriate.

**VAT / tax number:** VAT Reg. No is the VAT or tax number; use column STCEG (in KNA1, VBRK, PSIF_INV_HDR, BSAD as available).

**Annual sales revenue:** Use KNA1.UMSA1 for the annual sales revenue of the customer.

**Value determination:** Use VBRK and VBRP (best tables for value). Join VBRK.VBELN = VBRP.VBELN. VBRP has line-level NETWR and MATNR (product).

**Comparing sales orders and invoices:** Always show billing document number (VBRK.VBELN), sales order number (VBRP.AUBEL or VBAK.VBELN only), purchase order number (VBAK.BSTNK), and for process flow delivery number (LIKP.VBELN from LIKP table). Do not use VBRP.AUPOS or POSNR for order or invoice number — AUPOS is position number only. Join VBRP to VBAK: VBRP.AUBEL = VBAK.VBELN. To get delivery number: join VBRP to VBFA (VBRP.VBELN = VBFA.VBELN, VBRP.POSNR = VBFA.POSNN) and VBFA to LIKP (VBFA.VBELV = LIKP.VBELN).

**Process flow (required when referencing invoice/billing):** The VBFA table shows the flow distinctively (preceding document VBELV/POSNV, subsequent document VBELN/POSNN). When referencing invoice number or billing document, always include for the link of the process flow: (1) invoice/billing document number (VBRK.VBELN or VBRP.VBELN), (2) sales order number (VBRP.AUBEL or VBAK.VBELN), (3) delivery number from LIKP (LIKP.VBELN, via VBFA), (4) purchase order number from VBAK (VBAK.BSTNK). Use VBFA to link billing to delivery: VBFA.VBELN = VBRP.VBELN, VBFA.VBELV = LIKP.VBELN.

**Order, delivery, invoice — link by purchase order number:** The purchase order number (VBAK.BSTNK) is consistent through order, delivery, and invoice and links all three documents. When showing billing/invoice results always SELECT: billing document number (VBRK.VBELN or VBRP.VBELN), sales order number (VBRP.AUBEL or VBAK.VBELN only — never AUPOS; AUPOS is position number, not order or invoice number), delivery number (LIKP.VBELN), purchase order number (VBAK.BSTNK). Join VBRP to VBAK on VBRP.AUBEL = VBAK.VBELN. Join VBRP to VBFA and VBFA to LIKP to include delivery number (LIKP.VBELN).

**Invoice value per customer (name or number) and e-invoicing consistency:** Use e-invoicing tables PSIF_INV_HDR and PSIF_INV_ITEM with KNA1 for customer name and number. Join PSIF_INV_HDR.KUNNR = KNA1.KUNNR.

**Discrepancies (impact on payment terms and revenues):** Discrepancies are between (1) invoices created from VBRK and VBRP (billing) and (2) invoices sent and when they were sent out (PSIF_INV_HDR, PSIF_ACK). Compare on VBELN. Include reason for delay when available: use PSIF_ACK.ACK_TEXT or ACK_ID to attribute whether the issue is due to supplier delay or another reason. Include VBELN, value (NETWR), send date (e.g. PSIF_ACK.CRDAT/CRTIM), and delay reason so payment terms and revenue impact can be analysed and attributed (supplier delay vs other).

**Invoices created but not sent (by customer):** Start from VBRK (invoices created). Use a LEFT JOIN to PSIF_INV_HDR (or PSIF_ACK) on VBRK.VBELN = PSIF_INV_HDR.VBELN, and add a filter WHERE PSIF_INV_HDR.VBELN IS NULL (or PSIF_ACK.VBELN IS NULL) so only invoices that exist in VBRK but have no row in the sent table are returned. Join KNA1 on VBRK.KUNAG = KNA1.KUNNR for customer number and name. In the JSON spec: set the join to the sent table with "type": "left", and add a filter with "lhs": "PSIF_INV_HDR.VBELN", "operator": "IS NULL", "rhs": "NULL". Select columns from VBRK (VBELN, NETWR, FKDAT, KUNAG) and KNA1 (KUNNR, NAME1) so the result shows invoices by customer that were created but not sent.

**Paid on time vs delayed payments and which customers caused delay:** Compare invoices paid on time vs delayed: use VBRK (invoice, FKDAT = billing date, ZTERM = payment terms) and BSAD (clearing/payment: AUGDT = clearing date, VBELN, KUNNR). Join VBRK.VBELN = BSAD.VBELN and BSAD.KUNNR = KNA1.KUNNR for customer name (NAME1) and number. Compare AUGDT to due date (or FKDAT + terms) to flag paid on time vs delayed. Include customer number (KUNNR) and customer name (NAME1) so which customers have caused the delay in payments is visible; bring visibility by selecting VBELN, customer name, customer number, invoice value, billing date, clearing date (AUGDT), and whether paid on time or delayed.

**Credit risk and customer credit data:** Use KNKK (credit master data: credit limit KLIMK, credit exposure SKFOR = total value of open orders/deliveries, risk category CTLPC per credit control area KKBER), KNKA (central data: risk class, credit group, overall limits KLIMG/KLIME, WAERS), and T691A (risk categories; join KNKK.CTLPC = T691A.CTLPC AND KNKK.KKBER = T691A.KKBER). Join KNKK and KNKA to KNA1 on KUNNR. **Use only actual SAP column names:** From KNKK use KLIMK (credit limit), SKFOR (credit exposure — do not use a column named EXPOSURE), CTLPC (risk category — do not use RISK_CATEGORY). There is no column CHECK_RESULTS in KNKK; omit it or use other KNKK/KNKA fields from the column mappings only.""" + (
    " When UKM_BP_CMS_SGM is available (S/4HANA FSCM), also include it and join UKM_BP_CMS_SGM.PARTNER = KNA1.KUNNR; select CREDIT_LIMIT, BLOCK_REASON, and other FSCM fields so both ERP and FSCM credit data are shown."
    if fscm_credit_table_available() else
    " Do NOT include table UKM_BP_CMS_SGM — it is not available in this system (ERP only)."
) + """

**Comparisons by industry, customer, value, products:** Use VBRK and VBRP for value; KNA1 for customer/industry (BRSCH). Do NOT add a filter on country (LAND1) unless the user asks. Full data.

**Best products by value and industry:** Use VBRK and VBRP (columns: VBRP.MATNR, VBRP.NETWR or VBRK.NETWR); add KNA1 and join VBRK.KUNAG = KNA1.KUNNR for industry (KNA1.BRSCH). Join VBRK.VBELN = VBRP.VBELN. Do NOT add filters so the query returns data. Order by NETWR descending.

Task:
- Choose relevant columns.
- When the query involves billing documents or invoices (VBRK, VBRP): include VBAK and join VBRP.AUBEL = VBAK.VBELN; for process flow also include VBFA and LIKP (VBRP to VBFA, VBFA to LIKP) to get delivery number (LIKP.VBELN). In "columns" include at least: billing document number (VBRK.VBELN or VBRP.VBELN), sales order number (VBRP.AUBEL or VBAK.VBELN only — never use AUPOS or POSNR; AUPOS is position number only), delivery number (LIKP.VBELN), purchase order number (VBAK.BSTNK).
- When the query involves credit risk, customer credit, credit limits, exposure (e.g. open orders/deliveries), risk categories, or creditworthiness: include KNKK and KNKA with KNA1; join KNKK to T691A on KNKK.CTLPC = T691A.CTLPC AND KNKK.KKBER = T691A.KKBER. In "columns" use only actual SAP field names from the column mappings: from KNKK use KLIMK, SKFOR (credit exposure — never use column name EXPOSURE), CTLPC (risk category — never use RISK_CATEGORY); from KNKA use KLIMG, KLIME, WAERS; from KNA1 use KUNNR, NAME1. Do not use EXPOSURE, RISK_CATEGORY, or CHECK_RESULTS — those columns do not exist. For filters: if comparing credit exposure to credit limit use lhs "KNKK.SKFOR", operator ">", rhs "KNKK.KLIMK" (rhs must be the column reference, not the string 'KNKK.KLIMK'). Only add filters when the user explicitly requests a condition.""" + (
    " When UKM_BP_CMS_SGM is available, also include it and join UKM_BP_CMS_SGM.PARTNER = KNA1.KUNNR; add columns like CREDIT_LIMIT, BLOCK_REASON."
    if fscm_credit_table_available() else
    " Do not add UKM_BP_CMS_SGM (not available in this system)."
) + """
- When the user asks whether a specific invoice number has high credit risk: include VBRK (invoice), KNA1 (customer), KNKK and optionally KNKA and T691A (credit/risk). Join VBRK.KUNAG = KNA1.KUNNR and KNKK.KUNNR = KNA1.KUNNR. Add a filter with lhs "VBRK.VBELN", operator "=", rhs the invoice number as digits (e.g. 90035998) so only that invoice is checked. Select VBRK.VBELN, KNA1.KUNNR, KNA1.NAME1, KNKK.KLIMK, KNKK.SKFOR, KNKK.CTLPC so the user can see credit limit, exposure, and risk category for that invoice.
- All columns used in "columns", "filters", and "order_by" **must** be present in the column mappings
- For each column, use "name" exactly as in the column mappings for that table (e.g. KNKK: use KLIMK, SKFOR, CTLPC — never EXPOSURE, RISK_CATEGORY, or CHECK_RESULTS)
- Identify tables needed.
- Identify joins.
- For "best products by value and industry" leave filters empty. Add other filters only when the user explicitly requests them (e.g. a specific country).
- Return JSON:

{{
  "query": "...",  # optional raw SQL
  "tables": [{{ "name": "...", "description": "..." }}],
  "columns": [{{ "table": "...", "name": "...", "description": "..." }}],
  "joins": [{{ "left": "...", "right": "...", "on": "...", "type": "inner" or "left" (use "left" for created-but-not-sent) }}],
  "filters": [{{ "lhs": "...", "operator": "...", "rhs": "..." }}]  (use operator "IS NULL" and rhs "NULL" for created-but-not-sent; for column-to-column comparison use rhs as column reference e.g. "KNKK.KLIMK" not a quoted string),
"order_by": [{{ "table": "...", "column": "...", "direction": "..." }}],
  "limit": 100
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content =response.choices[0].message.content
    try:
        spec = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            spec = json.loads(m.group(0))
        else:
            return None
    spec = ensure_delivery_chain_in_spec(spec)
    return spec

def ensure_delivery_chain_in_spec(json_spec: dict) -> dict:
    """When VBRP and LIKP are in the spec, ensure the delivery link is complete: VBFA (VBRP->VBFA->LIKP), LIKP-LIPS, and LIKP.VBELN in columns."""
    if not json_spec:
        return json_spec
    columns = json_spec.get("columns", [])
    joins = json_spec.get("joins", [])
    tables_info = json_spec.get("tables", [])
    tables_from_cols = {c.get("table") for c in columns if c.get("table")}
    tables_from_joins = set()
    for j in joins:
        tables_from_joins.add(j.get("left"))
        tables_from_joins.add(j.get("right"))
    tables_from_info = {t.get("name") for t in tables_info if isinstance(t, dict) and t.get("name")}
    all_tables = tables_from_cols | tables_from_joins | tables_from_info
    if "VBRP" not in all_tables or "LIKP" not in all_tables:
        return json_spec
    join_pairs = {(j.get("left"), j.get("right")) for j in joins}
    added_joins = []
    if ("VBRP", "VBFA") not in join_pairs and ("VBFA", "VBRP") not in join_pairs:
        added_joins.append({"left": "VBRP", "right": "VBFA", "type": "left"})
    if ("VBFA", "LIKP") not in join_pairs and ("LIKP", "VBFA") not in join_pairs:
        added_joins.append({"left": "VBFA", "right": "LIKP", "type": "left"})
    if ("LIKP", "LIPS") not in join_pairs and ("LIPS", "LIKP") not in join_pairs:
        added_joins.append({"left": "LIKP", "right": "LIPS", "type": "left"})
    if "VBAK" in all_tables and "LIPS" in (all_tables | {j["right"] for j in added_joins}):
        if ("LIPS", "VBAK") not in join_pairs and ("VBAK", "LIPS") not in join_pairs:
            added_joins.append({"left": "LIPS", "right": "VBAK", "type": "left"})
    if added_joins:
        json_spec.setdefault("joins", [])
        json_spec["joins"].extend(added_joins)
    has_likp_vbeln = any(c.get("table") == "LIKP" and c.get("name") == "VBELN" for c in columns)
    if not has_likp_vbeln and "LIKP" in all_tables:
        json_spec.setdefault("columns", [])
        json_spec["columns"].append({"table": "LIKP", "name": "VBELN", "description": "delivery_number"})
    return json_spec

def convert_date_to_yyyymmdd(date_str: str) -> str:
    s = date_str.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except:
            pass
    if re.match(r"^\d{4}$", s):
        return f"{s}0101"
    if re.match(r"^\d{8}$", s):
        return s
    return s

def fix_date_filters(json_spec: dict):
    if not json_spec or "filters" not in json_spec:
        return json_spec
    for filt in json_spec["filters"]:
        lhs = filt.get("lhs", "")
        rhs = filt.get("rhs", "").strip("'\"")
        col = lhs.split(".")[-1]
        if col.upper() in DATE_COLUMNS:
            new_rhs = convert_date_to_yyyymmdd(rhs)
            filt["rhs"] = f"'{new_rhs}'"
    return json_spec

def format_table_name(tbl: str) -> str:
    if tbl == "PSIF_INV_HDR":
        return "[erp].[/PSIF/INV_HDR]"
    if tbl == "PSIF_INV_ITEM":
        return "[erp].[/PSIF/INV_ITEM]"
    if tbl == "PSIF_ACK":
        return "[erp].[/PSIF/ACK]"
    if tbl == "PSIF_SLS_HDR":
        return "[erp].[/PSIF/SLS_HDR]"
    if tbl == "PSIF_SLS_ITEM":
        return "[erp].[/PSIF/SLS_ITEM]"
    return f"[erp].[{tbl}]"


def fscm_credit_table_available() -> bool:
    """Return True if UKM_BP_CMS_SGM exists so we can use both ERP (KNKK/KNKA) and S/4HANA FSCM credit data."""
    if INCLUDE_FSCM_CREDIT_TABLE is True:
        return True
    if INCLUDE_FSCM_CREDIT_TABLE is False:
        return False
    try:
        if "_fscm_credit_table_available" in st.session_state:
            return st.session_state["_fscm_credit_table_available"]
    except Exception:
        pass
    try:
        conn = pyodbc.connect(CONN_STR)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM [erp].[UKM_BP_CMS_SGM] WHERE 1=0")
        conn.close()
        try:
            st.session_state["_fscm_credit_table_available"] = True
        except Exception:
            pass
        return True
    except Exception:
        try:
            st.session_state["_fscm_credit_table_available"] = False
        except Exception:
            pass
        return False

def make_alias_for_table(tbl: str, used_aliases: set):
    base = tbl[0].lower()
    alias = base
    i = 1
    while alias in used_aliases:
        alias = f"{base}{i}"
        i += 1
    used_aliases.add(alias)
    return alias

def replace_table_names_with_aliases(expr: str, alias_map: dict):
    if not expr:
        return expr
    out = expr
    for tbl, alias in alias_map.items():
        out = re.sub(rf"\b{tbl}\.", f"{alias}.", out)
    return out

# Known wrong column names in join conditions (LLM typos / wrong names)
# VBRK has KUNAG (sold-to), not KNCLI - apply whenever VBRK is in the join
JOIN_COLUMN_CORRECTIONS = [
    ("KNCLI", "KUNAG"),
]
def normalize_join_on(on_expr: str, left_table: str, right_table: str) -> str:
    if not on_expr:
        return on_expr
    out = on_expr
    # Apply global corrections (e.g. KNCLI -> KUNAG for any join involving VBRK)
    if "VBRK" in (left_table, right_table):
        for wrong, correct in JOIN_COLUMN_CORRECTIONS:
            out = re.sub(rf"\b{wrong}\b", correct, out)
    return out

# Preferred join conditions when LLM picks wrong key (e.g. customer vs document)
# PSIF_INV_HDR and VBRK link on billing document (VBELN), not customer
# ADDR1_DATA is linked to KNA1 in SAP (typically on ADRNR = address number)
# PSIF_INV_HDR and PSIF_INV_ITEM link on document number VBELN (products by value and industry)
# VBRK and VBRP: preferred for value determination (billing header + item)
# VBAK and VBAP: sales order header and item (VBELN link)
# PSIF_SLS_HDR and PSIF_SLS_ITEM: eOrder tables (counterpart to VBAK/VBAP)
# Sales order vs invoice: link invoice (VBRP) to sales order (VBAK) via VBRP.AUBEL = VBAK.VBELN
# Order-delivery-invoice: VBFA shows the flow distinctively; purchase order number (VBAK.BSTNK) links all three; delivery number from LIKP via VBFA (VBFA.VBELN = billing, VBFA.VBELV = delivery = LIKP.VBELN)
PREFERRED_JOIN_ONS = {
    ("PSIF_INV_HDR", "VBRK"): "PSIF_INV_HDR.VBELN = VBRK.VBELN",
    ("VBRK", "PSIF_INV_HDR"): "VBRK.VBELN = PSIF_INV_HDR.VBELN",
    ("VBRK", "PSIF_ACK"): "VBRK.VBELN = PSIF_ACK.VBELN",
    ("PSIF_ACK", "VBRK"): "PSIF_ACK.VBELN = VBRK.VBELN",
    ("VBRK", "BSAD"): "VBRK.VBELN = BSAD.VBELN",
    ("BSAD", "VBRK"): "BSAD.VBELN = VBRK.VBELN",
    ("BSAD", "KNA1"): "BSAD.KUNNR = KNA1.KUNNR",
    ("KNA1", "BSAD"): "KNA1.KUNNR = BSAD.KUNNR",
    ("VBRK", "VBRP"): "VBRK.VBELN = VBRP.VBELN",
    ("VBRP", "VBRK"): "VBRP.VBELN = VBRK.VBELN",
    ("VBRP", "VBAK"): "VBRP.AUBEL = VBAK.VBELN",
    ("VBAK", "VBRP"): "VBAK.VBELN = VBRP.AUBEL",
    ("VBRP", "VBAP"): "VBRP.AUBEL = VBAP.VBELN AND VBRP.AUPOS = VBAP.POSNR",
    ("VBAP", "VBRP"): "VBAP.VBELN = VBRP.AUBEL AND VBAP.POSNR = VBRP.AUPOS",
    ("VBAK", "VBAP"): "VBAK.VBELN = VBAP.VBELN",
    ("VBAP", "VBAK"): "VBAP.VBELN = VBAK.VBELN",
    ("VBAP", "VBEP"): "VBAP.VBELN = VBEP.VBELN AND VBAP.POSNR = VBEP.POSNR",
    ("VBEP", "VBAP"): "VBEP.VBELN = VBAP.VBELN AND VBEP.POSNR = VBAP.POSNR",
    ("VBAK", "VBEP"): "VBAK.VBELN = VBEP.VBELN",
    ("VBEP", "VBAK"): "VBEP.VBELN = VBAK.VBELN",
    ("VBAK", "VBPA"): "VBAK.VBELN = VBPA.VBELN",
    ("VBPA", "VBAK"): "VBPA.VBELN = VBAK.VBELN",
    ("VBAK", "VBUK"): "VBAK.VBELN = VBUK.VBELN",
    ("VBUK", "VBAK"): "VBUK.VBELN = VBAK.VBELN",
    ("VBAP", "VBUP"): "VBAP.VBELN = VBUP.VBELN AND VBAP.POSNR = VBUP.POSNR",
    ("VBUP", "VBAP"): "VBUP.VBELN = VBAP.VBELN AND VBUP.POSNR = VBAP.POSNR",
    ("VBAK", "VBFA"): "VBAK.VBELN = VBFA.VBELV",
    ("VBFA", "VBAK"): "VBFA.VBELV = VBAK.VBELN",
    ("VBAP", "VBFA"): "VBAP.VBELN = VBFA.VBELV AND VBAP.POSNR = VBFA.POSNV",
    ("VBFA", "VBAP"): "VBFA.VBELV = VBAP.VBELN AND VBFA.POSNV = VBAP.POSNR",
    ("PSIF_SLS_HDR", "PSIF_SLS_ITEM"): "PSIF_SLS_HDR.VBELN = PSIF_SLS_ITEM.VBELN",
    ("PSIF_SLS_ITEM", "PSIF_SLS_HDR"): "PSIF_SLS_ITEM.VBELN = PSIF_SLS_HDR.VBELN",
    ("PSIF_INV_HDR", "PSIF_INV_ITEM"): "PSIF_INV_HDR.VBELN = PSIF_INV_ITEM.VBELN",
    ("PSIF_INV_ITEM", "PSIF_INV_HDR"): "PSIF_INV_ITEM.VBELN = PSIF_INV_HDR.VBELN",
    ("PSIF_INV_HDR", "KNA1"): "PSIF_INV_HDR.KUNNR = KNA1.KUNNR",
    ("KNA1", "PSIF_INV_HDR"): "KNA1.KUNNR = PSIF_INV_HDR.KUNNR",
    ("VBRK", "KNA1"): "VBRK.KUNAG = KNA1.KUNNR",
    ("KNA1", "VBRK"): "KNA1.KUNNR = VBRK.KUNAG",
    ("LIKP", "LIPS"): "LIKP.VBELN = LIPS.VBELN",
    ("LIPS", "LIKP"): "LIPS.VBELN = LIKP.VBELN",
    ("VBRP", "VBFA"): "VBRP.VBELN = VBFA.VBELN AND VBRP.POSNR = VBFA.POSNN",
    ("VBFA", "VBRP"): "VBFA.VBELN = VBRP.VBELN AND VBFA.POSNN = VBRP.POSNR",
    ("VBFA", "LIKP"): "VBFA.VBELV = LIKP.VBELN",
    ("LIKP", "VBFA"): "LIKP.VBELN = VBFA.VBELV",
    ("LIPS", "VBAK"): "LIPS.VGBEL = VBAK.VBELN",
    ("VBAK", "LIPS"): "VBAK.VBELN = LIPS.VGBEL",
    ("LIPS", "VBAP"): "LIPS.VGBEL = VBAP.VBELN AND LIPS.VGPOS = VBAP.POSNR",
    ("VBAP", "LIPS"): "VBAP.VBELN = LIPS.VGBEL AND VBAP.POSNR = LIPS.VGPOS",
    ("ADDR1_DATA", "KNA1"): "ADDR1_DATA.ADRNR = KNA1.ADRNR",
    ("KNA1", "ADDR1_DATA"): "KNA1.ADRNR = ADDR1_DATA.ADRNR",
    ("KNKK", "KNA1"): "KNKK.KUNNR = KNA1.KUNNR",
    ("KNA1", "KNKK"): "KNA1.KUNNR = KNKK.KUNNR",
    ("KNKA", "KNA1"): "KNKA.KUNNR = KNA1.KUNNR",
    ("KNA1", "KNKA"): "KNA1.KUNNR = KNKA.KUNNR",
    ("UKM_BP_CMS_SGM", "KNA1"): "UKM_BP_CMS_SGM.PARTNER = KNA1.KUNNR",
    ("KNA1", "UKM_BP_CMS_SGM"): "KNA1.KUNNR = UKM_BP_CMS_SGM.PARTNER",
    ("KNKK", "T691A"): "KNKK.CTLPC = T691A.CTLPC AND KNKK.KKBER = T691A.KKBER",
    ("T691A", "KNKK"): "T691A.CTLPC = KNKK.CTLPC AND T691A.KKBER = KNKK.KKBER",
}

def json_to_sql(json_spec: dict) -> str:
    with st.expander("🔧 Debug: JSON Spec", expanded=False):
        st.json(json_spec or {})

    columns = json_spec.get("columns", [])
    joins = json_spec.get("joins", [])
    tables_info = json_spec.get("tables", [])
    limit = int(json_spec.get("limit", 100))

    # collect all tables
    all_tables = set()
    for j in joins:
        all_tables.add(j["left"])
        all_tables.add(j["right"])
    for c in columns:
        all_tables.add(c["table"])
    for t in tables_info:
        all_tables.add(t["name"])

    if not all_tables:
        st.error("No tables found in spec.")
        return ""

    # alias map
    used_aliases = set()
    alias_map = {tbl: make_alias_for_table(tbl, used_aliases) for tbl in sorted(all_tables)}

    # SELECT
    select_parts = []
    used_col_aliases = set()
    for col in columns:
        tbl, nm = col["table"], col["name"]
        alias = alias_map[tbl]
        human = col.get("description") or f"{tbl}_{nm}"
        human_safe = re.sub(r"[^\w]", "_", human)
        if human_safe in used_col_aliases:
            suffix = 1
            while f"{human_safe}_{suffix}" in used_col_aliases:
                suffix += 1
            human_safe = f"{human_safe}_{suffix}"
        used_col_aliases.add(human_safe)
        select_parts.append(f"    {alias}.[{nm}] AS [{human_safe}]")

    sql_lines = [f"SELECT TOP {limit}"]
    sql_lines.append(",\n".join(select_parts))

    # Base table
    base = joins[0]["left"] if joins else (tables_info[0]["name"] if tables_info else list(all_tables)[0])
    sql_lines.append(f"\nFROM {format_table_name(base)} AS {alias_map[base]}")

    added = {base}
    # Build reverse alias map (alias -> table) to find tables referenced in ON clauses
    reverse_alias_map = {alias: tbl for tbl, alias in alias_map.items()}

    def tables_referenced_in_on(on_expr_str: str) -> set:
        """Return set of table names that appear in on_expr (as table or alias prefix before a dot)."""
        if not on_expr_str:
            return set()
        prefixes = set(re.findall(r"\b(\w+)\.", on_expr_str))
        out = set()
        for px in prefixes:
            if px in reverse_alias_map:
                out.add(reverse_alias_map[px])
            elif px in all_tables:
                out.add(px)
        return out

    # Order joins so a table is never referenced in ON before it is added (avoids "identifier could not be bound")
    pending = [j for j in joins if j["right"] not in added]
    while pending:
        chosen = None
        for j in pending:
            left = j.get("left", base)
            if left not in added:
                continue
            right = j["right"]
            on_expr = PREFERRED_JOIN_ONS.get((left, right)) or PREFERRED_JOIN_ONS.get((right, left))
            if not on_expr:
                on_expr = normalize_join_on(j.get("on", ""), left, right)
            refs = tables_referenced_in_on(on_expr)
            # ON may reference left and right; right is being added — require all other refs to be in added
            if refs and not (refs - {right}).issubset(added):
                continue  # ON references a table not yet added — skip until later
            chosen = j
            break
        if chosen is None:
            chosen = pending[0]
        pending.remove(chosen)
        right = chosen["right"]
        if right in added:
            continue
        left = chosen.get("left", base)
        on_expr = PREFERRED_JOIN_ONS.get((left, right)) or PREFERRED_JOIN_ONS.get((right, left))
        if not on_expr:
            on_expr = normalize_join_on(chosen.get("on", ""), left, right)
        join_type = (chosen.get("type") or "inner").strip().lower()
        join_keyword = "LEFT JOIN" if join_type == "left" else "JOIN"
        sql_lines.append(f"\n{join_keyword} {format_table_name(right)} AS {alias_map[right]}")
        sql_lines.append(f"    ON {replace_table_names_with_aliases(on_expr, alias_map)}")
        added.add(right)

    # Add any joins that weren't processed (e.g. right already in added)
    for j in joins:
        right = j["right"]
        if right in added:
            continue
        left = j.get("left", base)
        on_expr = PREFERRED_JOIN_ONS.get((left, right)) or PREFERRED_JOIN_ONS.get((right, left))
        if not on_expr:
            on_expr = normalize_join_on(j.get("on", ""), left, right)
        join_type = (j.get("type") or "inner").strip().lower()
        join_keyword = "LEFT JOIN" if join_type == "left" else "JOIN"
        sql_lines.append(f"\n{join_keyword} {format_table_name(right)} AS {alias_map[right]}")
        sql_lines.append(f"    ON {replace_table_names_with_aliases(on_expr, alias_map)}")
        added.add(right)

    # Add missing tables (fallback): KNA1 links to VBRK on KUNAG = KUNNR; others on VBELN
    for tbl in (all_tables - added):
        on_expr = PREFERRED_JOIN_ONS.get((base, tbl)) or PREFERRED_JOIN_ONS.get((tbl, base))
        if not on_expr:
            on_expr = f"{alias_map[base]}.[VBELN] = {alias_map[tbl]}.[VBELN]"
        else:
            on_expr = replace_table_names_with_aliases(on_expr, alias_map)
        sql_lines.append(f"\nJOIN {format_table_name(tbl)} AS {alias_map[tbl]}")
        sql_lines.append(f"    ON {on_expr}")
        added.add(tbl)

    # WHERE: quote rhs so nvarchar columns (e.g. KUNNR) aren't converted to int (avoids 'R3002'->int error)
    # If rhs looks like a column reference (TABLE.COLUMN), use it as-is with alias replacement — do not quote as string
    from config import CUSTOMER_NUMBER_LENGTH
    def format_filter_rhs(rhs, lhs_sql="", alias_map=alias_map):
        if rhs is None:
            return None
        s = str(rhs).strip()
        if not s:
            return None
        if (s.startswith("'") and s.endswith("'")) or (s.startswith("N'") and s.endswith("'")):
            return rhs
        if s.upper() in ("NULL", "TRUE", "FALSE"):
            return s
        # Column-to-column comparison: rhs like "KNKK.KLIMK" or "TABLE.COL" — use as column reference, not string
        if "." in s and not s.isdigit():
            parts = s.split(".", 1)
            if len(parts) == 2 and parts[0].strip().upper() in [t.upper() for t in alias_map]:
                return replace_table_names_with_aliases(s, alias_map)
        # If filtering on KUNNR (customer number), pad with leading zeros to CUSTOMER_NUMBER_LENGTH
        if "KUNNR" in lhs_sql.upper() and s.isdigit():
            s = s.zfill(CUSTOMER_NUMBER_LENGTH)
        escaped = s.replace("'", "''")
        return f"N'{escaped}'"
    conds = []
    for f in json_spec.get("filters", []):
        lhs = replace_table_names_with_aliases(f.get("lhs"), alias_map)
        op, rhs = f.get("operator"), f.get("rhs")
        if not lhs or not op:
            continue
        op_upper = (op or "").strip().upper()
        # IS NULL: operator "IS NULL" or rhs is null/NULL
        if op_upper == "IS NULL" or (rhs is None) or (isinstance(rhs, str) and str(rhs).strip().upper() == "NULL"):
            conds.append(f"{lhs} IS NULL")
            continue
        if rhs is None:
            continue
        # Document number (invoice, order, delivery): match both 10-char and raw so e.g. 90035998 finds 0090035998
        rhs_str = str(rhs).strip().strip("'\"").strip()
        if op_upper == "=" and rhs_str.isdigit() and ("VBELN" in lhs.upper() or "AUBEL" in lhs.upper()):
            normalized = rhs_str.zfill(10)
            variants = [normalized, rhs_str] if normalized != rhs_str else [normalized]
            variants_escaped = [v.replace("'", "''") for v in variants]
            in_list = ", ".join([f"N'{v}'" for v in variants_escaped])
            conds.append(f"({lhs} IN ({in_list}))")
            continue
        rhs_sql = format_filter_rhs(rhs, lhs, alias_map)
        if rhs_sql is not None:
            conds.append(f"{lhs} {op} {rhs_sql}")
    if conds:
        sql_lines.append("\nWHERE " + " AND ".join(conds))

    # ORDER BY: only use columns that exist (selected table columns or output aliases)
    # Fallback: map revenue-like names to base column VBRK.NETWR when VBRK is in the query
    ORDER_BY_FALLBACK = {"total_revenue": ("VBRK", "NETWR"), "revenue": ("VBRK", "NETWR"), "net_revenue": ("VBRK", "NETWR")}
    selected_table_columns = {(col["table"], col["name"]) for col in columns}
    if "order_by" in json_spec:
        parts = []
        for ob in json_spec["order_by"]:
            if isinstance(ob, str):
                parts.append(ob)
            else:
                t, c, d = ob.get("table"), ob.get("column"), ob.get("direction", "ASC")
                if not c:
                    continue
                if (t and (t, c) in selected_table_columns and t in alias_map):
                    parts.append(f"{alias_map[t]}.[{c}] {d}")
                elif c in used_col_aliases:
                    parts.append(f"[{c}] {d}")
                else:
                    # Fallback: e.g. total_revenue -> base column VBRK.NETWR when in SELECT
                    fallback = ORDER_BY_FALLBACK.get(c)
                    if fallback:
                        ftbl, fcol = fallback
                        if ftbl in alias_map and (ftbl, fcol) in selected_table_columns:
                            parts.append(f"{alias_map[ftbl]}.[{fcol}] {d}")
        if parts:
            sql_lines.append("\nORDER BY " + ", ".join(parts))

    sql = "\n".join(sql_lines) + ";"
    with st.expander("🔧 Debug: Generated SQL", expanded=False):
        st.code(sql, language="sql")
    return sql

def strip_where_from_sql(sql: str) -> str:
    """Remove WHERE clause so the same query can be run without filters (to check if any data exists)."""
    if not sql or "WHERE" not in sql.upper():
        return sql
    upper = sql.upper()
    idx = upper.find("\nWHERE ")
    if idx == -1:
        idx = upper.find(" WHERE ")
    if idx == -1:
        return sql
    before_where = sql[:idx]
    after_where = sql[idx + 7:]  # after " WHERE "
    order_pos = after_where.upper().find("\nORDER BY")
    if order_pos != -1:
        after_where = after_where[order_pos:]
    else:
        semi = after_where.find(";")
        after_where = after_where[semi:] if semi != -1 else ";"
    return before_where + after_where


def get_core_billing_sql(limit: int = 100) -> str:
    """Return a minimal SQL that joins VBRK, VBRP, VBAK, KNA1 and optionally VBFA, LIKP for process flow (billing + sales order + PO + customer + delivery number)."""
    return f"""SELECT TOP {limit}
    v1.[VBELN] AS [Billing_document_number],
    v2.[AUBEL] AS [Sales_order_number],
    likp.[VBELN] AS [Delivery_number],
    v.[BSTNK] AS [Customer_purchase_order_number],
    v1.[NETWR] AS [Net_value],
    v1.[FKDAT] AS [Billing_date],
    k.[KUNNR] AS [Customer_number],
    k.[NAME1] AS [Customer_name]
FROM {format_table_name('VBRK')} AS v1
JOIN {format_table_name('VBRP')} AS v2 ON v1.VBELN = v2.VBELN
JOIN {format_table_name('VBAK')} AS v ON v2.AUBEL = v.VBELN
LEFT JOIN {format_table_name('VBFA')} AS vbfa ON v2.VBELN = vbfa.VBELN AND v2.POSNR = vbfa.POSNN
LEFT JOIN {format_table_name('LIKP')} AS likp ON vbfa.VBELV = likp.VBELN
JOIN {format_table_name('KNA1')} AS k ON v1.KUNAG = k.KUNNR
ORDER BY v1.NETWR DESC;"""


# Document number in SAP is often 10 characters (e.g. VBELN, AUBEL)
SALES_ORDER_NUMBER_LENGTH = 10
DOCUMENT_NUMBER_LENGTH = 10
POSITION_LENGTH = 6


def trace_document_number(value: str):
    """Trace where a document number appears across VBRK, VBRP, /PSIF/INV_HDR, /PSIF/INV_ITEM, VBAP, VBFA, VBAK, LIKP, LIPS, BSAD, BSID.
    Returns (normalized_value, results_dict) where results_dict keys are like 'VBRK.VBELN', 'VBRP.AUBEL', etc., and values are DataFrames (or None).
    Uses both normalized (10-char) and raw value so delivery/invoice numbers stored without leading zeros (e.g. 80003409) are found."""
    raw = (value or "").strip()
    if not raw:
        return None, {}
    normalized = raw.zfill(DOCUMENT_NUMBER_LENGTH) if raw.isdigit() else raw
    safe = normalized.replace("'", "''")
    # Variants: search with both normalized and raw so we find e.g. 80003409 in LIKP whether stored as 0080003409 or 80003409
    doc_variants = [normalized, raw] if raw != normalized else [normalized]
    doc_variants_safe = [v.replace("'", "''") for v in doc_variants]
    in_clause = ", ".join([f"N'{v}'" for v in doc_variants_safe])
    results = {}

    # VBRK.VBELN (billing document header)
    try:
        q = f"SELECT TOP 50 VBELN, KUNAG, FKDAT, NETWR FROM {format_table_name('VBRK')} WHERE VBELN = N'{safe}'"
        results["VBRK.VBELN"] = run_sql(q)
    except Exception:
        results["VBRK.VBELN"] = None

    # VBRP.VBELN (billing document item), VBRP.AUBEL (sales order), VBRP.AUPOS (position)
    try:
        q = f"SELECT TOP 50 VBELN, AUBEL, AUPOS, MATNR, NETWR FROM {format_table_name('VBRP')} WHERE VBELN = N'{safe}'"
        results["VBRP.VBELN"] = run_sql(q)
    except Exception:
        results["VBRP.VBELN"] = None
    try:
        q = f"SELECT TOP 50 VBELN, AUBEL, AUPOS, MATNR, NETWR FROM {format_table_name('VBRP')} WHERE AUBEL = N'{safe}'"
        results["VBRP.AUBEL"] = run_sql(q)
    except Exception:
        results["VBRP.AUBEL"] = None
    aupos_variants = [safe, raw]
    if raw.isdigit():
        aupos_variants.append(raw.zfill(POSITION_LENGTH))
    aupos_safe = [v.replace("'", "''") for v in aupos_variants]
    aupos_cond = " OR ".join([f"AUPOS = N'{v}'" for v in aupos_safe])
    try:
        q = f"SELECT TOP 50 VBELN, AUBEL, AUPOS, MATNR, NETWR FROM {format_table_name('VBRP')} WHERE {aupos_cond}"
        results["VBRP.AUPOS"] = run_sql(q)
    except Exception:
        results["VBRP.AUPOS"] = None

    # /PSIF/INV_HDR (e-invoice header)
    try:
        q = f"SELECT TOP 50 VBELN, KUNNR, NETWR FROM {format_table_name('PSIF_INV_HDR')} WHERE VBELN = N'{safe}'"
        results["PSIF_INV_HDR.VBELN"] = run_sql(q)
    except Exception:
        results["PSIF_INV_HDR.VBELN"] = None

    # /PSIF/INV_ITEM (e-invoice item)
    try:
        q = f"SELECT TOP 50 VBELN, MATNR, NETWR FROM {format_table_name('PSIF_INV_ITEM')} WHERE VBELN = N'{safe}'"
        results["PSIF_INV_ITEM.VBELN"] = run_sql(q)
    except Exception:
        results["PSIF_INV_ITEM.VBELN"] = None

    # VBAK.VBELN (sales order header)
    try:
        q = f"SELECT TOP 50 VBELN, BSTNK, KUNNR, AUDAT, NETWR FROM {format_table_name('VBAK')} WHERE VBELN = N'{safe}'"
        results["VBAK.VBELN"] = run_sql(q)
    except Exception:
        results["VBAK.VBELN"] = None

    # VBAP.VBELN (sales order item), VBAP.POSNR (position)
    try:
        q = f"SELECT TOP 50 VBELN, POSNR, MATNR, NETWR FROM {format_table_name('VBAP')} WHERE VBELN = N'{safe}'"
        results["VBAP.VBELN"] = run_sql(q)
    except Exception:
        results["VBAP.VBELN"] = None
    posnr_safe = [v.replace("'", "''") for v in aupos_variants]
    posnr_cond = " OR ".join([f"POSNR = N'{v}'" for v in posnr_safe])
    try:
        q = f"SELECT TOP 50 VBELN, POSNR, MATNR, NETWR FROM {format_table_name('VBAP')} WHERE {posnr_cond}"
        results["VBAP.POSNR"] = run_sql(q)
    except Exception:
        results["VBAP.POSNR"] = None

    # VBFA (document flow: preceding VBELV, subsequent VBELN) — use variants so delivery 80003409 is found
    try:
        q = f"SELECT TOP 50 VBELV, POSNV, VBELN, POSNN FROM {format_table_name('VBFA')} WHERE VBELV IN ({in_clause})"
        results["VBFA.VBELV"] = run_sql(q)
    except Exception:
        results["VBFA.VBELV"] = None
    try:
        q = f"SELECT TOP 50 VBELV, POSNV, VBELN, POSNN FROM {format_table_name('VBFA')} WHERE VBELN IN ({in_clause})"
        results["VBFA.VBELN"] = run_sql(q)
    except Exception:
        results["VBFA.VBELN"] = None

    # LIKP.VBELN (delivery header) — use variants so e.g. 80003409 shows under delivery header
    try:
        q = f"SELECT TOP 50 VBELN, KUNNR, LFDAT FROM {format_table_name('LIKP')} WHERE VBELN IN ({in_clause})"
        results["LIKP.VBELN"] = run_sql(q)
    except Exception:
        results["LIKP.VBELN"] = None

    # LIPS.VBELN (delivery item), LIPS.VGBEL (reference sales order) — use variants
    try:
        q = f"SELECT TOP 50 VBELN, POSNR, VGBEL, VGPOS, MATNR, NETWR FROM {format_table_name('LIPS')} WHERE VBELN IN ({in_clause})"
        results["LIPS.VBELN"] = run_sql(q)
    except Exception:
        results["LIPS.VBELN"] = None
    try:
        q = f"SELECT TOP 50 VBELN, POSNR, VGBEL, VGPOS, MATNR, NETWR FROM {format_table_name('LIPS')} WHERE VGBEL IN ({in_clause})"
        results["LIPS.VGBEL"] = run_sql(q)
    except Exception:
        results["LIPS.VGBEL"] = None

    # BSAD (cleared) and BSID (open): accounting doc; VBELN = billing, BELNR = accounting document number — use variants
    try:
        q = f"SELECT TOP 50 VBELN, BELNR, BUKRS, GJAHR, AUGDT, KUNNR FROM {format_table_name('BSAD')} WHERE VBELN IN ({in_clause})"
        results["BSAD.VBELN"] = run_sql(q)
    except Exception:
        results["BSAD.VBELN"] = None
    try:
        q = f"SELECT TOP 50 VBELN, BELNR, BUKRS, GJAHR, AUGDT, KUNNR FROM {format_table_name('BSAD')} WHERE BELNR IN ({in_clause})"
        results["BSAD.BELNR"] = run_sql(q)
    except Exception:
        results["BSAD.BELNR"] = None
    try:
        q = f"SELECT TOP 50 VBELN, BELNR, BUKRS, GJAHR, BUDAT, KUNNR FROM {format_table_name('BSID')} WHERE VBELN IN ({in_clause})"
        results["BSID.VBELN"] = run_sql(q)
    except Exception:
        results["BSID.VBELN"] = None
    try:
        q = f"SELECT TOP 50 VBELN, BELNR, BUKRS, GJAHR, BUDAT, KUNNR FROM {format_table_name('BSID')} WHERE BELNR IN ({in_clause})"
        results["BSID.BELNR"] = run_sql(q)
    except Exception:
        results["BSID.BELNR"] = None

    return normalized, results


def trace_sales_order_number(sales_order_num: str):
    """Trace where a sales order number appears: VBRP.AUBEL and VBAK.VBELN.
    Also check VBRP.AUPOS (reference item position number only — not order number or invoice number) so the user can see which column contains a given value."""
    raw = (sales_order_num or "").strip()
    if not raw:
        return None, None, None, None
    normalized = raw.zfill(SALES_ORDER_NUMBER_LENGTH) if raw.isdigit() else raw
    safe = normalized.replace("'", "''")
    # Search AUBEL (reference sales document)
    vbrp_sql = f"SELECT VBELN, AUBEL, AUPOS, MATNR, NETWR FROM {format_table_name('VBRP')} WHERE AUBEL = N'{safe}'"
    vbak_sql = f"SELECT VBELN, BSTNK, BSTDK, KUNNR, AUDAT, NETWR FROM {format_table_name('VBAK')} WHERE VBELN = N'{safe}'"
    vbrp_df = run_sql(vbrp_sql)
    vbak_df = run_sql(vbak_sql)
    # Also search AUPOS (position number only) so user can see if value appears as item position, not as order number
    aupos_variants = [safe, raw]
    if raw.isdigit():
        aupos_variants.append(raw.zfill(6))  # SAP often uses 6-char for position (e.g. 000041)
    aupos_safe = [v.replace("'", "''") for v in aupos_variants]
    aupos_cond = " OR ".join([f"AUPOS = N'{v}'" for v in aupos_safe])
    vbrp_aupos_sql = f"SELECT VBELN, AUBEL, AUPOS, MATNR, NETWR FROM {format_table_name('VBRP')} WHERE {aupos_cond}"
    vbrp_aupos_df = run_sql(vbrp_aupos_sql)
    return vbrp_df, vbak_df, normalized, vbrp_aupos_df


def get_document_flow_for_order(normalized_order: str):
    """Given a normalized sales order number, return the document flow: order → delivery → invoice → accounting document.
    Uses VBFA (preceding VBELV, subsequent VBELN), VBRP (AUBEL, VBELN), and BSAD (VBELN = billing, BELNR = accounting doc).
    Returns dict with order, deliveries, billings, accounting_docs, and flow_chain list of (stage, doc_number)."""
    if not (normalized_order or isinstance(normalized_order, str)):
        return None
    safe = str(normalized_order).strip().replace("'", "''")
    out = {"order": normalized_order, "deliveries": [], "billings": [], "accounting_docs": [], "flow_chain": [("Order", normalized_order)]}
    try:
        # Deliveries: VBFA where preceding = order (VBELV = order) -> subsequent = delivery (VBELN)
        vbfa_del_sql = f"SELECT DISTINCT VBELN AS delivery FROM {format_table_name('VBFA')} WHERE VBELV = N'{safe}'"
        df_del = run_sql(vbfa_del_sql)
        if not df_del.empty and "delivery" in df_del.columns:
            out["deliveries"] = df_del["delivery"].astype(str).str.strip().unique().tolist()
            for d in out["deliveries"]:
                out["flow_chain"].append(("Delivery", d))
        # Billings: VBRP where AUBEL = order
        vbrp_bill_sql = f"SELECT DISTINCT VBELN AS billing FROM {format_table_name('VBRP')} WHERE AUBEL = N'{safe}'"
        df_bill = run_sql(vbrp_bill_sql)
        if not df_bill.empty and "billing" in df_bill.columns:
            out["billings"] = df_bill["billing"].astype(str).str.strip().unique().tolist()
            for b in out["billings"]:
                out["flow_chain"].append(("Invoice", b))
            # Accounting documents: BSAD (cleared) and BSID (open) where VBELN = billing doc, BELNR = accounting document number; use variants for billing
            for b in out["billings"]:
                b_str = str(b).strip()
                b_variants = [b_str]
                if b_str.isdigit():
                    b_variants.append(b_str.zfill(DOCUMENT_NUMBER_LENGTH))
                b_variants = list(dict.fromkeys(b_variants))
                b_in = ", ".join([f"N'{v.replace(chr(39), chr(39)+chr(39))}'" for v in b_variants])
                for tbl in ["BSAD", "BSID"]:
                    try:
                        acct_sql = f"SELECT DISTINCT BELNR AS accounting_doc FROM {format_table_name(tbl)} WHERE VBELN IN ({b_in})"
                        df_acct = run_sql(acct_sql)
                        if not df_acct.empty and "accounting_doc" in df_acct.columns:
                            for _, row in df_acct.iterrows():
                                belnr = str(row["accounting_doc"]).strip()
                                if belnr and belnr not in out["accounting_docs"]:
                                    out["accounting_docs"].append(belnr)
                                    out["flow_chain"].append(("Accounting document", belnr))
                    except Exception:
                        pass
    except Exception:
        pass
    return out if (out["deliveries"] or out["billings"]) else None


def run_sql(sql: str) -> pd.DataFrame:
    try:
        conn = pyodbc.connect(CONN_STR)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"SQL execution error: {e}")
        return pd.DataFrame()


# -------------------------------
# PDF reading (for context in insights)
# -------------------------------
def load_pdf_text(file) -> str:
    """Extract text from an uploaded PDF file. file can be a file-like object or path."""
    try:
        from pypdf import PdfReader
        if hasattr(file, "seek"):
            file.seek(0)
        reader = PdfReader(file)
        text = []
        for p in reader.pages:
            text.append(p.extract_text() or "")
        return "\n".join(text).strip()
    except Exception as e:
        st.warning(f"Could not read PDF: {e}")
        return ""


# -------------------------------
# Multi-provider insights: ChatGPT, Claude, Gemini, Perplexity
# -------------------------------
INSIGHTS_PROMPT_TEMPLATE = """You are an expert analyst. Based on the following data and the user's question, provide concise insights from the perspective of:
- Industry trends (call out leading and lagging sectors; tie to any bar/pie charts shown above when the user asks for industry trends)
- Customer revenues and behavior
- Product and industry analysis
- Other vital business metrics

When the user asks for industry trends: summarize key patterns visible in the data (and in the charts/graphs), highlight top and bottom industries by value or share, and give 1–2 actionable recommendations.

User question: {user_query}

Data summary (columns and sample rows, max 50 rows):
{data_summary}
"""
EXTRA_CONTEXT_TEMPLATE = """

Additional context from uploaded PDF(s):
---
{pdf_text}
---
Use this context to enrich your analysis when relevant.
"""

def get_insights_from_provider(provider: str, user_query: str, df: pd.DataFrame, pdf_text: str = "") -> str:
    """Get analysis/insights from the selected provider (chatgpt, claude, gemini, perplexity)."""
    if df is None or df.empty:
        return "No data available to analyze."
    data_summary = f"Columns: {list(df.columns)}\n\nSample:\n{df.head(50).to_string()}"
    prompt = INSIGHTS_PROMPT_TEMPLATE.format(user_query=user_query, data_summary=data_summary)
    if pdf_text and pdf_text.strip():
        prompt += EXTRA_CONTEXT_TEMPLATE.format(pdf_text=pdf_text.strip()[:15000])
    provider = (provider or "chatgpt").strip().lower()
    try:
        if provider == "chatgpt" and OPENAI_API_KEY:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return (response.choices[0].message.content or "").strip()
        elif provider == "claude":
            import anthropic
            if not ANTHROPIC_API_KEY:
                return "Claude is not configured. Set ANTHROPIC_API_KEY in config or environment."
            c = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            msg = c.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return (msg.content[0].text if msg.content else "").strip()
        elif provider == "gemini":
            import google.generativeai as genai
            if not GOOGLE_GEMINI_API_KEY:
                return "Gemini is not configured. Set GOOGLE_GEMINI_API_KEY in config or environment."
            genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        elif provider == "perplexity":
            import httpx
            if not PERPLEXITY_API_KEY:
                return "Perplexity is not configured. Set PERPLEXITY_API_KEY in config or environment."
            r = httpx.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                },
                timeout=60.0,
            )
            if r.status_code != 200:
                return f"Perplexity API error: {r.status_code} — {r.text[:500]}"
            out = r.json()
            return (out.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        else:
            if provider == "chatgpt":
                return "ChatGPT (OpenAI) is not configured. Set OPENAI_API_KEY."
            return f"Unknown provider: {provider}. Use one of: chatgpt, claude, gemini, perplexity."
    except Exception as e:
        return f"Error from {provider}: {e}"


PROVIDER_DISPLAY_NAMES = {
    "chatgpt": "ChatGPT (OpenAI)",
    "claude": "Claude (Anthropic)",
    "gemini": "Gemini (Google)",
    "perplexity": "Perplexity",
}

def _configured_insight_providers():
    """Return list of provider keys that have API keys set."""
    out = []
    if OPENAI_API_KEY:
        out.append("chatgpt")
    if ANTHROPIC_API_KEY:
        out.append("claude")
    if GOOGLE_GEMINI_API_KEY:
        out.append("gemini")
    if PERPLEXITY_API_KEY:
        out.append("perplexity")
    return out

def get_insights_from_all_providers(user_query: str, df: pd.DataFrame, pdf_text: str = "") -> list:
    """Run all configured providers and return [(provider_display_name, text), ...] for successful responses."""
    providers = _configured_insight_providers()
    if not providers:
        return [("No provider", "No API keys configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_GEMINI_API_KEY, or PERPLEXITY_API_KEY.")]
    results = []
    for p in providers:
        text = get_insights_from_provider(p, user_query, df, pdf_text=pdf_text)
        if text and not text.startswith("Error from") and "not configured" not in text.lower():
            results.append((PROVIDER_DISPLAY_NAMES.get(p, p), text))
    if not results:
        return [("No result", "No provider returned a valid analysis. Check API keys and try again.")]
    return results

JUDGE_PROMPT = """You are a judge. Rank the following analyses of the same business data by quality of insights (industry trends, customer revenue, product/industry analysis, clarity, actionability). Reply with a single line: the exact provider names in order, best first, comma-separated. Example: ChatGPT (OpenAI), Claude (Anthropic), Gemini (Google)

User question: {user_query}

Analyses (each prefixed by [PROVIDER: name]):

{analyses_text}
"""

def pick_best_analysis(user_query: str, responses: list) -> tuple:
    """Given list of (provider_name, text), return (best_provider, best_text, alternatives).
    alternatives is list of (provider_name, text) for the rest, best alternative first."""
    if not responses:
        return None, "", []
    if len(responses) == 1:
        return responses[0][0], responses[0][1], []
    # Build text for judge: concatenate each analysis with [PROVIDER: name]
    analyses_text = "\n\n".join(
        f"[PROVIDER: {name}]\n{text[:4000]}" for name, text in responses
    )
    judge_prompt = JUDGE_PROMPT.format(user_query=user_query, analyses_text=analyses_text)
    try:
        if OPENAI_API_KEY:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
            )
            order_str = (r.choices[0].message.content or "").strip()
            order = [n.strip() for n in order_str.split(",") if n.strip()]
            name_to_response = {name: (name, text) for name, text in responses}
            def find_match(name_from_judge):
                if name_from_judge in name_to_response:
                    return name_from_judge
                name_lower = name_from_judge.lower()
                for rname in name_to_response:
                    if name_lower in rname.lower() or rname.lower() in name_lower:
                        return rname
                return None
            best_name = None
            for n in order:
                best_name = find_match(n)
                if best_name:
                    break
            if not best_name and responses:
                best_name = responses[0][0]
            if best_name:
                best = name_to_response[best_name]
                others = [(nm, tx) for nm, tx in responses if nm != best_name]
                return best[0], best[1], others
    except Exception:
        pass
    # Fallback: longest coherent response as best
    def _score(tup):
        name, text = tup
        if not text or len(text) < 100:
            return 0
        return len(text)
    sorted_responses = sorted(responses, key=_score, reverse=True)
    best_name, best_text = sorted_responses[0]
    return best_name, best_text, sorted_responses[1:]


def get_dynamic_analysis_plan(user_query: str, df: pd.DataFrame):
    cols = list(df.columns)
    sample_prompt = f"""
User query: "{user_query}"

Available columns in the dataset (use these exact names in your response):
{json.dumps(cols, indent=2)}

Task:
- Based on the user query and available data, suggest calculations or aggregations to perform.
- Suggest charts/graphs/diagrams that match the query. Prefer multiple visualizations (bar, pie, line) when useful.
- Industry trends: If the user asks for industry trends, or if the dataset has an industry-related column (e.g. BRSCH, industry_key, or any name containing "industry"), always suggest at least: (1) a bar chart of value/revenue by industry (x = industry column, y = value column, agg = sum), (2) a pie chart of share by industry (same columns), and optionally (3) a line chart over time by industry if a date column exists. Use only the exact column names listed above.
- Use only columns from the dataset. Return valid JSON in this format:

{{
  "calculations": [
    "sum of <column>",
    "sum of <value_column> grouped by <industry_or_category_column>"
  ],
  "visualizations": [
    {{ "type": "bar", "x": "<exact_column_name>", "y": "<exact_column_name>", "agg": "sum" }},
    {{ "type": "pie", "labels": "<exact_column_name>", "values": "<exact_column_name>", "agg": "sum" }},
    {{ "type": "line", "x": "<exact_column_name>", "y": "<exact_column_name>", "agg": "sum" }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": sample_prompt}],
        temperature=0.3
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group(0)) if match else {}





def _strip_calc_trailers(text: str) -> str:
    """Remove ' for ...', ' grouped by ...', ' group by ...' from calc column text."""
    for trailer in (" grouped by ", " group by ", " for "):
        if trailer in text:
            text = text.split(trailer)[0].strip()
    return text


def _resolve_calc_column(calc_desc: str, df_columns: list, after_key: str) -> str:
    """Extract column name from calc text (e.g. 'sum of Net_Value... for Customer 1175' -> 'Net_Value_in_Document_Currency')."""
    idx = calc_desc.lower().find(after_key.lower())
    if idx == -1:
        return None
    rest = calc_desc[idx + len(after_key):].strip()
    rest = _strip_calc_trailers(rest)
    if rest in df_columns:
        return rest
    # Try match: column name might be substring (e.g. rest has extra words)
    for c in df_columns:
        if c in rest or rest in c:
            return c
    return rest

def _is_industry_column(col: str, df_cols: list) -> bool:
    """True if col is an industry-like dimension (for tracking if we already showed industry chart)."""
    if not col:
        return False
    c = col.upper()
    if c in ("BRSCH", "INDUSTRY_KEY", "INDUSTRY") or "INDUSTRY" in c:
        return True
    return False


def _find_industry_column(df_cols: list) -> str:
    """Return first column that looks like industry (BRSCH, industry_key, or name containing industry)."""
    for c in df_cols:
        if c and (_is_industry_column(c, df_cols)):
            return c
    return ""


def _find_numeric_value_column(df_cols: list, df: pd.DataFrame = None) -> str:
    """Return a column name suitable for value/sum (NETWR, net value, or first numeric)."""
    prefer = []
    for c in df_cols:
        if not c:
            continue
        cu = c.upper()
        if "NETWR" in cu or "NET_VALUE" in cu or "VALUE" in cu or "REVENUE" in cu:
            prefer.append(c)
    if prefer:
        return prefer[0]
    if df is not None:
        for c in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
            except Exception:
                pass
    return ""


def perform_analysis_from_plan(df: pd.DataFrame, plan: dict):
    st.write("## 🧠 AI-Powered Data Analysis")
    df_cols = list(df.columns)

    # Calculations
    if "calculations" in plan:
        st.write("### 📊 Calculations")
        for calc in plan["calculations"]:
            try:
                # "sum of X grouped by Y" or "sum X group by Y"
                has_group = "group by" in calc.lower() or "grouped by" in calc.lower()
                if ("sum" in calc.lower() and has_group):
                    # Resolve group column: text after "grouped by" or "group by"
                    low = calc.lower()
                    group_col = ""
                    for sep in ("grouped by", "group by"):
                        if sep in low:
                            idx = low.find(sep) + len(sep)
                            group_col = calc[idx:].split("and")[0].strip()
                            break
                    # Resolve sum column: after "sum of" or "sum", then strip " grouped by ..." / " for ..."
                    if "sum of" in low:
                        sum_col = calc.split("sum of")[1].strip()
                    else:
                        sum_col = calc.split("sum")[1].strip()
                    sum_col = _strip_calc_trailers(sum_col)
                    if group_col and sum_col:
                        if group_col not in df_cols:
                            group_col = next((c for c in df_cols if group_col in c or c in group_col), group_col)
                        if sum_col not in df_cols:
                            sum_col = next((c for c in df_cols if sum_col in c or c in sum_col), sum_col)
                        if group_col in df_cols and sum_col in df_cols:
                            result = df.groupby(group_col)[sum_col].sum().reset_index()
                            st.write(f"**{calc}**")
                            st.dataframe(result)
                        else:
                            raise ValueError(f"Column not found: {sum_col!r} or {group_col!r}")
                    else:
                        raise ValueError("Could not parse sum column or group-by column")
                elif "sum of" in calc.lower():
                    col = _resolve_calc_column(calc, df_cols, "sum of") or _strip_calc_trailers(calc.split("sum of")[1].strip())
                    if col not in df_cols:
                        col = next((c for c in df_cols if col in c or c in col), col)
                    total = df[col].sum()
                    st.metric(f"Sum of {col}", f"{total:,.2f}")
                elif "avg" in calc.lower() or "average" in calc.lower():
                    col = _resolve_calc_column(calc, df_cols, "avg of") or _resolve_calc_column(calc, df_cols, "average of")
                    if col and col in df_cols:
                        st.metric(f"Average of {col}", f"{df[col].mean():,.2f}")
                # You can add more operations: min, max, etc.
            except Exception as e:
                st.warning(f"Could not perform: {calc} — {e}")

    # Visualizations
    charts_rendered = 0
    industry_chart_rendered = False
    if "visualizations" in plan:
        st.write("### 📈 Charts & graphs")
        for vis in plan["visualizations"]:
            try:
                # Pie uses labels/values; bar/line use x/y
                x = vis.get("x") or vis.get("labels")
                y = vis.get("y") or vis.get("values")
                if not x or not y:
                    continue
                # Resolve to actual column names if needed
                if x not in df_cols:
                    x = next((c for c in df_cols if x.lower() in c.lower() or c.lower() in x.lower()), x)
                if y not in df_cols:
                    y = next((c for c in df_cols if y.lower() in c.lower() or c.lower() in y.lower()), y)
                if x not in df_cols or y not in df_cols:
                    continue
                agg = vis.get("agg", "sum")
                vis_type = vis.get("type", "bar")

                df_copy = df.copy()
                if agg == "count":
                    df_vis = df_copy.groupby(x)[y].count().reset_index()
                elif agg == "sum":
                    df_vis = df_copy.groupby(x)[y].sum().reset_index()
                else:
                    df_vis = df_copy.groupby(x)[y].mean().reset_index()

                if vis_type == "bar":
                    chart = alt.Chart(df_vis).mark_bar().encode(x=x, y=y)
                elif vis_type == "line":
                    chart = alt.Chart(df_vis).mark_line().encode(x=x, y=y)
                elif vis_type == "pie":
                    chart = alt.Chart(df_vis).mark_arc().encode(
                        theta=alt.Theta(y, type="quantitative"),
                        color=alt.Color(x, type="nominal")
                    )
                else:
                    chart = alt.Chart(df_vis).mark_bar().encode(x=x, y=y)

                st.altair_chart(chart.properties(width="container"), use_container_width=True)
                charts_rendered += 1
                if _is_industry_column(x, df_cols):
                    industry_chart_rendered = True
            except Exception as e:
                st.warning(f"Could not render chart for: {vis} — {e}")

    # Industry trends: fallback bar + pie when data has industry but no industry chart was rendered
    industry_col = _find_industry_column(df_cols)
    value_col = _find_numeric_value_column(df_cols, df)
    if industry_col and value_col and not industry_chart_rendered:
        st.write("### 📊 Industry trends")
        try:
            df_vis = df.groupby(industry_col)[value_col].sum().reset_index()
            st.altair_chart(
                alt.Chart(df_vis).mark_bar().encode(x=industry_col, y=value_col).properties(title="Value by industry (bar)").properties(width="container"),
                use_container_width=True,
            )
            st.altair_chart(
                alt.Chart(df_vis).mark_arc().encode(
                    theta=alt.Theta(value_col, type="quantitative"),
                    color=alt.Color(industry_col, type="nominal"),
                ).properties(title="Share by industry (pie)").properties(width="container"),
                use_container_width=True,
            )
            charts_rendered += 2
        except Exception as e:
            st.warning(f"Could not render industry trend charts — {e}")

def stream_query_to_redpanda(user_query: str, sql_query: str, df: pd.DataFrame, insights: dict):
    if df.empty:
        st.warning("No data returned — skipping Redpanda stream.")
        return

    timestamp = datetime.now().isoformat()

    try:
        # 1. Stream SQL Query
        sql_payload = {
            "timestamp": timestamp,
            "user_query": user_query,
            "sql_query": sql_query
        }
        producer.produce(KAFKA_TOPICS["sql"], key="sql", value=json.dumps(sql_payload))

        # 2. Stream Query Results
        result_payload = {
            "timestamp": timestamp,
            "user_query": user_query,
            "row_count": len(df),
            "result_sample": df.head(10).to_dict(orient="records")
        }
        producer.produce(KAFKA_TOPICS["results"], key="results", value=json.dumps(result_payload))

        # 3. Stream Analysis Output
        insight_payload = {
            "timestamp": timestamp,
            "user_query": user_query,
            "insights": insights
        }
        producer.produce(KAFKA_TOPICS["insights"], key="insights", value=json.dumps(insight_payload))

        producer.flush()
        st.success("✅ Streamed query, result, and insights to Redpanda.")

    except Exception as e:
        st.warning(f"⚠️ Redpanda stream failed: {e}")


# functions.py


def decide_query_action(user_query: str, last_sql: str = "") -> dict:
    """
    Use LangChain memory to decide if the query is follow-up or needs new SQL.
    Returns action: "casual" | "reuse" | "new"
    """
    prompt = f"""
You are assisting with SQL query generation.

User query: "{user_query}"

Last SQL query (if any): "{last_sql}"

Task:
Decide and return exactly one action:

- "casual": follow-up chat, clarification, or question about previous result (no new SQL).
- "reuse": same question as before, just re-run the last SQL (e.g. "show again", "refresh").
- "new": new question that needs a new SQL query (different filters, columns, or comparison).

Return your decision as JSON only:
{{
  "action": "casual" | "reuse" | "new",
  "reason": "Explain in one sentence"
}}
"""
    response = conversation.predict(input=prompt)
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        out = json.loads(match.group(0)) if match else {"action": "new", "reason": "No structured output"}
    except Exception:
        out = {"action": "new", "reason": "Failed to parse response"}
    # Normalize: only allow casual, reuse, new (map "compare" or anything else to "new")
    allowed = ("casual", "reuse", "new")
    if out.get("action") not in allowed:
        out["action"] = "new"
        out["reason"] = (out.get("reason") or "") + " (treated as new query)"
    return out

def get_memory():
    return memory
def get_langchain_response(user_query: str, df: pd.DataFrame) -> str:

    """
    Answer user's follow-up questions based on the previous SQL result DataFrame.
    Uses manual chat history + vector store over DataFrame content.
    """
    if df is None or df.empty:
        return "⚠️ No data available to answer your question."

    # Step 1: Convert DataFrame rows into LangChain Documents
    documents = []
    for _, row in df.head(MAX_DOC_ROWS).iterrows():
        content = ", ".join([f"{col}: {val}" for col, val in row.items()])
        documents.append(Document(page_content=content))

    # Step 2: Embed the documents and create a FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Step 3: Use manual chat history
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # or conversation.llm if you initialized a LangChain model earlier
        retriever=retriever,
        return_source_documents=False
    )

    # Step 4: Run with explicit keys
    response = qa_chain.run({
        "question": user_query,
        "chat_history": chat_history
    })

    # Step 5: Update chat history for future context
    chat_history.append((user_query, response))

    return response

# test functions 

def split_comparison_query(user_query, llm):
    prompt_template =prompt_template = """
You are an assistant that splits a comparison query into multiple independent sub-queries.

Query: "{user_query}"

Return JSON in the format:
{{
  "sub_queries": ["...", "..."]
}}

If the query is not a comparison, return:
{{ "sub_queries": [] }}
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"user_query": user_query})
    try:
        result = json.loads(response)
        return result.get("sub_queries", [])
    except json.JSONDecodeError:
        return []


# 2. 🧠 Get SQL and corresponding DataFrame (with cache + fallback to regeneration)
def get_sql_and_df_for_query(user_sub_query, session_state, pick_tables_fn, load_column_mappings_fn,
                             generate_sql_json_fn, fix_date_filters_fn, json_to_sql_fn, run_sql_fn):
    if user_sub_query in session_state.query_to_sql_map:
        sql_query = session_state.query_to_sql_map[user_sub_query]
        df = run_sql_fn(sql_query)
        return sql_query, df

    pick_res = pick_tables_fn(user_sub_query)
    if not pick_res or "selected_tables" not in pick_res:
        raise ValueError("Could not identify relevant tables for sub-query.")

    selected = [t["name"] for t in pick_res["selected_tables"]]
    column_map = load_column_mappings_fn(selected)
    spec = generate_sql_json_fn(user_sub_query, selected, column_map)
    if not spec:
        raise ValueError("Failed to generate SQL specification for sub-query.")

    spec = fix_date_filters_fn(spec)
    sql_query = json_to_sql_fn(spec)
    df = run_sql_fn(sql_query)

    if df is not None and not df.empty:
        session_state.query_to_sql_map[user_sub_query] = sql_query

    return sql_query, df


# 3. ⚖️ Compare numeric metrics across multiple DataFrames
def compare_dataframes(dataframes: list, labels: list = None):
    if not dataframes or len(dataframes) < 2:
        return None, None, "Need at least 2 dataframes to compare."

    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(dataframes))]

    for i, df in enumerate(dataframes):
        df["__label__"] = labels[i]

    combined = pd.concat(dataframes, axis=0, ignore_index=True)
    numeric_cols = combined.select_dtypes(include='number').columns.tolist()
    summary_rows = []

    if numeric_cols:
        for col in numeric_cols:
            if len(dataframes) == 2:
                val1 = dataframes[0][col].sum()
                val2 = dataframes[1][col].sum()
                diff = val2 - val1
                pct = (diff / val1) * 100 if val1 != 0 else float('inf')
                summary_rows.append({
                    "metric": col,
                    labels[0]: val1,
                    labels[1]: val2,
                    "absolute_change": diff,
                    "percent_change": f"{pct:.2f} %"
                })
        summary_df = pd.DataFrame(summary_rows)
    else:
        summary_df = pd.DataFrame({"info": ["No numeric columns to compare."]})

    return combined, summary_df, None


# 4. 🧠 Generate natural language summary of comparison using LLM
def generate_comparison_summary(summary_df, llm):
    prompt_template = """
You are a helpful data analyst.

Given this summary DataFrame comparing two datasets:

{summary}

Write a short paragraph explaining the key differences.
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"summary": summary_df.to_string(index=False)})







#end