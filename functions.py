import ast
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
    from config import REVENUE_BILLING_CATEGORIES
except ImportError:
    REVENUE_BILLING_CATEGORIES = ("A", "B", "C", "D", "E", "I", "L", "W")
try:
    from config import INDUSTRY_SECTOR_LABELS, INDUSTRY_SECTOR_SOURCE
except ImportError:
    INDUSTRY_SECTOR_LABELS = {"M": "Mechanical engineering", "C": "Chemical", "A": "Plant engineering and construction", "P": "Pharmaceutical", "E": "Electrical engineering", "": "Not specified"}
    INDUSTRY_SECTOR_SOURCE = {"BRSCH": "Customer Master (KNA1)", "MBRSH": "Material Master (MARA)"}

# MARC.BESKZ (Procurement Type): show code + description whenever displayed
PROCUREMENT_TYPE_DISPLAY_LABELS = {
    "E": "E (In-house produced)",
    "F": "F (Externally procured)",
    "X": "X (Both)",
}
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
- For value determination, revenue, or "best products by value": prefer VBRK (header) and VBRP (item: NETWR, MATNR); join on VBELN. Revenue is shown only for billing category types A, B, C, D, E, I, L, W (VBRK.FKTYP); the system applies this filter automatically when querying value/revenue.
- For "best products" or "products by value and industry": use VBRK and VBRP for value; add KNA1 for industry (BRSCH); add MAKT and join MAKT.MATNR = VBRP.MATNR so results show material names (MAKTX) not only material numbers (MATNR). Join VBRK.VBELN = VBRP.VBELN. Do not add any WHERE filter so all data is returned.
- **Product/master data only (no revenue, no logistics):** When the user asks to show/list/filter **products by name only** (e.g. "show Harley products", "different Harley products", "Harley products") and does NOT ask for revenue, sales value, invoices, net/gross value, taxes, or delivery/logistics: use ONLY **master data tables** — MARA, MAKT, MEAN, MARM, MVKE. Do NOT use VBRK, VBRP, VBAK, VBAP, BSAD, BSID (sales/revenue) or LIKP, LIPS, VBFA, VTTK (logistics/delivery). Filter by product name (MAKT.MAKTX) so only matching materials are returned. **Do not repeat the same material for different languages:** always filter MAKT by one language (e.g. MAKT.SPRAS = 'E') so the same material number appears once; the focus of product analysis is to identify **same product name with different material numbers** (variants), not the same material in different languages.
- **Revenue / sales value (when explicitly asked):** When the user asks for **revenue**, **sales value**, **items sold**, **net value**, **gross value**, **taxes** on products, or **invoices**: use VBRK, VBRP, VBAK, VBAP, BSAD, BSID to derive value of sales, items sold, net/gross value and taxes. Do not add LIKP, LIPS, VTTK unless the user explicitly asks for delivery or logistics.
- **Delivery / logistics (when explicitly asked):** LIKP, LIPS, VBFA, VTTK are logistics/delivery tables. Use them only when the user explicitly asks for delivery, shipments, logistics, or process flow (order → delivery → invoice). They are not needed for "show products" or "revenue by product" alone.
- For product attributes, material master data, or product descriptions: include MARA (MATNR, MTART, MBRSH, MEINS, MATKL) and MAKT (MATNR, MAKTX = material description/product name); join MARA.MATNR and MAKT.MATNR to VBRP.MATNR, VBAP.MATNR, or LIPS.MATNR when combining with sales data; optionally MVKE for sales-org attributes. For product-only queries use only MARA, MAKT, MEAN, MARM, MVKE.
- For industry trends, industry analysis, or showing industry by graphs/charts/diagrams: use VBRK and VBRP for value and KNA1 for industry (BRSCH); include MAKT and join MAKT.MATNR = VBRP.MATNR so product names (MAKTX) appear with industry; join VBRK.KUNAG = KNA1.KUNNR and VBRK.VBELN = VBRP.VBELN.
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
- For EDI message status for validated or pre sales orders: use PSIF_SLS_MSG; link to PSIF_SLS_HDR on VBELN.
- For inbound invoices from vendors (header or item level): use PSIF_INV_HDR_IN and PSIF_INV_ITEM_I; join on document number (VBELN).
- For purchasing message status or invoice receipt: use PSIF_RBKP (invoice receipt header) and PSIF_RBKP_MSG (message status); join on BELNR and GJAHR.
- For product costing, cost of goods (COGM/COGS), cost estimates, standard price, or moving average price: For COGS always use net prices, moving average price (VERPR), or standard price (STPRS) from MBEW — per price control (VPRSV): use STPRS when VPRSV = 'S', VERPR when VPRSV = 'V'; include PEINH (price unit) for per-unit COGS. Key components — (1) Quantity structure (BOM & Routing): materials required and activities (machine, labor) × their costs. (2) Valuation: Standard Price (S) = constant planned cost; Moving Average (V) = updates with each goods receipt. (3) Costing variant (CK11N) defines rules. (4) Calculation: read BOM → fetch prices → component costs → activity/overhead → aggregate. Use MBEW (STPRS, VERPR, VPRSV, PEINH) and KEKO + KEPH; add CKIS, CKIT, CKHS, CKEP; join on KALNR. Add MARA and MAKT. Link KEKO.MATNR = MARA.MATNR = MBEW.MATNR, KEKO.BWKEY = MBEW.BWKEY. For BOM use STKO/STPO; for routing use PLKO/PLPO. For posted COGS from FI: use BKPF (document header), BSEG (line items: HKONT/SAKNR, amounts), SKA1/SKB1 (G/L account master for account description and company code), T001K (valuation area BWKEY to company code BUKRS), T030 (chart/posting config), and ACDOCA (S/4 HANA Universal Journal: RACCT, WTGBTR, filter by COGS account). **When the user asks for the cost of a specific product/material by name** (e.g. "what is the cost of Harley leather Jacket"): select ONLY costing tables (MBEW, KEKO, KEPH, MARA, MAKT, and optionally CKIS, CKHS, CKEP). Do NOT include VBRK, VBRP, VBAK, VBFA, LIKP, or LIPS — the answer is the material's cost/price from the material master and cost estimate, not billing or delivery documents.
- For verifying or correcting standard price data, or enabling meaningful standard price analysis: use MBEW (STPRS, VERPR, PEINH, VPRSV), MARA, and MAKT so the user can see material number, material description, standard price, moving average price, price unit (PEINH), and price control (S/V); join MBEW.MATNR = MARA.MATNR = MAKT.MATNR. For meaningful analysis exclude or flag rows where standard price is zero or null (unreleased or missing prices). Always include PEINH (price_unit) and note that price_unit defines the quantity per which the price applies.
- For material ledger or period valuation: use CKMLCR (period totals) with MBEW and MARA on MATNR/BWKEY.
- For BOM (bill of material): use STKO (BOM header: STLNR) and STPO (items: IDNRK = component material, MENGE); link STKO.STLNR = STPO.STLNR, STPO.IDNRK = MARA.MATNR.
- **Component data of main material:** To read component data of a main material use: **MAST** (Material to BOM Link: links MATNR, WERKS, STLNR), **STKO** (BOM header: status, alternative), **STPO** (BOM item: item number, component IDNRK, quantity MENGE, unit MEINS). Join MAST.MATNR = MARA.MATNR (main material), MAST.STLNR = STKO.STLNR = STPO.STLNR. Add MARA and MAKT for main material name; add MAKT on STPO.IDNRK for component names. Optionally **DF14L/DF14T** (PLM/PPM component/structure data) or **PAT03** (Hot Packages/components in system admin) when the user asks for PPM structures or package components.
- For routing and operations: use PLKO (routing header) and PLPO (operations); link PLKO to PLPO. Use CSLA for activity type master (LART).
- For cost object controlling, order costs, or cost reconciliation: use AUFK (order master: AUFNR, OBJNR), COBK (controlling document header), COEP (line items by period: OBJNR, WTGJB, PERIO); join COEP.OBJNR = AUFK.OBJNR and COEP to COBK. **AUFK has no MATNR column** — do not join AUFK to MARA or any table on MATNR; link AUFK only via KDAUF/KDPOS to VBAK/VBAP (make-to-order) or via OBJNR to COEP.
- **Products manufactured internally / procurement type / production orders:** In SAP the **primary field** for internal vs external is **MARC.BESKZ (Procurement Type**, MRP 2 view): E = In-house produced, F = Externally procured, X = Both. **MARC.SOBSL** = Special Procurement (e.g. phantom, subcontracting). **MBEW** (Accounting 1/Costing 1): VPRSV = price control (S = Standard for manufactured, V = Moving average for procured), STPRS = standard price, VERPR = moving average price, LOSGR = costing lot size. When the user asks about products manufactured internally, procurement type, production data, make-to-order, or dependencies: use **MARC** (BESKZ, SOBSL — join MARC.MATNR = MARA.MATNR, MARC.WERKS = plant), **MBEW** (VPRSV, STPRS, VERPR, LOSGR), **AUFK** (production orders), **VBAK**, **VBAP**, **COEP**, **MAST**, **STKO**, **STPO**, **MARA**, **MAKT**. Select MARC.BESKZ (and SOBSL) so the result shows the primary SAP indicator; MBEW for valuation. Optionally EKKO/EKPO for purchase orders. Always include MARA and MAKT for product names.
- For profitability analysis: Profitability depends on cost of goods, market demand, pricing strategies, and operational costs. In SAP ERP use FAGLFLEXA (New G/L line items / profitability); in S/4 HANA use ACDOCA (Universal Journal line items). Only include the table that exists in your system (FAGLFLEXA for ERP, ACDOCA for S/4 HANA). Where relevant also include: cost of goods — MBEW, KEKO/KEPH (standard price, cost estimates); pricing/revenue — VBRK/VBRP (billing value); demand — VBAK/VBAP, VBRP (order and sales volume); operational costs — COEP, AUFK (order/period costs). If both FAGLFLEXA and ACDOCA are in the list, prefer ACDOCA for S/4 HANA and FAGLFLEXA for ERP.
- For vendor master data (vendor name, payment terms, bank, purchasing org): use LFA1 (general: LIFNR, NAME1, LAND1), LFB1 (company code), LFM1 (purchasing org), LFBK (bank details), ADR6 (addresses/emails); join all to LFA1 on LIFNR.
- For purchase orders: use EKKO (header: EBELN, LIFNR, BEDAT) and EKPO (item: EBELN, EBELP, MATNR, MENGE, NETPR); join EKKO.EBELN = EKPO.EBELN; add LFA1 on EKKO.LIFNR for vendor name, MARA/MAKT on EKPO.MATNR for material. For delivery schedule use EKET; for history (GR/IR) use EKBE; for account assignment use EKKN. When the user asks who are the suppliers of these parts, or which vendors supply the listed parts, or for supplier name and country and what part they supply: use EKKO, EKPO, LFA1, MAKT, and **MARC**. Always select (1) vendor name — LFA1.NAME1 (vendor_name), (2) country — LFA1.LAND1 (country), (3) the part supplied — EKPO.MATNR (material_number) and MAKT.MAKTX (material_description), and (4) **procurement type — MARC.BESKZ (procurement_type)** so the user sees E/F/X (In-house/External/Both) per part. Join MARC on EKPO.MATNR = MARC.MATNR (and EKPO.WERKS = MARC.WERKS if EKPO has WERKS). Use **SELECT DISTINCT** on material number (or material_number), vendor (LIFNR or vendor_number), and optionally MARC.WERKS so the result has **one row per (material, vendor)** — no repeated vendor or country for the same part. Use only column names from the column mappings.
- For purchase requisitions: use EBAN (BANFN, BNFPO, MATNR, MENGE) and EBKN (account assignment); join EBAN.BANFN = EBKN.BANFN AND EBAN.BNFPO = EBKN.BNFPO.
- For material movements and inventory: use MKPF (material document header: MBLNR, MJAHR, BLDAT, BUDAT) and MSEG (line items: MATNR, MENGE, BWKEY, SHKZG); join MKPF.MBLNR = MSEG.MBLNR AND MKPF.MJAHR = MSEG.MJAHR. For batch stock use MCHB; for special stock with vendor use MSLB; for special stocks use MSKU; for project stock use MSPR.
- For material tax classification: use MLAN (MATNR, ALAND, TAXM1); link to MARA on MATNR. For forecast parameters use MAPR (MATNR, WERKS); link to MARA, MARC. For unit of measure (UoM) conversion: use MARM (MATNR, MEINS, UMREZ, UMREN — conversion to base = quantity × UMREZ/UMREN) and MEAN (MATNR, MEANM, MEINS); link MARM and MEAN to MARA on MATNR; add MARM/MEAN when calculating COGS or converting billed/delivered quantities between sales UoM and base or price unit.
- For other comparisons by industry, customer, value, or products: include the tables that hold that data (e.g. KNA1, VBRK, VBRP, VBAK, VBAP, VBEP, VBFA, VBPA, VBUK, VBUP, KNKK, KNKA, T691A, PSIF_SLS_HDR, PSIF_SLS_ITEM). Do not restrict to country-specific data unless the user explicitly asks for a specific country.
- For percentage of finished goods, percentage of raw materials and semi-finished goods, or sales value by material type: use VBRK, VBRP, and MARA. Join VBRP.MATNR = MARA.MATNR so MARA.MTART (material type) is available. SAP material types: FERT = finished goods, HALB = semi-finished, ROH = raw materials. Sales value = VBRP.NETWR. Add MAKT for material names if needed.
- For cost-benefit analysis, import vs domestic production or purchasing, or make-vs-buy decisions: include purchase orders EKKO and EKPO (purchase price NETPR, vendor LIFNR, material MATNR — reflects import or external purchase cost); include MBEW and optionally KEKO/KEPH (standard price STPRS, moving average VERPR — reflects domestic production or valuation cost); add MARA and MAKT for material names; add LFA1 for vendor name and country (LAND1) to distinguish import vs domestic vendors. Joining EKPO.MATNR = MARA.MATNR = MBEW.MATNR allows comparing purchase price with standard/domestic cost.
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
        q_lower = (user_query or "").lower().strip()
        # For "cost of [product name]" questions, use only costing tables — do not include billing/sales/delivery
        cost_of_product_keywords = ("cost of", "what is the cost", "cost for", "what's the cost", "price of", "what is the price", "show me cost")
        cost_by_product_number = ("product number" in q_lower or "material number" in q_lower) and ("cost" in q_lower or "price" in q_lower)
        costing_tables = {"KEKO", "MBEW", "KEPH", "CKIS", "CKHS", "CKEP"}
        process_flow_only = {"VBRK", "VBRP", "VBAK", "VBFA", "LIKP", "LIPS"}
        if (any(kw in q_lower for kw in cost_of_product_keywords) or cost_by_product_number) and (names & costing_tables):
            out["selected_tables"] = [t for t in out["selected_tables"] if t["name"] not in process_flow_only]
            names = {t["name"] for t in out["selected_tables"]}
        # For "show [X] products" / "different [X] products" (product/master data only): use ONLY master data tables; exclude sales/revenue and logistics
        master_data_tables = {"MARA", "MAKT", "MEAN", "MARM", "MVKE"}
        sales_revenue_tables = {"VBRK", "VBRP", "VBAK", "VBAP", "BSAD", "BSID"}
        logistics_tables = {"LIKP", "LIPS", "VBFA", "VTTK"}
        revenue_value_keywords = ("revenue", "sales value", "net value", "gross value", "taxes", "invoice", "invoices", "billing", "sold", "revenue by", "value of sales")
        delivery_logistics_keywords = ("delivery", "deliveries", "logistics", "shipment", "LIKP", "LIPS", "process flow", "order to delivery")
        product_name = _extract_product_name_from_query(user_query or "")
        bom_keywords = ("bom", "bill of material", "component", "components", "structure", "stko", "stpo", "used in manufacturing")
        wants_components = any(kw in q_lower for kw in bom_keywords)
        if product_name and (names & master_data_tables):
            if not any(kw in q_lower for kw in revenue_value_keywords) and not any(kw in q_lower for kw in delivery_logistics_keywords):
                # User asked for products by name only — keep only master data; remove sales/revenue and logistics
                # BUT if they asked for components/BOM (e.g. "products and components used in manufacturing the motorcycle"), keep MAST, STKO, STPO
                if not wants_components:
                    out["selected_tables"] = [t for t in out["selected_tables"] if t["name"] in master_data_tables]
                names = {t["name"] for t in out["selected_tables"]}
        # For product-by-name queries, do not include BOM/component tables so results only show that product's materials, not unrelated component numbers (e.g. 1320)
        if product_name and not wants_components:
            out["selected_tables"] = [t for t in out["selected_tables"] if t["name"] not in ("STPO", "STKO", "MAST")]
            names = {t["name"] for t in out["selected_tables"]}
        # When user asks for "products and components used in manufacturing [X]" or "components used in manufacturing [X]", force-add component tables so we read BOM data
        components_manufacturing_keywords = ("components used in manufacturing", "products and components used in manufacturing", "component used in manufacturing", "used in manufacturing the ")
        if any(kw in q_lower for kw in components_manufacturing_keywords) or (wants_components and "manufacturing" in q_lower):
            component_tables = [
                ("MAST", TABLE_DESCRIPTIONS.get("MAST", "Material to BOM link: MATNR (main material), WERKS, STLNR (BOM number).")),
                ("STKO", TABLE_DESCRIPTIONS.get("STKO", "BOM header: STLNR, STLAL, STLST.")),
                ("STPO", TABLE_DESCRIPTIONS.get("STPO", "BOM item: STLNR, POSNR, IDNRK (component material), MENGE, MEINS.")),
            ]
            for tbl, desc in component_tables:
                if tbl not in names and TABLE_DESCRIPTIONS.get(tbl):
                    out["selected_tables"].append({"name": tbl, "description": desc})
                    names.add(tbl)
            for tbl in ("MARA", "MAKT"):
                if tbl not in names and TABLE_DESCRIPTIONS.get(tbl):
                    out["selected_tables"].append({"name": tbl, "description": TABLE_DESCRIPTIONS.get(tbl, "")})
                    names.add(tbl)
        process_flow_tables = {"VBRK", "VBRP", "VBAK"}
        delivery_chain = [("VBFA", "Document flow: links invoice (VBRP) to delivery (LIKP)"), ("LIKP", "Delivery header; VBELN = delivery number"), ("LIPS", "Delivery item; VBELN = delivery, VGBEL = sales order")]
        if names & process_flow_tables:
            for tbl, desc in delivery_chain:
                if tbl not in names and tbl in TABLE_DESCRIPTIONS:
                    out["selected_tables"].append({"name": tbl, "description": desc})
        # Always add MAKT (material descriptions) when any product/material table is selected so analysis uses product names (MAKTX), not only material numbers
        product_material_tables = {"VBRP", "VBAP", "LIPS", "MARA", "PSIF_INV_ITEM", "PSIF_SLS_ITEM", "STPO", "CKIS", "KEKO", "MBEW", "CKMLCR", "EKPO", "EBAN", "MSEG", "MARM", "MEAN"}
        # When product data is present, MARA (with MTART, MATKL, MEINS) helps show what differentiates variant products (same name, different material number)
        if (names & product_material_tables) and "MAKT" not in names and "MAKT" in TABLE_DESCRIPTIONS:
            out["selected_tables"].append({"name": "MAKT", "description": "Material descriptions: MAKTX = product name; join MAKT.MATNR to table MATNR (or STPO.IDNRK) so all product analysis shows names/descriptions"})
        # Sales/revenue by material type: need VBRK, VBRP (for NETWR) and MARA (for MTART) so analysis is not limited
        sales_by_mt_keywords = ("sales by material type", "revenue by material type", "percentage of finished goods", "sales value by material type", "value by material type")
        q_low = (user_query or "").lower()
        if any(kw in q_low for kw in sales_by_mt_keywords):
            for tbl in ("VBRK", "VBRP", "MARA"):
                if tbl not in names and TABLE_DESCRIPTIONS.get(tbl):
                    out["selected_tables"].append({"name": tbl, "description": TABLE_DESCRIPTIONS.get(tbl, f"Required for {tbl}.")})
                    names.add(tbl)
        # Manufactured internally / production orders / procurement type: include MARC (BESKZ = primary), MBEW (valuation), and production tables
        production_order_keywords = ("manufactured internally", "production order", "production orders", "make to order", "make to stock", "linked production", "dependencies", "linked products", "through production orders", "procurement type", "internally or externally")
        if any(kw in q_low for kw in production_order_keywords):
            # MARC.BESKZ is the primary SAP field for internal vs external (E=In-house, F=External, X=Both); MBEW for costing/valuation
            production_tables = [
                ("MARC", TABLE_DESCRIPTIONS.get("MARC", "MRP 2: BESKZ = procurement type (E=In-house, F=External, X=Both), SOBSL = special procurement.")),
                ("MBEW", TABLE_DESCRIPTIONS.get("MBEW", "Accounting 1/Costing 1: VPRSV (S/V), STPRS, VERPR, LOSGR for valuation and costing.")),
                ("AUFK", TABLE_DESCRIPTIONS.get("AUFK", "Order master: production orders, AUFNR, AUART, KDAUF/KDPOS for sales order link.")),
                ("VBAK", TABLE_DESCRIPTIONS.get("VBAK", "Sales order header; link AUFK.KDAUF = VBAK.VBELN.")),
                ("VBAP", TABLE_DESCRIPTIONS.get("VBAP", "Sales order item; link AUFK.KDAUF = VBAP.VBELN, AUFK.KDPOS = VBAP.POSNR.")),
                ("COEP", TABLE_DESCRIPTIONS.get("COEP", "Cost object line items; link to AUFK.OBJNR for order costs.")),
                ("MAST", TABLE_DESCRIPTIONS.get("MAST", "Material to BOM link for components used in production.")),
                ("STKO", TABLE_DESCRIPTIONS.get("STKO", "BOM header.")),
                ("STPO", TABLE_DESCRIPTIONS.get("STPO", "BOM item: component material IDNRK, quantity MENGE.")),
            ]
            for tbl, desc in production_tables:
                if tbl not in names and TABLE_DESCRIPTIONS.get(tbl):
                    out["selected_tables"].append({"name": tbl, "description": desc})
                    names.add(tbl)
            for tbl in ("MARA", "MAKT"):
                if tbl not in names and TABLE_DESCRIPTIONS.get(tbl):
                    out["selected_tables"].append({"name": tbl, "description": TABLE_DESCRIPTIONS.get(tbl, "")})
                    names.add(tbl)
        # Procurement-only (which products internal/external): use ONLY master data — MARA, MAKT, MARC. Do not use AUFK, VBAK, VBAP, COEP, MBEW, MAST, STKO, STPO.
        if is_procurement_only_query(user_query or ""):
            procurement_master_only = {"MARA", "MAKT", "MARC"}
            out["selected_tables"] = [t for t in out["selected_tables"] if t["name"] in procurement_master_only]
            names = {t["name"] for t in out["selected_tables"]}
            for tbl in ("MARA", "MAKT", "MARC"):
                if tbl not in names and TABLE_DESCRIPTIONS.get(tbl):
                    desc = TABLE_DESCRIPTIONS.get(tbl) if tbl != "MARC" else (TABLE_DESCRIPTIONS.get("MARC") or "MRP 2: BESKZ = procurement type (E=In-house, F=External, X=Both), SOBSL.")
                    out["selected_tables"].append({"name": tbl, "description": desc})
                    names.add(tbl)
    return out

def _parse_json_spec(content: str):
    """Parse LLM JSON response; tolerate single quotes, trailing commas, and extract {...} if wrapped in text."""
    if not content or not content.strip():
        return None
    s = content.strip()
    # Try direct parse, then extract first {...}
    for attempt in range(2):
        if attempt == 1:
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if not m:
                return None
            s = m.group(0)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Remove trailing commas (invalid in JSON)
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Fix single-quoted property names: 'key': -> "key":
        s = re.sub(r"(\{|,)\s*'([^']*)'\s*:", r'\1 "\2":', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Try as Python literal (accepts single quotes, True/False/None)
        try:
            s_py = s.replace("true", "True").replace("false", "False").replace("null", "None")
            return ast.literal_eval(s_py)
        except (ValueError, SyntaxError):
            pass
    return None


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
    # When query is procurement-only and only master data tables are selected (MARA, MAKT, MARC), force SQL from these views only
    procurement_master_only = is_procurement_only_query(user_query or "") and set(selected_tables) <= {"MARA", "MAKT", "MARC"} and "MARC" in selected_tables
    procurement_master_instruction = ""
    if procurement_master_only:
        procurement_master_instruction = """

**PROCUREMENT-ONLY — use ONLY master data tables (MARA, MAKT, MARC):**
- Use **only** the tables listed above (MARA, MAKT, MARC). Do not reference any other tables (no VBRK, VBRP, AUFK, MBEW, etc.).
- Select: MARA.MATNR (material_number), MAKT.MAKTX (material_description), MARC.BESKZ (procurement_type), MARC.WERKS (plant).
- Joins: MARA.MATNR = MAKT.MATNR (and MAKT.SPRAS = 'E' for one language), MARA.MATNR = MARC.MATNR.
- MARC.BESKZ is the procurement type: E = In-house, F = External, X = Both. Leave filters empty so all materials with procurement type are returned (or add filter on MAKT.MAKTX only if the user specified a product name).
"""
    prompt = f"""
User query: "{user_query}"

Tables available:
{json.dumps({tbl: TABLE_DESCRIPTIONS.get(tbl, f"Table {tbl} (material master / product data).") for tbl in selected_tables}, indent=2)}

Column mappings:
{json.dumps(column_mappings, indent=2)}
{procurement_master_instruction}

**Customer number and name (always use KNA1):**
- Customer number: use {cust_tbl}.{cust_num_col} only.
- Customer name: use {cust_tbl}.{cust_name_col} only.
- When the query asks for customer, customer ID, customer number, or customer name, include {cust_tbl} and select {cust_num_col} and/or {cust_name_col} as appropriate.

**VAT / tax number:** VAT Reg. No is the VAT or tax number; use column STCEG (in KNA1, VBRK, PSIF_INV_HDR, BSAD as available).

**Annual sales revenue:** Use KNA1.UMSA1 for the annual sales revenue of the customer.

**Value determination and revenue:** Use VBRK and VBRP (best tables for value). Join VBRK.VBELN = VBRP.VBELN. VBRP has line-level NETWR and MATNR (product). Revenue is shown only for billing category types A, B, C, D, E, I, L, W (VBRK.FKTYP); the system automatically filters by these categories when NETWR/value is selected — do not add a conflicting FKTYP filter unless the user asks for a specific category.

**Highest sales by customer / sales by customer / revenue by customer:** When the user asks for highest sales by customer, sales by customer, revenue by customer, best sales totals per customer, or top customers by sales: use **only** VBRK, VBRP, and KNA1. Do **not** add LIKP, LIPS, VBFA, VBAK, EKKO, EKPO or other process-flow tables — they multiply rows and can leave the sales value column empty. Always select the **sales value column**: VBRP.NETWR or VBRK.NETWR (mapped as net_value_of_billing_item_in_document_currency / net_value_of_the_billing_item_in_document_currency) so the result has a populated value. Select KNA1.KUNNR (customer number), KNA1.NAME1 (customer name), and VBRP.NETWR or VBRK.NETWR. **When the user also asks for industry** (e.g. "sales by customer and industry", "best sales by customer and industry"): you **must** also select KNA1.BRSCH (industry_key) so the result shows industry data; BRSCH is in KNA1 and needs no extra table. Join VBRK.KUNAG = KNA1.KUNNR and VBRK.VBELN = VBRP.VBELN. When the user specifies a **year** (e.g. "year 2000", "in 2000"): add a filter on billing date — lhs "VBRK.FKDAT", operator "=", rhs the year as four digits (e.g. "2000"); the system converts this to a full-year range so all billing documents in that year are included. Add ORDER BY NETWR DESC so highest sales appear first. In "columns" you must include the NETWR column from VBRP or VBRK.

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

**Comparisons by industry, customer, value, products:** Use VBRK and VBRP for value; KNA1 for customer/industry (BRSCH). Include MAKT and join MAKT.MATNR = VBRP.MATNR so results show material names (MAKTX) not only MATNR. Revenue/value from VBRK/VBRP is restricted to billing categories A,B,C,D,E,I,L,W (VBRK.FKTYP); the system applies this automatically. Do NOT add a filter on country (LAND1) unless the user asks.

**Best products by value and industry:** Use VBRK and VBRP (columns: VBRP.MATNR, VBRP.NETWR or VBRK.NETWR); add KNA1 for industry (KNA1.BRSCH) and add MAKT for product names: select MAKT.MAKTX (material description) and join MAKT.MATNR = VBRP.MATNR so the result shows product names and industry clearly. Join VBRK.KUNAG = KNA1.KUNNR and VBRK.VBELN = VBRP.VBELN. Optionally filter MAKT.SPRAS = 'E' for English descriptions. Revenue is only for billing types A,B,C,D,E,I,L,W (applied automatically). Order by NETWR descending.

**Best selling product(s) by industry (not by customer):** When the user asks for **best selling product** by industry, **best products by industry**, **top products by industry**, or **which are the best selling products by industry** (with or without a year): the result must show **products** (material/product), not customers. Use VBRK, VBRP, KNA1, and MAKT. Select **VBRP.MATNR** (material_number), **MAKT.MAKTX** (material description / product name), **KNA1.BRSCH** (industry_key) for industry, and **VBRP.NETWR** or VBRK.NETWR for sales value. Join VBRK.VBELN = VBRP.VBELN, VBRK.KUNAG = KNA1.KUNNR (to get industry from customer), and MAKT.MATNR = VBRP.MATNR (and MAKT.SPRAS = 'E' for one language). When the user specifies a **year** (e.g. "year 2001"): add filter lhs "VBRK.FKDAT", operator "=", rhs the year as four digits (e.g. "2001"); the system converts to full-year range. Order by NETWR DESC. Do **not** use the "sales by customer" rule here — do **not** select only KUNNR/NAME1; the user asked for **product** by industry, so the result must include product (MATNR, MAKTX) and industry (BRSCH).

**How can we compete against products like X:** When the user asks **how can we compete against products like [X]** (e.g. "how can we compete against products like harley jackets"): (1) **Restrict results to that product** — add filter on **MAKT.MAKTX** (or material description) for the product name (e.g. "Harley") so the query returns only rows for that product and its variants; the user wants on-subject data, not broad/unrelated products. (2) Use **VBRK, VBRP, MAKT, and KNA1**. Select **VBRP.MATNR**, **MAKT.MAKTX** (material description), **VBRP.NETWR** or VBRK.NETWR (for calculations), and **KNA1.BRSCH** (industry_key) so insights can reference same industry and similar competitive products. Join VBRK.VBELN = VBRP.VBELN, MAKT.MATNR = VBRP.MATNR (MAKT.SPRAS = 'E'), VBRK.KUNAG = KNA1.KUNNR. Order by NETWR DESC. No date filter unless the user specifies a year or range.

**Best product data performance / product performance (with or without year):** When the user asks for **best product data performance**, **best product performance**, **product analysis and performance**, **top products**, or **comparative market** (with or without "web search" or "insights"): use **only** VBRK, VBRP, and MAKT (no LIKP, LIPS, VBFA, VBAK). Select **VBRP.MATNR** (material_number), **MAKT.MAKTX** (material description / product name), and **VBRP.NETWR** or VBRK.NETWR for sales value. Join VBRK.VBELN = VBRP.VBELN and MAKT.MATNR = VBRP.MATNR with MAKT.SPRAS = 'E'. **If the user specifies a year range** (e.g. "year 1992 to 2000", "1992 to 2000", "from 1992 to 2000"): add **two** filters — (1) lhs "VBRK.FKDAT", operator ">=", rhs the **start year** as four digits (e.g. "1992"); (2) lhs "VBRK.FKDAT", operator "<=", rhs the **end year** as four digits (e.g. "2000"). The system converts these to full date bounds (19920101 and 20001231). **If the user specifies a single year** (e.g. "year 1999"): add one filter lhs "VBRK.FKDAT", operator "=", rhs the year (e.g. "1999"); the system converts to full-year range. **If the user does NOT specify any year: do NOT add any filter on VBRK.FKDAT** so the query returns all-time best products. Order by NETWR DESC. Optionally add KNA1 and BRSCH if the user asks for market or industry context.
- **Product attributes / material master / product names:** When the user asks for product details, material names, material type, material group, or sales attributes: include MARA (link MARA.MATNR = VBRP.MATNR or VBAP.MATNR or LIPS.MATNR) and MAKT (link MAKT.MATNR = VBRP.MATNR or MARA.MATNR) so MAKTX (material description) is in the result; optionally MVKE for sales-org attributes. Use only column names from the column mappings.
- **Always use product names/descriptions in product data:** Whenever the query result includes material number (MATNR, material_number, or IDNRK in BOM), always include MAKT and select MAKT.MAKTX (material description) so analysis shows product names, not only codes. Join MAKT.MATNR to the table that has the material: VBRP.MATNR, VBAP.MATNR, LIPS.MATNR, MARA.MATNR, KEKO.MATNR, CKIS.MATNR, MBEW.MATNR; for BOM items use MAKT.MATNR = STPO.IDNRK. Optionally filter MAKT.SPRAS = 'E' for English. In "columns" always add MAKT.MAKTX when any material/product column is selected. When the same product name can appear for different material numbers (variant products — e.g. different colour, shape, or size, each with its own material number), include MARA and select MTART (material type), MATKL (material group), MEINS (base unit); if classification or variant attributes are available (e.g. colour, shape, size, or AUSP/CABN/characteristic columns), include them so the analysis can show what differentiates each variant.

Task:
- Choose relevant columns.
- When the user asks for **best selling product by industry**, **best products by industry**, or **which products by industry** (with or without a year): use VBRK, VBRP, KNA1, and MAKT; select VBRP.MATNR, MAKT.MAKTX (product name), KNA1.BRSCH (industry), and VBRP.NETWR or VBRK.NETWR; add year filter on VBRK.FKDAT when user specifies a year; order by NETWR DESC. Do **not** use the customer-only rule below — the result must show **products** and industry, not customers.
- When the user asks **how can we compete against products like X** (e.g. harley jackets): **restrict to that product** — add filter on MAKT.MAKTX for the product name (e.g. Harley) so results are only that product; use VBRK, VBRP, MAKT, and KNA1; select VBRP.MATNR, MAKT.MAKTX, VBRP.NETWR or VBRK.NETWR, and KNA1.BRSCH (industry); order by NETWR DESC.
- When the user asks for **best product data performance**, **product analysis and performance**, **product performance**, or **top products** (with or without "comparative market" or "insights"): use VBRK, VBRP, and MAKT; select VBRP.MATNR, MAKT.MAKTX (product name), and VBRP.NETWR or VBRK.NETWR; **if the user specifies a year range** (e.g. "year 1992 to 2000"): add **two** filters — VBRK.FKDAT >= start year (e.g. "1992") and VBRK.FKDAT <= end year (e.g. "2000"); **if a single year** (e.g. "year 1999"): add one filter VBRK.FKDAT = that year; **if no year**: do NOT add any date filter; order by NETWR DESC. Optionally add KNA1 and BRSCH if the user asks for market or industry context.
- When the user asks for **highest sales by customer**, **sales by customer**, or **revenue by customer** (and optionally **and industry**), and is **not** asking for product(s) by industry: use only VBRK, VBRP, and KNA1; select KNA1.KUNNR, KNA1.NAME1, and VBRP.NETWR or VBRK.NETWR; **if the user asks for industry** (e.g. "sales by customer and industry"), also select KNA1.BRSCH (industry_key); do NOT add VBAK, VBFA, LIKP, LIPS, EKKO, EKPO; order by NETWR DESC. Omit the process-flow rule below for this case.
- When the query involves billing documents or invoices (VBRK, VBRP) and the user did NOT ask specifically for sales by customer / revenue by customer: include VBAK and join VBRP.AUBEL = VBAK.VBELN; for process flow also include VBFA and LIKP (VBRP to VBFA, VBFA to LIKP) to get delivery number (LIKP.VBELN). In "columns" include at least: billing document number (VBRK.VBELN or VBRP.VBELN), sales order number (VBRP.AUBEL or VBAK.VBELN only — never use AUPOS or POSNR; AUPOS is position number only), delivery number (LIKP.VBELN), purchase order number (VBAK.BSTNK).
- When the query involves credit risk, customer credit, credit limits, exposure (e.g. open orders/deliveries), risk categories, or creditworthiness: include KNKK and KNKA with KNA1; join KNKK to T691A on KNKK.CTLPC = T691A.CTLPC AND KNKK.KKBER = T691A.KKBER. In "columns" use only actual SAP field names from the column mappings: from KNKK use KLIMK, SKFOR (credit exposure — never use column name EXPOSURE), CTLPC (risk category — never use RISK_CATEGORY); from KNKA use KLIMG, KLIME, WAERS; from KNA1 use KUNNR, NAME1. Do not use EXPOSURE, RISK_CATEGORY, or CHECK_RESULTS — those columns do not exist. For filters: if comparing credit exposure to credit limit use lhs "KNKK.SKFOR", operator ">", rhs "KNKK.KLIMK" (rhs must be the column reference, not the string 'KNKK.KLIMK'). Only add filters when the user explicitly requests a condition.""" + (
    " When UKM_BP_CMS_SGM is available, also include it and join UKM_BP_CMS_SGM.PARTNER = KNA1.KUNNR; add columns like CREDIT_LIMIT, BLOCK_REASON."
    if fscm_credit_table_available() else
    " Do not add UKM_BP_CMS_SGM (not available in this system)."
) + """
- When the user asks whether a specific invoice number has high credit risk: include VBRK (invoice), KNA1 (customer), KNKK and optionally KNKA and T691A (credit/risk). Join VBRK.KUNAG = KNA1.KUNNR and KNKK.KUNNR = KNA1.KUNNR. Add a filter with lhs "VBRK.VBELN", operator "=", rhs the invoice number as digits (e.g. 90035998) so only that invoice is checked. Select VBRK.VBELN, KNA1.KUNNR, KNA1.NAME1, KNKK.KLIMK, KNKK.SKFOR, KNKK.CTLPC so the user can see credit limit, exposure, and risk category for that invoice.
- **Product costing / cost of goods (COGM/COGS):** For COGS calculation always use net prices, moving average price (MBEW.VERPR), or standard price (MBEW.STPRS): when MBEW.VPRSV = 'S' use STPRS; when VPRSV = 'V' use VERPR; include MBEW.PEINH (price unit) for per-unit COGS. Quantity structure (BOM & Routing) × costs; costing variant (CK11N) defines rules. When the user asks about product costs, cost estimates, COGS, standard price, moving average, CK11N/CK13N/CK24/CK40N, or MM03 valuation: use MBEW (STPRS, VERPR, VPRSV, PEINH) and KEKO + KEPH; add CKIS, CKIT (itemization texts), CKHS, CKEP; join on KALNR. Add MARA and MAKT (MAKTX). Join KEKO.MATNR = MARA.MATNR = MBEW.MATNR, KEKO.BWKEY = MBEW.BWKEY, KEKO.MATNR = MAKT.MATNR. When the user asks for cost of a **specific material by name** (e.g. "what is the cost of Harley leather Jacket"): use ONLY costing tables (MBEW, KEKO, KEPH, MARA, MAKT) and select only cost/price columns (e.g. MAKTX, STPRS, VERPR, PEINH, VPRSV, KST001–KST003, cost component); do NOT include billing document number (VBELN), sales order number (AUBEL), or delivery number (LIKP.VBELN) — the answer is the material's cost/price, not sales or invoice data. Add filter on MAKT.MAKTX with operator "=" and rhs the user's phrase; the system applies case-insensitive matching. When the user asks for **cost of product number X** or **cost of material number X** (e.g. "show me cost of product number H10500", "cost of material number H10500"): use ONLY costing tables (MBEW, KEKO, KEPH, MARA, MAKT); add a filter lhs "MBEW.MATNR" (or "MARA.MATNR" if MBEW not in columns), operator "=", rhs the material number exactly as the user gave it (e.g. H10500); select MATNR, MAKTX (description), STPRS, VERPR, PEINH, VPRSV, BWKEY, KST001, KST002, KST003 so the result gives detailed cost for that one material only — do NOT return other materials. For BOM use STKO/STPO (IDNRK, MENGE) with MBEW; for routing use PLKO/PLPO. For FI/CO postings use COEP, COBK, AUFK. For **posted COGS from FI documents** use BKPF (header: BELNR, BUKRS, GJAHR), BSEG (line: HKONT/SAKNR, DMBTR, WRBTR), SKA1/SKB1 (G/L account master for account description), T001K (BWKEY to BUKRS for valuation area), T030 (chart/posting config), and ACDOCA (S/4: RACCT, WTGBTR, filter by COGS account). Use only column names from the column mappings.
- **Data needed for COGS / what is often missing:** To calculate COGS you need: (1) **Quantity sold/issued** — from VBRP.FKIMG (billed qty) or MSEG.MENGE (goods issue movements, BWART e.g. 601); (2) **Valuation price** — MBEW.STPRS or VERPR per VPRSV, and MBEW.PEINH; (3) **Valuation area (BWKEY)** — VBRP has no BWKEY; join VBRP → VBAP (AUBEL, AUPOS) to get plant (WERKS); use WERKS as BWKEY if valuation is at plant level, then join MBEW on MATNR and BWKEY; T001K links BWKEY to company code (BUKRS); (4) **Unit of measure (UoM) conversion** — use MARM (MATNR, MEINS, UMREZ, UMREN: base qty = quantity × UMREZ/UMREN) and MEAN (MATNR, MEINS) linked to MARA on MATNR; join VBRP or MSEG to MARM on MATNR and match MEINS to sales UoM to get conversion to base/price unit; (5) **Posted COGS** — for actual posted COGS use BKPF + BSEG (ERP: filter BSEG by COGS account HKONT/SAKNR) or ACDOCA (S/4 HANA: filter by RACCT = COGS account); use SKA1/SKB1 for account description and T001K for valuation area to company code. When the user asks about UoM conversion or COGS quantity conversion, include MARM and optionally MEAN.
- **Verify / correct standard price data, meaningful standard price analysis:** When the user asks to verify or correct standard price data, or to enable meaningful analysis of standard prices: use MBEW with MARA and MAKT. Select MBEW.MATNR, MAKT.MAKTX (material description), MBEW.STPRS (standard price), MBEW.VERPR (moving average price), MBEW.PEINH (price unit), MBEW.VPRSV (price control: S = standard, V = moving average), MBEW.BWKEY (valuation area). For meaningful analysis exclude or flag rows where standard price (STPRS) is zero or null (unreleased or missing prices): either add a SQL filter (e.g. STPRS IS NOT NULL AND STPRS > 0) to exclude such rows, or return all rows and the analysis will flag them. Always include PEINH (price_unit) and note in the result that price_unit defines the quantity per which the price applies. Use only column names from the column mappings.
- **Percentage of finished goods / raw materials and semi-finished goods:** When the user asks for percentage of finished goods, percentage of raw materials and semi-finished goods, or sales value breakdown by material type: use VBRK and VBRP for sales value (VBRP.NETWR); add MARA and join VBRP.MATNR = MARA.MATNR so MARA.MTART (material type) is available. SAP: MTART = 'FERT' = finished goods, 'HALB' = semi-finished, 'ROH' = raw materials. Select columns: MARA.MTART (material_type), VBRP.NETWR (sales value per line). Optionally add MAKT for material description. Leave filters empty so totals include all goods; the analysis will compute: Percentage of finished goods = (Total sales value where MTART = 'FERT') / (Total sales value of all goods) * 100; Percentage of raw materials and semi-finished goods = (Total sales value where MTART IN ('ROH','HALB')) / (Total sales value of all goods) * 100. Revenue is only for billing types A,B,C,D,E,I,L,W (applied automatically). Use only column names from the column mappings.
- **Profitability analysis:** Profitability depends on cost of goods, market demand, pricing strategies, and operational costs (not all may be in the data). When the user asks for profitability analysis or reporting: (1) Use FAGLFLEXA (ERP) or ACDOCA (S/4 HANA) for consolidated profitability line items; select columns from the column mappings (e.g. company code, profit center, segment, account, amount, currency). (2) Where the user needs to break down by these factors, also use: cost of goods (COGS) — use MBEW with net prices, standard price (STPRS), or moving average price (VERPR) per VPRSV; include PEINH for per-unit COGS; KEKO/KEPH for cost estimates; pricing/revenue — VBRK, VBRP (NETWR); demand/volume — VBRP, VBAK/VBAP; operational costs — COEP, AUFK. Join to MARA/MAKT where relevant. Use only column names from the column mappings.
- **Cost-benefit analysis (import vs domestic production or purchasing / make vs buy):** When the user asks for cost-benefit analysis, importing vs domestic production or purchasing, or make-vs-buy decisions: use EKKO and EKPO for purchase/import side (EKPO.NETPR = purchase price, EKPO.MATNR, EKPO.MENGE; EKKO.LIFNR = vendor; join EKKO.EBELN = EKPO.EBELN). Use MBEW for domestic/production cost side (MBEW.STPRS = standard price, MBEW.VERPR = moving average price; join MBEW.MATNR = EKPO.MATNR). Add MARA and MAKT for material description; add LFA1 for vendor name and LFA1.LAND1 (vendor country) to support import vs domestic comparison. Select columns so the result shows material, purchase price (from EKPO), standard price (from MBEW), and optionally vendor/country so the user can compare import cost vs domestic cost. Use only column names from the column mappings.
- **Component data of main material:** When the user asks for component data, components of a material, BOM components, or bill of material for a material: use **MAST** (Material to BOM Link: MATNR = main material, WERKS = plant, STLNR = BOM number), **STKO** (BOM header: STLNR, STLAL, STLST), **STPO** (BOM item: STLNR, POSNR, IDNRK, MENGE, MEINS). Join MAST.MATNR = MARA.MATNR (main material), MAST.STLNR = STKO.STLNR = STPO.STLNR; for component description join MAKT.MATNR = STPO.IDNRK; for main material description join MAKT.MATNR = MAST.MATNR (or MARA.MATNR). **MAST has no VBELN** — do not join MAST on VBELN. Use only real table names (MARA, MAKT, MAST, STKO, STPO) in "tables" and "joins"; do not use made-up names like "main_material" or "component_material". To get both main and component descriptions, use MARA once (for main material) and MAKT twice (once for main MAKT.MATNR = MAST.MATNR, once for component MAKT.MATNR = STPO.IDNRK). Select MAST.MATNR, MAST.WERKS, MAST.STLNR, STKO.STLAL, STKO.STLST, STPO.POSNR, STPO.IDNRK, STPO.MENGE, STPO.MEINS, and MAKT.MAKTX (from the MAKT joined to MAST for main, and from the MAKT joined to STPO for component — use different column aliases in the JSON so the two MAKTX columns are distinct). **When the user specifies a product name** (e.g. "motorcycle", "Harley", "motorcycle manufacturing"): add a filter on the **main material's** description so only BOMs for that product are returned — filter the MAKT that is joined to MAST.MATNR (main material description) with MAKT.MAKTX LIKE '%product_name%' and MAKT.SPRAS = 'E'. This prevents returning components from unrelated BOMs (e.g. food, IT) when the user asked for motorcycle or Harley components. Use only column names from the column mappings.
- **Products manufactured internally / procurement type / production orders / dependencies:** **Primary SAP field for internal vs external:** **MARC.BESKZ** (Procurement Type, MRP 2): E = In-house, F = External, X = Both. MARC.SOBSL = Special Procurement. **MBEW** (Accounting 1/Costing 1): VPRSV (S = Standard price for manufactured, V = Moving average for procured), STPRS, VERPR, LOSGR (costing lot size). When the user asks about products manufactured internally, procurement type, or production/dependencies: use **MARC** (select BESKZ, SOBSL; join MARC.MATNR = MARA.MATNR), **MBEW** (VPRSV, STPRS, VERPR, LOSGR), **AUFK**, **VBAK**, **VBAP**, **COEP**, **MAST**, **STKO**, **STPO**, **MARA**, **MAKT**. Select **MARC.BESKZ** (procurement_type) and optionally MARC.SOBSL so the result shows the primary indicator; include MBEW.VPRSV, STPRS, VERPR when costing is relevant. **Do not join AUFK to MARA on MATNR** (AUFK has no MATNR); to show production orders with materials use AUFK only with VBAK/VBAP (AUFK.KDAUF = VBAK.VBELN, AUFK.KDPOS = VBAP.POSNR) and then VBAP.MATNR, MAKT.MAKTX. **When the user asks for products manufactured internally for a specific product** (e.g. "for the motorcycle"): add filter MAKT.MAKTX LIKE '%product_name%' and MAKT.SPRAS = 'E' on the MAKT joined to the material (e.g. VBAP.MATNR). Add a **data_note** explaining: **Procurement Type (MARC.BESKZ)** is the primary SAP field for internal vs external: E = In-house produced, F = Externally procured, X = Both; MBEW (VPRSV, STPRS, VERPR, LOSGR) handles costing and valuation. Optionally also note: production order data (AUFK) and make-to-order/make-to-stock (KDAUF link) and BOM/purchase order dependencies. Use only column names from the column mappings.
- **Key fields for procurement (Material Master):** Use these when the user asks about purchasing, ordering, delivery time, valuation, or procurement details. **MARC (plant-specific):** Purchasing view — EKGRP = purchasing group (buyer responsibility), BSTME = order unit, MMSTA = plant-specific material status (procurement control), UEBTO/UNTTO = over/under-delivery tolerances, XCHAR = batch management; MRP views — BESKZ = procurement type (E/F/X), SOBSL = special procurement type, PLIFZ = planned delivery time (days), WEBAZ = goods receipt processing time (days), DISMM = MRP type. **MARA (client level):** MATKL = material group, MEINS = base unit of measure. **MBEW (valuation area):** BKLAS = valuation class (G/L for posting), VPRSV = price control (S = Standard, V = Moving average). Use only column names from the column mappings.
- All columns used in "columns", "filters", and "order_by" **must** be present in the column mappings
- For each column, use "name" exactly as in the column mappings for that table (e.g. KNKK: use KLIMK, SKFOR, CTLPC — never EXPOSURE, RISK_CATEGORY, or CHECK_RESULTS)
- Identify tables needed.
- Identify joins.
- For "best products by value and industry" leave filters empty. Add other filters only when the user explicitly requests them (e.g. a specific country).
- **When the user asks to show/list/filter products by a specific name** (e.g. "show Harley products", "Harley products", "analyse Harley", "analyze Harley", "different Harley products", "products named Harley", "show me all Harley product numbers and names"): add a filter so materials matching that name are returned (lhs "MAKT.MAKTX", operator "=", rhs the product name **only**, e.g. "Harley" — never use the full phrase like "analyse Harley"; the system converts to LIKE so all materials whose description contains the name are returned). For "all [name] product numbers and names" you MUST select both MATNR (material number) and MAKTX (material name) so the result lists every product number and name linked to that name. **Table usage:** If the user did NOT ask for revenue, sales value, invoices, net/gross value, taxes, or delivery/logistics: use ONLY **master data tables** MARA, MAKT, MEAN, MARM, MVKE — do not use VBRK, VBRP, VBAK, VBAP, BSAD, BSID (sales/revenue) or LIKP, LIPS, VBFA, VTTK (logistics). When the user **explicitly** asks for revenue, sales value, items sold, net/gross value, taxes: use VBRK, VBRP, VBAK, VBAP, BSAD, BSID to derive value. When the user **explicitly** asks for delivery or logistics: use LIKP, LIPS, VBFA, VTTK. LIKP, LIPS, VTTK have nothing to do with sales revenue directly — only add them when delivery/logistics is asked. **Product analysis — one language, no repeat by language:** Always add filter MAKT.SPRAS = 'E' (or one language) when MAKT is used so the same material is not repeated for different languages. The focus is to identify **same product name with different material numbers** (variants), not to show the same material in multiple languages.
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
    content = response.choices[0].message.content
    spec = _parse_json_spec(content)
    if spec is None:
        return None
    spec = ensure_delivery_chain_in_spec(spec)
    return spec

# Explanation for users when cost-of-goods queries return no data (SAP: Make vs. Buy, Standard vs. Actual)
COGS_CALCULATION_EXPLANATION = """
**How SAP calculates Cost of Goods Sold (COGS)**

In SAP, COGS depends on **procurement type** (Make vs. Buy) and **valuation method** (Standard vs. Actual) configured for the material.

---

**1. Manufactured internally (in-house production)**

For products made in-house, SAP calculates **Cost of Goods Manufactured (COGM)** using a **Material Cost Estimate with Quantity Structure**.

- **Bill of Materials (BOM):** Determines the cost of raw materials and semi-finished goods (tables **STKO/STPO**, component **IDNRK**, **MENGE**).
- **Routing / recipe:** Determines direct labour and machine activity costs by multiplying required time by planned activity rates from the cost centre (**PLKO/PLPO**, **CSLA**).
- **Overheads:** Calculated via a **Costing Sheet** that applies percentage or quantity-based surcharges for factory utilities and administration.
- **Settlement:** During production, actual costs are debited to the production order. When finished goods are received, the order is credited at the **Standard Price**; any difference is posted as a **Variance** at period-end.  
  Cost estimates are stored in **KEKO** (header) and **KEPH** (cost component split: material, labour, overhead — e.g. KST001–KST003).

---

**2. Procured from third-party vendors**

For traded goods or raw materials, cost is based on purchase price and landed costs.

- **Moving Average Price (V):** **VPRSV = 'V'** — cost is updated automatically with every goods receipt based on the actual invoice price (**MBEW.VERPR**).
- **Standard Price (S):** **VPRSV = 'S'** — procurement is valuated at a fixed price (**MBEW.STPRS**). Any difference between the purchase order price and the standard price is posted to a **Purchase Price Variance (PPV)** account.

**Price unit (PEINH)** in MBEW defines the quantity per which the price applies; per-unit cost = price ÷ PEINH.

---

**3. Subcontracting and other scenarios**

- **Subcontracting:** Cost = value of components provided to the vendor **plus** the **Subcontracting Service Fee** (external activity).
- **Co-products / by-products:** Costs are distributed using an **Apportionment Structure** (co-products) or the **Net Realizable Value** method, where the value of by-products reduces the total cost of the primary product.
- **Intercompany transfers:** Costs are transferred between plants or company codes using a **Transfer Price**, often with a **Partner Cost Component Split** to keep visibility of original production costs (**T001K** links valuation area **BWKEY** to company code **BUKRS**).

---

**4. Actual Costing (Material Ledger)**

If **Actual Costing** is active, SAP initially valuates all movements at standard price but recalculates an **actual price** at period-end. This process "rolls up" purchase and production variances to final inventory and COGS (e.g. **CKMLCR** for material ledger period data), giving more accurate financial results.

---

**How this app shows the data**

The app reads **valuation and cost estimate data** from SAP (e.g. **MBEW** for STPRS/VERPR/VPRSV/PEINH; **KEKO/KEPH** for cost estimates; **BKPF/BSEG** or **ACDOCA** for posted COGS). It does not compute COGS itself; it displays what is in these tables.

**Why you may see “no data available”**

- The **product name** you used does not match any **MAKT.MAKTX** (material description) in the system.  
- The material exists in **MAKT/MARA** but has **no row in MBEW or KEKO** (no valuation or cost estimate for that material/valuation area).  
- The material has **STPRS/VERPR = 0 or null** (price not yet released or missing).

**What to try**

- Use **“Show materials with standard price”** or **“List cost estimates”** to see which materials have cost data; then filter by product name.  
- Check the exact product name (spelling, language MAKT.SPRAS = 'E') and material number in your system.
"""


def get_cogs_calculation_answer_if_asked(user_query: str, df: pd.DataFrame) -> str | None:
    """If the user is asking how cost of goods is calculated for the product, return a full answer (explanation + note when all cost values are zero). Otherwise return None."""
    if not user_query or not isinstance(user_query, str) or df is None or df.empty:
        return None
    q = user_query.strip().lower()
    if not (
        ("cost of goods" in q or "cogs" in q or "cost of the product" in q)
        and ("calculat" in q or "how" in q or "what" in q or "explain" in q)
    ):
        return None
    # Check for cost-related columns and whether all values are zero/null
    cost_cols = []
    for c in df.columns:
        if not c:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if any(x in cu for x in ("STANDARD_PRICE", "STPRS", "MOVING_AVERAGE", "VERPR", "COST_COMPONENT", "KST001", "KST002", "KST003")):
            cost_cols.append(c)
    note = ""
    if cost_cols:
        try:
            all_zero = True
            for c in cost_cols:
                ser = pd.to_numeric(df[c], errors="coerce")
                if ser.notna().any() and (ser.fillna(0) != 0).any():
                    all_zero = False
                    break
            if all_zero:
                note = "\n\n---\n\n**About the table above:** The query returned materials and cost columns (standard price, moving average price, cost components), but **all values are 0 or empty** (prices not released or missing in SAP). The calculation method is as above; once prices are released in MM03 or cost estimates are run in CK11N/CK13N, the same query will show actual figures."
        except Exception:
            pass
    return (COGS_CALCULATION_EXPLANATION + note).strip() or None


def show_single_material_cost_summary(user_query: str, df: pd.DataFrame) -> bool:
    """When the user asked for cost of a specific product number and the result has one material (one row or same MATNR), show a detailed cost summary. Returns True if shown."""
    if df is None or df.empty or len(df) == 0:
        return False
    requested_matnr = _extract_material_number_from_query(user_query)
    cols = list(df.columns)
    matnr_col = _get_material_number_column(cols)
    if not matnr_col or matnr_col not in df.columns:
        return False
    # Single material: one row or all rows same MATNR
    unique_matnr = df[matnr_col].dropna().astype(str).str.strip().unique()
    if len(unique_matnr) != 1:
        return False
    matnr_val = str(unique_matnr[0]).strip()
    # Prefer first row (after deduplication we have one per material)
    row = df.iloc[0]
    # Build display: label -> value for material number, description, and cost fields
    cost_keywords = ("standard_price", "stprs", "moving_average", "verpr", "price_unit", "peinh", "price_control", "vprsv", "valuation_area", "bwkey", "cost_component", "kst001", "kst002", "kst003")
    display_labels = {
        "material_number": "Material number", "MATERIAL_NUMBER": "Material number", "matnr": "Material number", "MATNR": "Material number",
        "standard_price": "Standard price", "STANDARD_PRICE": "Standard price", "stprs": "Standard price", "STPRS": "Standard price",
        "moving_average_price": "Moving average price", "MOVING_AVERAGE_PRICE": "Moving average price", "verpr": "Moving average price", "VERPR": "Moving average price",
        "price_unit": "Price unit", "PRICE_UNIT": "Price unit", "peinh": "Price unit", "PEINH": "Price unit",
        "price_control": "Price control", "PRICE_CONTROL": "Price control", "vprsv": "Price control", "VPRSV": "Price control",
        "valuation_area": "Valuation area", "VALUATION_AREA": "Valuation area", "bwkey": "Valuation area", "BWKEY": "Valuation area",
        "cost_component_1": "Cost component 1", "KST001": "Cost component 1", "kst001": "Cost component 1",
        "cost_component_2": "Cost component 2", "KST002": "Cost component 2", "kst002": "Cost component 2",
        "cost_component_3": "Cost component 3", "KST003": "Cost component 3", "kst003": "Cost component 3",
    }
    desc_col = None
    for c in cols:
        cu = (c or "").upper().replace(" ", "_")
        if "MAKTX" in cu or "DESCRIPTION" in cu or "MATERIAL_DESCRIPTION" in cu:
            desc_col = c
            break
    lines = []
    lines.append(f"**Material number:** {matnr_val}")
    if desc_col and desc_col in df.columns:
        lines.append(f"**Description:** {row.get(desc_col, '')}")
    for c in cols:
        if c in (matnr_col, desc_col):
            continue
        cu = (c or "").upper().replace(" ", "_")
        if any(kw in cu for kw in cost_keywords):
            label = display_labels.get(cu, display_labels.get(c, c))
            val = row.get(c)
            if pd.isna(val):
                val = "—"
            else:
                try:
                    v = pd.to_numeric(val, errors="coerce")
                    if not pd.isna(v) and isinstance(val, (int, float)):
                        val = v
                except Exception:
                    pass
            lines.append(f"**{label}:** {val}")
    if len(lines) <= 2:
        return False
    st.write("### Cost for product number " + matnr_val)
    st.markdown("\n".join(lines))
    if requested_matnr and requested_matnr.upper() != matnr_val.upper():
        st.caption(f"Query was for product number **{requested_matnr}**; result shows material **{matnr_val}** (exact match from SAP).")
    return True


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

def _is_compete_against_products_query(user_query: str) -> bool:
    """True when the user asks how to compete against products like X (query results restricted to that product; insights on same-industry competitors)."""
    if not user_query or not isinstance(user_query, str):
        return False
    q = user_query.strip().lower()
    return any(
        phrase in q
        for phrase in (
            "compete against",
            "competing against",
            "compete with",
            "competing with",
            "how can we compete",
            "how to compete",
            "compete with products like",
            "compete against products like",
        )
    )


def _extract_product_name_from_query(user_query: str) -> str:
    """Extract product name when user asks for specific products (e.g. 'show Harley products' -> 'Harley'; 'compete against products like harley jackets' -> 'Harley'). Returns empty string if none."""
    if not user_query or not isinstance(user_query, str):
        return ""
    s = user_query.strip()
    if not s:
        return ""
    s_lower = s.lower()
    # "how can we compete against products like harley jackets" -> extract "Harley" (first word) so results restrict to that product
    if _is_compete_against_products_query(user_query):
        m = re.search(r"products like\s+([A-Za-z0-9_\-\s]+?)(?:\s+and|\s*$|\.|,|\?|\))", s_lower, re.IGNORECASE)
        if m:
            phrase = s[m.start(1):m.end(1)].strip()
            if phrase:
                first_word = phrase.split()[0] if phrase.split() else phrase
                if first_word.lower() not in ("the", "a", "an", "our", "my"):
                    return first_word.capitalize() if len(first_word) > 1 else first_word

    def _normalize_all(name: str) -> str:
        """Strip leading 'all ' so 'all Harley' -> 'Harley' for correct LIKE %Harley% (return all materials containing Harley)."""
        if not name:
            return name
        n = name.strip()
        if n.lower().startswith("all ") and len(n) > 4:
            return n[4:].strip()
        return n

    def _strip_analyse_prefix(name: str) -> str:
        """Strip leading 'analyse ' or 'analyze ' so we never use 'analyse Harley' as filter (use 'Harley' only)."""
        if not name:
            return name
        n = name.strip()
        if n.lower().startswith("analyse ") and len(n) > 8:
            return n[8:].strip()
        if n.lower().startswith("analyze ") and len(n) > 8:
            return n[8:].strip()
        return n

    # "show me all Harley product numbers and names", "all Harley products"
    m = re.search(r"(?:show\s+me?\s+)?(?:all\s+)?(.+?)\s+product(?:\s+numbers?\s+and\s+names?|\s*numbers?)?\s*\.?\s*$", s_lower, re.IGNORECASE)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        name = _normalize_all(name)
        if name and name.lower() not in ("all", "the", "my", "our") and len(name) > 0:
            return _strip_analyse_prefix(name)
    # "show Harley products", "show me Harley leather jacket products", "list Harley products", "analyse Harley products"
    m = re.search(r"(?:show\s+me?\s+)?(.+?)\s+products\b", s_lower, re.IGNORECASE | re.DOTALL)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        name = _normalize_all(name)
        if name and name.lower() not in ("all", "the", "my", "our"):
            return _strip_analyse_prefix(name)
    # "Harley products" at start or "products named Harley", "products called Harley"
    if s_lower.endswith(" products") and len(s) > 9:
        name = s[:-9].strip()
        name = _normalize_all(name)
        if name and name.lower() not in ("show", "list", "display", "get", "all", "the"):
            return _strip_analyse_prefix(name)
    for pattern in ("products named ", "products called "):
        if pattern in s_lower:
            idx = s_lower.index(pattern) + len(pattern)
            name = s[idx:].strip()
            if name:
                name = name.split(",")[0].split(".")[0].strip()
                name = _normalize_all(name)
                return _strip_analyse_prefix(name) if name else ""
    # "analyse Harley", "analyze Harley", "analyse X" -> extract X (product name); avoid using "analyse Harley" as filter
    m = re.search(r"\b(?:analyse|analyze)\s+([A-Za-z0-9_\-]+)(?:\s|$|,|\.)", s_lower, re.IGNORECASE)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        if name and len(name) <= 40 and name.lower() not in ("the", "a", "an", "all"):
            return name
    # "[Name] analysis" e.g. "Harley analysis", "Harley product analysis"
    m = re.search(r"(?:^|[\s,])([A-Za-z0-9_\-\s]+?)\s+analysis\s*\.?\s*$", s_lower, re.IGNORECASE)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        name = _normalize_all(name)
        skip = ("product", "data", "sales", "the", "a", "an", "query", "result")
        if name and name.lower() not in skip and len(name) <= 50:
            return _strip_analyse_prefix(name)
    # "components of X", "component of X", "BOM for X", "X manufactured internally", "X manufacturing"
    m = re.search(r"(?:components?|bom|bill\s+of\s+material)\s+(?:of|for)\s+(.+?)(?:\s+manufactured|\s*$|,)", s_lower, re.IGNORECASE | re.DOTALL)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        name = _normalize_all(name)
        if name and len(name) <= 80:
            return name.split(",")[0].strip()
    # "products manufactured internally for the motorcycle", "manufactured internally for the X"
    m = re.search(r"manufactured\s+internally\s+for\s+(?:the\s+)?([A-Za-z0-9_\-]+)(?:\s|$|,|\.)", s_lower, re.IGNORECASE)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        if name and len(name) <= 40 and name.lower() not in ("the", "a", "an"):
            return name
    m = re.search(r"(.+?)\s+manufactured\s+internally", s_lower, re.IGNORECASE)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        name = _normalize_all(name)
        # Avoid capturing "show me all products" — only use if it looks like a product name (single word or known term)
        if name and len(name) <= 80:
            first = name.split(",")[0].strip()
            if first.lower() not in ("show me all", "show me", "all", "the", "products", "product"):
                return _strip_analyse_prefix(first)
    m = re.search(r"(.+?)\s+manufacturing\s*\.?\s*$", s_lower, re.IGNORECASE)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        name = _normalize_all(name)
        skip = ("the", "a", "an")
        if name and len(name) <= 80 and name.lower() not in skip:
            return name.split(",")[0].strip()
    # "X manufacturing" anywhere (e.g. "not used in Motorcycle manufacturing why are they shown")
    m = re.search(r"\b([A-Za-z0-9_\-]+)\s+manufacturing\b", s_lower, re.IGNORECASE)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        if name and len(name) <= 40 and name.lower() not in ("the", "a", "an"):
            return name
    # "used in manufacturing the X", "products and components used in manufacturing the motorcycle"
    m = re.search(r"manufacturing\s+(?:the\s+)?([A-Za-z0-9_\-]+)(?:\s|$|,|\.)", s_lower, re.IGNORECASE)
    if m:
        name = s[m.start(1):m.end(1)].strip()
        if name and len(name) <= 40 and name.lower() not in ("the", "a", "an"):
            return name
    # Standalone "motorcycle" or "Harley" when query is about products/components (e.g. "why are these shown ... motorcycle manufacturing")
    for term in ("motorcycle", "harley"):
        if re.search(rf"\b{term}\b", s_lower) and any(k in s_lower for k in ("manufacturing", "components", "products", "shown", "shown here", "why")):
            return term.capitalize() if term == "harley" else term
    return ""


def _extract_material_number_from_query(user_query: str) -> str:
    """Extract material/product number when user asks for a specific one (e.g. 'cost of product number H10500' -> 'H10500'). Returns empty string if none."""
    if not user_query or not isinstance(user_query, str):
        return ""
    s = user_query.strip()
    if not s:
        return ""
    s_lower = s.lower()
    # "product number H10500", "material number H10500", "cost of the product number H10500"
    for pattern in (
        r"product\s+number\s+([A-Za-z0-9_\-]+)",
        r"material\s+number\s+([A-Za-z0-9_\-]+)",
        r"material\s+([A-Za-z0-9_\-]+)(?:\s|$|,)",
        r"product\s+([A-Za-z0-9_\-]+)(?:\s+number|\s*cost|$|,)",
    ):
        m = re.search(pattern, s_lower, re.IGNORECASE)
        if m:
            val = s[m.start(1):m.end(1)].strip()
            if val and len(val) <= 40 and val.lower() not in ("the", "a", "an", "cost", "price"):
                return val
    return ""


def inject_material_number_filter_if_needed(user_query: str, json_spec: dict) -> dict:
    """When user asks for cost (or data) of a specific product/material number (e.g. H10500), add MATNR filter so only that material is returned."""
    if not user_query or not json_spec:
        return json_spec
    matnr = _extract_material_number_from_query(user_query)
    if not matnr:
        return json_spec
    # Which table with MATNR is in the spec? Prefer MBEW for cost, then MARA, MAKT, KEKO
    tables = set()
    for c in json_spec.get("columns", []):
        if c.get("table"):
            tables.add(c.get("table"))
    for j in json_spec.get("joins", []):
        tables.add(j.get("left"))
        tables.add(j.get("right"))
    for t in json_spec.get("tables", []):
        if isinstance(t, dict) and t.get("name"):
            tables.add(t.get("name"))
    # Tables that have MATNR
    matnr_tables = [t for t in ("MBEW", "MARA", "MAKT", "KEKO", "CKIS", "VBRP", "VBAP", "LIPS") if t in tables]
    if not matnr_tables:
        return json_spec
    # Prefer first that appears (MBEW/MARA/MAKT/KEKO for cost queries)
    lhs_table = matnr_tables[0]
    lhs = f"{lhs_table}.MATNR"
    # Already have MATNR filter for this value?
    filters = json_spec.get("filters", [])
    for f in filters:
        if "MATNR" in str(f.get("lhs", "")).upper() and str(f.get("rhs", "")).strip().upper() == matnr.upper():
            return json_spec
    json_spec.setdefault("filters", [])
    json_spec["filters"].append({"lhs": lhs, "operator": "=", "rhs": matnr})
    return json_spec


# Tables that have MATNR (or IDNRK for STPO) and can join to MAKT for product-name filtering
_TABLES_THAT_JOIN_TO_MAKT = ("MARA", "VBRP", "VBAP", "LIPS", "MBEW", "KEKO", "EKPO", "CKIS", "STPO", "EBAN", "MSEG", "MAST")


def inject_product_name_filter_if_needed(user_query: str, json_spec: dict) -> dict:
    """When user asks for a specific product by name (e.g. 'show Harley products', 'compete against products like harley jackets') add MAKTX filter so results are restricted to that product."""
    if not user_query or not json_spec:
        return json_spec
    product_name = _extract_product_name_from_query(user_query)
    if not product_name:
        return json_spec
    # Collect tables in the spec
    columns = json_spec.get("columns", [])
    tables = {c.get("table") for c in columns if c.get("table")}
    for j in json_spec.get("joins", []):
        tables.add(j.get("left"))
        tables.add(j.get("right"))
    for t in json_spec.get("tables", []):
        if isinstance(t, dict) and t.get("name"):
            tables.add(t.get("name"))
    # If MAKT is not in the spec but we have a table that can join to MAKT, add MAKT so we can filter by product name (avoids irrelevant material numbers like 000000000000000023 in "Harley" results)
    if "MAKT" not in tables:
        partner = next((t for t in _TABLES_THAT_JOIN_TO_MAKT if t in tables), None)
        if partner is not None:
            json_spec.setdefault("columns", [])
            if not any(c.get("table") == "MAKT" and c.get("name") == "MAKTX" for c in json_spec["columns"]):
                json_spec["columns"].append({"table": "MAKT", "name": "MAKTX", "description": "material_description"})
            json_spec.setdefault("joins", [])
            if not any(j.get("right") == "MAKT" for j in json_spec["joins"]):
                json_spec["joins"].append({"left": partner, "right": "MAKT", "type": "inner"})
            tables.add("MAKT")
    if "MAKT" not in tables:
        return json_spec
    # Already a filter on material description?
    filters = json_spec.get("filters", [])
    has_maktx_filter = any(
        ("MAKTX" in str(f.get("lhs", "")).upper() or "MATERIAL_DESCRIPTION" in str(f.get("lhs", "")).upper())
        for f in filters
    )
    if has_maktx_filter:
        return json_spec
    # Add filter: MAKT.MAKTX = product_name (json_to_sql will apply LOWER + LIKE for MAKTX)
    json_spec.setdefault("filters", [])
    json_spec["filters"].append({"lhs": "MAKT.MAKTX", "operator": "=", "rhs": product_name})
    # For product analysis: one row per material, not per language — add MAKT.SPRAS = 'E' so same material is not repeated for different languages
    inject_makt_single_language_if_needed(json_spec)
    return json_spec


def inject_makt_single_language_if_needed(json_spec: dict, language: str = "E") -> dict:
    """When MAKT is in the spec, add MAKT.SPRAS = language (default 'E') if not already present. For product analysis we do not repeat the same material for different languages; focus is on same product name with different material numbers (variants)."""
    if not json_spec:
        return json_spec
    tables = set()
    for c in json_spec.get("columns", []):
        if c.get("table"):
            tables.add(c.get("table"))
    for j in json_spec.get("joins", []):
        tables.add(j.get("left"))
        tables.add(j.get("right"))
    for t in json_spec.get("tables", []):
        if isinstance(t, dict) and t.get("name"):
            tables.add(t.get("name"))
    if "MAKT" not in tables:
        return json_spec
    has_spras = any("SPRAS" in str(f.get("lhs", "")).upper() for f in json_spec.get("filters", []))
    if has_spras:
        return json_spec
    json_spec.setdefault("filters", [])
    json_spec["filters"].append({"lhs": "MAKT.SPRAS", "operator": "=", "rhs": language})
    return json_spec


def filter_dataframe_by_product_name_if_requested(user_query: str, df: pd.DataFrame) -> pd.DataFrame:
    """When the user asked for a product by name (e.g. Harley, motorcycle), keep only rows whose material description contains that name. For BOM/component results, filter by the MAIN material description so we keep only components of that product (and drop rows from unrelated BOMs like food or IT)."""
    if not user_query or df is None or df.empty:
        return df
    product_name = _extract_product_name_from_query(user_query)
    if not product_name:
        return df
    cols = list(df.columns)
    # For BOM/component data we must filter by MAIN material description, not component description
    desc_col = None
    if _result_looks_like_bom_or_components(cols):
        for c in cols:
            cu = (c or "").upper()
            if "MAIN" in cu and ("DESCRIPTION" in cu or "MAKTX" in cu) and "COMPONENT" not in cu:
                desc_col = c
                break
    if desc_col is None:
        for c in cols:
            cu = (c or "").upper()
            if cu == "MAKTX" or "MATERIAL_DESCRIPTION" in cu or (cu != "MATNR" and "DESCRIPTION" in cu and "MATERIAL" in cu):
                desc_col = c
                break
    if desc_col is None:
        return df
    try:
        mask = df[desc_col].astype(str).str.upper().str.contains(product_name.upper(), na=False, regex=False)
        return df.loc[mask].copy()
    except Exception:
        return df


def fix_date_filters(json_spec: dict):
    """Normalize date filters. Single year: replace FKDAT='YYYY' with >= YYYY0101 and <= YYYY1231. Year range (>=/<=): convert 4-digit year to YYYY0101/YYYY1231. If rhs is empty or invalid, drop the filter."""
    if not json_spec or "filters" not in json_spec:
        return json_spec
    new_filters = []
    for filt in json_spec["filters"]:
        lhs = filt.get("lhs", "")
        op = (filt.get("operator") or "=").strip()
        rhs_raw = (filt.get("rhs") or "").strip().strip("'\"")
        col = lhs.split(".")[-1].strip()
        if col.upper() in DATE_COLUMNS:
            if not rhs_raw or rhs_raw.upper() in ("NULL", "NONE", "''"):
                continue
            if re.match(r"^\d{4}$", rhs_raw):
                year = rhs_raw
                # Year range: "1992 to 2000" → LLM adds >= 1992 and <= 2000; keep operator, set full date
                if op == ">=":
                    new_filters.append({"lhs": lhs, "operator": ">=", "rhs": f"'{year}0101'"})
                    continue
                if op == "<=":
                    new_filters.append({"lhs": lhs, "operator": "<=", "rhs": f"'{year}1231'"})
                    continue
                # Single year equality: replace with full-year range
                new_filters.append({"lhs": lhs, "operator": ">=", "rhs": f"'{year}0101'"})
                new_filters.append({"lhs": lhs, "operator": "<=", "rhs": f"'{year}1231'"})
                continue
            new_rhs = convert_date_to_yyyymmdd(rhs_raw)
            if not new_rhs:
                continue
            filt["rhs"] = f"'{new_rhs}'"
        new_filters.append(filt)
    json_spec["filters"] = new_filters
    return json_spec

def _canonical_physical_table(tbl: str) -> str:
    """Return the physical SAP table name for SQL. Strips ' AS alias' and maps logical names to real tables."""
    if not tbl or not isinstance(tbl, str):
        return tbl or ""
    s = tbl.strip()
    if " AS " in s.upper():
        s = s.upper().split(" AS ", 1)[0].strip()
    s_upper = s.upper()
    if s_upper in ("MAIN_MATERIAL", "COMPONENT_MATERIAL"):
        return "MARA"
    if s_upper in ("MAKT_MAIN", "MAKT_COMPONENT"):
        return "MAKT"
    return s


def format_table_name(tbl: str) -> str:
    physical = _canonical_physical_table(tbl)
    if physical == "PSIF_INV_HDR":
        return "[erp].[/PSIF/INV_HDR]"
    if physical == "PSIF_INV_ITEM":
        return "[erp].[/PSIF/INV_ITEM]"
    if physical == "PSIF_ACK":
        return "[erp].[/PSIF/ACK]"
    if physical == "PSIF_SLS_HDR":
        return "[erp].[/PSIF/SLS_HDR]"
    if physical == "PSIF_SLS_ITEM":
        return "[erp].[/PSIF/SLS_ITEM]"
    if physical == "PSIF_SLS_MSG":
        return "[erp].[/PSIF/SLS_MSG]"
    if physical == "PSIF_INV_HDR_IN":
        return "[erp].[/PSIF/INV_HDR_IN]"
    if physical == "PSIF_INV_ITEM_I":
        return "[erp].[/PSIF/INV_ITEM_I]"
    if physical == "PSIF_RBKP_MSG":
        return "[erp].[/PSIF/RBKP_MSG]"
    if physical == "PSIF_RBKP":
        return "[erp].[/PSIF/RBKP]"
    return f"[erp].[{physical}]"


def _is_product_performance_query(user_query: str) -> bool:
    """True if the user is asking for product analysis/performance, best product data performance, or top products (with optional comparative market)."""
    if not user_query or not isinstance(user_query, str):
        return False
    q = user_query.lower().strip()
    if "product" not in q and "products" not in q:
        return False
    performance_phrases = (
        "product analysis", "product performance", "best product", "top product",
        "product data performance", "comparative market", "similar product"
    )
    return any(p in q for p in performance_phrases)


def get_product_performance_fallback_sql(user_query: str, with_date_filter: bool = True) -> str | None:
    """Build a known-good SQL for product performance (VBRK, VBRP, MAKT) with optional year range. If with_date_filter=False, omit date filter (all-time). Returns None if not a product-performance query."""
    if not _is_product_performance_query(user_query):
        return None
    q = user_query.strip()
    start_year = end_year = None
    if with_date_filter:
        range_m = re.search(r"(?:year\s+)?(\d{4})\s+to\s+(\d{4})", q, re.IGNORECASE)
        if range_m:
            start_year, end_year = range_m.group(1), range_m.group(2)
        else:
            single_m = re.search(r"year\s+(\d{4})\b", q, re.IGNORECASE)
            if single_m:
                start_year = end_year = single_m.group(1)
    vbrk = format_table_name("VBRK")
    vbrp = format_table_name("VBRP")
    makt = format_table_name("MAKT")
    fktyp_in = ", ".join(f"N'{c}'" for c in REVENUE_BILLING_CATEGORIES)
    where_parts = [f"vbrk.[FKTYP] IN ({fktyp_in})"]
    if with_date_filter and start_year and end_year:
        where_parts.append(f"vbrk.[FKDAT] >= N'{start_year}0101'")
        where_parts.append(f"vbrk.[FKDAT] <= N'{end_year}1231'")
    where_sql = " AND ".join(where_parts)
    sql = f"""SELECT TOP 100
  vbrp.[MATNR] AS [material_number],
  makt.[MAKTX] AS [material_description],
  vbrp.[NETWR] AS [net_value]
FROM {vbrk} AS vbrk
JOIN {vbrp} AS vbrp ON vbrk.[VBELN] = vbrp.[VBELN]
JOIN {makt} AS makt ON makt.[MATNR] = vbrp.[MATNR] AND makt.[SPRAS] = N'E'
WHERE {where_sql}
ORDER BY vbrp.[NETWR] DESC"""
    return sql


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
        out = re.sub(rf"\b{re.escape(tbl)}\.", f"{alias}.", out)
        # If spec used "TABLE AS logical_name", ON clauses may reference logical_name (e.g. MAKT_main); replace that too
        if " AS " in tbl:
            suffix = tbl.upper().split(" AS ", 1)[1].strip()
            if suffix and re.match(r"^[A-Za-z0-9_]+$", suffix):
                out = re.sub(rf"\b{re.escape(suffix)}\.", f"{alias}.", out, flags=re.IGNORECASE)
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
    # Material master: MARA (general), MVKE (sales); link to document lines on MATNR
    ("VBRP", "MARA"): "VBRP.MATNR = MARA.MATNR",
    ("MARA", "VBRP"): "MARA.MATNR = VBRP.MATNR",
    ("VBAP", "MARA"): "VBAP.MATNR = MARA.MATNR",
    ("MARA", "VBAP"): "MARA.MATNR = VBAP.MATNR",
    ("LIPS", "MARA"): "LIPS.MATNR = MARA.MATNR",
    ("MARA", "LIPS"): "MARA.MATNR = LIPS.MATNR",
    ("MARA", "MVKE"): "MARA.MATNR = MVKE.MATNR",
    ("MVKE", "MARA"): "MVKE.MATNR = MARA.MATNR",
    ("VBRP", "MVKE"): "VBRP.MATNR = MVKE.MATNR",
    ("MVKE", "VBRP"): "MVKE.MATNR = VBRP.MATNR",
    ("VBAP", "MVKE"): "VBAP.MATNR = MVKE.MATNR",
    ("MVKE", "VBAP"): "MVKE.MATNR = VBAP.MATNR",
    # MAKT = material descriptions (MAKTX = product name); join on MATNR
    ("VBRP", "MAKT"): "VBRP.MATNR = MAKT.MATNR",
    ("MAKT", "VBRP"): "MAKT.MATNR = VBRP.MATNR",
    ("MARA", "MAKT"): "MARA.MATNR = MAKT.MATNR",
    ("MAKT", "MARA"): "MAKT.MATNR = MARA.MATNR",
    ("VBAP", "MAKT"): "VBAP.MATNR = MAKT.MATNR",
    ("MAKT", "VBAP"): "MAKT.MATNR = VBAP.MATNR",
    ("LIPS", "MAKT"): "LIPS.MATNR = MAKT.MATNR",
    ("MAKT", "LIPS"): "MAKT.MATNR = LIPS.MATNR",
    ("KEKO", "MAKT"): "KEKO.MATNR = MAKT.MATNR",
    ("MAKT", "KEKO"): "MAKT.MATNR = KEKO.MATNR",
    ("CKIS", "MAKT"): "CKIS.MATNR = MAKT.MATNR",
    ("MAKT", "CKIS"): "MAKT.MATNR = CKIS.MATNR",
    ("MBEW", "MAKT"): "MBEW.MATNR = MAKT.MATNR",
    ("MAKT", "MBEW"): "MAKT.MATNR = MBEW.MATNR",
    ("STPO", "MAKT"): "STPO.IDNRK = MAKT.MATNR",
    ("MAKT", "STPO"): "MAKT.MATNR = STPO.IDNRK",
    ("EKPO", "MAKT"): "EKPO.MATNR = MAKT.MATNR",
    ("MAKT", "EKPO"): "MAKT.MATNR = EKPO.MATNR",
    ("EBAN", "MAKT"): "EBAN.MATNR = MAKT.MATNR",
    ("MAKT", "EBAN"): "MAKT.MATNR = EBAN.MATNR",
    ("MSEG", "MAKT"): "MSEG.MATNR = MAKT.MATNR",
    ("MAKT", "MSEG"): "MAKT.MATNR = MSEG.MATNR",
    # PSIF: EDI/sales messages, inbound invoices, invoice receipt
    ("PSIF_SLS_HDR", "PSIF_SLS_MSG"): "PSIF_SLS_HDR.VBELN = PSIF_SLS_MSG.VBELN",
    ("PSIF_SLS_MSG", "PSIF_SLS_HDR"): "PSIF_SLS_MSG.VBELN = PSIF_SLS_HDR.VBELN",
    ("PSIF_INV_HDR_IN", "PSIF_INV_ITEM_I"): "PSIF_INV_HDR_IN.VBELN = PSIF_INV_ITEM_I.VBELN",
    ("PSIF_INV_ITEM_I", "PSIF_INV_HDR_IN"): "PSIF_INV_ITEM_I.VBELN = PSIF_INV_HDR_IN.VBELN",
    ("PSIF_RBKP", "PSIF_RBKP_MSG"): "PSIF_RBKP.BELNR = PSIF_RBKP_MSG.BELNR AND PSIF_RBKP.GJAHR = PSIF_RBKP_MSG.GJAHR",
    ("PSIF_RBKP_MSG", "PSIF_RBKP"): "PSIF_RBKP_MSG.BELNR = PSIF_RBKP.BELNR AND PSIF_RBKP_MSG.GJAHR = PSIF_RBKP.GJAHR",
    # Product Costing (CO-PC-PCP): link on KALNR
    ("KEKO", "KEPH"): "KEKO.KALNR = KEPH.KALNR",
    ("KEPH", "KEKO"): "KEPH.KALNR = KEKO.KALNR",
    ("KEKO", "CKIS"): "KEKO.KALNR = CKIS.KALNR",
    ("CKIS", "KEKO"): "CKIS.KALNR = KEKO.KALNR",
    ("KEKO", "CKHS"): "KEKO.KALNR = CKHS.KALNR",
    ("CKHS", "KEKO"): "CKHS.KALNR = KEKO.KALNR",
    ("KEKO", "CKEP"): "KEKO.KALNR = CKEP.KALNR",
    ("CKEP", "KEKO"): "CKEP.KALNR = KEKO.KALNR",
    ("KEKO", "MARA"): "KEKO.MATNR = MARA.MATNR",
    ("MARA", "KEKO"): "MARA.MATNR = KEKO.MATNR",
    ("KEKO", "MBEW"): "KEKO.MATNR = MBEW.MATNR AND KEKO.BWKEY = MBEW.BWKEY",
    ("MBEW", "KEKO"): "MBEW.MATNR = KEKO.MATNR AND MBEW.BWKEY = KEKO.BWKEY",
    ("CKIS", "MARA"): "CKIS.MATNR = MARA.MATNR",
    ("MARA", "CKIS"): "MARA.MATNR = CKIS.MATNR",
    ("CKIT", "CKIS"): "CKIT.KALNR = CKIS.KALNR AND CKIT.POSNR = CKIS.POSNR",
    ("CKIS", "CKIT"): "CKIS.KALNR = CKIT.KALNR AND CKIS.POSNR = CKIT.POSNR",
    ("CKIT", "KEKO"): "CKIT.KALNR = KEKO.KALNR",
    ("KEKO", "CKIT"): "KEKO.KALNR = CKIT.KALNR",
    # FI / G/L for COGS: BKPF, BSEG, SKA1, SKB1, T001K, ACDOCA
    ("BKPF", "BSEG"): "BKPF.BELNR = BSEG.BELNR AND BKPF.BUKRS = BSEG.BUKRS AND BKPF.GJAHR = BSEG.GJAHR",
    ("BSEG", "BKPF"): "BSEG.BELNR = BKPF.BELNR AND BSEG.BUKRS = BKPF.BUKRS AND BSEG.GJAHR = BKPF.GJAHR",
    ("BSEG", "SKB1"): "BSEG.BUKRS = SKB1.BUKRS AND (BSEG.HKONT = SKB1.SAKNR OR BSEG.SAKNR = SKB1.SAKNR)",
    ("SKB1", "BSEG"): "SKB1.BUKRS = BSEG.BUKRS AND (SKB1.SAKNR = BSEG.HKONT OR SKB1.SAKNR = BSEG.SAKNR)",
    ("SKB1", "SKA1"): "SKB1.SAKNR = SKA1.SAKNR",
    ("SKA1", "SKB1"): "SKA1.SAKNR = SKB1.SAKNR",
    ("T001K", "MBEW"): "T001K.BWKEY = MBEW.BWKEY",
    ("MBEW", "T001K"): "MBEW.BWKEY = T001K.BWKEY",
    ("ACDOCA", "SKB1"): "ACDOCA.RBUKRS = SKB1.BUKRS AND ACDOCA.RACCT = SKB1.SAKNR",
    ("SKB1", "ACDOCA"): "SKB1.BUKRS = ACDOCA.RBUKRS AND SKB1.SAKNR = ACDOCA.RACCT",
    # Material Ledger & Valuation
    ("MBEW", "MARA"): "MBEW.MATNR = MARA.MATNR",
    ("MARA", "MBEW"): "MARA.MATNR = MBEW.MATNR",
    ("CKMLCR", "MBEW"): "CKMLCR.MATNR = MBEW.MATNR AND CKMLCR.BWKEY = MBEW.BWKEY",
    ("MBEW", "CKMLCR"): "MBEW.MATNR = CKMLCR.MATNR AND MBEW.BWKEY = CKMLCR.BWKEY",
    ("CKMLCR", "MARA"): "CKMLCR.MATNR = MARA.MATNR",
    ("MARA", "CKMLCR"): "MARA.MATNR = CKMLCR.MATNR",
    ("MARC", "MARA"): "MARC.MATNR = MARA.MATNR",
    ("MARA", "MARC"): "MARA.MATNR = MARC.MATNR",
    ("MARC", "VBAP"): "MARC.MATNR = VBAP.MATNR AND MARC.WERKS = VBAP.WERKS",
    ("VBAP", "MARC"): "VBAP.MATNR = MARC.MATNR AND VBAP.WERKS = MARC.WERKS",
    # BOM and Material-to-BOM link (component data of main material)
    ("STKO", "STPO"): "STKO.STLNR = STPO.STLNR",
    ("STPO", "STKO"): "STPO.STLNR = STKO.STLNR",
    ("STPO", "MARA"): "STPO.IDNRK = MARA.MATNR",
    ("MARA", "STPO"): "MARA.MATNR = STPO.IDNRK",
    ("MAST", "MARA"): "MAST.MATNR = MARA.MATNR",
    ("MARA", "MAST"): "MARA.MATNR = MAST.MATNR",
    ("MAST", "STKO"): "MAST.STLNR = STKO.STLNR",
    ("STKO", "MAST"): "STKO.STLNR = MAST.STLNR",
    ("MAST", "STPO"): "MAST.STLNR = STPO.STLNR",
    ("STPO", "MAST"): "STPO.STLNR = MAST.STLNR",
    ("MAST", "MAKT"): "MAST.MATNR = MAKT.MATNR",
    ("MAKT", "MAST"): "MAKT.MATNR = MAST.MATNR",
    # Routing (typical keys; adjust if your system uses PLNTY/PLNNR/PLNKN)
    ("PLKO", "PLPO"): "PLKO.PLNTY = PLPO.PLNTY AND PLKO.PLNNR = PLPO.PLNNR AND PLKO.PLNAL = PLPO.PLNAL",
    ("PLPO", "PLKO"): "PLPO.PLNTY = PLKO.PLNTY AND PLPO.PLNNR = PLKO.PLNNR AND PLPO.PLNAL = PLKO.PLNAL",
    # Cost Object Controlling
    ("COEP", "COBK"): "COEP.KOKRS = COBK.KOKRS AND COEP.BELNR = COBK.BELNR AND COEP.GJAHR = COBK.GJAHR",
    ("COBK", "COEP"): "COBK.KOKRS = COEP.KOKRS AND COBK.BELNR = COEP.BELNR AND COBK.GJAHR = COEP.GJAHR",
    ("AUFK", "COEP"): "AUFK.OBJNR = COEP.OBJNR",
    ("COEP", "AUFK"): "COEP.OBJNR = AUFK.OBJNR",
    # Production order to sales order (make-to-order: KDAUF = sales order, KDPOS = item)
    ("AUFK", "VBAK"): "AUFK.KDAUF = VBAK.VBELN",
    ("VBAK", "AUFK"): "VBAK.VBELN = AUFK.KDAUF",
    ("AUFK", "VBAP"): "AUFK.KDAUF = VBAP.VBELN AND AUFK.KDPOS = VBAP.POSNR",
    ("VBAP", "AUFK"): "VBAP.VBELN = AUFK.KDAUF AND VBAP.POSNR = AUFK.KDPOS",
    # Material Master (MLAN, MAPR, MARM, MEAN — UoM)
    ("MLAN", "MARA"): "MLAN.MATNR = MARA.MATNR",
    ("MARA", "MLAN"): "MARA.MATNR = MLAN.MATNR",
    ("MAPR", "MARA"): "MAPR.MATNR = MARA.MATNR",
    ("MARA", "MAPR"): "MARA.MATNR = MAPR.MATNR",
    ("MARM", "MARA"): "MARM.MATNR = MARA.MATNR",
    ("MARA", "MARM"): "MARA.MATNR = MARM.MATNR",
    ("MEAN", "MARA"): "MEAN.MATNR = MARA.MATNR",
    ("MARA", "MEAN"): "MARA.MATNR = MEAN.MATNR",
    ("VBRP", "MARM"): "VBRP.MATNR = MARM.MATNR",
    ("MARM", "VBRP"): "MARM.MATNR = VBRP.MATNR",
    ("MSEG", "MARM"): "MSEG.MATNR = MARM.MATNR",
    ("MARM", "MSEG"): "MARM.MATNR = MSEG.MATNR",
    ("EKPO", "MARM"): "EKPO.MATNR = MARM.MATNR",
    ("MARM", "EKPO"): "MARM.MATNR = EKPO.MATNR",
    ("MAPR", "MARC"): "MAPR.MATNR = MARC.MATNR AND MAPR.WERKS = MARC.WERKS",
    ("MARC", "MAPR"): "MARC.MATNR = MAPR.MATNR AND MARC.WERKS = MAPR.WERKS",
    # Vendor Master
    ("LFA1", "LFB1"): "LFA1.LIFNR = LFB1.LIFNR",
    ("LFB1", "LFA1"): "LFB1.LIFNR = LFA1.LIFNR",
    ("LFA1", "LFM1"): "LFA1.LIFNR = LFM1.LIFNR",
    ("LFM1", "LFA1"): "LFM1.LIFNR = LFA1.LIFNR",
    ("LFA1", "LFBK"): "LFA1.LIFNR = LFBK.LIFNR",
    ("LFBK", "LFA1"): "LFBK.LIFNR = LFA1.LIFNR",
    ("EKKO", "LFA1"): "EKKO.LIFNR = LFA1.LIFNR",
    ("LFA1", "EKKO"): "LFA1.LIFNR = EKKO.LIFNR",
    ("LFA1", "ADR6"): "LFA1.ADRNR = ADR6.ADRNR",
    ("ADR6", "LFA1"): "ADR6.ADRNR = LFA1.ADRNR",
    # Purchasing
    ("EKKO", "EKPO"): "EKKO.EBELN = EKPO.EBELN",
    ("EKPO", "EKKO"): "EKPO.EBELN = EKKO.EBELN",
    ("EKPO", "EKET"): "EKPO.EBELN = EKET.EBELN AND EKPO.EBELP = EKET.EBELP",
    ("EKET", "EKPO"): "EKET.EBELN = EKPO.EBELN AND EKET.EBELP = EKPO.EBELP",
    ("EKPO", "EKKN"): "EKPO.EBELN = EKKN.EBELN AND EKPO.EBELP = EKKN.EBELP",
    ("EKKN", "EKPO"): "EKKN.EBELN = EKPO.EBELN AND EKKN.EBELP = EKPO.EBELP",
    ("EKPO", "MARA"): "EKPO.MATNR = MARA.MATNR",
    ("MARA", "EKPO"): "MARA.MATNR = EKPO.MATNR",
    ("EKBE", "EKPO"): "EKBE.EBELN = EKPO.EBELN AND EKBE.EBELP = EKPO.EBELP",
    ("EKPO", "EKBE"): "EKPO.EBELN = EKBE.EBELN AND EKPO.EBELP = EKBE.EBELP",
    ("EBAN", "EBKN"): "EBAN.BANFN = EBKN.BANFN AND EBAN.BNFPO = EBKN.BNFPO",
    ("EBKN", "EBAN"): "EBKN.BANFN = EBAN.BANFN AND EBKN.BNFPO = EBAN.BNFPO",
    ("EBAN", "MARA"): "EBAN.MATNR = MARA.MATNR",
    ("MARA", "EBAN"): "MARA.MATNR = EBAN.MATNR",
    # Inventory Management
    ("MKPF", "MSEG"): "MKPF.MBLNR = MSEG.MBLNR AND MKPF.MJAHR = MSEG.MJAHR",
    ("MSEG", "MKPF"): "MSEG.MBLNR = MKPF.MBLNR AND MSEG.MJAHR = MKPF.MJAHR",
    ("MSEG", "MARA"): "MSEG.MATNR = MARA.MATNR",
    ("MARA", "MSEG"): "MARA.MATNR = MSEG.MATNR",
    ("MCHB", "MARA"): "MCHB.MATNR = MARA.MATNR",
    ("MARA", "MCHB"): "MARA.MATNR = MCHB.MATNR",
    ("MCHB", "MARC"): "MCHB.MATNR = MARC.MATNR AND MCHB.WERKS = MARC.WERKS",
    ("MARC", "MCHB"): "MARC.MATNR = MCHB.MATNR AND MARC.WERKS = MCHB.WERKS",
    ("MSLB", "LFA1"): "MSLB.LIFNR = LFA1.LIFNR",
    ("LFA1", "MSLB"): "LFA1.LIFNR = MSLB.LIFNR",
    ("MSLB", "MARA"): "MSLB.MATNR = MARA.MATNR",
    ("MARA", "MSLB"): "MARA.MATNR = MSLB.MATNR",
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

    # For master-data-only queries (MARA, MAKT, MEAN, MARM, MVKE), use DISTINCT to avoid repeated material numbers from 1:N joins (e.g. MAKT per language, MARM per UoM, MEAN per EAN, MVKE per sales org)
    master_data_only = {"MARA", "MAKT", "MEAN", "MARM", "MVKE"}
    use_distinct = all_tables.issubset(master_data_only) and len(all_tables) > 1
    sql_lines = [f"SELECT {'DISTINCT ' if use_distinct else ''}TOP {limit}"]
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
        # If no join is ready, add a "dependency" first: a table that is referenced in ON by others (e.g. KEKO before KEPH)
        if chosen is None:
            needed = set()
            for j in pending:
                left, right = j.get("left", base), j["right"]
                on_expr = PREFERRED_JOIN_ONS.get((left, right)) or PREFERRED_JOIN_ONS.get((right, left)) or normalize_join_on(j.get("on", ""), left, right)
                refs = tables_referenced_in_on(on_expr)
                needed |= (refs - {right} - added)
            for j in pending:
                left, right = j.get("left", base), j["right"]
                if right not in needed or left not in added:
                    continue
                on_expr = PREFERRED_JOIN_ONS.get((left, right)) or PREFERRED_JOIN_ONS.get((right, left)) or normalize_join_on(j.get("on", ""), left, right)
                refs = tables_referenced_in_on(on_expr)
                if (refs - {right}).issubset(added):
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

    # Add any joins that weren't processed, in dependency order (e.g. KEKO before KEPH when ON references k.KALNR)
    remaining = [j for j in joins if j["right"] not in added]
    while remaining:
        made_progress = False
        for j in remaining[:]:
            right = j["right"]
            left = j.get("left", base)
            on_expr = PREFERRED_JOIN_ONS.get((left, right)) or PREFERRED_JOIN_ONS.get((right, left))
            if not on_expr:
                on_expr = normalize_join_on(j.get("on", ""), left, right)
            refs = tables_referenced_in_on(on_expr)
            if refs and not (refs - {right}).issubset(added):
                continue  # wait until dependencies are added
            if left not in added:
                continue
            join_type = (j.get("type") or "inner").strip().lower()
            join_keyword = "LEFT JOIN" if join_type == "left" else "JOIN"
            sql_lines.append(f"\n{join_keyword} {format_table_name(right)} AS {alias_map[right]}")
            sql_lines.append(f"    ON {replace_table_names_with_aliases(on_expr, alias_map)}")
            added.add(right)
            remaining.remove(j)
            made_progress = True
        if not made_progress and remaining:
            # force-add first remaining (may produce invalid SQL if ON refs missing table)
            j = remaining[0]
            left, right = j.get("left", base), j["right"]
            on_expr = PREFERRED_JOIN_ONS.get((left, right)) or PREFERRED_JOIN_ONS.get((right, left)) or normalize_join_on(j.get("on", ""), left, right)
            join_type = (j.get("type") or "inner").strip().lower()
            join_keyword = "LEFT JOIN" if join_type == "left" else "JOIN"
            sql_lines.append(f"\n{join_keyword} {format_table_name(right)} AS {alias_map[right]}")
            sql_lines.append(f"    ON {replace_table_names_with_aliases(on_expr, alias_map)}")
            added.add(right)
            remaining.pop(0)

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
        # IN operator: rhs must be a list of values, e.g. ['FERT','HALB','ROH'] or string "('FERT', 'HALB', 'ROH')" — generate IN (N'v1', N'v2', ...)
        if op_upper == "IN":
            in_values = []
            if isinstance(rhs, (list, tuple)):
                in_values = [str(v).strip().strip("'\"").strip() for v in rhs if v is not None and str(v).strip()]
            else:
                raw = str(rhs).strip()
                if raw.startswith("(") and raw.endswith(")"):
                    inner = raw[1:-1].strip()
                    for part in re.split(r"[,;]", inner):
                        v = part.strip().strip("'\"").strip()
                        if v:
                            in_values.append(v)
            if in_values:
                in_escaped = [v.replace("'", "''") for v in in_values]
                in_list = ", ".join([f"N'{e}'" for e in in_escaped])
                conds.append(f"({lhs} IN ({in_list}))")
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
            # Material name/description (MAKTX): case-insensitive exact match OR contains (e.g. "Lower shaft winding" matches "Motor - Lower shaft winding")
            lhs_upper = lhs.upper()
            if op_upper == "=" and isinstance(rhs, str) and rhs_sql and (
                "MAKTX" in lhs_upper or "MATERIAL_DESCRIPTION" in lhs_upper
            ):
                rhs_str = str(rhs).strip().strip("'\"").strip()
                like_escaped = rhs_str.replace("'", "''")
                like_pattern = f"N'%{like_escaped}%'"
                conds.append(f"(LOWER({lhs}) = LOWER({rhs_sql}) OR {lhs} LIKE {like_pattern})")
            else:
                conds.append(f"{lhs} {op} {rhs_sql}")
    # Revenue is shown only for billing category types A,B,C,D,E,I,L,W (VBRK.FKTYP)
    selected_table_columns = {(col["table"], col["name"]) for col in columns}
    if "VBRK" in alias_map:
        has_netwr = ("VBRK", "NETWR") in selected_table_columns or ("VBRP", "NETWR") in selected_table_columns
        has_fktyp_filter = any("FKTYP" in str(f.get("lhs", "")).upper() for f in json_spec.get("filters", []))
        if has_netwr and not has_fktyp_filter:
            in_list = ", ".join(f"N'{c}'" for c in REVENUE_BILLING_CATEGORIES)
            vbrk_alias = alias_map["VBRK"]
            conds.append(f"{vbrk_alias}.[FKTYP] IN ({in_list})")
    if conds:
        sql_lines.append("\nWHERE " + " AND ".join(conds))

    # ORDER BY: only use columns that exist (selected table columns or output aliases)
    # Fallback: map revenue-like names to base column VBRK.NETWR when VBRK is in the query; prefer VBRP for line-level value
    ORDER_BY_FALLBACK = {
        "total_revenue": ("VBRK", "NETWR"), "revenue": ("VBRK", "NETWR"), "net_revenue": ("VBRK", "NETWR"),
        "net_value_of_billing_item_in_document_currency": ("VBRP", "NETWR"),
        "net_value_of_the_billing_item_in_document_currency": ("VBRK", "NETWR"),
    }
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
    # MAST has no VBELN; remove any JOINs that reference VBELN when query uses MAST (avoids Invalid column name 'VBELN')
    if "MAST" in all_tables:
        sql = _remove_joins_on_vbeln_when_mast(sql)
    with st.expander("🔧 Debug: Generated SQL", expanded=False):
        st.code(sql, language="sql")
    return sql

def _remove_joins_on_vbeln_when_mast(sql: str) -> str:
    """Remove JOINs that use VBELN when the query uses MAST (MAST has no VBELN). Replace removed aliases in the rest of the SQL with a remaining same-table alias so SELECT/WHERE stay valid."""
    if not sql or "VBELN" not in sql.upper():
        return sql
    if "MAST" not in sql.upper():
        return sql
    lines = sql.split("\n")
    out = []
    removed_join_aliases = []  # (physical_table, alias) for each removed JOIN
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"\s*JOIN\s+", line, re.IGNORECASE):
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if "ON" in next_line.upper() and "VBELN" in next_line.upper():
                m = re.search(r"\[erp\]\.\[?(\w+)\]?\s+AS\s+(\w+)", line, re.IGNORECASE)
                if m:
                    removed_join_aliases.append((m.group(1).upper(), m.group(2)))
                i += 2
                continue
        out.append(line)
        i += 1
    sql_after_removal = "\n".join(out)
    if not removed_join_aliases:
        return sql_after_removal
    # Build remaining (table, alias) from FROM/JOIN lines
    remaining_by_table = {}
    for ln in out:
        m = re.search(r"\[erp\]\.\[?(\w+)\]?\s+AS\s+(\w+)", ln, re.IGNORECASE)
        if m:
            tbl, alias = m.group(1).upper(), m.group(2)
            remaining_by_table.setdefault(tbl, []).append(alias)
    # Map each removed alias to a remaining same-table alias (use first available)
    alias_to_replacement = {}
    used_remaining = {}
    for tbl, alias in removed_join_aliases:
        remaining = remaining_by_table.get(tbl, [])
        if not remaining:
            continue
        idx = used_remaining.get(tbl, 0)
        if idx < len(remaining):
            alias_to_replacement[alias] = remaining[idx]
            used_remaining[tbl] = idx + 1
    for old_alias, new_alias in alias_to_replacement.items():
        sql_after_removal = re.sub(rf"\b{re.escape(old_alias)}\.", f"{new_alias}.", sql_after_removal)
    return sql_after_removal


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


def _inject_ekko_lfa1_when_l_referenced(sql: str) -> str:
    """When SELECT references l.LIFNR, l.NAME1, l.LAND1 (LFA1) but query has no LFA1 join, inject EKKO and LFA1. EKPO has no LIFNR; vendor comes from EKKO and LFA1."""
    if not sql:
        return sql
    sql_upper = sql.upper()
    # Referenced alias l with LFA1-like columns
    has_l_lifnr = re.search(r"\bl\.\s*\[?LIFNR\]?|\bl\]\.\s*\[?LIFNR\]?", sql, re.IGNORECASE)
    has_l_name1 = re.search(r"\bl\.\s*\[?NAME1\]?|\bl\]\.\s*\[?NAME1\]?", sql, re.IGNORECASE)
    has_l_land1 = re.search(r"\bl\.\s*\[?LAND1\]?|\bl\]\.\s*\[?LAND1\]?", sql, re.IGNORECASE)
    if not (has_l_lifnr or has_l_name1 or has_l_land1):
        return sql
    if "LFA1" in sql_upper or "LFA1" in sql:
        return sql
    if "EKPO" not in sql_upper and "EKPO" not in sql:
        return sql
    # Resolve EKPO alias (e in typical generated SQL)
    ekpo_alias = "e"
    m = re.search(r"\[?EKPO\]?\s+AS\s+(\w+)", sql, re.IGNORECASE)
    if m:
        ekpo_alias = m.group(1)
    inject = f" JOIN [erp].[EKKO] AS k ON {ekpo_alias}.EBELN = k.EBELN JOIN [erp].[LFA1] AS l ON k.LIFNR = l.LIFNR"
    # Insert before WHERE or ORDER BY or at end
    if re.search(r"\s+WHERE\s+", sql, re.IGNORECASE):
        sql = re.sub(r"(\s+)(WHERE\s+)", r"\1" + inject + r" \2", sql, count=1, flags=re.IGNORECASE)
    elif re.search(r"\s+ORDER\s+BY\s+", sql, re.IGNORECASE):
        sql = re.sub(r"(\s+)(ORDER\s+BY\s+)", r"\1" + inject + r" \2", sql, count=1, flags=re.IGNORECASE)
    else:
        sql = sql.rstrip() + inject + "\n"
    return sql


def _fix_ekpo_ekorg_join(sql: str) -> str:
    """EKPO has no EKORG; EKORG is on EKKO. Rewrite JOIN EKPO ON ... EKORG = e.EKORG to use EKKO."""
    if not sql or "EKPO" not in sql.upper() or "EKORG" not in sql.upper():
        return sql
    # JOIN [erp].[EKPO] AS e ON l2.EKORG = e.EKORG -> JOIN EKKO AS k ON l2.EKORG = k.EKORG JOIN EKPO AS e ON k.EBELN = e.EBELN
    m = re.search(
        r"JOIN\s+\[erp\]\.\[EKPO\]\s+AS\s+(\w+)\s+ON\s+(\w+)\.\[?EKORG\]?\s*=\s*\1\.\[?EKORG\]?",
        sql,
        re.IGNORECASE,
    )
    if m:
        ekpo_alias, other_alias = m.group(1), m.group(2)
        old_join = m.group(0)
        new_join = (
            f"JOIN [erp].[EKKO] AS k ON {other_alias}.[EKORG] = k.[EKORG] "
            f"JOIN [erp].[EKPO] AS {ekpo_alias} ON k.EBELN = {ekpo_alias}.EBELN"
        )
        sql = sql.replace(old_join, new_join, 1)
        return sql
    # Reverse: ON e.EKORG = l2.EKORG
    m = re.search(
        r"JOIN\s+\[erp\]\.\[EKPO\]\s+AS\s+(\w+)\s+ON\s+\1\.\[?EKORG\]?\s*=\s*(\w+)\.\[?EKORG\]?",
        sql,
        re.IGNORECASE,
    )
    if m:
        ekpo_alias, other_alias = m.group(1), m.group(2)
        old_join = m.group(0)
        new_join = (
            f"JOIN [erp].[EKKO] AS k ON k.[EKORG] = {other_alias}.[EKORG] "
            f"JOIN [erp].[EKPO] AS {ekpo_alias} ON k.EBELN = {ekpo_alias}.EBELN"
        )
        sql = sql.replace(old_join, new_join, 1)
    return sql


def _fix_ekpo_waers_in_sql(sql: str) -> str:
    """Replace EKPO.WAERS with EKKO.WAERS (WAERS exists on EKKO, not EKPO, in SQL Server)."""
    if not sql or "WAERS" not in sql.upper() or "EKPO" not in sql.upper() or "EKKO" not in sql.upper():
        return sql
    # Find EKPO alias: FROM [erp].[EKPO] AS xxx or JOIN [erp].[EKPO] AS xxx
    ekpo_alias = None
    m = re.search(r"\[?EKPO\]?\s+AS\s+(\w+)", sql, re.IGNORECASE)
    if m:
        ekpo_alias = m.group(1)
    # Find EKKO alias
    ekko_alias = None
    m = re.search(r"\[?EKKO\]?\s+AS\s+(\w+)", sql, re.IGNORECASE)
    if m:
        ekko_alias = m.group(1)
    if not ekpo_alias or not ekko_alias or ekpo_alias == ekko_alias:
        return sql
    # Replace ekpo_alias.[WAERS] or ekpo_alias.WAERS with ekko_alias.[WAERS]
    sql = re.sub(rf"\b{re.escape(ekpo_alias)}\.[\[]?WAERS[\]]?", f"{ekko_alias}.[WAERS]", sql, flags=re.IGNORECASE)
    return sql


def _strip_aufk_gstrp_gltrp_in_sql(sql: str) -> str:
    """Remove GSTRP and GLTRP from SELECT (AUFK may not have these columns in some systems)."""
    if not sql or ("GSTRP" not in sql.upper() and "GLTRP" not in sql.upper()):
        return sql
    # Remove ", alias.[GSTRP] AS [Start_date]" or ", alias.[GLTRP] AS [End_date]" (and similar) from SELECT
    for col in ("GSTRP", "GLTRP"):
        sql = re.sub(r",\s*[\w]+\.\[?" + col + r"\]?\s+AS\s+\[[^\]]*\]", "", sql, flags=re.IGNORECASE)
    return sql


def _remove_aufk_join_on_matnr(sql: str) -> str:
    """AUFK has no MATNR column; remove JOIN AUFK ON ... MATNR and SELECT columns from that alias."""
    if not sql or "AUFK" not in sql.upper():
        return sql
    # Find JOIN/LEFT JOIN to AUFK with ON ... alias.MATNR (invalid: AUFK has no MATNR)
    m = re.search(
        r"((?:LEFT\s+)?JOIN\s+\[?erp\]?\.\[?AUFK\]?\s+AS\s+(\w+)\s+ON\s+[^\n]+\.MATNR\s*=\s*\2\.MATNR)\s*",
        sql,
        re.IGNORECASE,
    )
    if not m:
        return sql
    alias = m.group(2)
    # Remove the invalid JOIN line
    sql = sql.replace(m.group(1), "", 1)
    # Remove from SELECT any column from this alias (e.g. ", a.[AUFNR] AS [order_number]")
    sql = re.sub(r",\s*" + re.escape(alias) + r"\.\[?[\w]+\]?\s+AS\s+\[[^\]]*\]", "", sql, flags=re.IGNORECASE)
    # Remove trailing comma before FROM if we removed the last column
    sql = re.sub(r",\s+FROM\b", " FROM", sql, flags=re.IGNORECASE)
    return sql


def is_procurement_only_query(user_query: str) -> bool:
    """
    True when the user is asking only which products are procured internally vs externally.
    Used to restrict analysis to procurement-type only (no COGS, sales-by-type, or generic SAP blurb).
    """
    q = (user_query or "").strip().lower()
    has_procurement = (
        "procured internally" in q
        or "procured externally" in q
        or "internally and externally" in q
        or "internal and external" in q
        or "procurement type" in q
        or "which are internal" in q
        or "which are external" in q
        or "which products are procured" in q
    )
    # Optional list reference (from list below / which products listed)
    list_ref = (
        "from the list below" in q
        or "from the list above" in q
        or "from these products" in q
        or "which of these" in q
        or "which products listed" in q
        or "from the previous" in q
        or "from the prior" in q
    )
    return bool(has_procurement and (list_ref or "which products" in q or "which are" in q))


def is_from_list_below_procurement_query(user_query: str) -> bool:
    """
    Detect if the user is asking which products from the prior result list
    are procured internally vs externally (from the list below/above).
    """
    q = (user_query or "").strip().lower()
    list_ref = (
        "from the list below" in q
        or "from the list above" in q
        or "from these products" in q
        or "which of these" in q
        or "which products listed" in q
        or "from the previous" in q
        or "from the prior" in q
    )
    procurement = (
        "procured internally" in q
        or "procured externally" in q
        or "internally and externally" in q
        or "internal and external" in q
        or "procurement type" in q
        or "which are internal" in q
        or "which are external" in q
        or "which products are procured" in q
    )
    return bool(list_ref and procurement)


def get_material_numbers_from_dataframe(df: pd.DataFrame) -> list:
    """
    Extract unique material numbers from a result dataframe.
    Looks for columns: material_number, MATNR, main_material_number,
    component_material, component_material_number, etc.
    """
    if df is None or df.empty:
        return []
    cols = [c for c in df.columns if c]
    # Prefer exact matches, then case-insensitive
    mat_cols = []
    for c in cols:
        lower = c.lower()
        if lower in (
            "material_number",
            "matnr",
            "main_material_number",
            "component_material",
            "component_material_number",
            "idnrk",
        ):
            mat_cols.append(c)
    if not mat_cols:
        for c in cols:
            if "material" in c.lower() and "number" in c.lower():
                mat_cols.append(c)
                break
            if c.upper() == "MATNR":
                mat_cols.append(c)
                break
    out = []
    for c in mat_cols:
        try:
            vals = df[c].dropna().astype(str).str.strip()
            out.extend(vals[vals != ""].tolist())
        except Exception:
            continue
    return list(dict.fromkeys(out))  # unique, preserve order


def query_procurement_type_for_materials(matnr_list: list) -> tuple:
    """
    Run a targeted query: MARA + MAKT + MARC for the given material numbers
    to return material_number, material_description, procurement_type (BESKZ), plant.
    BESKZ: E = In-house, F = External, X = Both.
    Returns (dataframe, actual_sql_used).
    """
    if not matnr_list:
        return pd.DataFrame(), ""
    # Limit to avoid huge IN clause (e.g. 500)
    matnr_list = list(matnr_list)[:500]
    # SQL Server: use N'...' for NVARCHAR; escape single quotes in values
    safe = []
    for m in matnr_list:
        s = str(m).strip()
        if not s:
            continue
        s = s.replace("'", "''")
        safe.append(f"N'{s}'")
    if not safe:
        return pd.DataFrame(), ""
    in_clause = ", ".join(safe)
    sql = f"""
    SELECT TOP 1000
        m1.[MATNR] AS [material_number],
        m.[MAKTX] AS [material_description],
        m2.[BESKZ] AS [procurement_type],
        m2.[WERKS] AS [plant]
    FROM [erp].[MARA] AS m1
    JOIN [erp].[MAKT] AS m ON m1.MATNR = m.MATNR AND m.SPRAS = N'E'
    JOIN [erp].[MARC] AS m2 ON m1.MATNR = m2.MATNR
    WHERE m1.MATNR IN ({in_clause})
    ORDER BY m.[MAKTX], m1.[MATNR]
    """
    df = run_sql(sql)
    return df, sql.strip()


def _inject_distinct_for_supplier_parts_sql(sql: str) -> str:
    """When query uses EKPO and LFA1 (supplier-of-parts style) but has no DISTINCT, add DISTINCT to reduce repeated vendor rows."""
    if not sql or "DISTINCT" in sql.upper():
        return sql
    sql_upper = sql.upper()
    if "EKPO" not in sql_upper or "LFA1" not in sql_upper:
        return sql
    # SELECT TOP N -> SELECT DISTINCT TOP N
    if re.search(r"\bSELECT\s+TOP\s+", sql, re.IGNORECASE):
        sql = re.sub(r"(\bSELECT)\s+(TOP\s+)", r"\1 DISTINCT \2", sql, count=1, flags=re.IGNORECASE)
    return sql


def run_sql(sql: str) -> pd.DataFrame:
    sql = _inject_ekko_lfa1_when_l_referenced(sql)
    sql = _fix_ekpo_ekorg_join(sql)
    sql = _inject_distinct_for_supplier_parts_sql(sql)
    sql = _fix_ekpo_waers_in_sql(sql)
    sql = _strip_aufk_gstrp_gltrp_in_sql(sql)
    sql = _remove_aufk_join_on_matnr(sql)
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
INSIGHTS_PROMPT_TEMPLATE = """You are an expert analyst. Your analysis is **required** to add **plausibility and official recognition** to the raw query result. The query result alone is not sufficient — your interpretation, validation, and reasoning are what make the insights plausible and hold recognition. Use **all available data sources** to enrich and validate the analysis.

**Data sources available:**
1. **Query result data** (below) — from the database query the user ran.
2. **SQL context** (if provided) — the query that produced the data; use it to interpret what the columns and filters mean.
3. **Uploaded PDF(s)** (if provided) — use to add market, competitive, or external context.

**Your role:** Provide additional intelligence and plausibility so the combined output (query result + your analysis) can be relied upon. State conclusions clearly, flag uncertainties, and ground recommendations in the data so the insights hold official recognition.

**For market and competitive analysis:** Combine (1) the internal query result, (2) any uploaded PDFs (reports, market data), and (3) your reasoning. Compare results across the sources when relevant. Do not rely on the query result alone — use every source provided and state what comes from internal data vs uploaded context.

**When the user asks for comparative market, web search on similar products, or market insights:** Use the **product names** (and material numbers) from the query result data below. (1) Provide a **comparative market view**: how do similar products or product categories perform in the market; trends for that year or segment. (2) **If your model supports web search** (e.g. Perplexity): use it to look up similar products, market data, or benchmarks and summarize findings; then combine that with the internal data for a single set of insights. (3) Clearly separate **internal data** (from the query result) vs **market/web context** (from search or PDFs) so the user sees both. Give 2–4 concrete insights and 1–2 recommendations.

**When the user asks "how can we compete against products like [X]" (e.g. Harley jackets):** (1) **Restrict your analysis to the product discussed** — the query result is already filtered to that product; base all calculations, metrics, and conclusions on this data only; do not introduce unrelated or off-subject products. (2) **Include calculations** — summarize key metrics from the result (e.g. total/avg value, volumes, by material or customer if present) so the answer is grounded in numbers. (3) **Similar competitive products from the same industry** — use the industry (BRSCH) from the result and your knowledge or web search to name **similar competitive products from the same industry** and their relevance (e.g. same segment, same customer need); keep this focused and aligned with the product (e.g. for Harley jackets: same-industry apparel/outerwear competitors). (4) **Avoid vague or off-subject data** — do not list random products or industries; every point should relate to the product the user asked about and how to compete against products like it.

Provide concise insights from the perspective of:
- Industry trends (call out leading and lagging sectors; tie to any bar/pie charts when the user asks for industry trends)
- Customer revenues and behavior
- Product and industry analysis
- Market and competitive context when relevant (using all provided sources)
- Other vital business metrics

When the user asks for industry trends: summarize key patterns visible in the data (and in the charts/graphs), highlight top and bottom industries by value or share, and give 1–2 actionable recommendations.

When the data or question involves production orders, products manufactured internally, procurement type, or internal vs external: (1) In SAP the **primary field** is **MARC.BESKZ (Procurement Type**, MRP 2 view): E = In-house produced, F = Externally procured, X = Both; MARC.SOBSL = Special Procurement. (2) **MBEW** (Accounting 1/Costing 1): VPRSV = price control (S = Standard for manufactured, V = Moving average for procured), STPRS, VERPR, LOSGR = costing lot size. (3) Production order data (AUFK) and make-to-order/make-to-stock (KDAUF link) and BOM/purchase order dependencies support the picture; use order_number to trace linked products and components.

User question: {user_query}
{sql_context_section}
Data summary (columns and sample rows, max 50 rows):
{data_summary}
"""
SQL_CONTEXT_SECTION = """
**SQL that produced this data (for context):**
```
{sql_query}
```
"""
EXTRA_CONTEXT_TEMPLATE = """

**Additional context from uploaded PDF(s):**
---
{pdf_text}
---
Use this context to enrich your analysis. For market and competitive analysis, synthesize the query result with this PDF context and compare or combine the insights.
"""

def get_insights_from_provider(provider: str, user_query: str, df: pd.DataFrame, pdf_text: str = "", sql_query: str = "") -> str:
    """Get analysis/insights from the selected provider. Uses query result, optional SQL context, and optional PDF(s) to enrich insights (e.g. market/competitive analysis)."""
    if df is None or df.empty:
        return "No data available to analyze."
    data_summary = f"Columns: {list(df.columns)}\n\nSample:\n{df.head(50).to_string()}"
    sql_context_section = ""
    if sql_query and sql_query.strip() and "(procurement type for materials" not in (sql_query or ""):
        sql_context_section = SQL_CONTEXT_SECTION.format(sql_query=sql_query.strip()[:4000])
    prompt = INSIGHTS_PROMPT_TEMPLATE.format(user_query=user_query, data_summary=data_summary, sql_context_section=sql_context_section)
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

def get_insights_from_all_providers(user_query: str, df: pd.DataFrame, pdf_text: str = "", sql_query: str = "") -> list:
    """Run all configured AI providers (ChatGPT, Claude, Gemini, Perplexity) and return [(provider_display_name, text), ...]. This step is required to add plausibility and official recognition to the query result — raw data alone is not sufficient. Pass sql_query and pdf_text so insights use all sources (query result + SQL context + PDFs)."""
    providers = _configured_insight_providers()
    if not providers:
        return [("No provider", "No API keys configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_GEMINI_API_KEY, or PERPLEXITY_API_KEY.")]
    results = []
    for p in providers:
        text = get_insights_from_provider(p, user_query, df, pdf_text=pdf_text, sql_query=sql_query)
        if text and not text.startswith("Error from") and "not configured" not in text.lower():
            results.append((PROVIDER_DISPLAY_NAMES.get(p, p), text))
    if not results:
        return [("No result", "No provider returned a valid analysis. Check API keys and try again.")]
    return results

JUDGE_PROMPT = """You are a judge. Rank the following analyses by quality of insights (industry trends, customer revenue, product/industry analysis, market/competitive use of data sources, clarity, actionability). When the user asked for market or competitive analysis, prefer analyses that use and compare multiple data sources (query result + any PDF/context). Reply with a single line: the exact provider names in order, best first, comma-separated. Example: ChatGPT (OpenAI), Claude (Anthropic), Gemini (Google)

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
    # Check if dataset has product/material ID but no product description (limits product analysis)
    col_str = " ".join(str(c).upper() for c in cols)
    has_material_id = "MATNR" in col_str or "MATERIAL_NUMBER" in col_str or "material_number" in col_str
    has_product_desc = "MAKTX" in col_str or "MATERIAL_DESCRIPTION" in col_str or "description" in col_str.lower()
    product_data_limited = has_material_id and not has_product_desc
    procurement_only = is_procurement_only_query(user_query or "")
    has_procurement_col = "PROCUREMENT_TYPE" in col_str or "BESKZ" in col_str
    procurement_only_instruction = ""
    if procurement_only and has_procurement_col:
        procurement_only_instruction = """
**IMPORTANT — Procurement-only query:** The user is asking ONLY which products are procured internally vs externally. You MUST:
- Suggest ONLY visualizations by procurement_type (e.g. bar chart: x = procurement_type, y = material_number or material_description, agg = count; or pie: labels = procurement_type, values = count). Use column names from the dataset (e.g. procurement_type, material_description).
- Do NOT add data_notes about COGS, cost of goods, sales by material type, variant products, or costing/valuation. At most one short data_note: "Procurement Type (MARC.BESKZ): E = In-house, F = External, X = Both."
- Do NOT add calculations unrelated to procurement (no sum of value, no material type percentages). If any calculation, only count by procurement_type.
- Keep "calculations" minimal or empty; keep "data_notes" to one short BESKZ note only.
"""
    sample_prompt = f"""
User query: "{user_query}"

Available columns in the dataset (use these exact names in your response):
{json.dumps(cols, indent=2)}
{procurement_only_instruction}

Task:
- Based on the user query and available data, suggest calculations or aggregations to perform.
- Suggest charts/graphs/diagrams that match the query. Prefer multiple visualizations (bar, pie, line) when useful.
- **Always use product names/descriptions for product data:** When the dataset has both a material ID column (MATNR, material_number, IDNRK) and a product name/description column (MAKTX, material_description, or any column name containing "description"), always use the product name/description column (e.g. MAKTX) for product-related visualizations and calculations—use it for x-axis, labels, and group-by so that charts and tables show product names, not material numbers.
- **Always use customer names for customer data:** When the dataset has both customer number (KUNNR, customer_number) and customer name (NAME1, customer_name, or any column containing "customer" and "name"), always use the customer name column for customer-related visualizations and calculations—use it for x-axis, labels, and group-by so that charts, graphs, and AI analysis show customer names, not customer numbers.
- Industry trends: If the user asks for industry trends, or if the dataset has an industry-related column (e.g. BRSCH, industry_key, or any name containing "industry"), always suggest at least: (1) a bar chart of value/revenue by industry (x = industry column, y = value column, agg = sum), (2) a pie chart of share by industry (same columns), and optionally (3) a line chart over time by industry if a date column exists. Use only the exact column names listed above.
- Product analysis: If the dataset has MAKTX or material_description, use that column (not MATNR) for product charts and group-bys. If the dataset has material/product columns but no product description column, set "data_notes" to a one-sentence note that the lack of product descriptions limits analysis and that MAKT could enrich it. Variant products: When the same product name appears for different material numbers, these are variant products (e.g. different colour, shape, or size, each with its own material number). If the dataset has material type (MTART), material group (MATKL), base unit (MEINS), plant (WERKS), or variant attributes (colour, shape, size, or classification/characteristic columns), add a "data_notes" item with this exact wording: "The section **Variant products: what differentiates them?** lists each material number and the attributes that distinguish it (material type, material group, base unit, plant, and any variant attributes such as colour, shape, or size)."
- Standard price / cost data: If the dataset has standard_price (STPRS) or moving_average_price (VERPR), add a data_notes item: for meaningful analysis exclude or flag rows where standard_price is zero or null (unreleased or missing prices). Always add a note that price_unit (PEINH) defines the quantity per which the price applies. When charting or summing standard price, filter to rows with standard_price > 0 and standard_price IS NOT NULL so results are meaningful.
- Sales by material type / percentage of finished goods: If the dataset has material_type (MTART) and a sales/value column (e.g. NETWR, Net_Value, or column name containing value/revenue): add calculations — (1) Percentage of finished goods = (Total sales value where material_type = 'FERT') / (Total sales value of all goods) * 100; (2) Percentage of raw materials and semi-finished goods = (Total sales value where material_type IN ('ROH','HALB')) / (Total sales value of all goods) * 100. Suggest a pie chart of sales value by material_type (labels = material_type or mapped labels: FERT = Finished goods, HALB = Semi-finished, ROH = Raw materials), and a bar chart of total value by material type. If the dataset has material_type (MTART) but NO sales/value column (no NETWR, Net_Value, or value/revenue column): add exactly one data_note: "The lack of a sales/value column limits analysis of sales by material type. Ask **Sales value by material type** or **Revenue by material type** to include billing data (VBRP.NETWR) and get percentages and charts." Do not add other data_notes about sales by material type in that case.
- **Production orders / manufactured internally / procurement type / dependencies:** If the dataset has **procurement_type (MARC.BESKZ)** or columns indicating production orders (order_number, order_type, AUFNR, AUART, sales_order_number, KDAUF, object_number, OBJNR) or BOM/component columns: add a data_note: "**Procurement Type (MARC.BESKZ)** is the primary SAP field for internal vs external: **E** = In-house produced, **F** = Externally procured, **X** = Both. MBEW (VPRSV, STPRS, VERPR, LOSGR) handles costing and valuation. Production order data (AUFK) and make-to-order/make-to-stock (KDAUF) and BOM/purchase orders support the picture." Suggest visualizations by procurement_type or order_type when present.
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
  ],
  "data_notes": ["<optional note when product data is limited or other data caveat>"]
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
    """Remove ' for ...', ' grouped by ...', ' group by ...', ' where ...' from calc column text."""
    for trailer in (" grouped by ", " group by ", " for ", " where "):
        if trailer in text:
            text = text.split(trailer)[0].strip()
    return text


def _format_numeric(value) -> str:
    """Format a value for display; coerce to float if possible to avoid 'Unknown format code f' for str."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        n = float(value)
        return f"{n:,.2f}"
    except (TypeError, ValueError):
        return str(value)


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

def _get_procurement_type_column(df_columns: list) -> str:
    """Return the procurement type (BESKZ) column name if present, else empty string."""
    for c in (df_columns or []):
        if not c:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("BESKZ", "PROCUREMENT_TYPE") or "PROCUREMENT_TYPE" in cu:
            return c
    return ""


def apply_procurement_type_display(df: pd.DataFrame) -> pd.DataFrame:
    """If df has a procurement_type (BESKZ) column, return a copy with values shown as 'E (In-house produced)', etc. Otherwise return df unchanged."""
    if df is None or df.empty:
        return df
    col = _get_procurement_type_column(list(df.columns))
    if not col or col not in df.columns:
        return df
    out = df.copy()

    def _label(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        s = str(val).strip().upper()
        return PROCUREMENT_TYPE_DISPLAY_LABELS.get(s, s)

    out[col] = out[col].apply(_label)
    return out


def apply_industry_display(df: pd.DataFrame) -> pd.DataFrame:
    """If df has an industry column (BRSCH, industry_key, etc.), return a copy with codes replaced by full descriptions (e.g. M → Mechanical engineering). Otherwise return df unchanged."""
    if df is None or df.empty:
        return df
    industry_col = _find_industry_column(list(df.columns))
    if not industry_col or industry_col not in df.columns:
        return df
    out = df.copy()

    def _label(code):
        s = (code or "").strip()
        return INDUSTRY_SECTOR_LABELS.get(s, INDUSTRY_SECTOR_LABELS.get((s or "").upper(), s or "Not specified"))

    out[industry_col] = out[industry_col].astype(str).map(_label)
    return out


def _is_industry_column(col: str, df_cols: list) -> bool:
    """True if col is an industry-like dimension (for tracking if we already showed industry chart)."""
    if not col:
        return False
    c = col.upper()
    if c in ("BRSCH", "MBRSH", "INDUSTRY_KEY", "INDUSTRY") or "INDUSTRY" in c:
        return True
    return False


def _map_industry_to_display(df: pd.DataFrame, industry_col: str):
    """Map industry sector codes (e.g. M, C) to readable labels. Returns (df_with_display_col, display_col_name, source_text)."""
    display_col = "Industry_sector"
    source_text = INDUSTRY_SECTOR_SOURCE.get(industry_col.upper(), industry_col)
    if industry_col not in df.columns:
        return df, industry_col, source_text
    def _label(code):
        s = (code or "").strip()
        return INDUSTRY_SECTOR_LABELS.get(s, INDUSTRY_SECTOR_LABELS.get(s.upper(), s or "Not specified"))

    mapped = df[industry_col].astype(str).map(_label)
    out = df.copy()
    out[display_col] = mapped
    return out, display_col, source_text


def _find_industry_column(df_cols: list) -> str:
    """Return first column that looks like industry (BRSCH, industry_key, or name containing industry)."""
    for c in df_cols:
        if c and (_is_industry_column(c, df_cols)):
            return c
    return ""


def _is_product_material_column(col: str, df_cols: list) -> bool:
    """True if col is a material/product ID column (MATNR, material_number, IDNRK). Prefer showing MAKTX instead."""
    if not col:
        return False
    c = col.upper()
    if c in ("MATNR", "MATERIAL_NUMBER", "IDNRK") or "MATERIAL_NUMBER" in c:
        return True
    return False


def _get_product_display_column(df_cols: list) -> str:
    """Return the product name/description column (MAKTX or material_description) if present, else empty. Use for display instead of MATNR."""
    for c in df_cols:
        if not c:
            continue
        cu = c.upper()
        if cu == "MAKTX" or "MATERIAL_DESCRIPTION" in cu or (cu != "MATNR" and "DESCRIPTION" in cu and "MATERIAL" in cu):
            return c
    return ""


def _is_customer_number_column(col: str, df_cols: list) -> bool:
    """True if col is customer number (KUNNR, customer_number). Prefer showing customer name instead."""
    if not col:
        return False
    cu = (col or "").upper().replace(" ", "_")
    return cu in ("KUNNR", "CUSTOMER_NUMBER") or "CUSTOMER_NUMBER" in cu


def _get_customer_name_column(df_cols: list) -> str:
    """Return the customer name column (NAME1, customer_name) if present, else empty. Use for display instead of KUNNR."""
    for c in df_cols:
        if not c:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("NAME1", "CUSTOMER_NAME") or (cu != "KUNNR" and "CUSTOMER" in cu and "NAME" in cu):
            return c
    return ""


def _get_material_number_column(df_cols: list) -> str:
    """Return the material number column (MATNR, material_number, IDNRK) if present, else empty."""
    for c in df_cols:
        if not c:
            continue
        cu = (c or "").upper()
        if cu in ("MATNR", "MATERIAL_NUMBER", "IDNRK") or "MATERIAL_NUMBER" in cu:
            return c
    return ""


def _get_material_description_column(df_cols: list) -> str:
    """Return the material description column (MAKTX, material_description) if present, else empty."""
    for c in df_cols:
        if not c:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("MAKTX", "MATERIAL_DESCRIPTION") or "MATERIAL_DESCRIPTION" in cu or "MAKTX" in cu:
            return c
    return ""


def _get_vendor_identifier_column(df_cols: list) -> str:
    """Return a column that identifies vendor (LIFNR, vendor_number, vendor_name, NAME1) if present, else empty. Prefer vendor number for deduplication."""
    for c in df_cols:
        if not c:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("LIFNR", "VENDOR_NUMBER") or cu == "VENDOR_NUMBER":
            return c
    for c in df_cols:
        if not c:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("NAME1", "VENDOR_NAME", "VENDOR_NAME_1") or "VENDOR_NAME" in cu:
            return c
    return ""


def _normalize_material_description(series: pd.Series) -> pd.Series:
    """Strip and collapse spaces so 'Harley  leather jacket' and 'Harley leather jacket' group together."""
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def _format_matnr_for_display(series: pd.Series) -> pd.Series:
    """Format SAP material numbers for display: strip leading zeros from all-numeric values (e.g. 000000000000000038 -> 38); leave alphanumeric as-is (e.g. H10500)."""
    def _one(val):
        s = str(val).strip()
        if not s:
            return s
        if s.isdigit():
            return s.lstrip("0") or "0"
        return s
    return series.astype(str).apply(_one)


# SAP / common column names that often differentiate variant products (same description, different material number)
# Includes: material master (type, group, UoM, plant) and variant attributes (colour, shape, size, classification)
VARIANT_DIFF_COLUMN_PATTERNS = (
    "MTART", "MATERIAL_TYPE", "MATKL", "MATERIAL_GROUP", "MEINS", "BASE_UOM", "BASE_UNIT",
    "WERKS", "PLANT", "MBRSH", "SPRAS", "LANGUAGE", "BWKEY", "VALUATION_AREA",
    "BWTAR", "VALUATION_TYPE", "MSTAE", "XCHPF", "LVORM",
    # Variant attributes: colour, shape, size (and SAP classification / characteristic names)
    "COLOR", "COLOUR", "FARBE", "SHAPE", "FORM", "SIZE", "GROESSE", "DIMENSION",
    "ATBEZ", "ATWRT", "ATINN", "CHAR", "CHARACTERISTIC", "ATTRIBUTE", "VARIANT",
    "MERKMAL", "VALUE_CHAR", "CHAR_VALUE", "ATNAM", "ATFOR"
)


def _get_variant_differentiator_columns(df: pd.DataFrame, df_cols: list) -> list:
    """Return column names that exist in df and can differentiate variant products (material type, group, UoM, plant; or attributes like colour, shape, size)."""
    out = []
    col_upper = {c: (c or "").upper().replace(" ", "_") for c in df_cols}
    for c in df_cols:
        if not c or c not in df.columns:
            continue
        cu = col_upper.get(c, (c or "").upper().replace(" ", "_"))
        for pat in VARIANT_DIFF_COLUMN_PATTERNS:
            if pat in cu or cu == pat:
                out.append(c)
                break
        # Also match column names that contain common variant attribute words (colour, shape, size)
        if c not in out and c in df.columns:
            for word in ("COLOR", "COLOUR", "SHAPE", "SIZE", "ATTRIBUTE", "CHARACTERISTIC", "VARIANT"):
                if word in cu:
                    out.append(c)
                    break
    return out


def _result_looks_like_bom_or_components(df_cols: list) -> bool:
    """True if the result set looks like BOM/component data (not a product-variant list). Variant section should be skipped then so it links to the main query."""
    col_str = " ".join((c or "").upper().replace(" ", "_") for c in df_cols)
    bom_component_indicators = (
        "IDNRK", "COMPONENT_MATERIAL", "BILL_OF_MATERIAL", "ITEM_NUMBER", "POSNR",
        "COMPONENT_QUANTITY", "COMPONENT_DESCRIPTION", "MAIN_MATERIAL", "MAIN_MATERIAL_DESCRIPTION",
        "BOM_NUMBER", "STLNR"
    )
    return any(ind in col_str for ind in bom_component_indicators)


def _show_variant_differentiation_table(df: pd.DataFrame, product_display_col: str, matnr_col: str, df_cols: list) -> bool:
    """If same product name appears for different material numbers (variant products), show a table of what differentiates them. Only shown when the data is product/variant list (not BOM/component) so it links to the main query. Returns True if shown."""
    if not product_display_col or not matnr_col or product_display_col not in df.columns or matnr_col not in df.columns:
        return False
    if _result_looks_like_bom_or_components(df_cols):
        return False
    df_n = df.copy()
    df_n[product_display_col] = _normalize_material_description(df_n[product_display_col])
    dup = df_n.groupby(product_display_col)[matnr_col].nunique()
    if (dup > 1).any():
        diff_cols = _get_variant_differentiator_columns(df_n, df_cols)
        display_cols = [matnr_col, product_display_col] + [c for c in diff_cols if c in df_n.columns]
        variant_rows = df_n.drop_duplicates(subset=[product_display_col, matnr_col], keep="first")[display_cols]
        st.write("### 🔀 Variant products: what differentiates them?")
        st.caption("The table below refers to the **products in your query result**: when the same product name has **different material numbers**, each row is a **variant** (e.g. different colour, shape, or size). It lists **material number**, **description**, and the attributes that distinguish each variant: **material type** (MTART), **material group** (MATKL), **base unit** (MEINS), **plant** (WERKS), and any variant attributes such as colour, shape, or size.")
        st.dataframe(variant_rows, use_container_width=True, hide_index=True)
        return True
    return False


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


def _deduplicate_material_price_rows(df: pd.DataFrame, df_cols: list) -> tuple[pd.DataFrame, bool]:
    """When result has material number + (standard/moving average price and/or material description), keep one row per material (last update).
    Removes repeated same material number with same value and collapses multiple language rows to one. Returns (deduplicated_df, was_deduped)."""
    if df.empty:
        return df, False
    matnr_col = _get_material_number_column(df_cols)
    if not matnr_col or matnr_col not in df.columns:
        return df, False
    std_col = None
    mov_col = None
    desc_col = None
    for c in df_cols:
        if not c or c not in df.columns:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("STPRS", "STANDARD_PRICE") or "STANDARD_PRICE" in (cu or ""):
            std_col = c
        if cu in ("VERPR", "MOVING_AVERAGE_PRICE") or "MOVING_AVERAGE" in (cu or ""):
            mov_col = c
        if cu in ("MAKTX", "MATERIAL_DESCRIPTION") or "DESCRIPTION" in (cu or "") or "MAKTX" in (cu or ""):
            desc_col = c
    if not std_col:
        for c in df_cols:
            if c and "standard" in (c or "").lower() and "price" in (c or "").lower():
                std_col = c
                break
    if not mov_col:
        for c in df_cols:
            if c and ("moving" in (c or "").lower() and "average" in (c or "").lower()) or (c or "").upper() == "VERPR":
                mov_col = c
                break
    if not desc_col:
        for c in df_cols:
            if c and ("material" in (c or "").lower() and "description" in (c or "").lower()) or (c or "").upper() == "MAKTX":
                desc_col = c
                break
    has_price = (std_col and std_col in df.columns) or (mov_col and mov_col in df.columns)
    has_description = desc_col and desc_col in df.columns
    if not has_price and not has_description:
        return df, False
    if not df[matnr_col].duplicated().any():
        return df, False
    out = df.copy()
    # Prefer English when same material has multiple language descriptions: put ASCII/Latin descriptions first
    if desc_col and desc_col in out.columns:
        try:
            # Prefer rows where description is mostly ASCII (English) over CJK or other scripts
            s = out[desc_col].astype(str)
            is_ascii = s.str.replace(r"[ -~]", "", regex=True).str.len() <= (s.str.len() * 0.1)
            out = out.assign(_lang_order=is_ascii.astype(int)).sort_values("_lang_order", ascending=False).drop(columns=["_lang_order"])
        except Exception:
            pass
    # Prefer "last update": sort by last-change date descending if present
    date_col = None
    for c in df_cols:
        if not c or c not in out.columns:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("LAEDA", "AEDAT", "LAST_CHANGE_DATE", "DATE_OF_LAST_CHANGE") or (
            "LAST" in (cu or "") and "CHANGE" in (cu or "") and "DATE" in (cu or "")
        ):
            try:
                pd.to_datetime(out[c], errors="coerce")
                date_col = c
                break
            except Exception:
                pass
    if date_col:
        out = out.sort_values(by=date_col, ascending=False, na_position="last")
    out = out.drop_duplicates(subset=[matnr_col], keep="first")
    return out, True


def _deduplicate_supplier_per_part_rows(df: pd.DataFrame, df_cols: list) -> tuple[pd.DataFrame, bool]:
    """When result has material (part) and vendor/supplier columns from purchase data, keep one row per (material, vendor) to avoid repeated vendor entries per part. Uses material_number or material_description as part key. Returns (deduplicated_df, was_deduped)."""
    if df.empty:
        return df, False
    vendor_col = _get_vendor_identifier_column(df_cols)
    if not vendor_col or vendor_col not in df.columns:
        return df, False
    # Prefer material number for key; fall back to material description so we deduplicate even when query returns only MAKTX
    mat_col = _get_material_number_column(df_cols)
    if not mat_col or mat_col not in df.columns:
        mat_col = _get_material_description_column(df_cols)
    if not mat_col or mat_col not in df.columns:
        return df, False
    key = [mat_col, vendor_col]
    if not df.duplicated(subset=key).any():
        return df, False
    out = df.drop_duplicates(subset=key, keep="first")
    return out, True


def is_sales_by_customer_query(user_query: str) -> bool:
    """True if the user is asking for sales/revenue by customer (totals per customer), not for product by industry."""
    if not user_query or not user_query.strip():
        return False
    q = user_query.lower().strip()
    # If user asked for product(s) by industry, do NOT aggregate by customer (result must show products)
    if "product" in q and ("by industry" in q or "per industry" in q):
        return False
    if "best selling" in q and "industry" in q:
        return False
    return (
        "by customer" in q
        or "sales by customer" in q
        or "revenue by customer" in q
        or ("highest sales" in q and "product" not in q and "industry" not in q)
        or "best sales" in q
        or "sales totals per customer" in q
        or "totals per customer" in q
        or "top customers" in q
        or "customer sales" in q
        or "customer revenue" in q
    )


def _aggregate_by_customer_sales(df: pd.DataFrame, df_cols: list) -> tuple[pd.DataFrame, bool]:
    """When result has customer and a value column with repeated customers, aggregate to one row per customer with summed value. Returns (aggregated_df, was_aggregated)."""
    if df.empty:
        return df, False
    customer_num_col = None
    for c in df_cols:
        if not c or c not in df.columns:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("KUNNR", "CUSTOMER_NUMBER") or cu == "CUSTOMER_NUMBER":
            customer_num_col = c
            break
    customer_name_col = _get_customer_name_column(df_cols)
    value_col = _find_numeric_value_column(df_cols, df)
    group_col = customer_num_col if (customer_num_col and customer_num_col in df.columns) else (customer_name_col if (customer_name_col and customer_name_col in df.columns) else None)
    if not group_col or not value_col or value_col not in df.columns:
        return df, False
    if not df[group_col].duplicated().any():
        return df, False
    agg_dict = {value_col: "sum"}
    if customer_num_col and customer_num_col in df.columns and customer_name_col and customer_name_col in df.columns and group_col == customer_num_col:
        agg_dict[customer_name_col] = "first"
    elif group_col == customer_name_col and customer_num_col and customer_num_col in df.columns:
        agg_dict[customer_num_col] = "first"
    # Preserve industry (BRSCH / industry_key) so "sales by customer and industry" still shows industry after aggregation
    for c in df.columns:
        if c in agg_dict or c == group_col:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("BRSCH", "INDUSTRY_KEY", "MBRSH") or "INDUSTRY" in cu:
            agg_dict[c] = "first"
            break
    try:
        out = df.groupby(group_col, dropna=False).agg(agg_dict).reset_index()
        val_cols = [c for c in out.columns if c != group_col and (pd.api.types.is_numeric_dtype(out[c]) or "total_" in (c or "").lower())]
        if val_cols:
            out = out.sort_values(by=val_cols[0], ascending=False).reset_index(drop=True)
        return out, True
    except Exception:
        return df, False


def perform_analysis_from_plan(df: pd.DataFrame, plan: dict, user_query: str = ""):
    st.write("## 🧠 AI-Powered Data Analysis")
    df_cols = list(df.columns)
    mt_col = None  # set by Sales-by-material-type section when not procurement_only_mode
    val_col = None
    procurement_only = is_procurement_only_query(user_query or "")
    has_beskz = any(
        (c or "").upper().replace(" ", "_") in ("BESKZ", "PROCUREMENT_TYPE") or "PROCUREMENT_TYPE" in (c or "").upper()
        for c in df_cols
    )
    procurement_only_mode = procurement_only and has_beskz

    # One row per material for standard price / moving average (no repeated same material with same value)
    df, material_price_deduped = _deduplicate_material_price_rows(df, df_cols)
    if material_price_deduped and not procurement_only_mode:
        st.caption("**Material valuation:** One entry per material (latest / last update). Duplicate rows with the same material number and same standard price or moving average have been removed so each product appears once.")

    # Precompute industry sector display (codes → readable labels) and source explanation
    industry_col = _find_industry_column(df_cols)
    df_for_industry = df
    industry_display_col = industry_col or ""
    industry_source_text = INDUSTRY_SECTOR_SOURCE.get((industry_col or "").upper(), industry_col or "—")
    if industry_col:
        df_for_industry, industry_display_col, industry_source_text = _map_industry_to_display(df, industry_col)
    industry_source_shown = False
    # Prefer product name/description (MAKTX) over material number (MATNR) for all product-related display
    product_display_col = _get_product_display_column(df_cols)
    # Prefer customer name (NAME1) over customer number (KUNNR) for charts and analysis
    customer_name_col = _get_customer_name_column(df_cols)

    # Data limitation notes — for procurement-only, show only BESKZ-related note; skip COGS/sales/variant notes
    data_notes = plan.get("data_notes") or []
    if procurement_only_mode:
        data_notes = [n for n in data_notes if n and ("BESKZ" in str(n) or "procurement" in str(n).lower() or "internal" in str(n).lower() or "external" in str(n).lower())]
        if not data_notes:
            data_notes = ["Procurement Type (MARC.BESKZ): **E** = In-house produced, **F** = Externally procured, **X** = Both."]
    for note in data_notes:
        if note and str(note).strip():
            st.info(f"ℹ️ {note.strip()}")
    # Fallback: if dataset has material/product ID but no description, show a short caveat (skip in procurement-only)
    if not procurement_only_mode and (not plan.get("data_notes") or not data_notes):
        col_str = " ".join(str(c).upper() for c in df_cols)
        if ("MATNR" in col_str or "MATERIAL_NUMBER" in col_str) and "MAKTX" not in col_str and "DESCRIPTION" not in col_str:
            st.caption("ℹ️ This dataset contains material numbers but no product descriptions. For richer product analysis, consider adding MAKT (material descriptions) or material master tables to your query.")
    # Why products are shown as numbers (e.g. 000000000000000038) and why "no data available" can appear (skip in procurement-only)
    if not procurement_only_mode:
        _matnr_only = _get_material_number_column(df_cols) and not product_display_col
        if _matnr_only and _get_material_number_column(df_cols) in df.columns:
            st.info(
            "**Why are products shown with numbers (e.g. 000000000000000038)?** "
            "The analysis shows **SAP material numbers (MATNR)** because the query did not include the **MAKT** table or the **MAKTX** (material description) column. "
            "To see **product names** instead of numbers: include **MAKT** in your query, or ask for *materials with description* or *cost of product [name]*. "
            "When only material numbers are in the data, the **Insights** section may answer *no data available* for questions about product names, because names are not in the result set."
            )
        # Explain the link between material numbers (MATNR) and product names (e.g. Harley Leather jacket)
        if product_display_col and _get_material_number_column(df_cols) and _get_material_number_column(df_cols) in df_cols:
            st.caption("**Link between material numbers and product names:** The numbers (e.g. 000000000000000058, 000000000000000059) are **SAP material numbers (MATNR)** — the unique ID for each product. The product name (e.g. Harley Leather jacket) comes from the **material description (MAKTX)** in table MAKT, linked by MATNR: **MAKT.MATNR** = material number, **MAKT.MAKTX** = description. So each row’s material number is the ID of that product; the description column is the name you searched for.")

        # Why repeated material numbers (when same MATNR appears multiple times)
        _matnr_col = _get_material_number_column(df_cols)
        if _matnr_col and _matnr_col in df.columns and df[_matnr_col].duplicated().any():
            st.caption("**Why repeated material numbers?** Product analysis uses **one language** (MAKT.SPRAS = 'E') so the same material is not repeated for different languages. The focus is to identify **same product name with different material numbers** (variants). If the same material still appears on several rows, it is from other 1:N joins (e.g. **MARM** per UoM, **MEAN** per EAN, **MVKE** per sales org). The query uses **DISTINCT** when only master tables are used to reduce duplicates.")

    # Procurement Type (MARC.BESKZ): primary SAP field for internal vs external
    PROCUREMENT_TYPE_LABELS = {"E": "In-house produced", "F": "Externally procured", "X": "Both"}
    _beskz_col = None
    for c in df_cols:
        if not c:
            continue
        cu = (c or "").upper().replace(" ", "_")
        if cu in ("BESKZ", "PROCUREMENT_TYPE") or "PROCUREMENT_TYPE" in cu:
            _beskz_col = c
            break
    if _beskz_col:
        if procurement_only_mode:
            st.caption("**Procurement Type (MARC.BESKZ):** **E** = In-house produced, **F** = Externally procured, **X** = Both.")
        else:
            st.caption(
                "**Procurement Type (MARC.BESKZ**, MRP 2 view) is the **primary SAP field** for whether a product is manufactured internally or externally: "
                "**E** = In-house produced, **F** = Externally procured, **X** = Both. "
                "Costing and valuation use **MBEW** (Accounting 1/Costing 1): VPRSV = price control (S = Standard, V = Moving average), STPRS = standard price, VERPR = moving average price, LOSGR = costing lot size."
            )

    # Production orders / manufactured internally (skip in procurement-only)
    if not procurement_only_mode:
        _prod_order_indicators = ("order_number", "order_type", "AUFNR", "AUART", "sales_order_number", "KDAUF", "object_number", "OBJNR", "start_date", "end_date")
        _col_str_upper = " ".join(str(c).upper().replace(" ", "_") for c in df_cols)
        if any(ind in _col_str_upper for ind in _prod_order_indicators) or ("ORDER" in _col_str_upper and ("TYPE" in _col_str_upper or "NUMBER" in _col_str_upper)):
            st.info(
                "**In SAP, the primary field** for internal vs external is **MARC.BESKZ (Procurement Type**, MRP 2): **E** = In-house, **F** = External, **X** = Both. "
                "**Production order data** (order_type, order_number) and **MBEW** (price control, standard/moving average price) support the picture. "
                "**Make-to-order:** production orders linked to sales orders (KDAUF). **Make-to-stock:** not linked to a sales order. "
                "**Dependencies:** production orders use BOM components (MAST, STPO) and may trigger purchase orders (EKKO, EKPO); use **order_number** to trace linked products and components."
            )

    # Standard price / cost (skip in procurement-only)
    if not procurement_only_mode:
        _std_price_col = None
        for c in df_cols:
            if not c:
                continue
            cu = (c or "").upper().replace(" ", "_")
            if cu in ("STPRS", "STANDARD_PRICE") or "STANDARD_PRICE" in cu:
                _std_price_col = c
                break
        if _std_price_col is None:
            for c in df_cols:
                if c and "standard" in (c or "").lower() and "price" in (c or "").lower():
                    _std_price_col = c
                    break
        if _std_price_col:
            try:
                ser = pd.to_numeric(df[_std_price_col], errors="coerce")
                null_or_zero = ser.isna() | (ser == 0)
                n_invalid = int(null_or_zero.sum())
                n_valid = int((~null_or_zero).sum())
                st.caption("**Standard price / cost:** For meaningful analysis, exclude or flag rows where standard price is zero or null (unreleased or missing prices). "
                           "**Price unit (PEINH)** defines the quantity per which the price applies.")
                if n_invalid > 0:
                    st.caption(f"⚠️ **Flag:** {n_invalid} row(s) have zero or null standard price (unreleased/missing); {n_valid} row(s) have valid standard price. Filter to valid rows for totals and charts.")
            except Exception:
                pass

    # Variant products (skip in procurement-only)
    if not procurement_only_mode and product_display_col and product_display_col in df_cols:
        matnr_col = _get_material_number_column(df_cols)
        try:
            _show_variant_differentiation_table(df, product_display_col, matnr_col, df_cols)
        except Exception:
            pass

    # Sales by material type (skip in procurement-only — not relevant)
    if not procurement_only_mode:
        mt_col = None
        val_col = None
        for c in df_cols:
            cu = (c or "").strip()
            if not cu:
                continue
            if cu.upper() in ("MTART", "MATERIAL_TYPE") or (("MATERIAL" in cu.upper() and "TYPE" in cu.upper())):
                mt_col = c
                break
        if not mt_col:
            for c in df_cols:
                if c and "material_type" in (c or "").lower():
                    mt_col = c
                    break
        for c in df_cols:
            if c is None:
                continue
            try:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    continue
            except Exception:
                continue
            cu = (c or "").upper()
            if cu == "NETWR" or "NET_VALUE" in cu or ("VALUE" in cu and "NET" in cu) or "REVENUE" in cu or "SALES" in cu:
                val_col = c
                break
        if not val_col:
            for c in df_cols:
                if c is None or c == mt_col:
                    continue
                try:
                    if pd.api.types.is_numeric_dtype(df[c]) and ("value" in (c or "").lower() or "net" in (c or "").lower() or "revenue" in (c or "").lower() or (c or "").upper() == "NETWR"):
                        val_col = c
                        break
                except Exception:
                    pass
        if mt_col and val_col:
            try:
                # Coerce value column to numeric (DB may return string)
                val_series = pd.to_numeric(df[val_col], errors="coerce").fillna(0)
                total = val_series.sum()
                if total and total != 0:
                    # Normalize material type: strip, upper; SAP often stores 4-char (e.g. "FERT", "ROH ")
                    mt_norm = df[mt_col].astype(str).str.strip().str.upper().str[:4]
                    fert_mask = mt_norm == "FERT"
                    sales_fert = val_series.loc[fert_mask].sum()
                    # Non-finished = all that is not FERT (ROH, HALB, and any other types sold in SAP)
                    sales_non_fert = total - sales_fert
                    pct_fert = (sales_fert / total) * 100
                    pct_non_fert = (sales_non_fert / total) * 100
                    st.write("### 📊 Sales by material type (finished vs raw / semi-finished / other)")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Total sales value (all goods)", _format_numeric(total))
                    with c2:
                        st.metric("Percentage finished goods (FERT)", f"{pct_fert:.1f}%")
                    with c3:
                        st.metric("Percentage raw + semi-finished + other (non-FERT)", f"{pct_non_fert:.1f}%")
                    summary = pd.DataFrame({
                        "Category": ["Finished goods (FERT)", "Raw, semi-finished & other (non-FERT)", "Total (all goods)"],
                        "Sales value": [sales_fert, sales_non_fert, total],
                        "Percentage (%)": [pct_fert, pct_non_fert, 100.0],
                    })
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                    # Breakdown by each material type in the data (so user sees ROH, HALB, FERT, and any others)
                    by_type = pd.DataFrame({"Material type": mt_norm, "_val": val_series}).groupby("Material type")["_val"].sum().reset_index()
                    by_type.columns = ["Material type", "Sales value"]
                    by_type["Sales value"] = by_type["Sales value"].astype(float)
                    by_type["Percentage (%)"] = (by_type["Sales value"] / total * 100).round(1)
                    by_type = by_type.sort_values("Sales value", ascending=False)
                    st.write("**Breakdown by material type** (all types in your data)")
                    st.dataframe(by_type, use_container_width=True, hide_index=True)
                    st.caption("Finished goods = FERT. Raw, semi-finished & other = all non-FERT (ROH, HALB, and any other material types sold). Percentages = (category sales / total sales) × 100.")
            except Exception as e:
                st.warning(f"Could not compute material-type percentages: {e}")
        elif mt_col:
            # Material type present but no sales/value column — show actionable suggestion
            st.info(
                "**Sales by material type** needs a sales/value column (e.g. **VBRP.NETWR**). "
                "Run a new query such as **\"Sales value by material type\"** or **\"Revenue by material type\"** so the app includes billing data (VBRK, VBRP) with material type (MARA.MTART) and can show percentages and charts."
            )

    # Calculations
    if "calculations" in plan:
        st.write("### 📊 Calculations")
        for calc in plan["calculations"]:
            try:
                low = (calc or "").lower()
                # "sum of X where material_type = 'FERT'" or "where material_type IN ('ROH', 'HALB')"
                if "sum" in low and " where material_type" in low and mt_col and val_col:
                    part = calc.split("sum of")[-1].strip() if "sum of" in low else ""
                    sum_col = _strip_calc_trailers(part.split(" where ")[0].strip())
                    if sum_col not in df_cols:
                        sum_col = next((c for c in df_cols if (sum_col in c or c in sum_col) and c != mt_col), sum_col)
                    if sum_col not in df_cols:
                        continue
                    ser = pd.to_numeric(df[sum_col], errors="coerce").fillna(0)
                    mt_vals = df[mt_col].astype(str).str.strip().str.upper().str[:4]
                    if "= 'fert'" in low or "= 'fert' " in low:
                        s = ser.loc[mt_vals == "FERT"].sum()
                        st.metric("Sum (material_type = FERT)", _format_numeric(s))
                    elif "in ('roh', 'halb')" in low or "in ( 'roh' , 'halb' )" in low:
                        s = ser.loc[mt_vals.isin(("ROH", "HALB"))].sum()
                        st.metric("Sum (material_type IN ROH, HALB)", _format_numeric(s))
                    else:
                        st.metric(f"**{calc}**", _format_numeric(ser.sum()))
                    continue
                # "sum of X grouped by Y" or "sum X group by Y"
                has_group = "group by" in low or "grouped by" in low
                if ("sum" in low and has_group):
                    # Resolve group column: text after "grouped by" or "group by"
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
                        # Use product name/description (MAKTX) instead of material number for group-by when available
                        if _is_product_material_column(group_col, df_cols) and product_display_col and product_display_col in df_cols:
                            group_col = product_display_col
                        # Calculations must show customer names, not customer numbers
                        if _is_customer_number_column(group_col, df_cols) and customer_name_col and customer_name_col in df_cols:
                            group_col = customer_name_col
                        if group_col in df_cols and sum_col in df_cols:
                            ser = pd.to_numeric(df[sum_col], errors="coerce").fillna(0)
                            df_calc = df.assign(_sum=ser)
                            # When grouping by product description, use distinct labels if same name = different materials
                            if group_col == product_display_col:
                                df_calc = df_calc.copy()
                                df_calc[group_col] = _normalize_material_description(df_calc[group_col])
                                matnr_col = _get_material_number_column(df_cols)
                                if matnr_col and matnr_col in df_calc.columns:
                                    dup_desc = df_calc.groupby(group_col)[matnr_col].nunique()
                                    if (dup_desc > 1).any():
                                        df_calc["_group_label_"] = df_calc[group_col].astype(str) + " (" + df_calc[matnr_col].astype(str).str.strip() + ")"
                                        result = df_calc.groupby("_group_label_")["_sum"].sum().reset_index().rename(columns={"_group_label_": group_col, "_sum": sum_col})
                                    else:
                                        result = df_calc.groupby(group_col)["_sum"].sum().reset_index().rename(columns={"_sum": sum_col})
                                else:
                                    result = df_calc.groupby(group_col)["_sum"].sum().reset_index().rename(columns={"_sum": sum_col})
                            else:
                                result = df_calc.groupby(group_col)["_sum"].sum().reset_index().rename(columns={"_sum": sum_col})
                            if _is_industry_column(group_col, df_cols):
                                if not industry_source_shown:
                                    st.caption(f"**Industry sector** (source: {industry_source_text}). Codes are shown as full names.")
                                    industry_source_shown = True
                                result = result.copy()
                                result[group_col] = result[group_col].astype(str).str.strip().map(
                                    lambda x: INDUSTRY_SECTOR_LABELS.get(x, INDUSTRY_SECTOR_LABELS.get((x or "").upper(), x or "Not specified"))
                                )
                            st.write(f"**{calc}**")
                            st.dataframe(result)
                        else:
                            raise ValueError(f"Column not found: {sum_col!r} or {group_col!r}")
                    else:
                        raise ValueError("Could not parse sum column or group-by column")
                elif "sum of" in low:
                    col = _resolve_calc_column(calc, df_cols, "sum of") or _strip_calc_trailers(calc.split("sum of")[1].strip())
                    if col not in df_cols:
                        col = next((c for c in df_cols if col in c or c in col), col)
                    if col in df_cols:
                        total = pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
                        st.metric(f"Sum of {col}", _format_numeric(total))
                elif "avg" in low or "average" in low:
                    col = _resolve_calc_column(calc, df_cols, "avg of") or _resolve_calc_column(calc, df_cols, "average of")
                    if col and col in df_cols:
                        mean_val = pd.to_numeric(df[col], errors="coerce").mean()
                        st.metric(f"Average of {col}", _format_numeric(mean_val))
                # You can add more operations: min, max, etc.
            except Exception as e:
                st.warning(f"Could not perform: {calc} — {e}")

    # Visualizations
    charts_rendered = 0
    industry_chart_rendered = False
    if "visualizations" in plan:
        vis_list = plan["visualizations"]
        if procurement_only_mode:
            # Only show charts that use procurement_type (internal vs external)
            def _is_procurement_vis(v):
                x = (str(v.get("x") or v.get("labels") or "")).upper().replace(" ", "_")
                return x in ("PROCUREMENT_TYPE", "BESKZ") or "PROCUREMENT" in x
            vis_list = [v for v in vis_list if v and _is_procurement_vis(v)]
            # If no procurement chart was suggested, add a simple count by procurement_type
            if not vis_list and has_beskz and _beskz_col and _beskz_col in df_cols:
                vis_list = [{"type": "bar", "x": _beskz_col, "y": _beskz_col, "agg": "count"}]
        if vis_list:
            st.write("### 📈 Charts & graphs")
        for vis in vis_list:
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

                # Use industry display labels when x is industry column; use product name (MAKTX) when x is material column; use customer name when x is customer number
                use_df = df_for_industry if _is_industry_column(x, df_cols) else df
                use_x = industry_display_col if _is_industry_column(x, df_cols) and industry_display_col else x
                is_product_chart = _is_product_material_column(x, df_cols) and product_display_col and product_display_col in df_cols
                if is_product_chart:
                    use_x = product_display_col
                # Charts and graphs must show customer names, not customer numbers
                if _is_customer_number_column(use_x, df_cols) and customer_name_col and customer_name_col in df_cols:
                    use_x = customer_name_col
                if _is_industry_column(x, df_cols) and not industry_source_shown:
                    st.caption(f"**Industry sector** (source: {industry_source_text}). Single-letter codes (e.g. M, C) are shown as full names.")
                    industry_source_shown = True

                df_copy = use_df.copy()
                chart_label_col = use_x
                # When x-axis is procurement type (BESKZ), show code + description (e.g. "E (In-house produced)")
                if _beskz_col and use_x == _beskz_col and _beskz_col in df_copy.columns:
                    df_copy["_procurement_display"] = df_copy[_beskz_col].astype(str).str.strip().str.upper().map(
                        lambda v: PROCUREMENT_TYPE_DISPLAY_LABELS.get(v, v) if v else ""
                    )
                    use_x = "_procurement_display"
                # When x-axis is material number (MATNR) and we have no product description: show formatted numbers (e.g. 38 instead of 000000000000000038)
                matnr_col = _get_material_number_column(df_cols)
                if matnr_col and use_x == matnr_col and matnr_col in df_copy.columns and not (product_display_col and product_display_col in df_copy.columns):
                    df_copy["_matnr_display"] = _format_matnr_for_display(df_copy[matnr_col])
                    use_x = "_matnr_display"
                # Product/material charts: normalize description and, if same name appears for different materials, show "Description (MATNR)" so it's clear
                if is_product_chart and product_display_col in df_copy.columns:
                    df_copy[product_display_col] = _normalize_material_description(df_copy[product_display_col])
                    matnr_col = _get_material_number_column(df_cols)
                    if matnr_col and matnr_col in df_copy.columns:
                        # Same description for different MATNR? Use "Description (MATNR)" so each bar/slice is clearly one material
                        dup_desc = df_copy.groupby(product_display_col)[matnr_col].nunique()
                        has_duplicate_names = (dup_desc > 1).any()
                        if has_duplicate_names:
                            chart_label_col = "_chart_product_label_"
                            matnr_display = _format_matnr_for_display(df_copy[matnr_col])
                            df_copy[chart_label_col] = df_copy[product_display_col].astype(str) + " (" + matnr_display.astype(str) + ")"
                            use_x = chart_label_col
                        else:
                            use_x = product_display_col
                    else:
                        use_x = product_display_col
                if agg == "count":
                    df_vis = df_copy.groupby(use_x)[y].count().reset_index()
                elif agg == "sum":
                    df_vis = df_copy.groupby(use_x)[y].sum().reset_index()
                else:
                    df_vis = df_copy.groupby(use_x)[y].mean().reset_index()

                # When x and y are the same column, groupby().count().reset_index() yields two columns with the same name → rename the value column by position
                use_y = y
                if use_x == y and len(df_vis.columns) >= 2:
                    value_col_name = "_value_" if agg == "sum" else "_count_" if agg == "count" else "_mean_"
                    df_vis.columns = [df_vis.columns[0], value_col_name]
                    use_y = value_col_name

                if vis_type == "bar":
                    chart = alt.Chart(df_vis).mark_bar().encode(x=use_x, y=use_y)
                elif vis_type == "line":
                    chart = alt.Chart(df_vis).mark_line().encode(x=use_x, y=use_y)
                elif vis_type == "pie":
                    chart = alt.Chart(df_vis).mark_arc().encode(
                        theta=alt.Theta(use_y, type="quantitative"),
                        color=alt.Color(use_x, type="nominal")
                    )
                else:
                    chart = alt.Chart(df_vis).mark_bar().encode(x=use_x, y=use_y)

                st.altair_chart(chart.properties(width="container"), use_container_width=True)
                if is_product_chart:
                    st.caption("**Product / material (variants):** Each bar or slice = one material. Same name with different material numbers = **variant products** (e.g. colour, shape, or size). The section **Variant products: what differentiates them?** above lists material type, group, base unit, plant, and variant attributes for each.")
                charts_rendered += 1
                if _is_industry_column(x, df_cols):
                    industry_chart_rendered = True
            except Exception as e:
                st.warning(f"Could not render chart for: {vis} — {e}")

    # Industry trends: fallback bar + pie when data has industry but no industry chart was rendered
    value_col = _find_numeric_value_column(df_cols, df)
    if industry_col and value_col and not industry_chart_rendered:
        st.write("### 📊 Industry trends")
        if not industry_source_shown:
            st.caption(f"**Industry sector** (source: {industry_source_text}). Single-letter codes (e.g. M, C) are shown as full names.")
        try:
            df_vis = df_for_industry.groupby(industry_display_col)[value_col].sum().reset_index()
            st.altair_chart(
                alt.Chart(df_vis).mark_bar().encode(x=industry_display_col, y=value_col).properties(title="Value by industry (bar)").properties(width="container"),
                use_container_width=True,
            )
            st.altair_chart(
                alt.Chart(df_vis).mark_arc().encode(
                    theta=alt.Theta(value_col, type="quantitative"),
                    color=alt.Color(industry_display_col, type="nominal"),
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