# config.example.py
# Copy this file to config.py and fill in your values. Do not commit config.py.

import os

# -------------------------------
# API keys (use env vars in production)
# -------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")

# -------------------------------
# SQL Server (SAP ERP)
# -------------------------------
SQL_USERNAME = os.environ.get("SQL_USERNAME", "your_db_user")
SQL_PASSWORD = os.environ.get("SQL_PASSWORD", "your_db_password")
SQL_SERVER = os.environ.get("SQL_SERVER", "your-server\\ERP")
SQL_DATABASE = os.environ.get("SQL_DATABASE", "ERP")

# -------------------------------
# Kafka (optional)
# -------------------------------
KAFKA_PRODUCER_CONFIG = {
    "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"),
}
KAFKA_TOPICS = {
    "sql": "sql-query",
    "results": "query-results",
    "insights": "final_output"
}

# -------------------------------
# Paths
# -------------------------------
MAPPINGS_FOLDER = "table_mapping"

# -------------------------------
# Connection string
# -------------------------------
CONN_STR = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SQL_SERVER};"
    f"DATABASE={SQL_DATABASE};"
    f"UID={SQL_USERNAME};"
    f"PWD={SQL_PASSWORD};"
    "Encrypt=no;"
    "TrustServerCertificate=yes;"
)

# -------------------------------
# Table descriptions (SAP tables — same as production config)
# -------------------------------
TABLE_DESCRIPTIONS = {
    "PSIF_INV_HDR": "e-invoicing header (/PSIF/INV_HDR): value (NETWR), customer (KUNNR), VBELN; invoices sent; use with PSIF_INV_ITEM and VBRK/VBRP to identify discrepancies (created vs sent); impacts payment terms and revenues",
    "PSIF_INV_ITEM": "e-invoicing line items (/PSIF/INV_ITEM): MATNR, NETWR, quantities; link to PSIF_INV_HDR via VBELN; use for consistency and discrepancy identification (value and quantities)",
    "PSIF_ACK": "status of invoices sent out and when sent (table /PSIF/ACK); VBELN = billing document; ACK_TEXT or ACK_ID can indicate reason for delay (e.g. supplier delay or other); use with VBRK to compare created vs sent and to attribute delay reason; impacts payment terms and revenues",
    "BSAD": "Accounting: cleared items; AUGDT = clearing/payment date, VBELN = billing document, KUNNR = customer; use with VBRK and KNA1 to compare invoices paid on time vs delayed and which customers caused payment delay (visibility)",
    "VBRK": "Billing Document: invoice created (header); VBELN = billing document number; BSTNK_VF = customer purchase order number; for process flow always show with: sales order number (VBRP.AUBEL / VBAK.VBELN), delivery number (LIKP.VBELN via VBFA), purchase order number (VBAK.BSTNK); link to sales order via VBRP.AUBEL = VBAK.VBELN; link billing to delivery via VBFA (VBFA.VBELN = VBRP.VBELN, VBFA.VBELV = LIKP.VBELN)",
    "VBRP": "Billing Document: Item Data (invoice line); VBELN = billing/invoice document number, AUBEL = reference sales order number, AUPOS = reference item position number only (do not use AUPOS as order number or invoice number); for process flow always show with: billing document (VBELN), sales order number (AUBEL), delivery number (LIKP.VBELN via VBFA), purchase order number (VBAK.BSTNK); use with VBRK, VBAK, VBFA, LIKP",
    "LIKP": "SD Document: Delivery Header Data; VBELN = delivery number (required for process flow when referencing invoice/billing); link to LIPS on VBELN; link from billing via VBFA (VBFA.VBELV = LIKP.VBELN when VBFA.VBELN = billing doc); link to sales order via LIPS.VGBEL = VBAK.VBELN; purchase order number (VBAK.BSTNK) links order, delivery, invoice",
    "LIPS": "SD Document: Delivery Item data; VBELN = delivery number, VGBEL = reference document (sales order number), VGPOS = reference item; link to order via LIPS.VGBEL = VBAK.VBELN and LIPS.VGPOS = VBAP.POSNR; use for matching values, products, positions across order-delivery-invoice",
    "VBAP": "Sales Document: Item Data (sales order item); use with VBAK; sales and purchase orders use VBAK, VBAP, VBEP, VBFA, VBPA, VBUK, VBUP",
    "VBAK": "Sales Document: Header Data (sales order header); BSTNK = customer purchase order number, BSTDK = purchase order date, BSARK = purchase order type; use with VBAP; sales and purchase orders stored in VBAK, VBAP, VBEP, VBFA, VBPA, VBUK, VBUP",
    "VBEP": "Sales document schedule lines (dates, quantities); link to VBAP on VBELN and POSNR; part of sales/purchase order set with VBAK, VBAP, VBFA, VBPA, VBUK, VBUP",
    "VBFA": "Document flow table — shows the flow distinctively (preceding and subsequent documents); VBELV/POSNV = preceding document, VBELN/POSNN = subsequent document; use VBFA to get delivery number for a billing document: VBFA.VBELN = VBRP.VBELN, VBFA.VBELV = LIKP.VBELN (delivery number); required for process flow when showing invoice with delivery number; the definitive link between order, delivery, and invoice",
    "VBPA": "Document partner (sold-to, ship-to, etc.); link to VBAK on VBELN (POSNR 000000 for header); part of sales/purchase order set",
    "VBUK": "Sales document header status; link to VBAK on VBELN; part of sales/purchase order set",
    "VBUP": "Sales document item status; link to VBAP on VBELN and POSNR; part of sales/purchase order set",
    "PSIF_SLS_HDR": "eOrder header: table /PSIF/SLS_HDR; electronic sales order header (counterpart to VBAK)",
    "PSIF_SLS_ITEM": "eOrder item: table /PSIF/SLS_ITEM; electronic sales order item (counterpart to VBAP); use with PSIF_SLS_HDR",
    "KNA1": "General Data in Customer Master; customer number = KUNNR, customer name = NAME1 (always use KNA1 for customer number and name)",
    "ADDR1_DATA": "customer name and address, linked to KNA1 (use for customer name)",
    "KNKK": "Credit master data (customer credit data, SD-BF-CM / FI-AR): credit limit (KLIMK), used credit, credit exposure (SKFOR = total value of open orders/deliveries), risk category per credit control area (KKBER, CTLPC); link to KNA1 on KUNNR for customer name; link to T691A on CTLPC and KKBER for risk category definitions",
    "KNKA": "Credit management: central data; general credit data such as risk class and credit group; KUNNR = customer; overall credit limits (KLIMG, KLIME), currency (WAERS); link to KNA1 on KUNNR for customer credit and risk",
    "T691A": "Credit management risk categories; defines the risk categories used for credit checks; CTLPC = risk category, KKBER = credit control area; link from KNKK on KNKK.CTLPC = T691A.CTLPC AND KNKK.KKBER = T691A.KKBER",
    "UKM_BP_CMS_SGM": "Credit limit data (S/4HANA FSCM only — table may not exist in ERP/ECC); PARTNER = business partner/customer number; credit limit, calculated limit, credit segment, block reason, risk/critical flag; link to KNA1 on KNA1.KUNNR = PARTNER. The app supports both ERP (KNKK, KNKA) and S/4HANA FSCM: it auto-detects if this table exists and includes it for credit queries when available, otherwise uses only KNKK and KNKA.",
}

CUSTOMER_NUMBER_COLUMN = ("KNA1", "KUNNR")
CUSTOMER_NAME_COLUMN = ("KNA1", "NAME1")
CUSTOMER_NUMBER_LENGTH = 10
VAT_TAX_NUMBER_COLUMN = "STCEG"
ANNUAL_SALES_REVENUE_COLUMN = ("KNA1", "UMSA1")
VBAK_CUSTOMER_PO_NUMBER = "BSTNK"
VBAK_CUSTOMER_PO_DATE = "BSTDK"
VBAK_CUSTOMER_PO_TYPE = "BSARK"
INCLUDE_FSCM_CREDIT_TABLE = None
DATE_COLUMNS = {"FKDAT", "BUDAT", "BLDAT", "ZBDAT", "GSTRP"}
MAX_DOC_ROWS = 50

# LangChain (requires openai from above)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
