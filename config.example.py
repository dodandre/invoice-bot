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
    "MARA": "Material Master (General Data, client level): MATNR; MTART, MBRSH; MATKL = material group, MEINS = base unit of measure; link to MARC, MBEW, MAKT on MATNR.",
    "MAKT": "Material Descriptions: MATNR = material number, MAKTX = material description (product name text), SPRAS = language; link to VBRP, MARA, VBAP, LIPS on MAKT.MATNR = table.MATNR so results show product names instead of only material numbers; use SPRAS = 'E' for English when needed.",
    "MVKE": "Material Master: Sales Data for Sales Org; MATNR = material number, VKORG = sales organization, VTWEG = distribution channel; link to MARA on MARA.MATNR = MVKE.MATNR; use for product sales attributes by sales org/distribution channel.",
    "PSIF_SLS_MSG": "EDI message status for validated orders (table /PSIF/SLS_MSG); VBELN = sales document; shows message status for pre sales orders; link to PSIF_SLS_HDR on VBELN.",
    "PSIF_INV_HDR_IN": "Inbound invoices from vendors header (table /PSIF/INV_HDR_IN); VBELN or BELNR = document number, LIFNR = vendor; link to PSIF_INV_ITEM_I on document number.",
    "PSIF_INV_ITEM_I": "Inbound invoices from vendors item level (table /PSIF/INV_ITEM_I); document number, POSNR, MATNR, NETWR; link to PSIF_INV_HDR_IN on document number.",
    "PSIF_RBKP_MSG": "Purchasing messages status (table /PSIF/RBKP_MSG); BELNR, GJAHR = invoice document; message status for invoice receipt; link to PSIF_RBKP on BELNR and GJAHR.",
    "PSIF_RBKP": "Document Header: Invoice Receipt (table /PSIF/RBKP); BELNR = invoice document number, GJAHR = fiscal year, LIFNR = vendor; link to PSIF_RBKP_MSG on BELNR and GJAHR.",
    # Product Costing (CO-PC-PCP): KEKO, KEPH, CKIS, CKIT, KALA, CKHS, CKEP
    "KEKO": "Product Costing Header: KALNR = cost estimate number, MATNR, BWKEY, BWTAR; link to KEPH, CKIS, CKHS, CKEP on KALNR; link to MARA, MBEW on MATNR.",
    "KEPH": "Product Costing Cost Components (Cost of Goods Mfd); link to KEKO on KALNR.",
    "CKIS": "Product Costing Itemization: detailed cost breakdown; KALNR, POSNR; link to KEKO on KALNR.",
    "CKIT": "Texts for CKIS (cost estimate itemization texts): KALNR, POSNR, LTEXT, SPRAS; link to CKIS on KALNR and POSNR; link to KEKO on KALNR.",
    "KALA": "Costing Run: general data/parameters.",
    "CKHS": "Cost Estimate Header; link to KEKO (KALNR).",
    "CKEP": "Costing Items Subtotals; link to cost estimate (KALNR)/CKHS.",
    # FI document & G/L for COGS: BKPF, BSEG, SKA1, SKB1, T001K, T030, ACDOCA
    "BKPF": "Accounting document header (FI): BELNR, BUKRS, GJAHR, BLDAT, BUDAT, WAERS; link to BSEG on BELNR, BUKRS, GJAHR; use for posted COGS documents and FI line items.",
    "BSEG": "Accounting document segment (FI line items): BELNR, BUKRS, GJAHR, BUZEI, HKONT or SAKNR (G/L account), DMBTR, WRBTR, VBELN (billing doc); link to BKPF on BELNR, BUKRS, GJAHR; link to SKB1 on BUKRS and HKONT/SAKNR; use with BKPF for posted COGS by account.",
    "SKA1": "G/L account master (chart of accounts): KTOPL, SAKNR, TXT50 (account description); link to SKB1 on SAKNR; use with SKB1 to get account descriptions for COGS and P&L accounts.",
    "SKB1": "G/L account master (company code): BUKRS, SAKNR, WAERS; link to BKPF/BSEG on BUKRS; link to SKA1 on SAKNR for account description; use for COGS account determination by company code.",
    "T001K": "Valuation area to company code: BWKEY (valuation area), BUKRS (company code); link T001K.BWKEY = MBEW.BWKEY to get company code for valuation area; use for COGS and material valuation.",
    "T030": "Chart of accounts / posting configuration (e.g. accounts for exchange rate differences); KTOPL, BSCHL, KOART, SAKON; use with SKA1 for account determination.",
    # Material Ledger & Master Data
    "MBEW": "Material Master Accounting/Costing: MATNR, BWKEY, BWTAR; STPRS, VERPR; link to MARA on MATNR.",
    "CKMLCR": "Material Ledger Period Totals; MATNR, BWKEY, period; link to MBEW, MARA on MATNR.",
    "MARC": "Material Master Plant level: Purchasing — EKGRP, BSTME, MMSTA, UEBTO, UNTTO, XCHAR; MRP — BESKZ (E/F/X), SOBSL, PLIFZ, WEBAZ, DISMM; MATNR, WERKS; link to MARA on MATNR.",
    # BOM & Routing
    "STKO": "BOM Header: STLNR, STLAL, STLST; link to STPO on STLNR; use with MAST for component data of main material.",
    "STPO": "BOM Item: STLNR, POSNR, IDNRK = component material, MENGE, MEINS; link to STKO on STLNR, MARA/MAKT on IDNRK.",
    "MAST": "Material to BOM Link: MATNR (main material), WERKS, STLNR (BOM number); join MAST.STLNR = STKO.STLNR = STPO.STLNR to read component data.",
    "DF14L": "PLM/PPM component data; use with DF14T for texts.",
    "DF14T": "PLM/PPM texts for DF14L; link per system keys.",
    "PAT03": "Hot Packages/components in system administration.",
    "PLKO": "Routing Header; link to PLPO.",
    "PLPO": "Routing Operations; link to PLKO.",
    "CSLA": "Activity Type Master: LART; for cost center/activity pricing.",
    # Cost Object Controlling (CO-PC-OBJ)
    "AUFK": "Order Master Data: internal and production orders; AUFNR, AUART, KDAUF/KDPOS for sales order link (make-to-order); link to VBAK/VBAP, COEP, MAST/STPO for dependencies and linked products.",
    "COBK": "Controlling Document Header; link to COEP, AUFK.",
    "COEP": "Cost Object Line Items by Period: OBJNR, WTGJB, PERIO; link to COBK, AUFK.",
    # Profitability analysis
    "FAGLFLEXA": "Profitability analysis / New G/L line items (SAP ERP): use for profitability reporting and analysis in ERP; link to company code, profit center, segment, account, etc. Add table_mapping/FAGLFLEXA.json with your column names if using this table.",
    "ACDOCA": "Universal Journal line items (S/4 HANA): use for profitability analysis, COGS reporting, and FI/CO line items; RACCT = G/L account, RBUKRS, RYEAR/GJAHR, BELNR, DOCLN, WTGBTR (amount). Filter by COGS account for posted COGS. Add table_mapping/ACDOCA.json with your column names if using this table.",
    # Material Master (additional): MLAN (tax), MAPR (forecast), MARM/MEAN (UoM)
    "MLAN": "Tax classification for material: MATNR, ALAND, TAXM1; link to MARA on MATNR.",
    "MAPR": "Forecast parameters: MATNR, WERKS; link to MARA, MARC.",
    "MARM": "Units of measure for material: MATNR, MEINS (alternative UoM), UMREZ (numerator), UMREN (denominator); conversion to base = quantity × UMREZ/UMREN; link to MARA on MARA.MATNR = MARM.MATNR; use for UoM conversion in COGS and quantity reporting.",
    "MEAN": "International article numbers (EAN/UPC) and unit of measure: MATNR, MEANM (EAN), MEINS (UoM); link to MARA on MATNR; use with MARM for UoM conversion.",
    # Vendor Master
    "LFA1": "General vendor data: LIFNR, NAME1, LAND1; link to LFB1, LFM1, EKKO on LIFNR.",
    "LFB1": "Vendor company code: LIFNR, BUKRS, ZTERM; link to LFA1 on LIFNR.",
    "LFM1": "Vendor purchasing org: LIFNR, EKORG, EKGRP; link to LFA1 on LIFNR.",
    "LFBK": "Vendor bank details: LIFNR, BANKS, BANKL, BANKN; link to LFA1 on LIFNR.",
    "ADR6": "Vendor/BP addresses (emails); link to LFA1 via ADRNR/LIFNR.",
    # Purchasing
    "EKKO": "PO header: EBELN, LIFNR, BEDAT, BUKRS; link to EKPO on EBELN.",
    "EKPO": "PO item: EBELN, EBELP, MATNR, MENGE, NETPR; link to EKKO on EBELN, MARA on MATNR.",
    "EKET": "PO delivery schedule: EBELN, EBELP, EINDT, MENGE; link to EKPO.",
    "EKBE": "PO history (GR/IR): EBELN, EBELP, DMBTR; link to EKKO/EKPO.",
    "EKKN": "PO account assignment: EBELN, EBELP, KOSTL, SAKTO; link to EKPO.",
    "EBAN": "Purchase requisition: BANFN, BNFPO, MATNR, MENGE; link to EBKN.",
    "EBKN": "PR account assignment: BANFN, BNFPO; link to EBAN.",
    # Inventory Management
    "MSEG": "Material document line: MBLNR, MJAHR, ZEILE, MATNR, MENGE, BWKEY; link to MKPF on MBLNR, MJAHR.",
    "MKPF": "Material document header: MBLNR, MJAHR, BLDAT, BUDAT; link to MSEG.",
    "MSKU": "Special stocks (customer/vendor); link to MSEG/MARA.",
    "MSLB": "Special stock with vendor: LIFNR, MATNR, WERKS; link to LFA1, MARA.",
    "MCHB": "Batch stock: MATNR, WERKS, CHARG; link to MARA, MARC.",
    "MSPR": "Project stock; link to MSEG/MARA.",
}

# Industry sector (BRSCH / MBRSH): map single-letter codes to readable names for charts/tables.
# Override INDUSTRY_SECTOR_LABELS in config.py if your SAP uses different or extra codes.
INDUSTRY_SECTOR_LABELS = {
    "1": "Sector 1",
    "A": "Plant engineering and construction",
    "C": "Chemical",
    "E": "Electrical engineering",
    "F": "Food / FMCG",
    "M": "Mechanical engineering",
    "P": "Pharmaceutical",
    "R": "Retail",
    "S": "Services",
    "T": "Transport / Logistics",
    "V": "Vehicle manufacturing",
    "": "Not specified",
}
INDUSTRY_SECTOR_SOURCE = {
    "BRSCH": "Customer Master (KNA1)",
    "MBRSH": "Material Master (MARA)",
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
# Revenue is shown only for these billing category types (VBRK.FKTYP). Use when querying value/revenue from VBRK/VBRP.
REVENUE_BILLING_CATEGORIES = ("A", "B", "C", "D", "E", "I", "L", "W")
MAX_DOC_ROWS = 50

# LangChain (requires openai from above)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
