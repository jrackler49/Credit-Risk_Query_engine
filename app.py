"""
Northbridge Bank Credit Risk Query Engine
Streamlit app that lets business users ask routine commercial-lending portfolio
questions in plain English and receive verified, auditable answers.

Deployment notes:
- Place credit_risk_portfolio.db in the same directory as this file.
- Set OPENAI_API_KEY (and OPENAI_API_BASE if using a custom endpoint) as a
  Streamlit secret (Settings > Secrets) or environment variable before running.
"""

import os
import re
import json
import sqlite3
import sqlparse
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Northbridge Credit Risk Query Engine", layout="wide")
st.title("Northbridge Bank - Credit Risk Query Engine")
st.caption(
    "Ask a routine commercial lending portfolio question in plain English. "
    "Every answer shows the SQL used, the raw data, and a confidence score."
)

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
OPENAI_API_BASE = st.secrets.get("OPENAI_API_BASE", os.environ.get("OPENAI_API_BASE"))

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not configured. Add it under Settings > Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

# llm handles generation/classification; evaluator_llm is a distinct, stronger model
# so the same model isn't grading its own SQL in the validation gate.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------------------------------------------------------------------
# Database connection (read-only) and schema context
# ---------------------------------------------------------------------------
DB_PATH = "credit_risk_portfolio.db"


@st.cache_resource
def get_connection():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)


conn = get_connection()

database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# ---------------------------------------------------------------------------
# Verified Query Template Library
# ---------------------------------------------------------------------------
sql_1 = """
SELECT
    sm.sector_name,
    ROUND(SUM(lm.total_outstanding) / 1e6, 2) AS total_outstanding_mn,
    ROUND(SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                    THEN lm.total_outstanding ELSE 0 END) / 1e6, 2) AS npa_outstanding_mn
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_mn DESC
"""

sql_2 = """
SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC
"""

sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1e6, 2) AS ead_mn,
    ROUND(SUM(ecl_amount) / 1e6, 2) AS ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""

sql_4 = """
SELECT
    sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""

sql_5 = """
SELECT
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1e6, 2) AS outstanding_mn,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY lm.total_outstanding DESC
LIMIT 10
"""

sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5
"""

sql_7 = """
SELECT
    lm.loan_account_number,
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1e6, 2) AS outstanding_mn,
    lm.days_past_due,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC
"""

sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END
"""

sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""

sql_11 = """
SELECT
    CASE WHEN sm.is_sensitive_sector = 1 THEN 'Sensitive' ELSE 'Non-Sensitive' END AS sector_sensitivity,
    COUNT(*) AS loan_count,
    ROUND(SUM(lm.total_outstanding) / 1e6, 2) AS total_outstanding_mn,
    ROUND(SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                    THEN lm.total_outstanding ELSE 0 END) / 1e6, 2) AS npa_outstanding_mn
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sector_sensitivity
ORDER BY total_outstanding_mn DESC
"""

sql_12 = """
SELECT
    CASE WHEN is_secured = 1 THEN 'Secured' ELSE 'Unsecured' END AS security_status,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn,
    ROUND(SUM(CASE WHEN asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN total_outstanding ELSE 0 END) / 1e6, 2) AS npa_outstanding_mn,
    ROUND(100.0 * SUM(CASE WHEN asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN total_outstanding ELSE 0 END) / SUM(total_outstanding), 2) AS npa_rate_pct
FROM loan_master
GROUP BY security_status
ORDER BY total_outstanding_mn DESC
"""

sql_13 = """
SELECT
    loan_account_number,
    borrower_name,
    maturity_date,
    ROUND(total_outstanding / 1e6, 2) AS outstanding_mn,
    asset_classification
FROM loan_master
WHERE julianday(maturity_date) - julianday('2025-09-30') BETWEEN 0 AND 90
ORDER BY maturity_date ASC
"""

sql_14 = """
SELECT
    internal_rating,
    COUNT(*) AS borrower_count,
    ROUND(AVG(pd_estimate) * 100, 2) AS avg_pd_pct
FROM borrower_rating
WHERE rating_date = '2025-09-30'
GROUP BY internal_rating
ORDER BY internal_rating
"""

sql_15 = """
SELECT
    p.loan_account_number,
    lm.borrower_name,
    p.ecl_amount,
    p.provision_held,
    ROUND(p.ecl_amount - p.provision_held, 2) AS shortfall_usd
FROM provisioning p
JOIN loan_master lm ON p.loan_account_number = lm.loan_account_number
WHERE p.reporting_date = '2025-09-30' AND p.provision_held < p.ecl_amount
ORDER BY shortfall_usd DESC
"""

verified_query_library = {
    'VQ1': {'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors', 'sql': sql_1},
    'VQ2': {'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)', 'sql': sql_2},
    'VQ3': {'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter', 'sql': sql_3},
    'VQ4': {'description': 'Average provision coverage ratio by sector for the latest reporting quarter', 'sql': sql_4},
    'VQ5': {'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level', 'sql': sql_5},
    'VQ6': {'description': 'Top 5 largest exposures aggregated at the business group level', 'sql': sql_6},
    'VQ7': {'description': 'All overdue loan accounts with their days past due and asset classification', 'sql': sql_7},
    'VQ8': {'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio', 'sql': sql_8},
    'VQ9': {'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle', 'sql': sql_9},
    'VQ10': {'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time', 'sql': sql_10},
    'VQ11': {'description': 'Exposure and NPA breakdown between regulator-sensitive and non-sensitive sectors', 'sql': sql_11},
    'VQ12': {'description': 'Secured vs unsecured exposure with NPA rate comparison', 'sql': sql_12},
    'VQ13': {'description': 'Loans maturing within the next 90 days, for rollover and refinancing risk monitoring', 'sql': sql_13},
    'VQ14': {'description': 'Borrower count and average probability of default by internal credit rating grade', 'sql': sql_14},
    'VQ15': {'description': 'Loans where provision held is less than expected credit loss, indicating a provisioning shortfall', 'sql': sql_15},
}

# ---------------------------------------------------------------------------
# Pipeline tools (identical logic to the notebook)
# ---------------------------------------------------------------------------

def classify_intent(user_question, query_library):
    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    # Optimized prompt from the lightweight optimization loop (0.80 -> 1.00).
    # Built with .replace() rather than an f-string because the OUTPUT section
    # contains literal JSON braces.
    classification_prompt_template = """You are an intent router for a credit-risk analytics query engine at a commercial bank.

Given a user's natural-language question, decide whether it can be fully answered by one of the pre-approved VERIFIED QUERY templates below, or whether it requires a freshly GENERATED SQL query.

Choose "verified" ONLY if a template matches the question's metric, grouping, and level of aggregation. If the question asks for a filter, calculation, or combination that no template covers, choose "generated" instead of forcing a partial match. 

For example, if the question is about the total amount in a specific category and the template covers that category, select "verified." However, if the question involves a breakdown or a specific condition not covered by any template, select "generated."

### USER QUESTION

{user_question}

### AVAILABLE VERIFIED QUERY TEMPLATES

{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{"route": "verified" or "generated", "query_id": "VQ1" ... "VQ15" or null, "match_reason": "one short sentence"} 
Do not include any other text."""

    classification_prompt = (
        classification_prompt_template
        .replace('{user_question}', user_question)
        .replace('{library_descriptions}', library_descriptions)
    )

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    generation_prompt = f"""
You are a SQL analyst writing a single SQLite-compatible, READ-ONLY SQL query for a
commercial lending credit-risk database.

Rules:
- Use ONLY the tables and columns listed in the schema below.
- Write exactly one SELECT (or WITH ... SELECT) statement. No DDL/DML keywords
  (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, REPLACE, ATTACH) are allowed.
- Do not include a trailing semicolon, markdown code fences, or any commentary - return SQL only.
- Use explicit table aliases and JOIN conditions.
- Convert monetary amounts to millions (divide by 1e6 and ROUND to 2 decimals) unless the
  question clearly implies a different unit.
- Apply the NPA definition and the latest reporting_date / rating_date given in the schema
  whenever the question involves NPA status, IFRS 9 staging, or ratings.

### USER QUESTION
{user_question}

{schema_context}
"""
    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    result = {'passed': False, 'failed_check': None, 'details': '', 'relevance_confidence': None}

    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    sqlparse.parse(candidate_sql)[0]

    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage - judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
You are reviewing a candidate SQL query for correctness before it is executed against a
credit-risk database, on behalf of a bank's analytics team.

Assess whether the SQL correctly and completely answers the user's question - including
using the right tables, the right metric/aggregation, and correct date, NPA, or staging
filter logic where relevant.

{track_context}

{user_question}

{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    retry_prompt = f"""
You are a SQL analyst correcting a SQLite query that failed validation.

Rewrite it as a single, READ-ONLY (SELECT/WITH only) SQL query that preserves the original
user intent and directly fixes the reported validation error. Use only tables and columns
present in the schema below. Return ONLY the corrected SQL - no markdown fences, no commentary,
no trailing semicolon.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}
"""
    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    result = {'dataframe': None, 'reasonable': True, 'warnings': []}
    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    if df.empty:
        result['warnings'].append('Query returned an empty result')
    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')
    if len(result['warnings']) > 2:
        result['reasonable'] = False
    return result


def generate_response(user_question, dataframe, route, query_id=None):
    response_prompt = f"""
You are a credit-risk analyst summarizing query results for a business user at a bank who
does not read SQL.

Write a concise 2-4 sentence narrative that directly answers the user's question using EXACT
figures from the data below. Reference only the rows/values relevant to what was asked - do
not restate the entire table or describe its structure. Use plain business language.

### USER QUESTION
{user_question}

{dataframe.to_string()}
"""
    return llm.invoke(response_prompt).content.strip()


def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=False):
    log = {
        'user_question': user_question, 'route': None, 'query_id': None, 'match_reason': None,
        'candidate_sql': None, 'gate_result': None, 'retry_used': False, 'escalated': False,
        'executed_sql': None, 'row_count': None, 'confidence': None, 'narrative': None,
    }

    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if not gate['passed'] and log['route'] == 'generated':
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        return {'log': log, 'dataframe': None, **log}

    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative
    log['confidence'] = gate.get('relevance_confidence')

    return {'log': log, 'dataframe': df, **log}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This PoC routes routine questions to pre-approved SQL templates and generates "
        "read-only SQL for novel questions, with validation, retry, and human escalation."
    )
    st.subheader("Verified Query Library")
    for qid, entry in verified_query_library.items():
        st.caption(f"**{qid}**: {entry['description']}")

question = st.text_input(
    "Ask a portfolio question",
    placeholder="e.g. How much of our book is in real estate, and how much of that is non-performing?",
)

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if st.button("Submit", type="primary") and question:
    with st.spinner("Routing, validating, and executing query..."):
        result = run_pipeline(question, conn, verified_query_library, database_schema)
    st.session_state.audit_log.append(result)

    if result['escalated']:
        st.warning(result['narrative'])
    else:
        st.subheader("Answer")
        st.write(result['narrative'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Route", result['route'])
        col2.metric("Confidence", f"{result['confidence']:.2f}" if isinstance(result['confidence'], float) else result['confidence'])
        col3.metric("Rows Returned", result['row_count'])

        with st.expander("SQL query used"):
            st.code(result['executed_sql'], language="sql")

        with st.expander("Raw data returned"):
            st.dataframe(result['dataframe'], use_container_width=True)

if st.session_state.audit_log:
    st.divider()
    st.subheader("Audit Trail (this session)")
    audit_rows = [
        {
            "Question": r['user_question'],
            "Route": r['route'],
            "Query ID": r['query_id'],
            "Escalated": r['escalated'],
            "Confidence": r['confidence'],
            "Rows": r['row_count'],
        }
        for r in st.session_state.audit_log
    ]
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
