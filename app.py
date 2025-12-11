import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd

# -----------------------------
# BASIC PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="MySQL Database Viewer",
    layout="wide"
)

st.title("📊 MySQL Database Viewer")
st.write(
    "This app connects to your MySQL server and shows table data. "
    "Select a database and a table from the sidebar."
)

# -----------------------------
# LOAD DB SETTINGS FROM SECRETS
# -----------------------------
if "mysql" not in st.secrets:
    st.error(
        "❌ `mysql` section not found in secrets.\n\n"
        "Please add it in your secrets as:\n\n"
        "[mysql]\n"
        "host = \"...\"\n"
        "port = 3306\n"
        "user = \"...\"\n"
        "password = \"...\"\n"
        "# database = \"optional_db_name\""
    )
    st.stop()

DB_CONFIG = st.secrets["mysql"]

HOST = DB_CONFIG.get("host")
PORT = DB_CONFIG.get("port", 3306)
USER = DB_CONFIG.get("user")
PASSWORD = DB_CONFIG.get("password")
DB_NAME = DB_CONFIG.get("database", None)  # optional

missing_fields = [k for k in ["host", "user", "password"] if k not in DB_CONFIG]
if missing_fields:
    st.error(
        f"❌ Missing fields in [mysql] secrets: {', '.join(missing_fields)}\n\n"
        "Please update your secrets and rerun the app."
    )
    st.stop()


# -----------------------------
# CONNECTION HELPERS
# -----------------------------
@st.cache_resource(show_spinner=False)
def create_server_connection():
    """
    Create a connection to the MySQL server.
    Cached by Streamlit so it isn't recreated on every rerun.
    """
    try:
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DB_NAME if DB_NAME else None,
            connection_timeout=10,
        )
        return conn
    except Error:
        return None


def get_connection_or_fail():
    """
    Ensure we have a live MySQL connection.
    If not, show a clear error and stop the app.
    """
    conn = create_server_connection()

    if conn is None or not conn.is_connected():
        try:
            conn = mysql.connector.connect(
                host=HOST,
                port=PORT,
                user=USER,
                password=PASSWORD,
                database=DB_NAME if DB_NAME else None,
                connection_timeout=10,
            )
        except Error as e:
            st.error(
                "❌ Could not connect to the MySQL server.\n\n"
                "Please check:\n"
                "- That the server is running\n"
                "- Host/port/user/password are correct\n"
                "- That remote connections are allowed\n"
                "- Any firewall / security group rules\n\n"
                f"Technical error: `{e}`"
            )
            st.stop()

    if not conn.is_connected():
        st.error("❌ MySQL connection is not available even after retrying.")
        st.stop()

    return conn


@st.cache_data(show_spinner=False)
def list_databases_cached():
    """
    Cached wrapper to list databases (uses a fresh lightweight connection).
    """
    try:
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            connection_timeout=10,
        )
        system_dbs = {"information_schema", "mysql", "performance_schema", "sys"}
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        dbs = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return [db for db in dbs if db not in system_dbs]
    except Error:
        return []


def get_databases():
    """
    If DB_NAME is set, we work in single-DB mode.
    Otherwise we list all non-system DBs.
    """
    if DB_NAME:
        return [DB_NAME]
    return list_databases_cached()


def use_database(conn, db_name):
    """
    Set the active database for the existing connection.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE `{db_name}`;")
        cursor.close()
    except Error as e:
        st.error(f"Error selecting database `{db_name}`: `{e}`")
        st.stop()


@st.cache_data(show_spinner=False)
def get_tables_cached(db_name: str):
    """
    Cached list of tables for a given database.
    Uses a fresh short-lived connection.
    """
    try:
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=db_name,
            connection_timeout=10,
        )
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return tables
    except Error as e:
        st.error(f"Error while listing tables for `{db_name}`: `{e}`")
        return []


@st.cache_data(show_spinner=True)
def fetch_table_data_cached(
    db_name: str,
    table_name: str,
    limit: int,
    offset: int,
):
    """
    Cached fetch for a slice of table data.
    This keeps repeated views fast.
    """
    try:
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=db_name,
            connection_timeout=10,
        )
        cursor = conn.cursor()
        query = f"SELECT * FROM `{table_name}` LIMIT %s OFFSET %s"
        cursor.execute(query, (limit, offset))
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        df = pd.DataFrame(rows, columns=col_names)
        return df
    except Error as e:
        st.error(f"Error reading table `{table_name}`: `{e}`")
        return pd.DataFrame()


# -----------------------------
# MAIN APP LOGIC
# -----------------------------

conn = get_connection_or_fail()

with st.sidebar:
    st.header("⚙️ Connection")
    st.markdown(
        f"**Host:** `{HOST}`  \n"
        f"**Port:** `{PORT}`  \n"
        f"**User:** `{USER}`"
    )

    st.header("🗄️ Database selection")

    databases = get_databases()
    if not databases:
        st.error("No databases found or you don't have permission to list them.")
        st.stop()

    if len(databases) == 1:
        selected_db = databases[0]
        st.markdown(f"Using database: **`{selected_db}`**")
    else:
        selected_db = st.selectbox("Choose a database:", databases)

    st.divider()
    st.header("📂 Table & rows")

    # Get tables for selected DB
    tables = get_tables_cached(selected_db)
    if not tables:
        st.error(f"No tables found in database `{selected_db}`.")
        st.stop()

    selected_table = st.selectbox("Choose a table:", tables)

    # How many rows per page
    page_size = st.number_input(
        "Rows per page",
        min_value=50,
        max_value=2000,     # hard cap to avoid crashing
        value=200,
        step=50,
        help="Number of rows to load at once. Larger values may be slower."
    )

    # Which page
    page_number = st.number_input(
        "Page number (starting from 1)",
        min_value=1,
        value=1,
        step=1,
        help="Use this to move through the table in chunks."
    )

    # Calculate offset for SQL
    offset = int((page_number - 1) * page_size)

# Switch DB on the main connection (mainly to verify it works)
use_database(conn, selected_db)

st.subheader(f"📚 Database: `{selected_db}`")
st.markdown(f"### 📂 Table: `{selected_table}`")

st.caption(
    f"Showing **page {page_number}**, up to **{int(page_size)}** rows "
    f"(offset {offset})."
)

# Fetch only the selected slice of the selected table
df = fetch_table_data_cached(
    db_name=selected_db,
    table_name=selected_table,
    limit=int(page_size),
    offset=offset,
)

if df.empty:
    st.info("No data returned (or failed to load data) for this slice.")
else:
    st.dataframe(df, use_container_width=True)

# Try to close the long-lived connection politely
try:
    if conn is not None and conn.is_connected():
        conn.close()
except Exception:
    pass
