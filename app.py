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
    "This app connects to your MySQL server and shows the tables and their data. "
    "Use the sidebar to select a database."
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
        # We will handle the error gracefully in get_connection_or_fail()
        return None


def get_connection_or_fail():
    """
    Ensure we have a live MySQL connection.
    If not, show a clear error and stop the app.
    """
    conn = create_server_connection()

    # If connection object is None or not connected, try once to reconnect
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
                "- That remote connections (e.g. from Streamlit Cloud) are allowed\n"
                "- Any firewall / security group rules\n\n"
                f"Technical error: `{e}`"
            )
            st.stop()

    if not conn.is_connected():
        st.error("❌ MySQL connection is not available even after retrying.")
        st.stop()

    return conn


def get_databases(conn):
    """
    Return a list of databases on the server, excluding system databases.
    If DB_NAME is set (single DB mode), just return that.
    """
    if DB_NAME:
        return [DB_NAME]

    if conn is None or not conn.is_connected():
        return []

    system_dbs = {"information_schema", "mysql", "performance_schema", "sys"}

    try:
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        dbs = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return [db for db in dbs if db not in system_dbs]
    except Error as e:
        st.error(f"Error while listing databases: `{e}`")
        return []


def use_database(conn, db_name):
    """
    Set the active database for the existing connection.
    """
    if conn is None or not conn.is_connected():
        st.error("Connection lost before selecting database.")
        st.stop()

    try:
        cursor = conn.cursor()
        cursor.execute(f"USE `{db_name}`;")
        cursor.close()
    except Error as e:
        st.error(f"Error selecting database `{db_name}`: `{e}`")
        st.stop()


def get_tables(conn):
    """
    Return a list of tables in the currently selected database.
    """
    if conn is None or not conn.is_connected():
        st.error("Connection lost while trying to list tables.")
        st.stop()

    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables
    except Error as e:
        st.error(f"Error while listing tables: `{e}`")
        return []


def fetch_table_data(conn, table_name, limit=500):
    """
    Fetch up to `limit` rows from a table as a DataFrame.
    """
    if conn is None or not conn.is_connected():
        st.error(f"Connection lost while reading table `{table_name}`.")
        st.stop()

    try:
        cursor = conn.cursor()
        query = f"SELECT * FROM `{table_name}` LIMIT %s"
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        cursor.close()
        df = pd.DataFrame(rows, columns=col_names)
        return df
    except Error as e:
        st.error(f"Error reading table `{table_name}`: `{e}`")
        return pd.DataFrame()


# -----------------------------
# MAIN APP LOGIC
# -----------------------------

# 1. Get a working connection (or stop with a clear error)
conn = get_connection_or_fail()

with st.sidebar:
    st.header("⚙️ Connection")
    st.markdown(
        f"**Host:** `{HOST}`  \n"
        f"**Port:** `{PORT}`  \n"
        f"**User:** `{USER}`"
    )

    st.header("🗄️ Database selection")

    databases = get_databases(conn)

    if not databases:
        st.error("No databases found or you don't have permission to list them.")
        st.stop()

    if len(databases) == 1:
        selected_db = databases[0]
        st.markdown(f"Using database: **`{selected_db}`**")
    else:
        selected_db = st.selectbox("Choose a database:", databases)

    row_limit = st.number_input(
        "Rows to show per table",
        min_value=10,
        max_value=5000,
        value=500,
        step=50,
        help="Maximum number of rows to load for each table."
    )

# 2. Ensure the chosen database is active
use_database(conn, selected_db)

# 3. Get tables
tables = get_tables(conn)

if not tables:
    st.warning(f"No tables found in database `{selected_db}`.")
    st.stop()

st.subheader(f"📚 Database: `{selected_db}`")
st.write(f"Found **{len(tables)}** tables.")

# 4. Create tabs for each table
tabs = st.tabs(tables)

for table_name, tab in zip(tables, tabs):
    with tab:
        st.markdown(f"### 📂 Table: `{table_name}`")
        df = fetch_table_data(conn, table_name, limit=row_limit)
        if df.empty:
            st.info("No data (or failed to load data) for this table.")
        else:
            st.write(f"Showing up to **{len(df)}** rows.")
            st.dataframe(df, use_container_width=True)

# 5. (Optional) Close connection when the script finishes
try:
    if conn is not None and conn.is_connected():
        conn.close()
except Exception:
    pass
