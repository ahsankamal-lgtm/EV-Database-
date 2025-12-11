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
# CONNECTION SETTINGS
# -----------------------------
HOST = "3.12.253.224"
PORT = 3306
USER = "traccar_user"
PASSWORD = "Wavetec@123"


# -----------------------------
# CONNECTION HELPERS
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_server_connection():
    """
    Connect to the MySQL server (no database selected yet).
    """
    try:
        conn = mysql.connector.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD
        )
        return conn
    except Error as e:
        st.error(f"Error connecting to MySQL server: {e}")
        return None


def get_databases(conn):
    """
    Return a list of databases on the server, excluding system databases.
    """
    if conn is None:
        return []

    system_dbs = {"information_schema", "mysql", "performance_schema", "sys"}
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")
    dbs = [row[0] for row in cursor.fetchall()]
    cursor.close()
    # Filter out system DBs
    return [db for db in dbs if db not in system_dbs]


def use_database(conn, db_name):
    """
    Set the active database for the existing connection.
    """
    cursor = conn.cursor()
    cursor.execute(f"USE `{db_name}`;")
    cursor.close()


def get_tables(conn):
    """
    Return a list of tables in the currently selected database.
    """
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tables


def fetch_table_data(conn, table_name, limit=500):
    """
    Fetch up to `limit` rows from a table.
    """
    cursor = conn.cursor()
    query = f"SELECT * FROM `{table_name}` LIMIT %s"
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    cursor.close()
    df = pd.DataFrame(rows, columns=col_names)
    return df


# -----------------------------
# MAIN APP LOGIC
# -----------------------------
conn = get_server_connection()

if conn is None:
    st.stop()

with st.sidebar:
    st.header("⚙️ Connection")
    st.markdown(f"**Host:** `{HOST}`  \n**Port:** `{PORT}`  \n**User:** `{USER}`")

    st.header("🗄️ Database selection")
    databases = get_databases(conn)

    if not databases:
        st.error("No non-system databases found on this server.")
        st.stop()

    selected_db = st.selectbox("Choose a database:", databases)

    row_limit = st.number_input(
        "Rows to show per table",
        min_value=10,
        max_value=5000,
        value=500,
        step=50,
        help="Maximum number of rows to load for each table."
    )

# Use the selected database
use_database(conn, selected_db)

# Get all tables
tables = get_tables(conn)

if not tables:
    st.warning(f"No tables found in database `{selected_db}`.")
    st.stop()

st.subheader(f"📚 Database: `{selected_db}`")
st.write(f"Found **{len(tables)}** tables.")

# Create one tab per table
tabs = st.tabs(tables)

for table_name, tab in zip(tables, tabs):
    with tab:
        st.markdown(f"### 📂 Table: `{table_name}`")
        try:
            df = fetch_table_data(conn, table_name, limit=row_limit)
            st.write(f"Showing up to **{len(df)}** rows.")
            st.dataframe(df, use_container_width=True)
        except Error as e:
            st.error(f"Error reading table `{table_name}`: {e}")

# Close the connection when Streamlit session ends
# (Streamlit will handle cleanup, but this is safe)
if conn.is_connected():
    conn.close()
