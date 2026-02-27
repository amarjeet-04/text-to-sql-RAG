import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

HOST     = os.getenv("DB_HOST", "localhost")
PORT     = os.getenv("DB_PORT", "1433")
USER     = os.getenv("DB_USER", "sa")
PASSWORD = os.getenv("DB_PASSWORD", "")
DATABASE = os.getenv("DB_NAME", os.getenv("DB_DATABASE", "master"))

cs = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={HOST},{PORT};"
    f"DATABASE={DATABASE};"
    f"UID={USER};PWD={PASSWORD};"
    f"Encrypt=yes;"
    f"TrustServerCertificate=yes;"
    f"Connection Timeout=10;"
)

cn = pyodbc.connect(cs)
print("connected", cn.cursor().execute("SELECT 1").fetchone())
cn.close()
