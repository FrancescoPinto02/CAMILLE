import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Connection Pool Configuration
pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="camille_db_pool",
    pool_size=5,
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST")
)


def get_connection():
    try:
        conn = pool.get_connection()
        if conn.is_connected():
            print("Connection obtained from pool")
            return conn
        else:
            print("Failed to obtain connection")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
