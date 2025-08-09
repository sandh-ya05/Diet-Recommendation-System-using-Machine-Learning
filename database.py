import sqlite3
import hashlib
from datetime import datetime

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    conn = sqlite3.connect('healthyplate.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Admin table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Meals table (for admin-added meals)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT NOT NULL,
            calories INTEGER NOT NULL,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (created_by) REFERENCES admins (id)
        )
    ''')
    
    # User diet results table - UPDATED: Removed waist_size and hip_size
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diet_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            height REAL NOT NULL,
            weight REAL NOT NULL,
            fitness_goal TEXT NOT NULL,
            food_preference TEXT NOT NULL,
            activity_level TEXT DEFAULT 'moderate',
            bmi REAL NOT NULL,
            body_type TEXT NOT NULL,
            daily_calorie_target INTEGER NOT NULL,
            protein_target REAL NOT NULL,
            carbs_target REAL NOT NULL,
            fat_target REAL NOT NULL,
            fiber_target REAL NOT NULL,
            sugar_target REAL NOT NULL,
            sodium_target REAL NOT NULL,
            breakfast TEXT NOT NULL,
            lunch TEXT NOT NULL,
            dinner TEXT NOT NULL,
            snack TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert default admin if not exists
    cursor.execute('SELECT COUNT(*) FROM admins WHERE username = ?', ('admin',))
    if cursor.fetchone()[0] == 0:
        hashed_password = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute('INSERT INTO admins (username, password) VALUES (?, ?)', 
                      ('admin', hashed_password))
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == '__main__':
    init_db()
