from flask import Flask, request, jsonify, session, render_template, send_file, redirect, url_for
from flask_cors import CORS
import sqlite3
import hashlib
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import json

# Import functions from database.py and ml_model.py
from database import init_db
from ml_model import load_models, generate_personalized_diet_plan

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'healthyplate_secret_key_2025'
CORS(app)

# Helper function to calculate BMI and body type
def calculate_bmi_and_body_type(height_cm, weight_kg):
    """Calculate BMI and determine body type"""
    bmi = weight_kg / ((height_cm / 100) ** 2)
    
    if bmi < 18.5:
        body_type = "Underweight"
    elif bmi < 25:
        body_type = "Normal weight"
    elif bmi < 30:
        body_type = "Overweight"
    else:
        body_type = "Obese"
    
    return round(bmi, 2), body_type

# Routes
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/login')
def login_page():
  return render_template('login.html')

@app.route('/register')
def register_page():
  return render_template('register.html')

@app.route('/adminlogin')
def admin_login_page():
  return render_template('adminlogin.html')

# New route for the diet form, accessible only after user login
@app.route('/diet-form')
def diet_form_page():
  if 'user_id' not in session:
      return redirect(url_for('login_page'))
  return render_template('diet-form.html') # Renamed from healthyplate.html

@app.route('/admin')
def admin_dashboard():
  if 'admin_id' not in session:
      return redirect(url_for('admin_login_page'))
  return render_template('admin.html')

@app.route('/api/auth-status')
def auth_status():
  if 'user_id' in session:
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      cursor.execute('SELECT username, first_name, last_name FROM users WHERE id = ?', (session['user_id'],))
      user = cursor.fetchone()
      conn.close()
      
      if user:
          return jsonify({
              'authenticated': True,
              'user_type': 'user',
              'user': {
                  'id': session['user_id'],
                  'username': user[0],
                  'first_name': user[1],
                  'last_name': user[2]
              }
          })
  elif 'admin_id' in session:
      return jsonify({
          'authenticated': True,
          'user_type': 'admin',
          'admin': {
              'id': session['admin_id'],
              'username': session.get('admin_username', 'admin')
          }
      })
  
  return jsonify({'authenticated': False})

@app.route('/api/register', methods=['POST'])
def register():
  try:
      data = request.json
      username = data.get('username')
      email = data.get('email')
      password = data.get('password')
      first_name = data.get('first_name')
      last_name = data.get('last_name')
      
      if not all([username, email, password, first_name, last_name]):
          return jsonify({'success': False, 'message': 'All fields are required'})
      
      hashed_password = hashlib.sha256(password.encode()).hexdigest()
      
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      
      try:
          cursor.execute('''
              INSERT INTO users (username, email, password, first_name, last_name)
              VALUES (?, ?, ?, ?, ?)
          ''', (username, email, hashed_password, first_name, last_name))
          conn.commit()
          return jsonify({'success': True, 'message': 'Registration successful'})
      except sqlite3.IntegrityError:
          return jsonify({'success': False, 'message': 'Username or email already exists'})
      finally:
          conn.close()
          
  except Exception as e:
      return jsonify({'success': False, 'message': str(e)})

@app.route('/api/login', methods=['POST'])
def login():
  try:
      data = request.json
      username_or_email = data.get('username')
      password = data.get('password')
      
      if not username_or_email or not password:
          return jsonify({'success': False, 'message': 'Username/email and password are required'})
      
      hashed_password = hashlib.sha256(password.encode()).hexdigest()
      
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      
      cursor.execute('''
          SELECT id, username, first_name, last_name FROM users 
          WHERE (username = ? OR email = ?) AND password = ?
      ''', (username_or_email, username_or_email, hashed_password))
      
      user = cursor.fetchone()
      conn.close()
      
      if user:
          session['user_id'] = user[0]
          session['username'] = user[1]
          session['user_type'] = 'user'
          return jsonify({
              'success': True, 
              'message': 'Login successful',
              'user': {
                  'id': user[0],
                  'username': user[1],
                  'first_name': user[2],
                  'last_name': user[3]
              }
          })
      else:
          return jsonify({'success': False, 'message': 'Invalid credentials'})
          
  except Exception as e:
      return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
  try:
      data = request.json
      username = data.get('username')
      password = data.get('password')
      
      if not username or not password:
          return jsonify({'success': False, 'message': 'Username and password are required'})
      
      hashed_password = hashlib.sha256(password.encode()).hexdigest()
      
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      
      cursor.execute('SELECT id, username FROM admins WHERE username = ? AND password = ?', 
                    (username, hashed_password))
      admin = cursor.fetchone()
      conn.close()
      
      if admin:
          session['admin_id'] = admin[0]
          session['admin_username'] = admin[1]
          session['user_type'] = 'admin'
          return jsonify({'success': True, 'message': 'Admin login successful'})
      else:
          return jsonify({'success': False, 'message': 'Invalid admin credentials'})
          
  except Exception as e:
      return jsonify({'success': False, 'message': str(e)})

@app.route('/api/logout', methods=['POST'])
def logout():
  session.clear()
  return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/diet-recommendation', methods=['POST'])
def diet_recommendation():
  try:
      if 'user_id' not in session:
          return jsonify({'success': False, 'message': 'Please login first'})
      
      data = request.json
      
      # Extract and convert units
      age = data.get('age')
      weight = data.get('weight')
      height = data.get('height')
      waist_size = data.get('waist_size')
      hip_size = data.get('hip_size')
      
      height_unit = data.get('height_unit', 'cm')
      weight_unit = data.get('weight_unit', 'kg')

      # Convert height to cm
      if height_unit == 'ft_in':
          feet = data.get('height_ft')
          inches = data.get('height_in')
          height_cm = (feet * 30.48) + (inches * 2.54)
      else: # 'cm'
          height_cm = height

      # Convert weight to kg
      if weight_unit == 'lbs':
          weight_kg = weight * 0.453592
      else: # 'kg'
          weight_kg = weight
      
      # Validation with new ranges
      if not (1 <= age <= 120):
          return jsonify({'success': False, 'message': 'Age must be between 1 and 120 years'})
      if not (1 <= weight_kg <= 300): # Assuming 300kg is the max for kg
          return jsonify({'success': False, 'message': 'Weight must be between 1 and 300 kg'})
      if not (50 <= height_cm <= 250): # Assuming 50cm is min for cm
          return jsonify({'success': False, 'message': 'Height must be between 50 and 250 cm'})
      if not (50 <= waist_size <= 200):
          return jsonify({'success': False, 'message': 'Waist size must be between 50 and 200 cm'})
      if not (60 <= hip_size <= 200):
          return jsonify({'success': False, 'message': 'Hip size must be between 60 and 200 cm'})
      
      # Calculate BMI and body type
      bmi, body_type = calculate_bmi_and_body_type(height_cm, weight_kg)
      
      # Load models
      models = load_models()
      
      if models is None:
          return jsonify({'success': False, 'message': 'ML models not available'})
      
      # Prepare user data for ML model
      user_data = {
          'gender': data.get('gender'),
          'age': age,
          'height': height_cm, # Use converted height
          'weight': weight_kg, # Use converted weight
          'activity_level': data.get('activity_level', 'moderate'),
          'food_preference': data.get('food_preference'),
          'fitness_goal': data.get('fitness_goal')
      }
      
      # Generate personalized diet plan
      diet_plan = generate_personalized_diet_plan(models, user_data)
      
      # Save to database
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      
      cursor.execute('''
          INSERT INTO diet_results 
          (user_id, name, age, gender, height, weight, waist_size, hip_size, 
           fitness_goal, food_preference, activity_level, bmi, body_type,
           daily_calorie_target, protein_target, carbs_target, fat_target, 
           fiber_target, sugar_target, sodium_target,
           breakfast, lunch, dinner, snack)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ''', (session['user_id'], data.get('name'), age, data.get('gender'), 
            height_cm, weight_kg, waist_size, hip_size, data.get('fitness_goal'),
            data.get('food_preference'), data.get('activity_level', 'moderate'),
            bmi, body_type, diet_plan['daily_calories'], diet_plan['protein'],
            diet_plan['carbohydrates'], diet_plan['fat'], diet_plan['fiber'],
            diet_plan['sugar'], diet_plan['sodium'],
            diet_plan['breakfast'], diet_plan['lunch'], diet_plan['dinner'], diet_plan['snack']))
      
      result_id = cursor.lastrowid
      conn.commit()
      conn.close()
      
      return jsonify({
          'success': True,
          'result_id': result_id,
          'bmi': bmi,
          'body_type': body_type,
          'diet_plan': diet_plan
          # Model accuracy is no longer sent to frontend
      })
      
  except Exception as e:
      return jsonify({'success': False, 'message': str(e)})
  
  #download diet plan

@app.route('/api/download-diet-plan/<int:result_id>')
def download_diet_plan(result_id):
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Please login first'})

        conn = sqlite3.connect('healthyplate.db')
        cursor = conn.cursor()

        # Fetch the diet result
        cursor.execute('''
            SELECT * FROM diet_results 
            WHERE id = ? AND user_id = ?
        ''', (result_id, session['user_id']))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return jsonify({'success': False, 'message': 'Diet plan not found'})

        # Build the PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("HealthyPlate - Personalized Diet Plan", styles['Title']))
        story.append(Spacer(1, 0.2 * inch))

        # User info
        user_info = f"""
        <b>Name:</b> {result[2]}<br/>
        <b>Age:</b> {result[3]}<br/>
        <b>Gender:</b> {result[4].title()}<br/>
        <b>Height:</b> {round(result[5], 2)} cm<br/>
        <b>Weight:</b> {round(result[6], 2)} kg<br/>
        <b>Waist Size:</b> {result[7]} cm<br/>
        <b>Hip Size:</b> {result[8]} cm<br/>
        <b>Fitness Goal:</b> {result[9].replace('_', ' ').title()}<br/>
        <b>Food Preference:</b> {result[10].replace('_', ' ').title()}<br/>
        <b>Activity Level:</b> {result[11].replace('_', ' ').title()}<br/>
        <b>BMI:</b> {result[12]} ({result[13]})<br/>
        """
        story.append(Paragraph(user_info, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Nutritional Targets
        nutrition_info = f"""
        <b>Daily Nutritional Targets:</b><br/>
        <b>Calories:</b> {result[14]} kcal<br/>
        <b>Protein:</b> {result[15]}g<br/>
        <b>Carbs:</b> {result[16]}g<br/>
        <b>Fat:</b> {result[17]}g<br/>
        <b>Fiber:</b> {result[18]}g<br/>
        <b>Sugar:</b> {result[19]}g<br/>
        <b>Sodium:</b> {result[20]}mg
        """
        story.append(Paragraph(nutrition_info, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Diet plan meals
        story.append(Paragraph("Recommended Diet Plan", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))

        meals = {
            'Breakfast': result[21],
            'Lunch': result[22],
            'Dinner': result[23],
            'Snack': result[24]
        }

        for meal, desc in meals.items():
            meal_text = f"<b>{meal}:</b> {desc if desc else 'Not provided'}"
            story.append(Paragraph(meal_text, styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

        doc.build(story)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'healthyplate_diet_plan_{result_id}.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({'success': False, 'message': f'Error generating PDF: {str(e)}'})



# Admin routes

@app.route('/api/admin/users', methods=['GET'])
def get_users():
  try:
      if 'admin_id' not in session:
          return jsonify({'success': False, 'message': 'Admin access required'})
      
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      
      cursor.execute('SELECT id, username, email, first_name, last_name, created_at FROM users ORDER BY created_at DESC')
      users = cursor.fetchall()
      conn.close()
      
      users_list = []
      for user in users:
          users_list.append({
              'id': user[0],
              'username': user[1],
              'email': user[2],
              'first_name': user[3],
              'last_name': user[4],
              'created_at': user[5]
          })
      
      return jsonify({'success': True, 'users': users_list})
      
  except Exception as e:
      return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/users', methods=['POST'])
def add_user():
    try:
        if 'admin_id' not in session:
            return jsonify({'success': False, 'message': 'Admin access required'})
        
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        
        if not all([username, email, password, first_name, last_name]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect('healthyplate.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password, first_name, last_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, hashed_password, first_name, last_name))
            conn.commit()
            return jsonify({'success': True, 'message': 'User added successfully'})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'message': 'Username or email already exists'})
        finally:
            conn.close()
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        if 'admin_id' not in session:
            return jsonify({'success': False, 'message': 'Admin access required'})
        
        data = request.json
        username = data.get('username')
        email = data.get('email')
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        password = data.get('password') # Optional: only update if provided
        
        conn = sqlite3.connect('healthyplate.db')
        cursor = conn.cursor()
        
        update_fields = []
        params = []
        
        if username:
            update_fields.append("username = ?")
            params.append(username)
        if email:
            update_fields.append("email = ?")
            params.append(email)
        if first_name:
            update_fields.append("first_name = ?")
            params.append(first_name)
        if last_name:
            update_fields.append("last_name = ?")
            params.append(last_name)
        if password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            update_fields.append("password = ?")
            params.append(hashed_password)
        
        if not update_fields:
            return jsonify({'success': False, 'message': 'No fields provided for update'})
            
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
        params.append(user_id)
        
        try:
            cursor.execute(query, tuple(params))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({'success': False, 'message': 'User not found or no changes made'})
            return jsonify({'success': True, 'message': 'User updated successfully'})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'message': 'Username or email already exists'})
        finally:
            conn.close()
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
  try:
      if 'admin_id' not in session:
          return jsonify({'success': False, 'message': 'Admin access required'})
      
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      
      cursor.execute('DELETE FROM diet_results WHERE user_id = ?', (user_id,))
      cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
      
      conn.commit()
      conn.close()
      
      return jsonify({'success': True, 'message': 'User deleted successfully'})
      
  except Exception as e:
      return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/diet-results', methods=['GET'])
def get_diet_results():
  try:
      if 'admin_id' not in session:
          return jsonify({'success': False, 'message': 'Admin access required'})
      
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      
      cursor.execute('''
          SELECT dr.*, u.username, u.email 
          FROM diet_results dr
          JOIN users u ON dr.user_id = u.id
          ORDER BY dr.created_at DESC
      ''')
      results = cursor.fetchall()
      conn.close()
      
      results_list = []
      for result in results:
          results_list.append({
              'id': result[0],
              'user_id': result[1],
              'name': result[2],
              'age': result[3],
              'gender': result[4],
              'height': result[5],
              'weight': result[6],
              'bmi': result[11],
              'body_type': result[12],
              'fitness_goal': result[8],
              'food_preference': result[9],
              'daily_calories': result[13],
              'protein': result[14],
              'carbs': result[15],
              'fat': result[16],
              'fiber': result[17],
              'sugar': result[18],
              'sodium': result[19],
              'breakfast': result[20],
              'lunch': result[21],
              'dinner': result[22],
              'snack': result[23],
              'created_at': result[24],
              'username': result[25],
              'email': result[26]
          })
      
      return jsonify({'success': True, 'results': results_list})
      
  except Exception as e:
      return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/diet-results/<int:result_id>', methods=['DELETE'])
def delete_diet_result(result_id):
  try:
      if 'admin_id' not in session:
          return jsonify({'success': False, 'message': 'Admin access required'})
      
      conn = sqlite3.connect('healthyplate.db')
      cursor = conn.cursor()
      
      cursor.execute('DELETE FROM diet_results WHERE id = ?', (result_id,))
      conn.commit()
      conn.close()
      
      return jsonify({'success': True, 'message': 'Diet result deleted successfully'})
      
  except Exception as e:
      return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
  init_db() # Initialize database
  load_models() # Load/train ML models on startup
  app.run(debug=True, host='0.0.0.0', port=5000)
