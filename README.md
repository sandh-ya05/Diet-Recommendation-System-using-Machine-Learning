# HealthyPlate - Diet Recommendation System

A comprehensive diet recommendation system that uses machine learning and real nutrition data to provide personalized diet plans based on user characteristics and goals.

## Features

### User Features
- User registration and authentication (username/email + password)
- Multi-step health assessment form with validation
- BMI calculation and body type classification
- Personalized diet recommendations using real nutrition dataset
- Detailed nutritional targets (calories, protein, carbs, fat, fiber, sugar, sodium)
- PDF download of complete diet plans
- View the past history
- Responsive design with proper validation ranges

### Admin Features
- Admin authentication and dashboard
- View and manage users
- View detailed user diet results with nutritional information
- Real-time data management

### Technical Features
- Flask backend with comprehensive REST API
- SQLite database with proper relationships
- Machine Learning using KNN and Random Forest
- Integration with actual Food & Nutrition CSV dataset
- Session handling and authentication
- Input validation with proper ranges
- Responsive frontend using Bootstrap and Tailwind CSS

## Dataset Integration

The system now uses your actual Food & Nutrition CSV dataset with the following features:
- **Real meal suggestions**: Breakfast, lunch, dinner, and snack recommendations
- **Accurate nutritional data**: Daily calorie targets, protein, carbohydrates, fat, fiber, sugar, sodium
- **Personalized matching**: Based on gender, activity level, dietary preference, and BMI
- **ML-powered predictions**: Using actual nutrition patterns from the dataset

## Installation

1. Clone the repository
2. Install Python dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Run the application:
   \`\`\`bash
   python app.py
   \`\`\`

4. Access the application at `http://localhost:5000`

## Default Credentials

### Admin Login
- Username: `admin`
- Password: `admin123`


## Machine Learning Models

The system uses two ML algorithms trained on your actual nutrition dataset:
1. **K-Nearest Neighbors (KNN)**: For pattern recognition in similar user profiles
2. **Random Forest**: For more accurate diet category and nutritional target prediction

## Dataset Features Used

- **Gender**: Male/Female
- **Activity Level**: Sedentary, Light, Moderate, Active, Very Active
- **Dietary Preference**: Vegetarian, Vegan, Non-Vegetarian, Pescatarian, Gluten-Free
- **Nutritional Targets**: Daily calories, protein, carbohydrates, fat, fiber, sugar, sodium
- **Meal Suggestions**: Actual breakfast, lunch, dinner, and snack recommendations

## API Endpoints

### Authentication
- `POST /api/register` - User registration
- `POST /api/login` - User login
- `POST /api/admin/login` - Admin login
- `POST /api/logout` - Logout
- `GET /api/auth-status` - Check authentication status

### Diet Recommendation
- `POST /api/diet-recommendation` - Get personalized diet plan
- `GET /api/download-diet-plan/<id>` - Download diet plan as PDF

### Admin Endpoints
- `GET /api/admin/meals` - Get all admin-added meals
- `POST /api/admin/meals` - Add new meal
- `PUT /api/admin/meals/<id>` - Update meal
- `DELETE /api/admin/meals/<id>` - Delete meal
- `GET /api/admin/users` - Get all users
- `DELETE /api/admin/users/<id>` - Delete user
- `GET /api/admin/diet-results` - Get all diet results
- `DELETE /api/admin/diet-results/<id>` - Delete diet result

## Database Schema

### Users Table
- id, username, email, password, first_name, last_name, created_at

### Admins Table
- id, username, password, created_at


### Diet Results Table
- id, user_id, name, age, gender, height, weight
- fitness_goal, food_preference, activity_level, bmi, body_type
- daily_calorie_target, protein_target, carbs_target, fat_target, fiber_target, sugar_target, sodium_target
- breakfast, lunch, dinner, snack, created_at

## Security Features

- Password hashing using SHA-256
- Session-based authentication with proper logout handling
- Input validation and sanitization
- CSRF protection
- Proper error handling and user feedback

## Responsive Design

The application is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones

## Contributors

- Satkar Shrestha
- Sandhya Gharti

## License

© 2025 All rights reserved.
\`\`\`

```python file="requirements.txt"
Flask==2.3.3
Flask-CORS==4.0.0
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
reportlab==4.0.4
requests==2.31.0
