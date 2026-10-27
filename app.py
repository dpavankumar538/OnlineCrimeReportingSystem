# app.py
# ==============================================================
# MAIN FLASK APPLICATION — Online Crime Reporting System
# ==============================================================



from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify)
from flask_mysqldb import MySQL
from functools import wraps
import bcrypt
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from config import Config
# ----------------------------------------------------------
# 1. Initialize Flask App
# ----------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)

# ----------------------------------------------------------
# 2. Initialize MySQL
# ----------------------------------------------------------
mysql = MySQL(app)

# ----------------------------------------------------------
# 3. HELPER FUNCTIONS
# ----------------------------------------------------------

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower()
            in app.config['ALLOWED_EXTENSIONS'])


# ----------------------------------------------------------
# 4. DECORATORS
# ----------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get('role') not in roles:
                flash('Access denied.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated
    return decorator


# ==============================================================
# SECTION A: PUBLIC ROUTES
# ==============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email     = request.form.get('email', '').strip()
        phone     = request.form.get('phone', '').strip()
        address   = request.form.get('address', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')

        if not full_name or not email or not password:
            flash('Full name, email, and password are required.', 'danger')
            return redirect(url_for('register'))

        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))

        hashed = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt()
        )

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                flash('Email already registered. Please log in.', 'warning')
                cur.close()
                return redirect(url_for('register'))

            cur.execute("""
                INSERT INTO users
                    (full_name, email, phone, address, password, role)
                VALUES (%s, %s, %s, %s, %s, 'user')
            """, (full_name, email, phone, address,
                  hashed.decode('utf-8')))

            mysql.connection.commit()
            cur.close()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            flash(f'Database error: {str(e)}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard_redirect'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Email and password are required.', 'danger')
            return redirect(url_for('login'))

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()

            if user and bcrypt.checkpw(
                password.encode('utf-8'),
                user['password'].encode('utf-8')
            ):
                session['user_id']   = user['id']
                session['user_name'] = user['full_name']
                session['role']      = user['role']
                flash(f"Welcome back, {user['full_name']}!", 'success')
                return redirect(url_for('dashboard_redirect'))
            else:
                flash('Invalid email or password.', 'danger')

        except Exception as e:
            flash(f'Login error: {str(e)}', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard_redirect():
    role = session.get('role')
    if role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif role == 'police':
        return redirect(url_for('police_dashboard'))
    else:
        return redirect(url_for('user_dashboard'))


# ==============================================================
# SECTION B: USER ROUTES
# ==============================================================

@app.route('/user/dashboard')
@login_required
@role_required('user')
def user_dashboard():
    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT status, COUNT(*) as count
        FROM crime_reports
        WHERE user_id = %s
        GROUP BY status
    """, (session['user_id'],))
    stats = cur.fetchall()

    cur.execute("""
        SELECT * FROM crime_reports
        WHERE user_id = %s
        ORDER BY reported_at DESC
        LIMIT 5
    """, (session['user_id'],))
    recent = cur.fetchall()
    cur.close()

    return render_template('user/dashboard.html',
                           stats=stats, recent=recent)


@app.route('/user/report', methods=['GET', 'POST'])
@login_required
@role_required('user')
def report_crime():
    if request.method == 'POST':
        crime_type  = request.form.get('crime_type', '').strip()
        description = request.form.get('description', '').strip()
        location    = request.form.get('location', '').strip()
        district    = request.form.get('district', '').strip()

        if not crime_type or not description or not location:
            flash('Crime type, description, and location are required.',
                  'danger')
            return redirect(url_for('report_crime'))

        evidence_filename = None
        if 'evidence' in request.files:
            file = request.files['evidence']
            if file and file.filename != '' and allowed_file(file.filename):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_name = (f"{session['user_id']}_"
                             f"{timestamp}_{file.filename}")
                save_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], safe_name
                )
                file.save(save_path)
                evidence_filename = safe_name

        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO crime_reports
                    (user_id, crime_type, description,
                     location, district, evidence_file)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (session['user_id'], crime_type, description,
                  location, district, evidence_filename))
            mysql.connection.commit()
            cur.close()
            flash('Crime report submitted successfully! '
                  'Authorities will review it shortly.', 'success')
            return redirect(url_for('my_reports'))

        except Exception as e:
            flash(f'Submission error: {str(e)}', 'danger')

    return render_template('user/report_crime.html')


@app.route('/user/my-reports')
@login_required
@role_required('user')
def my_reports():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT cr.*, u.full_name as officer_name
        FROM crime_reports cr
        LEFT JOIN users u ON cr.assigned_to = u.id
        WHERE cr.user_id = %s
        ORDER BY cr.reported_at DESC
    """, (session['user_id'],))
    reports = cur.fetchall()
    cur.close()
    return render_template('user/my_reports.html', reports=reports)


# ==============================================================
# SECTION C: POLICE ROUTES
# ==============================================================

@app.route('/police/dashboard')
@login_required
@role_required('police')
def police_dashboard():
    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT cr.*, u.full_name as reporter_name
        FROM crime_reports cr
        JOIN users u ON cr.user_id = u.id
        WHERE cr.assigned_to = %s
        ORDER BY cr.reported_at DESC
    """, (session['user_id'],))
    assigned_cases = cur.fetchall()

    cur.execute("""
        SELECT cr.*, u.full_name as reporter_name
        FROM crime_reports cr
        JOIN users u ON cr.user_id = u.id
        WHERE cr.status = 'Pending' AND cr.assigned_to IS NULL
        ORDER BY cr.reported_at DESC
    """)
    pending_cases = cur.fetchall()
    cur.close()

    return render_template('police/dashboard.html',
                           assigned_cases=assigned_cases,
                           pending_cases=pending_cases)


@app.route('/police/update-case/<int:report_id>',
           methods=['GET', 'POST'])
@login_required
@role_required('police')
def update_case(report_id):
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        new_status = request.form.get('status')
        note       = request.form.get('note', '').strip()

        # Handle media file upload on case update
        media_filename = None
        if 'media_file' in request.files:
            file = request.files['media_file']
            if (file and file.filename != ''
                    and allowed_file(file.filename)):
                timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_name  = (f"case_{report_id}_"
                              f"{timestamp}_{file.filename}")
                save_path  = os.path.join(
                    app.config['UPLOAD_FOLDER'], safe_name
                )
                file.save(save_path)
                media_filename = safe_name

        # Update report status
        cur.execute("""
            UPDATE crime_reports
            SET status = %s, assigned_to = %s
            WHERE id = %s
        """, (new_status, session['user_id'], report_id))

        # Log the update with optional media
        if note or media_filename:
            note_text = note if note else '(Media evidence uploaded)'
            cur.execute("""
                INSERT INTO case_updates
                    (report_id, updated_by, note, media_file)
                VALUES (%s, %s, %s, %s)
            """, (report_id, session['user_id'],
                  note_text, media_filename))

        mysql.connection.commit()
        cur.close()
        flash('Case updated successfully.', 'success')
        return redirect(url_for('police_dashboard'))

    # GET — load case details and updates
    cur.execute(
        "SELECT * FROM crime_reports WHERE id = %s", (report_id,)
    )
    case = cur.fetchone()

    cur.execute("""
        SELECT cu.*, u.full_name
        FROM case_updates cu
        JOIN users u ON cu.updated_by = u.id
        WHERE cu.report_id = %s
        ORDER BY cu.updated_at DESC
    """, (report_id,))
    updates = cur.fetchall()
    cur.close()

    return render_template('police/update_case.html',
                           case=case, updates=updates)


# ==============================================================
# SECTION D: ADMIN ROUTES
# ==============================================================

@app.route('/admin/dashboard')
@login_required
@role_required('admin')
def admin_dashboard():
    cur = mysql.connection.cursor()

    cur.execute(
        "SELECT COUNT(*) as total FROM users WHERE role='user'"
    )
    total_users = cur.fetchone()['total']

    cur.execute(
        "SELECT COUNT(*) as total FROM users WHERE role='police'"
    )
    total_police = cur.fetchone()['total']

    cur.execute("SELECT COUNT(*) as total FROM crime_reports")
    total_reports = cur.fetchone()['total']

    cur.execute("""
        SELECT COUNT(*) as total FROM crime_reports
        WHERE status = 'Pending'
    """)
    pending = cur.fetchone()['total']

    # Count both Resolved AND Closed as completed
    cur.execute("""
        SELECT COUNT(*) as total FROM crime_reports
        WHERE status IN ('Resolved', 'Closed')
    """)
    resolved = cur.fetchone()['total']

    cur.execute("""
        SELECT crime_type, COUNT(*) as count
        FROM crime_reports
        GROUP BY crime_type
        ORDER BY count DESC
    """)
    crime_types = cur.fetchall()

    cur.execute("""
        SELECT cr.*, u.full_name as reporter_name
        FROM crime_reports cr
        JOIN users u ON cr.user_id = u.id
        ORDER BY cr.reported_at DESC
        LIMIT 10
    """)
    recent_reports = cur.fetchall()
    cur.close()

    return render_template('admin/dashboard.html',
                           total_users=total_users,
                           total_police=total_police,
                           total_reports=total_reports,
                           pending=pending,
                           resolved=resolved,
                           crime_types=crime_types,
                           recent_reports=recent_reports)


@app.route('/admin/all-reports')
@login_required
@role_required('admin')
def all_reports():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT cr.*, u.full_name as reporter_name,
               p.full_name as officer_name
        FROM crime_reports cr
        JOIN users u ON cr.user_id = u.id
        LEFT JOIN users p ON cr.assigned_to = p.id
        ORDER BY cr.reported_at DESC
    """)
    reports = cur.fetchall()
    cur.close()
    return render_template('admin/all_reports.html', reports=reports)


@app.route('/admin/manage-users')
@login_required
@role_required('admin')
def manage_users():
    cur = mysql.connection.cursor()
    # Only fetch citizens — not police or admin
    cur.execute("""
        SELECT id, full_name, email, phone, address, role, created_at
        FROM users
        WHERE role = 'user'
        ORDER BY address ASC, created_at DESC
    """)
    users = cur.fetchall()
    cur.close()
    return render_template('admin/manage_users.html', users=users)


@app.route('/admin/manage-police')
@login_required
@role_required('admin')
def manage_police():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT u.id, u.full_name, u.email, u.phone,
               po.badge_number, po.station, po.district
        FROM users u
        JOIN police_officers po ON u.id = po.user_id
        WHERE u.role = 'police'
        ORDER BY po.district ASC, u.full_name ASC
    """)
    officers = cur.fetchall()
    cur.close()
    return render_template('admin/manage_police.html',
                           officers=officers)


@app.route('/admin/add-police', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def add_police():
    if request.method == 'POST':
        full_name    = request.form.get('full_name', '').strip()
        email        = request.form.get('email', '').strip()
        phone        = request.form.get('phone', '').strip()
        password     = request.form.get('password', '')
        badge_number = request.form.get('badge_number', '').strip()
        station      = request.form.get('station', '').strip()
        district     = request.form.get('district', '').strip()

        hashed = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt()
        ).decode('utf-8')

        try:
            cur = mysql.connection.cursor()

            # Check duplicate email
            cur.execute(
                "SELECT id FROM users WHERE email = %s", (email,)
            )
            if cur.fetchone():
                flash('Email already registered.', 'warning')
                cur.close()
                return redirect(url_for('manage_police'))

            cur.execute("""
                INSERT INTO users
                    (full_name, email, phone, password, role)
                VALUES (%s, %s, %s, %s, 'police')
            """, (full_name, email, phone, hashed))

            new_user_id = cur.lastrowid

            cur.execute("""
                INSERT INTO police_officers
                    (user_id, badge_number, station, district)
                VALUES (%s, %s, %s, %s)
            """, (new_user_id, badge_number, station, district))

            mysql.connection.commit()
            cur.close()
            flash('Police officer added successfully.', 'success')
            return redirect(url_for('manage_police'))

        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')

    return redirect(url_for('manage_police'))


@app.route('/admin/delete-user/<int:user_id>')
@login_required
@role_required('admin')
def delete_user(user_id):
    """
    Safe delete — removes all related records first
    to avoid foreign key constraint errors.
    """
    try:
        cur = mysql.connection.cursor()

        # Step 1: Get all reports by this user
        cur.execute(
            "SELECT id FROM crime_reports WHERE user_id = %s",
            (user_id,)
        )
        reports = cur.fetchall()

        # Step 2: Delete case_updates for those reports
        for r in reports:
            cur.execute(
                "DELETE FROM case_updates WHERE report_id = %s",
                (r['id'],)
            )

        # Step 3: Delete case_updates made BY this user
        cur.execute(
            "DELETE FROM case_updates WHERE updated_by = %s",
            (user_id,)
        )

        # Step 4: Unassign cases assigned to this user
        cur.execute(
            """UPDATE crime_reports
               SET assigned_to = NULL
               WHERE assigned_to = %s""",
            (user_id,)
        )

        # Step 5: Delete this user's crime reports
        cur.execute(
            "DELETE FROM crime_reports WHERE user_id = %s",
            (user_id,)
        )

        # Step 6: Delete police officer profile if exists
        cur.execute(
            "DELETE FROM police_officers WHERE user_id = %s",
            (user_id,)
        )

        # Step 7: Now safely delete the user
        cur.execute(
            "DELETE FROM users WHERE id = %s AND role != 'admin'",
            (user_id,)
        )

        mysql.connection.commit()
        cur.close()
        flash('User removed successfully.', 'success')

    except Exception as e:
        flash(f'Error deleting user: {str(e)}', 'danger')

    # Redirect to correct page based on where the request came from
    ref = request.referrer or ''
    if 'manage-police' in ref:
        return redirect(url_for('manage_police'))
    return redirect(url_for('manage_users'))


@app.route('/admin/assign-officer/<int:report_id>',
           methods=['GET', 'POST'])
@login_required
@role_required('admin')
def assign_officer(report_id):
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        officer_id = request.form.get('officer_id')
        if not officer_id:
            flash('Please select an officer.', 'danger')
            return redirect(
                url_for('assign_officer', report_id=report_id)
            )

        cur.execute("""
            UPDATE crime_reports
            SET assigned_to = %s, status = 'Under Investigation'
            WHERE id = %s
        """, (officer_id, report_id))

        cur.execute("""
            INSERT INTO case_updates (report_id, updated_by, note)
            VALUES (%s, %s, %s)
        """, (report_id, session['user_id'],
              'Case assigned to officer by Admin.'))

        mysql.connection.commit()
        cur.close()
        flash('Officer assigned successfully!', 'success')
        return redirect(url_for('all_reports'))

    # GET
    cur.execute(
        "SELECT * FROM crime_reports WHERE id = %s", (report_id,)
    )
    case = cur.fetchone()

    cur.execute("""
        SELECT u.id, u.full_name, u.email,
               po.badge_number, po.station, po.district
        FROM users u
        JOIN police_officers po ON u.id = po.user_id
        WHERE u.role = 'police'
        ORDER BY po.district ASC, u.full_name ASC
    """)
    officers = cur.fetchall()
    cur.close()

    return render_template('admin/assign_officer.html',
                           case=case, officers=officers)


# ==============================================================
# SECTION E: ML PREDICTION ROUTES
# ==============================================================

@app.route('/admin/prediction')
@login_required
@role_required('admin')
def crime_prediction():
    predictions = []
    error_msg   = None

    model_path   = os.path.join('ml_model', 'crime_model.pkl')
    encoder_path = os.path.join('ml_model', 'label_encoder.pkl')
    district_path= os.path.join('ml_model', 'district_encoder.pkl')
    domain_path  = os.path.join('ml_model', 'domain_encoder.pkl')
    weapon_path  = os.path.join('ml_model', 'weapon_encoder.pkl')
    gender_path  = os.path.join('ml_model', 'gender_encoder.pkl')
    freq_path    = os.path.join('ml_model', 'city_crime_freq.json')
    fcols_path   = os.path.join('ml_model', 'feature_cols.json')

    if not os.path.exists(model_path):
        error_msg = ('ML model not trained yet. '
                     'Run: python ml_model/train_model.py')
    elif not os.path.exists(freq_path):
        error_msg = ('Frequency data missing. '
                     'Run: python ml_model/train_model.py')
    else:
        try:
            model          = joblib.load(model_path)
            crime_encoder  = joblib.load(encoder_path)
            city_encoder   = joblib.load(district_path)
            domain_encoder = joblib.load(domain_path)
            weapon_encoder = joblib.load(weapon_path)
            gender_encoder = joblib.load(gender_path)

            with open(freq_path, 'r') as f:
                city_freq = json.load(f)

            with open(fcols_path, 'r') as f:
                feature_cols = json.load(f)

            # Safe encoder defaults
            try:
                weapon_enc = int(
                    weapon_encoder.transform(['Knife'])[0]
                )
            except Exception:
                weapon_enc = 0

            try:
                gender_enc = int(
                    gender_encoder.transform(['M'])[0]
                )
            except Exception:
                gender_enc = 0

            cur = mysql.connection.cursor()
            cur.execute("""
                SELECT
                    district,
                    COUNT(*) AS report_count,
                    CAST(AVG(HOUR(reported_at)) AS UNSIGNED)
                        AS avg_hour,
                    CAST(AVG(MONTH(reported_at)) AS UNSIGNED)
                        AS avg_month
                FROM crime_reports
                WHERE district IS NOT NULL AND district != ''
                GROUP BY district
                ORDER BY report_count DESC
            """)
            districts = cur.fetchall()
            cur.close()

            # If no district data yet, show all cities from model
            cities_to_predict = []
            if not districts:
                for city in sorted(city_encoder.classes_):
                    cities_to_predict.append({
                        'name'        : city,
                        'report_count': 0,
                        'avg_hour'    : 20,
                        'avg_month'   : datetime.now().month
                    })
            else:
                for row in districts:
                    cities_to_predict.append({
                        'name'        : row['district'],
                        'report_count': row['report_count'],
                        'avg_hour'    : row['avg_hour'] or 20,
                        'avg_month'   : row['avg_month'] or 6
                    })

            for city_data in cities_to_predict:
                try:
                    city_name = city_data['name']

                    # Match city name case-insensitively
                    city_match = None
                    for c in city_encoder.classes_:
                        if c.lower() == city_name.lower():
                            city_match = c
                            break

                    if city_match is None:
                        continue

                    city_enc    = int(
                        city_encoder.transform([city_match])[0]
                    )
                    hour        = int(city_data['avg_hour'])
                    month       = int(city_data['avg_month'])
                    day_of_week = datetime.now().weekday()
                    time_period = (0 if hour < 6  else
                                   1 if hour < 12 else
                                   2 if hour < 18 else 3)
                    is_weekend  = 1 if day_of_week >= 5 else 0

                    # FIX: Use pd.DataFrame with column names
                    # This completely eliminates the UserWarning
                    input_df = pd.DataFrame(
                        [[city_enc, hour, month, day_of_week,
                          time_period, is_weekend, weapon_enc,
                          gender_enc, 2, 10]],
                        columns=feature_cols
                    )

                    domain_proba = model.predict_proba(input_df)[0]
                    domain_preds = []
                    for idx, prob in enumerate(domain_proba):
                        label = domain_encoder.inverse_transform(
                            [idx]
                        )[0]
                        domain_preds.append({
                            'domain': label,
                            'pct'   : round(float(prob) * 100, 1)
                        })
                    domain_preds.sort(key=lambda x: -x['pct'])
                    top_domain = domain_preds[0]['domain']

                    # Get top crime from real frequency data
                    if city_match in city_freq:
                        top_crime = city_freq[city_match]['top_crime']
                    else:
                        top_crime = 'UNKNOWN'

                    predictions.append({
                        'district'    : city_match,
                        'report_count': city_data['report_count'],
                        'predicted'   : top_crime,
                        'top_domain'  : top_domain,
                        'domain_preds': domain_preds[:4]
                    })

                except Exception:
                    continue

        except Exception as e:
            error_msg = f'Prediction error: {str(e)}'

    return render_template('admin/prediction.html',
                           predictions=predictions,
                           error_msg=error_msg)

@app.route('/predict-crime', methods=['POST'])
@login_required
def predict_crime_ajax():
    """
    Uses real historical frequency data from city_crime_freq.json
    for accurate crime breakdown.
    Domain model predicts risk category (Fire/Violent/Traffic/Other).
    pd.DataFrame used for prediction — no UserWarning.
    """
    model_path   = os.path.join('ml_model', 'crime_model.pkl')
    domain_path  = os.path.join('ml_model', 'domain_encoder.pkl')
    city_path    = os.path.join('ml_model', 'district_encoder.pkl')
    weapon_path  = os.path.join('ml_model', 'weapon_encoder.pkl')
    gender_path  = os.path.join('ml_model', 'gender_encoder.pkl')
    freq_path    = os.path.join('ml_model', 'city_crime_freq.json')
    fcols_path   = os.path.join('ml_model', 'feature_cols.json')

    if not os.path.exists(model_path):
        return jsonify({
            'success': False,
            'error': ('Model not trained yet. '
                      'Run: python ml_model/train_model.py')
        })

    if not os.path.exists(freq_path):
        return jsonify({
            'success': False,
            'error': ('Frequency data missing. '
                      'Run: python ml_model/train_model.py')
        })

    try:
        model          = joblib.load(model_path)
        domain_encoder = joblib.load(domain_path)
        city_encoder   = joblib.load(city_path)
        weapon_encoder = joblib.load(weapon_path)
        gender_encoder = joblib.load(gender_path)

        with open(freq_path, 'r') as f:
            city_freq = json.load(f)

        with open(fcols_path, 'r') as f:
            feature_cols = json.load(f)

        district = request.form.get('district', '').strip()
        hour     = int(request.form.get('hour', 20))
        month    = int(request.form.get('month',
                       datetime.now().month))

        if not district:
            return jsonify({
                'success': False,
                'error': 'Please enter a district name.'
            })

        # Case-insensitive city match
        city_match = None
        for c in city_encoder.classes_:
            if c.lower() == district.lower():
                city_match = c
                break

        if city_match is None:
            known = ', '.join(sorted(city_encoder.classes_))
            return jsonify({
                'success': False,
                'error': (f'"{district}" not in model. '
                          f'Available cities: {known}')
            })

        city_enc    = int(city_encoder.transform([city_match])[0])
        day_of_week = datetime.now().weekday()
        time_period = (0 if hour < 6  else
                       1 if hour < 12 else
                       2 if hour < 18 else 3)
        is_weekend  = 1 if day_of_week >= 5 else 0

        # Safe encoder transforms with fallback
        try:
            weapon_enc = int(weapon_encoder.transform(['Knife'])[0])
        except Exception:
            weapon_enc = 0

        try:
            gender_enc = int(gender_encoder.transform(['M'])[0])
        except Exception:
            gender_enc = 0

        age_group = 2  # 30-45 default

        # Build pandas DataFrame with correct column names
        # This completely eliminates the UserWarning
        input_df = pd.DataFrame(
            [[city_enc, hour, month, day_of_week,
              time_period, is_weekend, weapon_enc,
              gender_enc, age_group, 10]],
            columns=feature_cols
        )

        # Predict crime domain probabilities
        domain_proba = model.predict_proba(input_df)[0]
        domain_preds = []
        for idx, prob in enumerate(domain_proba):
            label = domain_encoder.inverse_transform([idx])[0]
            domain_preds.append({
                'domain': label,
                'pct':    round(float(prob) * 100, 1)
            })
        domain_preds.sort(key=lambda x: -x['pct'])
        top_domain = domain_preds[0]['domain']

        # Get real crime frequency breakdown from historical data
        if city_match in city_freq:
            breakdown_raw = city_freq[city_match]['breakdown']

            # Boost crimes that match the predicted domain
            violent_crimes = {
                'ASSAULT', 'ROBBERY', 'HOMICIDE',
                'SEXUAL ASSAULT', 'KIDNAPPING',
                'DOMESTIC VIOLENCE', 'VEHICLE - STOLEN',
                'IDENTITY THEFT'
            }
            fire_crimes    = {'ARSON', 'FIREARM OFFENSE'}
            traffic_crimes = {'TRAFFIC VIOLATION'}

            adjusted = {}
            for crime, pct in breakdown_raw.items():
                boost = 1.0
                if 'Violent' in top_domain and crime in violent_crimes:
                    boost = 1.4
                elif 'Fire' in top_domain and crime in fire_crimes:
                    boost = 1.4
                elif 'Traffic' in top_domain and crime in traffic_crimes:
                    boost = 1.6
                adjusted[crime] = pct * boost

            # Normalise to 100%
            total = sum(adjusted.values())
            breakdown = {
                k: round(v / total * 100, 1)
                for k, v in adjusted.items()
            }
            sorted_crimes = sorted(
                breakdown.items(), key=lambda x: -x[1]
            )
            top_crime = sorted_crimes[0][0]
            top_pct   = sorted_crimes[0][1]

        else:
            sorted_crimes = [('DATA NOT AVAILABLE', 100.0)]
            top_crime     = 'DATA NOT AVAILABLE'
            top_pct       = 0.0

        top8 = sorted_crimes[:8]

        return jsonify({
            'success'     : True,
            'district'    : city_match,
            'top_crime'   : top_crime,
            'top_pct'     : top_pct,
            'top_domain'  : top_domain,
            'domain_preds': domain_preds[:4],
            'breakdown'   : [
                {'crime': k, 'pct': v} for k, v in top8
            ]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
# ==============================================================
# SECTION F: SETUP ROUTE (run once only)
# ==============================================================

# @app.route('/setup-admin')
# def setup_admin():
#     """
#     Visit this URL once to set the admin password.
#     Then comment out or remove this route.
#     """
#     hashed = bcrypt.hashpw(
#         b'admin123', bcrypt.gensalt()
#     ).decode('utf-8')
#     try:
#         cur = mysql.connection.cursor()
#         cur.execute("""
#             UPDATE users SET password = %s
#             WHERE email = 'admin@crimeportal.com'
#         """, (hashed,))
#         mysql.connection.commit()
#         cur.close()
#         return 'Admin password set! <a href="/login">Login now</a>'
#     except Exception as e:
#         return f'Error: {str(e)}'


# ==============================================================
# RUN
# ==============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)