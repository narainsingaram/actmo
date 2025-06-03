from flask import Blueprint, render_template, request, redirect, session
from pymongo import MongoClient
import bcrypt
import certifi

register_bp = Blueprint('register', __name__)

client = MongoClient(
    "mongodb+srv://narainsingaram:narain2007@idk.7pveyga.mongodb.net/?retryWrites=true&w=majority&appName=idk",
    tlsCAFile=certifi.where()
)

db = client["auth_app"]
users_collection = db["users"]

@register_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        existing_user = users_collection.find_one({'username': request.form['username']})
        if existing_user:
            return 'Username already exists!'

        hashed_password = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({'username': request.form['username'], 'password': hashed_password})
        return redirect('/login')
    return render_template('register.html')

@register_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = users_collection.find_one({'username': request.form['username']})
        if user and bcrypt.checkpw(request.form['password'].encode('utf-8'), user['password']):
            session['user'] = request.form['username']
            return redirect('/')
        return 'Invalid credentials!'
    return render_template('login.html')

@register_bp.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')
