from flask import Flask, render_template, request, redirect, session
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

client = MongoClient("mongodb+srv://narainsingaram:narain2007@idk.7pveyga.mongodb.net/?retryWrites=true&w=majority&appName=idk")
db = client["auth_app"]
users_collection = db["users"]

@app.route('/')
def index():
    if 'user' in session:
        return f"Logged in as {session['user']}"
    return redirect("/login")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        existing_user = users_collection.find_one({'username': request.form['username']})
        if existing_user:
            return 'Username already exists!'

        hashed_password = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({'username': request.form['username'], 'password': hashed_password})
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users_collection.find_one({'username': request.form['username']})
        if user and bcrypt.checkpw(request.form['password'].encode('utf-8'), user['password']):
            session['user'] = request.form['username']
            return redirect('/')
        return 'Invalid credentials!'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)