import time
import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import DateField
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from functools import wraps
import sqlite3
conn = sqlite3.connect('db/database.db', check_same_thread=False)

#### Defining Flask App
app = Flask(__name__)

app.config['SECRET_KEY'] = '8IR4M7-R3c74GjTHhKzWODaYVHuPGqn4w92DHLqeYJA'

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

class DateForm(FlaskForm):
    dt = DateField('Pick a Date')

#### If these directories don't exist, create them
def createDir():
    datetoday = date.today().strftime("%m_%d_%y")
    if not os.path.isdir('Attendance'):
        os.makedirs('Attendance')
    if not os.path.isdir('static'):
        os.makedirs('static')
    if not os.path.isdir('static/faces'):
        os.makedirs('static/faces')
    if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
            f.write('Name,Roll,Time')
createDir()
#### extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

#### Add Attendance of a specific user
def add_attendance(name):
    datetoday = date.today().strftime("%m_%d_%y")
    username = name.split('_')[0]
    roll = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(roll) not in list(df['Roll']):
        cursor = conn.cursor()
        sql = 'INSERT INTO Attendance(rollno,name,marked_date,marked_time) VALUES(?,?,?,?)'
        data_tuple = (roll, username, datetoday, current_time,)
        cursor.execute(sql, data_tuple)
        conn.commit()
        cursor.close()
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{roll},{current_time}')
        return True    
    else:
        False

################## ROUTING FUNCTIONS ##############################

@app.route('/')
def index():
    createDir()
    return render_template('index.html')

@app.route('/login_student', methods=['GET', 'POST'])
def login_student():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Student where email = ? and password = ?", (email, password))
        data = cursor.fetchone()
        if data is not None:
            session['std_logged_in'] = True
            session['uid'] = data[0]
            session['roll_no'] = data[1]
            session['uname'] = data[2]
            session['email'] = data[3]
            session['pic_path'] = data[5]
            flash('You are now logged in', 'success')
            return redirect(url_for('student'))
        else:
            error = 'Invalid login'
            return render_template('login_student.html', error=error)
    return render_template('login_student.html')

@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Admin where email = ? and password = ?", (email, password))
        data = cursor.fetchone()
        if data is not None:
            session['uid'] = data[0]
            session['uname'] = data[1]
            session['email'] = data[2]
            session['is_admin'] = True
            session['fty_logged_in'] = True
            flash('You are now logged in', 'success')
            return redirect(url_for('admin'))
        else:
            error = 'Invalid login'
            return render_template('login_admin.html', error=error)

    return render_template('login_admin.html')

def is_student_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'std_logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login!', 'danger')
            return redirect(url_for('login_student'))
    return wrap

def is_admin_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'fty_logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login!', 'danger')
            return redirect(url_for('login_admin'))
    return wrap

@app.route('/student')
@is_student_logged_in
def student():
    rollno = session['roll_no']
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM Attendance where rollno = ?", (rollno,))
    data = cursor.fetchone()
    cursor.close()
    return render_template('dashboard_student.html', att_count=data[0])

@app.route('/view_students')
@is_admin_logged_in
def view_students():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Student order by id desc")
    students = cursor.fetchall()
    cursor.close()
    return render_template('view_students.html', students=students)

def get_attendance(date, user):
    sql = "SELECT * FROM Attendance where marked_date = ? order by id"
    data_tuple = (date,)
    if user == "student":
        sql = "SELECT * FROM Attendance where marked_date = ? and rollno = ? order by id"
        data_tuple = (date,session['roll_no'])
    cursor = conn.cursor()
    cursor.execute(sql, data_tuple)
    attendance = cursor.fetchall()
    cursor.close()
    return attendance

@app.route('/attendance', methods=['GET', 'POST'])
@is_admin_logged_in
def attendance():
    form = DateForm()
    datetoday = date.today().strftime("%m_%d_%y")
    datelabel = "Today's"
    if form.validate_on_submit():
        datetoday = form.dt.data.strftime("%m_%d_%y")
        datelabel = form.dt.data.strftime("%d-%m-%Y")
    if datelabel == date.today().strftime("%d-%m-%Y"):
        datelabel = "Today's"
    attendance = get_attendance(datetoday, "admin")
    return render_template('attendance_admin.html', attendance=attendance, form=form, datelabel=datelabel)

@app.route('/my_attendance', methods=['GET', 'POST'])
@is_student_logged_in
def my_attendance():
    form = DateForm()
    datetoday = date.today().strftime("%m_%d_%y")
    datelabel = "Today's"
    if form.validate_on_submit():
        datetoday = form.dt.data.strftime("%m_%d_%y")
        datelabel = form.dt.data.strftime("%d-%m-%Y")
    if datelabel == date.today().strftime("%d-%m-%Y"):
        datelabel = "Today's"
    attendance = get_attendance(datetoday, "student")
    return render_template('attendance_student.html', attendance=attendance, form=form, datelabel=datelabel)

@app.route('/my_profile', methods=['GET', 'POST'])
@is_student_logged_in
def my_profile():
    sql = "SELECT * FROM Student where rollno = ?"
    data_tuple = (session['roll_no'],)
    cursor = conn.cursor()
    cursor.execute(sql, data_tuple)
    data = cursor.fetchone()
    cursor.close()
    return render_template('student_profile.html', student=data)


@app.route('/admin')
@is_admin_logged_in
def admin():
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM Student")
    data = cursor.fetchone()
    cursor.close()
    return render_template('dashboard_admin.html', students_count=data[0])

@app.route("/logout")
def logout():
    if 'std_logged_in' or 'fty_logged_in' in session:
        session.clear()
    return redirect(url_for('index'))

@app.route('/student_registration', methods=['GET', 'POST'])
@is_admin_logged_in
def register_student():
    nimgs = 10
    if request.method == 'POST':
        email = request.form['email']
        rollno = request.form['rollno']
        name = request.form['name']
        password = request.form['password']
        pic_path = f'static/images/users/{rollno}-{name}.jpg'
        registered_on = datetime.now()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Student where email = ?", (email,))
        data = cursor.fetchone()
        if data is None:
            sql = 'INSERT INTO Student(name,email,rollno,password,pic_path,registered_on) VALUES(?,?,?,?,?,?)'
            data_tuple = (name, email, rollno, password, pic_path, registered_on)
            cursor.execute(sql, data_tuple)
            conn.commit()
            cursor.close()
            if os.path.isfile('static/images/users/temp.jpg'):
                os.rename('static/images/users/temp.jpg',pic_path)
            if 'img_captured' in session:
                session.pop('img_captured')
            userimagefolder = 'static/faces/'+name+'_'+str(rollno)
            if not os.path.isdir(userimagefolder):
                os.makedirs(userimagefolder)
            i, j = 0, 0
            cap = cv2.VideoCapture(0)
            while 1:
                _,frame = cap.read()
                faces = extract_faces(frame)
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                    if j % 5 == 0:
                        name_jpg = name+'_'+str(i)+'.jpg'
                        cv2.imwrite(userimagefolder+'/'+name_jpg, frame[y:y+h, x:x+w])
                        i+=1
                    j+=1
                if j == nimgs*5:
                    break
                cv2.imshow('Student Registration',frame)
                if cv2.waitKey(1)==27:
                    break
            cap.release()
            cv2.destroyAllWindows()
            print('Training Model')
            train_model()
            flash('Student registration successful', 'success')
            return render_template('register_student.html')
        else:
            flash('Student with this email already exists!', 'danger')
    if os.path.isfile('static/images/users/temp.jpg'):
        temp_pic = True
    else:
        temp_pic = False
    return render_template('register_student.html', temp_pic=temp_pic)

@app.route("/capture_image")
@is_admin_logged_in
def capture_image():
    session['dt'] = datetime.now()
    path = 'static/images/users'
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Display the resulting frame
        cv2.imshow('Press c to capture image', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(os.path.join(path, 'temp.jpg'), frame)
            time.sleep(2)
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    session['img_captured'] = True
    return redirect(url_for('register_student'))

#### This function will run when we click on Take Attendance Button
@app.route('/mark_attendance',methods=['GET'])
def mark_attendance():
    ATTENDENCE_MARKED = False
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        flash("This face is not in database, need to register first", "danger")
        return redirect(url_for('attendance'))
        
    cap = cv2.VideoCapture(0)
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                status = add_attendance(identified_person)
                ATTENDENCE_MARKED = True
                if status:
                    current_time_ = datetime.now().strftime("%H:%M:%S")
                    flash(f"Attendence marked for {identified_person}, at {current_time_} ", "success")
                else:
                    flash(f"Student {identified_person} has already marked attendence for the day", "danger")
                break
        if ATTENDENCE_MARKED:
            break

        # Display the resulting frame
        cv2.imshow('Attendance Check, press "a" to exit', frame)
        cv2.putText(frame,'hello',(30,30),cv2.FONT_HERSHEY_COMPLEX,2,(255, 255, 255))
        
        # Wait for the user to press 'q' to quit
        # if cv2.waitKey(1) == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('attendance'))
    
#### Our main function which runs the Flask App
app.run(debug=True,port=1000)
if __name__ == '__main__':
    pass
