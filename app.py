
from flask import Flask, render_template, request, session, redirect, url_for, send_file,jsonify
from fpdf import FPDF
import secrets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import base64
from datetime import timedelta

app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 
app.config['SESSION_PERMANENT'] = True  # Enable session persistence
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
# Load the dataset
df = pd.read_csv('Copy of dataset(1).csv',header=0)

@app.route('/')
def index():
     return render_template('login.html')
   



'''
@app.route("/login", methods=["POST"])
def login():
    student_id = request.form["student_id"]
    student = df[df["Student Id"] == student_id]

    if not student.empty:
        session["Student Id"] = student_id  # Store Student Id in session
        return redirect(url_for("dashboard"))
    
    return "Invalid Student ID", 401

'''
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        usn = request.form['usn']

        student_data = df[(df['Name'] == name) & (df['Student Id'] == usn)]
        if student_data.empty:
            return "Invalid credentials! Please check your details."

        session.permanent = True 
        # Store user details in session
        session['name'] = name
        print(f"Session name: {session['name']}")
        session['usn'] = usn
        print(f"Session name: {session['usn']}")

        return redirect(url_for('dashboard'))

    # Render login form when accessed via GET request
    return render_template('login.html')
    '''
    return 
        <form method="post">
            Name: <input type="text" name="name" required><br>
            USN: <input type="text" name="usn" required><br>
            <input type="submit" value="login">
        </form>
    '''
    



@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
     
     if 'name' not in session or 'usn' not in session:
        return redirect(url_for('login'))  # Redirect if not logged in

     name = session['name']
     usn = session['usn']
     '''if request.method == 'POST':
        # Get Name and USN from login form
        name = request.form['name']
        usn = request.form['usn']'''
     df = pd.read_csv('Copy of dataset(1).csv', header=0)
        # Filter dataset for the specific student
     student_data = df[(df['Name'] == name) & (df['Student Id'] == usn)]

     if student_data.empty:
            return "No data found for the provided Name and USN. Please check your details and try again."

        # Store name and USN in session
       # session['name'] = name
     '''
         session['usn'] = usn

    elif 'name' in session and 'usn' in session:
        # If session exists, retrieve stored student details
        name = session['name']
        usn = session['usn']
        student_data = df[(df['Name'] == name) & (df['Student Id'] == usn)]
    else:
        return redirect(url_for('login'))  # Redirect if no session data found
'''
    # Extract student details
     attendance = student_data['Attendence Rate'].values[0]
     unit_test_marks = student_data[['IoT-U1', 'Software Testing-U1', 'DAP-U1', 'NoSQL-U1', 
                                    'NoSQL-U2', 'IoT-U2', 'Software Testing-U2', 'DAP-U2']].values.flatten()
    
     technical_skill = student_data[['Java', 'Python', 'Web Technology', 'SQL']].mean(axis=1).values[0]

     student_row = df[df['Name'] == name]
     Java = student_row['Java'].values[0]
     Python = student_row['Python'].values[0]
     Web_Technology = student_row['Web Technology'].values[0]
     SQL = student_row['SQL'].values[0]

     sem1 = student_data['1st Sem Marks'].values[0]
     sem2 = student_data['2nd Sem Marks'].values[0]

    # Generate graphs as base64
     attendance_img = generate_attendance_graph(attendance)
     unit_test_img = generate_unit_test_graph(unit_test_marks)
     skills_img = generate_skills_line_chart(Java, Python, Web_Technology, SQL, name)
     final_unit_test_img = generate_final_unit_test_graph_for_student(name, usn)
     subject_wise_img = generate_subject_wise_graph_for_student(name, usn)
     sem_wise_img = generate_horizontal_bar_chart(name, sem1, sem2)

     return render_template('dashboard.html', 
                            name=name, 
                           usn=usn,
                           attendance_img=attendance_img,
                           unit_test_img=unit_test_img,
                           final_unit_test_img=final_unit_test_img,
                           subject_wise_img=subject_wise_img,
                           skills_img=skills_img,
                           sem_wise_img=sem_wise_img)

@app.route('/logout')
def logout():
    session.pop('name', None)
    session.pop('usn', None)
    return redirect(url_for('login'))  # Redirect to login page after logout


def generate_attendance_graph(attendance):
    # Attendance Graph (Pie Chart)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([attendance, 100 - attendance], labels=['Attendance', 'Remaining'], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF6347'])
    ax.set_title('Attendance')

    # Save to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    
    # Encode the image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()
    
    return 'data:image/png;base64,' + img_base64

def generate_unit_test_graph(unit_test_marks):
    # Unit Test Comparison Graph
    subjects = ['IoT-U1', 'Software Testing-U1', 'DAP-U1', 'NoSQL-U1', 'NoSQL-U2', 'IoT-U2', 'Software Testing-U2', 'DAP-U2']
    colors = ['blue', 'blue', 'blue', 'blue', 
              'green', 'green', 'green', 'green']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars=ax.bar(subjects, unit_test_marks, color=colors)
    ax.set_title('Unit Test Comparison')
    ax.set_xticklabels(subjects, rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f"{height:.1f}", ha='center', fontsize=12)

    

    # Save to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    
    # Encode the image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()
    
    return 'data:image/png;base64,' + img_base64


import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def refresh_dataset():
    global df  # Ensure df is updated globally
    df = pd.read_csv("Copy of dataset(1).csv") 

def generate_final_unit_test_graph_for_student(student_name, student_usn):
    # List of subjects for Unit Test 1 and Unit Test 2
    unit_test_1_subjects = ['IoT-U1', 'Software Testing-U1', 'DAP-U1', 'NoSQL-U1']
    unit_test_2_subjects = ['IoT-U2', 'Software Testing-U2', 'DAP-U2', 'NoSQL-U2']
    refresh_dataset()
    # Filter dataset for the specific student
    student_data = df[(df['Name'] == student_name) & (df['Student Id'] == student_usn)]
    
    # Check if student exists
    if student_data.empty:
        return None  # Return None if student not found

    # Calculate the average marks for Unit Test 1 and Unit Test 2 for the student
    average_ut1 = np.mean(student_data[unit_test_1_subjects].to_numpy())  # Convert to NumPy array and calculate mean
    average_ut2 = np.mean(student_data[unit_test_2_subjects].to_numpy())

    # Calculate percentage (Assuming Max Marks = 50 per subject)
    percentage_ut1 = (average_ut1 / 50) * 100
    percentage_ut2 = (average_ut2 / 50) * 100

    # Plotting the individual student's performance
    labels = ['Unit Test 1', 'Unit Test 2']
    averages = [percentage_ut1, percentage_ut2]
    colors = ['skyblue', 'lightgreen']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, averages, color=colors)
    ax.set_ylim([0, 100])  # Set y-axis limit to 100%
    ax.set_ylabel('Average Percentage (%)')
    ax.set_title(f'Performance of {student_name} in Unit Tests',pad=35)

    # Display actual percentage values on bars
    for i, v in enumerate(averages):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12)

    # Save to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    # Encode the image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()
    
    return 'data:image/png;base64,' + img_base64


import matplotlib.pyplot as plt
import io
import base64

def refresh_dataset():
    global df  # Ensure df is updated globally
    df = pd.read_csv("Copy of dataset(1).csv")  # Or fetch from DB


def generate_subject_wise_graph_for_student(student_name, student_usn):
    refresh_dataset()
     # List of subjects (Unit 1 and Unit 2 for each subject)
    subjects = ['IoT-U1', 'IoT-U2', 'Software Testing-U1', 'Software Testing-U2', 'DAP-U1', 'DAP-U2', 'NoSQL-U1', 'NoSQL-U2']

    # Filter dataset for the specific student
    student_data =df[(df['Name'] == student_name) & (df['Student Id'] == student_usn)]
    
    # Check if student exists
    if student_data.empty:
        return None  # Return None if student not found

    # Extract marks for the student
    subject_marks = {subject: student_data[subject].values[0] for subject in subjects}

    # Custom function to display actual marks
    def mark_labels(pct, values):
        absolute = int(round(pct / 100. * sum(values)))
        return f"{absolute}/50" # Display actual marks

    # Create a pie chart for the student's marks distribution
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(subject_marks.values(), labels=subject_marks.keys(),
                                      autopct=lambda pct: mark_labels(pct, subject_marks.values()), 
                                      startangle=90, 
                                      colors=['#FF6347', '#FF6347', '#FF9800', '#FF9800', '#9C27B0', '#9C27B0', '#3F51B5', '#3F51B5'])

                                      #colors=['#FF6347', '#4CAF50', '#FF9800', '#00BCD4', '#9C27B0', '#FFEB3B', '#3F51B5', '#00E5FF'])
    
    # Increase font size for better readability
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(12)

    ax.set_title(f'Subject-wise Marks for {student_name}')

    # Save the chart to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    # Encode the image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()

    return 'data:image/png;base64,' + img_base64



def generate_skills_line_chart(java, python,sql, web_tech,student_name):
    labels = ['Java', 'Python', 'SQL','Web Technology']
    skill_values = [java, python,sql, web_tech]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(labels, skill_values, marker='o', linestyle='-', color='red')

    for i, txt in enumerate(skill_values):
        ax.annotate(txt, (labels[i], skill_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    ax.set_ylim([0, 10])
    ax.set_ylabel('Skill Level')
    ax.set_title(f'Technical Skills of {student_name}')

    # Save to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()

    return 'data:image/png;base64,' + img_base64

import matplotlib.pyplot as plt
import io
import base64

def generate_horizontal_bar_chart(student_name, sem1_marks, sem2_marks):
    semesters = ["Semester 1", "Semester 2"]  # Y-axis labels
    marks = [sem1_marks, sem2_marks]  # Marks data

    fig, ax = plt.subplots(figsize=(10, 4))

    # Create horizontal bars
    bars=ax.barh(semesters, marks, color=['blue', 'green'])

    # Add labels and title
    ax.set_title(f"{student_name}'s Semester Performance")
    ax.set_xlabel("Marks")
    ax.set_ylabel("Semesters")

    for bar in bars:
        width = bar.get_width()  # Get the bar width (marks)
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,  # Adjust position
                f"{width:.1f}", ha='left', va='center', fontsize=12)


    # Save plot to BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    # Encode plot as base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()

    return 'data:image/png;base64,' + img_base64





@app.route('/admin')
def admin():
    df = pd.read_csv('Copy of dataset(1).csv', header=0)
    if 'username' not in session:
        return redirect(url_for('admin_login'))  # Redirect to login if not logged in
    return render_template('admin.html')  # Render admin dashboard


@app.route('/students', methods=['GET'])
def get_students():
    df = pd.read_csv('Copy of dataset(1).csv', header=0)
    df = df.fillna("")
    students = df.to_dict(orient='records')
    return jsonify(students)
'''
@app.route('/highest-scorer', methods=['GET'])
def highest_scorer():
    df = pd.read_csv('Copy of dataset(1).csv', header=0)
    df["Total Score"] = (df["1st Sem Marks"] + df["2nd Sem Marks"] + 
                         df["IoT-U1"] + df["Software Testing-U1"] + df["DAP-U1"] + df["NoSQL-U1"] +
                         df["NoSQL-U2"] + df["IoT-U2"] + df["Software Testing-U2"] + df["DAP-U2"]
                         )
    top_student = df.loc[df["Total Score"].idxmax()]
    return jsonify({"ID": top_student["Student Id"], "Name": top_student["Name"], "Total Score": int(top_student["Total Score"])})
'''
@app.route('/highest-scorers', methods=['GET'])
def highest_scorers():
    df = pd.read_csv('Copy of dataset(1).csv', header=0)

    # Calculate total score for Unit Test 1
    df["Total Score U1"] = df["IoT-U1"] + df["Software Testing-U1"] + df["DAP-U1"] + df["NoSQL-U1"]
    top_scorer_u1 = df.loc[df["Total Score U1"].idxmax()]

    # Calculate total score for Unit Test 2
    df["Total Score U2"] = df["IoT-U2"] + df["Software Testing-U2"] + df["DAP-U2"] + df["NoSQL-U2"]
    top_scorer_u2 = df.loc[df["Total Score U2"].idxmax()]

    return jsonify({
        "Top Scorer U1": {
            "ID": top_scorer_u1["Student Id"], 
            "Name": top_scorer_u1["Name"], 
            "Total Score": int(top_scorer_u1["Total Score U1"])
        },
        "Top Scorer U2": {
            "ID": top_scorer_u2["Student Id"], 
            "Name": top_scorer_u2["Name"], 
            "Total Score": int(top_scorer_u2["Total Score U2"])
        }
    })

@app.route('/performance-graph')
def generate_student_percentage_graph():
    df = pd.read_csv('Copy of dataset(1).csv', header=0)
    student_names = df['Name']
    
    # Calculate percentage for each student
    subject_columns = ['IoT-U1', 'Software Testing-U1', 'DAP-U1', 'NoSQL-U1', 
                       'IoT-U2', 'Software Testing-U2', 'DAP-U2', 'NoSQL-U2']
    
    total_marks = df[subject_columns].sum(axis=1)  # Total marks obtained
    max_marks = len(subject_columns) * 50 # Assuming each subject has max 100 marks
    percentages = (total_marks / max_marks * 100)  # Convert to percentage list

    ffig, ax = plt.subplots(figsize=(15, 6))  # Larger figure
    bars = ax.bar(student_names, percentages, color='teal')

# Add labels above bars
    for bar, percentage in zip(bars, percentages):
     height = bar.get_height()
     ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{percentage:.1f}%', ha='center', fontsize=6, fontweight='bold')

# Set labels and title
    ax.set_xlabel('Students')
    ax.set_ylabel('Percentage (%)', labelpad=15)  # Move Y-label outward
    ax.set_title('Overall Percentage of Students')
    ax.set_ylim([0, 110])  # Keep space above bars

# Rotate X-labels vertically
    ax.set_xticklabels(student_names, rotation=90, ha='right', fontsize=8)

# Adjust layout to avoid cropping
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.2)  # Adjust margins

    img_base64 = save_plot_as_base64(ffig)



    
    first_sem = df['1st Sem Marks']
    second_sem = df['2nd Sem Marks']
    attendance = df['Attendence Rate']
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    x = np.arange(len(student_names))  # X-axis positions
    width = 0.2  # Bar width

    ax2.bar(x - width, first_sem, width, label='1st Sem Marks', color='royalblue')
    ax2.bar(x, second_sem, width, label='2nd Sem Marks', color='teal')
    ax2.bar(x + width, attendance, width, label='Attendance (%)', color='orange')

    ax2.set_xticks(x)
    ax2.set_xticklabels(student_names, rotation=90, ha='right', fontsize=8)
    ax2.set_ylabel('Marks / Attendance (%)')
    ax2.set_title('1st Sem, 2nd Sem & Attendance')
    ax2.legend()
    grouped_bar_graph = save_plot_as_base64(fig2)
    
    return jsonify({'graph': img_base64,
                    'grouped_bar_graph': grouped_bar_graph})  

# Helper function to save Matplotlib plot as base64
def save_plot_as_base64(fig):
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()
    return 'data:image/png;base64,' + img_base64



@app.route("/view_report", methods=["GET"])
def view_report():
    if 'name' not in session or 'usn' not in session:
        return redirect(url_for('dashboard'))  # Redirect if session data is missing
    # Get student details from request parameters
    name = session["name"]
    usn = session["usn"]

    if not name or not usn:
        return redirect(url_for("dashboard"))  # Redirect if no student data

    # Fetch student data from DataFrame
    student_data = df[(df["Name"] == name) & (df["Student Id"] == usn)]

    if student_data.empty:
        return "No data found for the provided Name and USN."

    student_data = student_data.to_dict(orient="records")

# Check if student data is empty
    if not student_data:
        return "No student data found", 404

    student_data = student_data[0]  # Extract first student record

# Ensure column names are correct
    print(student_data.keys())  # Debugging

# List of subjects (make sure they exist in student_data)
    subjects = ["IoT-U1", "Software Testing-U1", "DAP-U1", "NoSQL-U1"]
    subjects1 = ["IoT-U2", "Software Testing-U2", "DAP-U2", "NoSQL-U2"]

# Fetch marks safely
    marks = [student_data.get(sub, 0) for sub in subjects]  # Use .get() to avoid KeyError
    marks1 = [student_data.get(sub, 0) for sub in subjects1] 

# Generate graph
    plt.figure(figsize=(6, 4))
    plt.bar(subjects, marks, color="blue")
    plt.xlabel("Subjects")
    plt.ylabel("Marks")
    plt.title("Unit Test 1 Marks")
    plt.savefig("static/unit_test1_graph.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(subjects1, marks1, color="green")
    plt.xlabel("Subjects")
    plt.ylabel("Marks")
    plt.title("Unit Test 2 Marks")
    plt.savefig("static/unit_test2_graph.png")
    plt.close()


    return render_template("report.html", student=student_data)

from fpdf import FPDF
import os
from flask import Flask, request, send_file, redirect, url_for


@app.route("/download_report")
def download_report():
    # Ensure user is logged in
    if "name" not in session or "usn" not in session:
        return redirect(url_for("login"))  # Redirect to login if session is missing

    # Retrieve student details from session
    name = session["name"]
    usn = session["usn"]

    # Fetch student data from DataFrame
    student_data = df[(df["Name"] == name) & (df["Student Id"] == usn)]

    if student_data.empty:
        return "Student not found", 404

    student_data = student_data.to_dict(orient="records")[0]

    # Create PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Student Performance Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(10)

    # Centering table
    page_width = pdf.w
    table_width = 140
    margin_x = (page_width - table_width) / 2

    # Student Details Table
    pdf.set_x(margin_x)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(60, 10, "Name", border=1, align="C")
    pdf.cell(40, 10, "USN", border=1, align="C")
    pdf.cell(40, 10, "Attendance", border=1, align="C")
    pdf.ln()

    pdf.set_x(margin_x)
    pdf.set_font("Arial", "", 12)
    pdf.cell(60, 10, student_data["Name"], border=1, align="C")
    pdf.cell(40, 10, student_data["Student Id"], border=1, align="C")
    pdf.cell(40, 10, f"{student_data.get('Attendence Rate', 'N/A')}%", border=1, align="C")
    pdf.ln(15)

    # Subject Marks Table
    pdf.set_x(margin_x)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(60, 10, "Subject", border=1, align="C")
    pdf.cell(40, 10, "Unit Test 1", border=1, align="C")
    pdf.cell(40, 10, "Unit Test 2", border=1, align="C")
    pdf.ln()

    subjects = ["IoT", "Software Testing", "DAP", "NoSQL"]
    pdf.set_font("Arial", "", 12)
    for sub in subjects:
        pdf.set_x(margin_x)
        pdf.cell(60, 10, sub, border=1, align="C")
        pdf.cell(40, 10, str(student_data.get(f"{sub}-U1", "N/A")), border=1, align="C")
        pdf.cell(40, 10, str(student_data.get(f"{sub}-U2", "N/A")), border=1, align="C")
        pdf.ln()

    pdf.ln(10)

    # Ensure graphs exist before adding them
    graph_path1 = "static/unit_test1_graph.png"
    graph_path2 = "static/unit_test2_graph.png"

    if os.path.exists(graph_path1) and os.path.exists(graph_path2):
        pdf.cell(200, 10, "Performance Graphs", ln=True, align="C")

        y_position = pdf.get_y() + 5
        graph_width = 80
        total_graph_width = (2 * graph_width) + 10
        margin_graph_x = (page_width - total_graph_width) / 2

        pdf.image(graph_path1, x=margin_graph_x, y=y_position, w=graph_width)
        pdf.image(graph_path2, x=margin_graph_x + graph_width + 10, y=y_position, w=graph_width)

    # Save PDF
    pdf_path = f"static/{usn}_report.pdf"
    pdf.output(pdf_path)

    return send_file(pdf_path, as_attachment=True)


@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        subject = request.form['subject']
        password = request.form['password']

        # Corrected teacher data structure
        teachers = {
            "Dr.Rajeshwari M": {"subject": "IoT", "password": "Rajeshwari2024"},
            "Anil Kumar K": {"subject": ["Software Testing", "Class Advisor"], "password": "Anilkumar2024"},
            "Ramesha K": {"subject": "DAP", "password": "Ramesha2024"},
            "Neema H": {"subject": "NoSQL", "password": "Neema2024"}
        }

        # Check if username exists
        if username in teachers:
            teacher_info = teachers[username]  # Retrieve the dictionary for the teacher

            # Validate subject and password
            if (subject in teacher_info["subject"] if isinstance(teacher_info["subject"], list) else subject == teacher_info["subject"]) and password == teacher_info["password"]:
                session.permanent = True
                session['username'] = username
                session['teacher_subject'] = subject  # Store subject in session

                print(f"Session username: {session['username']}")
                print(f"Session subject: {session['teacher_subject']}")

                return redirect(url_for('admin'))

        return "Invalid credentials! Please check your details."

    return render_template('teacherlogin.html')

@app.route('/teacher-role')
def get_teacher_role():
    teacher_role = session.get("teacher_subject", "Teacher")  # Default to "Teacher"
    return jsonify({"role": teacher_role})


DATA_FILE = "Copy of dataset(1).csv"
@app.route('/student/<string:studentID>', methods=['GET'])
def get_student(studentID):
    DATA_FILE = "dataset.csv"
    df = pd.read_csv(DATA_FILE)

    student_data = df[df["Student Id"] == studentID].to_dict(orient="records")
    
    if not student_data:
        return jsonify({"error": "Student not found"}), 404

    return jsonify(student_data[0])  # Return as JSON
 


@app.route('/update-student/<string:studentID>', methods=['PUT'])
def update_student(studentID):
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Received update request for student: {studentID}")

        if studentID not in df["Student Id"].astype(str).values:
            return jsonify({"error": "Student not found"}), 404

        updated_data = request.json
        print(f"Updated data received: {updated_data}")

        student_index = df.index[df["Student Id"].astype(str) == studentID].tolist()[0]

        print("Before update:", df.loc[student_index])  # Debugging log

        for key, value in updated_data.items():
            if key in df.columns:
                current_value = str(df.at[student_index, key])
                new_value = str(value)

                if current_value != new_value:
                    df.at[student_index, key] = new_value
                    print(f"Updated {key}: {current_value} -> {new_value}")

        print("After update:", df.loc[student_index])  # Debugging log

        df.to_csv(DATA_FILE, index=False)  # Save changes
        print("CSV updated successfully!")  # Debugging log

        return jsonify({"message": f"Updated marks successfully for {studentID}"})

    except Exception as e:
        print(f"Error updating student marks: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/update-students', methods=['PUT'])
def update_students():
    try:
        df = pd.read_csv(DATA_FILE)
        updated_data = request.json
        print(f"Received batch update: {updated_data}")  # Debugging

        for student in updated_data:
            studentID = student["Student Id"]
            if studentID in df["Student Id"].astype(str).values:
                student_index = df.index[df["Student Id"].astype(str) == studentID].tolist()[0]
                for key, value in student.items():
                    if key in df.columns and key != "Student Id":
                        df.at[student_index, key] = value
                        print(f"Updated {key} for {studentID}: {value}")  # Debugging

        print("Data before saving:")
        print(df.head())  # Debugging
        
        df.to_csv(DATA_FILE, index=False)
        print("CSV file updated successfully!")  # Debugging

        return jsonify({"message": "Marks updated successfully!"})

    except Exception as e:
        print(f"Error updating student marks: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500



@app.route('/update-marks')
def update_marks():
    if 'username' not in session:
        return redirect(url_for('login'))  # Redirect if not logged in

    role = session.get('role')

    df = pd.read_csv("Copy of dataset(1).csv")
    students = df.to_dict(orient="records")

    return render_template("edit_marks.html", students=students, role=role)
DATA_FILE="Copy of dataset(1).csv"
@app.route('/edit_marks')
def edit_marks():
    if 'username' not in session or 'teacher_subject' not in session:
        print("Redirecting to login: Session missing required keys.")  # Debuggin
        return redirect(url_for('admin_login'))  # Redirect if not logged in

    subject = session['teacher_subject']
    
    df = pd.read_csv(DATA_FILE)
    
    students = df[['Student Id', 'Name', f"{subject}-U1", f"{subject}-U2"]].rename(
        columns={f"{subject}-U1": "UT1", f"{subject}-U2": "UT2"}
    ).to_dict(orient='records')

    return render_template('edit_marks.html', students=students, subject=subject)

dataset_path = 'Copy of dataset().csv'

def load_data():
    return pd.read_csv(dataset_path)
def save_data(df):
    df.to_csv(dataset_path, index=False)

    
@app.route('/modify-student', methods=['POST'])
def modify_student():
    data = request.json
    print("Received Data:", data)  # Debugging output

    student_id = data.get("Student Id")
    field = data.get("field")
    new_value = data.get("value")

    print(f"Student ID: {student_id}, Field: {field}, New Value: {new_value}")  # Debugging output

    if student_id and field:
        # Check if student exists
        if student_id in df["Student Id"].values:
            df.loc[df["Student Id"] == student_id, field] = new_value
            df.to_csv("Copy of dataset(1).csv", index=False)  # Save to file
            print("Data updated successfully.")  # Debugging output
            return jsonify({"success": True})
        else:
            print("Error: Student ID not found.")  # Debugging output
            return jsonify({"success": False, "error": "Student ID not found"}), 400

    print("Error: Missing student ID or field.")  # Debugging output
    return jsonify({"success": False, "error": "Invalid input"}), 400


@app.route('/report')
def report_page():
    return render_template('teacherreport.html')  

@app.route('/report-data')
def get_report_data():
    teacher_subject = session.get("teacher_subject") 
    print(teacher_subject)  # Example: "DAP"

    if not teacher_subject:
        return jsonify({"error": "Unauthorized access"}), 403

    # Construct column names for Unit Test 1 and Unit Test 2
    subject_u1 = f"{teacher_subject}-U1"
    subject_u2 = f"{teacher_subject}-U2"

    # Check if the subject exists in the dataset
    if subject_u1 not in df.columns or subject_u2 not in df.columns:
        return jsonify({"error": "Subject not found in dataset"}), 404

    total_students = len(df)

    # Calculate average marks
    avg_marks_u1 = df[subject_u1].mean() if not df[subject_u1].isnull().all() else 0
    avg_marks_u2 = df[subject_u2].mean() if not df[subject_u2].isnull().all() else 0

    # Count students in different mark ranges for U1 and U2
    def count_students_in_range(marks_series):
        return {
            "40-50": sum((marks_series >= 40) & (marks_series < 50)),
            "30-40": sum((marks_series >= 30) & (marks_series < 40)),
            "20-30": sum((marks_series >= 20) & (marks_series < 30)),
            "Below 20": sum(marks_series < 20)
        }

    mark_ranges_u1 = count_students_in_range(df[subject_u1])
    mark_ranges_u2 = count_students_in_range(df[subject_u2])

    return jsonify({
        "total_students": total_students,
        "average_marks": {subject_u1: avg_marks_u1, subject_u2: avg_marks_u2},
        "mark_ranges_u1": mark_ranges_u1,
        "mark_ranges_u2": mark_ranges_u2
    })


if __name__ == '__main__':
    app.run(debug=True)
