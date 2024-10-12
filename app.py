import os
import logging
import sys
from flask import Flask, render_template, request, redirect, url_for, flash
from google.generativeai import GenerativeModel, upload_file
import sqlite3
from PIL import Image
import pathlib
import dotenv
import os
import json 

dotenv.load_dotenv()
import google.generativeai as genai

genai.configure(api_key=os.environ['API_KEY'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images/uploads'
app.secret_key = "supersecretkey"

PERSIST_DIR = "./storage"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize SQLite DB
def init_db():
    conn = sqlite3.connect('home_reports.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT,
                        image_path TEXT,
                        fields JSON)''')
    conn.commit()
    conn.close()


@app.template_filter('loads')
def loads_filter(s):
    import json
    return json.loads(s)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file_handler():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        file_path = f"{'static'}/{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(file_path)
        extracted_data = extract_report_data(file_path)
        save_to_db(file.filename, f"{app.config['UPLOAD_FOLDER']}/{file.filename}", extracted_data)
        return redirect(url_for('dashboard'))
    
    
@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('home_reports.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports")
    reports = cursor.fetchall()
    conn.close()

    # Preprocess and group reports by report_type in the backend
    report_types = {}
    unique_fields = set()
    for report in reports:
        # report[0]: id
        # report[1]: file_name
        # report[2]: image_path
        # report[3]: fields (JSON string)

        # Load the JSON fields into a dictionary
        try:
            fields_dict = json.loads(report[3])
        except json.JSONDecodeError:
            fields_dict = {}

        # Get the report_type, default to 'Other Reports' if not present
        report_type = fields_dict.get('report_type', 'Other Reports')

        # Initialize the list for this report_type if not already present
        if report_type not in report_types:
            report_types[report_type] = []

        # Append the report and its fields_dict to the list
        report_types[report_type].append({
            'report': report,
            'fields_dict': fields_dict
        })

        # Collect unique fields
        unique_fields.update(fields_dict.keys())

    total_reports = len(reports)
    total_report_types = len(report_types)
    total_fields = len(unique_fields) - (1 if 'report_type' in unique_fields else 0)  # Exclude 'report_type' field

    return render_template('dashboard.html', report_types=report_types, total_reports=total_reports, total_report_types=total_report_types, total_fields=total_fields)

# Extract data from the uploaded file using Gemini Vision
def extract_report_data(file_path):
    myfile = upload_file(pathlib.Path(file_path))
    model = GenerativeModel("gemini-1.5-flash", generation_config=genai.GenerationConfig(
        response_mime_type="application/json"
    ))
   
    response = model.generate_content([myfile, "\n\n", f"Extract details from this image and return response as a JSON object with key value pairs, where report_type is a mandatory key, with values 'Select from termite_report,natural_hazard_disclosure,transfer_disclosure_statement,hoa_documents,preliminary_title_report,lead_based_paint_disclosure,water_heater_smoke_detector_compliance,closing_statement,home_inspection_report', followed by extracting values for any 5 of the folliowing keys  that match with the report_type,  flood_zone, earthquake_zone, fire_zone, landslide_risk, radon_gas_presence, other_natural_hazards, known_defects, plumbing_issues, electrical_issues, structural_problems, appliances_condition, roof_condition, hvac_condition, repairs_done, remodels_done, neighborhood_issues, zoning_violations, disputes_with_neighbors, hoa_fees, cc_and_rs, hoa_rules, pending_assessments, financial_status_of_hoa, liens_on_property, easements, encumbrances, ownership_history, lead_paint_presence, areas_with_lead_paint, water_heater_strapped, smoke_detectors_installed, final_sale_price, closing_costs, credits_or_debits_to_buyer_or_seller, general_property_condition, recommended_repairs, major_deficiencies, safety_hazards, estimated_repair_costs, wall_type, material_used, termite_status, water_line_type "])
    return json.loads(response.text)

# Save extracted data into the database
def save_to_db(file_name,  image_path, data):
    conn = sqlite3.connect('home_reports.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO reports (file_name, image_path, fields)
                      VALUES (?, ?, ?)''', (file_name,  image_path, json.dumps(data)))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)