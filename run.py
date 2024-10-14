from flask import Flask, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import zipfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

from llama_index.llms.openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import dotenv
import json
import uuid
import shutil
from flask_sqlalchemy import SQLAlchemy
dotenv.load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.environ['API_KEY']

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'zip'}
INDEXES_ROOT = 'indexes'

# Ensure upload and index folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEXES_ROOT, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///properties.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database model for property details
class PropertyDetails(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    index_folder_name = db.Column(db.String(100), nullable=False) 
    address = db.Column(db.String(200))
    exterior_walls = db.Column(db.String(100))
    exterior_windows = db.Column(db.String(100))
    exterior_doors = db.Column(db.String(100))
    roof_type_and_age = db.Column(db.String(100))
    fencing_type = db.Column(db.String(100))
    garage_type = db.Column(db.String(100))
    lot_size = db.Column(db.String(100))
    house_size = db.Column(db.String(100))
    bedrooms = db.Column(db.String(10))
    bathrooms = db.Column(db.String(10))
    lot_topography = db.Column(db.String(100))
    driveway = db.Column(db.String(100))
    walkway_and_sidewalks = db.Column(db.String(100))
    porch_deck_and_patio_covers = db.Column(db.String(100))
    fascia_eaves_and_rafters = db.Column(db.String(100))
    built_year = db.Column(db.String(20))
    interior_details = db.Column(db.String(500))
    electrical_panel_rating = db.Column(db.String(100))
    heating_and_cooling = db.Column(db.String(100))
    fireplace_or_chimney = db.Column(db.String(100))
    plumbing = db.Column(db.String(100))
    utilities = db.Column(db.String(100))
    appliances = db.Column(db.String(100))
    # Add other fields as needed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_property_to_db(property_data, index_folder_name):
    print(property_data)
    property_details = PropertyDetails(
        address=property_data.get('address'),
        index_folder_name=index_folder_name,
        exterior_walls=property_data.get('exterior_walls'),
        exterior_windows=property_data.get('exterior_windows'),
        exterior_doors=property_data.get('exterior_doors'),
        roof_type_and_age=property_data.get('roof_type_and_age'),
        fencing_type=property_data.get('fencing_type'),
        garage_type=property_data.get('garage_type'),
        lot_size=property_data.get('lot_size'),
        house_size=property_data.get('house_size'),
        bedrooms=property_data.get('bedrooms'),
        bathrooms=property_data.get('bathrooms'),
        lot_topography=property_data.get('lot_topography'),
        driveway=property_data.get('driveway'),
        walkway_and_sidewalks=property_data.get('walkway_and_sidewalks'),
        porch_deck_and_patio_covers=property_data.get('porch_deck_and_patio_covers'),
        fascia_eaves_and_rafters=property_data.get('fascia_eaves_and_rafters'),
        built_year=property_data.get('built_year'),
    )
    db.session.add(property_details)
    db.session.commit()



def create_index_from_pdfs(directory_path, index_folder_path, index_folder_name):
    system_prompt = (
        "Extract all property details including address, year built, lot size, house size, bedrooms, bathrooms, "
        "areas, construction type, foundation, wall and ceiling materials, garage details, exterior and interior "
        "features, roof details, electrical panel rating, heating, cooling, fireplace, insulation, plumbing, utilities, "
        "and appliance specifications from the provided document."
    )
    llm = OpenAI(api_key=os.environ['API_KEY'], temperature=0.1, model="gpt-4", system_prompt=system_prompt)

    # Manually collect all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {directory_path}.")

    print(f"Found PDF files: {pdf_files}")

    # Use SimpleDirectoryReader with the list of PDF files
    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
    print('Documents loaded.')

    # Create the index
    index = VectorStoreIndex.from_documents(documents, llm=llm)
    print('Index created.')

    # Persist the index to the unique folder
    index.storage_context.persist(persist_dir=index_folder_path)
    print('Index saved.')
    # Query the LLM for property details and save to the database
    llm = OpenAI(api_key=os.environ['API_KEY'], temperature=0.1, model="gpt-4")
    sllm = llm.as_structured_llm(output_cls=PropertyModel)

    # Create the query engine using the structured LLM
    query_engine = index.as_query_engine(llm=sllm)
    response = query_engine.query("Extract the details for the property including address, built year, lot size (sqft), house size (sqft), number of bedrooms, bathrooms, areas (kitchen, dining room, living room, laundry room, garage with type, garage door type, and opener status, deck, gazebo, pool), construction type, foundation, walls, ceiling, attic (with access location), crawl space or basement (with access location), exterior (walls, windows, doors, roof type and age, rain gutters, fencing type and location), interior (walls, ceiling, flooring for each area), electrical panel rating, heating and cooling systems (source, system type, manufacturer, and location), fireplace or chimney, plumbing (water heater details, supply piping, main valve location), utilities (electricity, gas, water, sewer, provider names), and appliances (cooktop type, refrigerator, dishwasher, microwave, oven, washer, dryer details).")
    structured_output = response.response.dict() # Convert to dictionary
    print(structured_output)
    property_data = structured_output
    # get the index folder name
    # index_folder_name  = index_folder_path.split('/')[-1]
    # Save property details to the database
    save_property_to_db(property_data, index_folder_name)

def extract_and_create_index(zip_path, index_folder_path, index_folder_name):
    extract_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted', index_folder_name)
    os.makedirs(extract_folder, exist_ok=True)
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Debug: List extracted files and directories
    print(f"Extracted contents of ZIP file to {extract_folder}:")
    for root, dirs, files in os.walk(extract_folder):
        level = root.replace(extract_folder, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

    # Create index from PDFs
    create_index_from_pdfs(extract_folder, index_folder_path, index_folder_name)

    # Delete the extracted PDFs
    shutil.rmtree(extract_folder)


class PropertyModel(BaseModel):
    address: Optional[str]
    exterior_walls: Optional[str]
    exterior_windows: Optional[str]
    exterior_doors: Optional[str]
    roof_type_and_age: Optional[str]
    rain_gutters: Optional[str]
    fencing_type: Optional[str]
    fencing_location: Optional[str]
    garage_type: Optional[str]
    garage_door_type: Optional[str]
    garage_opener_status: Optional[str]
    lot_topography: Optional[str]
    driveway: Optional[str]
    walkway_and_sidewalks: Optional[str]
    porch_deck_and_patio_covers: Optional[str]
    fascia_eaves_and_rafters: Optional[str]
    built_year: Optional[str]
    lot_size: Optional[str]
    house_size: Optional[str]
    bedrooms: Optional[str]
    bathrooms: Optional[str]
    interior_details: Optional[str]
    electrical_panel_rating: Optional[str]
    heating_and_cooling: Optional[str]
    fireplace_or_chimney: Optional[str]
    plumbing: Optional[str]
    utilities: Optional[str]
    appliances: Optional[str]

def get_query_engine(index_folder_name):
    index_folder_path = os.path.join(INDEXES_ROOT, index_folder_name)
    storage_context = StorageContext.from_defaults(persist_dir=index_folder_path)
    index = load_index_from_storage(storage_context)

    llm = OpenAI(api_key=os.environ['API_KEY'], temperature=0.1, model="gpt-4")
    sllm = llm.as_structured_llm(output_cls=PropertyModel)

    # Create the query engine using the structured LLM
    query_engine = index.as_query_engine(llm=sllm)
    return query_engine


def get_query_engine2(index_folder_name):
    index_folder_path = os.path.join(INDEXES_ROOT, index_folder_name)
    storage_context = StorageContext.from_defaults(persist_dir=index_folder_path)
    index = load_index_from_storage(storage_context)

    llm = OpenAI(api_key=os.environ['API_KEY'], temperature=0.1, model="gpt-4")
    # sllm = llm.as_structured_llm(output_cls=PropertyModel)

    # Create the query engine using the structured LLM
    query_engine = index.as_query_engine(llm=llm)
    return query_engine

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(zip_path)
            index_folder_name = str(uuid.uuid4())
            index_folder_path = os.path.join(INDEXES_ROOT, index_folder_name)
            os.makedirs(index_folder_path, exist_ok=True)
            extract_and_create_index(zip_path, index_folder_path, index_folder_name)
            os.remove(zip_path)
            # Redirect to dashboard
            return redirect(url_for('dashboard'))
    return render_template('upload.html')


@app.route('/query/<index_folder_name>', methods=['GET', 'POST'])
def query_index(index_folder_name):
    if request.method == 'POST':
        user_query = request.form['query']
        # Get query engine for the index folder
        try:
            query_engine = get_query_engine2(index_folder_name)
            response = query_engine.query(user_query)
            # Get structured output
            structured_output = response.response  # This is an instance of PropertyModel
            # # Convert to dict
            # response_dict = structured_output.dict()
            # # Convert dict to JSON string for display
            # response_json = json.dumps(response_dict, indent=4)
            # Render template
            return render_template('query.html', index_folder_name=index_folder_name, response=structured_output, query=user_query)
        except Exception as e:
            return f"An error occurred: {e}"
    else:
        # Render a form to accept the query
        return render_template('query.html', index_folder_name=index_folder_name)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    # Read all index folders and retrieve property details
    properties = PropertyDetails.query.all()
    
    return render_template('dashboard2.html', properties=properties)



if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Creates the database tables
    app.run(debug=True)
