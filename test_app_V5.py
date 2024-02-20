import os
import re
import time
import uuid
import json
import pprint

# import json
# import spacy
# import openai
import base64
import shutil
import tempfile
import pypandoc
import subprocess
import streamlit as st
from io import BytesIO
from joblib import load
from pathlib import Path
from zipfile import ZipFile
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# from prompts import queries_list
from subprocess import PIPE, run
from datetime import datetime, timedelta
from pdf_extraction import get_pdf_content
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from pdfminer.high_level import extract_text
# from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from sklearn.feature_extraction.text import TfidfVectorizer
from kor import create_extraction_chain, Object, Text 
from langchain.chat_models import ChatAnthropic
import os
from bs4 import BeautifulSoup
from joblib import load
from bs4 import BeautifulSoup

anthropic = Anthropic()

llm = ChatAnthropic(
    model_name="claude-v1.3-100k",
    temperature=0,
    max_tokens=8000)

schema_1 = Object(
    id="recommendation name",
    description="""
        description of the recommendation
    """,
    attributes=[
        Text(
            id="title",
            description="recommendation name"
        )
    ],
    examples=[
        (
            """Review Will
As your personal circumstances change, it is important that your Will is kept up to date to ensure your assets are transferred to your intended beneficiaries according to your wishes. 
Colleen, we recommend you seek legal advice to have your Will reviewed.
Why this benefits you	•	Having an appropriate estate plan in place ensures the ownership and control of your assets are transferred to your intended beneficiaries according to your wishes. This will also ensure your family is not faced with difficult decisions or doubts about your intentions upon death. 
•	This will give you peace of mind on matters relating to your estate, both financial and personal. 
•	A robust estate plan may minimise the tax payable on the income and capital gains earned on assets transferred.

Things you should consider	•	Not all assets are included in your estate for distribution via your Will. Non-estate assets include superannuation, account-based pensions, life insurance policies, jointly owned assets and assets owned through a trust and/or company.
•	If you were to pass away without a valid Will in place, the distribution of your assets will be determined by your relevant State or Territory, which may not reflect your wishes and could result in delays, conflicts and excess costs, which could otherwise have been avoided.
•	Your Will should be reviewed at least once every few years or when your circumstances change to ensure it reflects your current wishes and remains legal and valid.

""",
            [
                {"title": """Review Will"""
                },
            ]
        )
    ]
)

summarization_query_1_4 = """summarize 'why this benefits you','"Things you should consider","Alternatives considered" and  ""recommendations" sections from the above json and ammend under each section. 
give the summary under each section of the json and output the same json object.I want the output as a pure json object with all the sections from the original json summarized. No need to write anything other than the json object in the final output"""

summarization_query_2 = """summarize 'Advice to help you achieve this','"Benefits","Things you should know" and  ""Alternative strategies considered" sections from the above json and ammend under each section. 
give the summary under each section of the json and output the same json object.I want the output as a pure json object with all the sections from the original json summarized. No need to write anything other than the json object in the final output"""

summarization_query_3 = """summarize 'Reasons','Risks and Consequences' sections from the above json and ammend under each section. 
give the summary under each section of the json and output the same json object.I want the output as a pure json object with all the sections from the original json summarized. No need to write anything other than the json object in the final output"""

st.set_page_config(page_title="Office to PDF Converter",
                   page_icon='📄',
                   layout='wide',
                   initial_sidebar_state='expanded')

# apply custom css if needed
with open(Path('utils/style.css')) as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


@st.cache_resource(ttl=60*60*24)
def cleanup_tempdir() -> None:
    '''Cleanup temp dir for all user sessions.
    Filters the temp dir for uuid4 subdirs.
    Deletes them if they exist and are older than 1 day.
    '''
    deleteTime = datetime.now() - timedelta(days=1)
    # compile regex for uuid4
    uuid4_regex = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    uuid4_regex = re.compile(uuid4_regex)
    tempfiledir = Path(tempfile.gettempdir())
    if tempfiledir.exists():
        subdirs = [x for x in tempfiledir.iterdir() if x.is_dir()]
        subdirs_match = [x for x in subdirs if uuid4_regex.match(x.name)]
        for subdir in subdirs_match:
            itemTime = datetime.fromtimestamp(subdir.stat().st_mtime)
            if itemTime < deleteTime:
                shutil.rmtree(subdir)


@st.cache_data(show_spinner=False)
def make_tempdir() -> Path:
    '''Make temp dir for each user session and return path to it
    returns: Path to temp dir
    '''
    if 'tempfiledir' not in st.session_state:
        tempfiledir = Path(tempfile.gettempdir())
        tempfiledir = tempfiledir.joinpath(
            f"{uuid.uuid4()}")   # make unique subdir
        # make dir if not exists
        tempfiledir.mkdir(parents=True, exist_ok=True)
        st.session_state['tempfiledir'] = tempfiledir
    return st.session_state['tempfiledir']


def store_file_in_tempdir(tmpdirname: Path, uploaded_file: BytesIO) -> Path:
    '''Store file in temp dir and return path to it
    params: tmpdirname: Path to temp dir
            uploaded_file: BytesIO object
    returns: Path to stored file
    '''
    # store file in temp dir
    tmpfile = tmpdirname.joinpath(uploaded_file.name)
    with open(tmpfile, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return tmpfile


def convert_doc_to_pdf_native(doc_file: Path, output_dir: Path = Path("."), timeout: int = 60):
    """Converts a doc file to pdf using libreoffice without msoffice2pdf.
    Calls libroeoffice (soffice) directly in headless mode.
    params: doc_file: Path to doc file
            output_dir: Path to output dir
            timeout: timeout for subprocess in seconds
    returns: (output, exception)
            output: Path to converted file
            exception: Exception if conversion failed
    """
    exception = None
    output = None
    try:
        process = run(['soffice', '--headless', '--convert-to',
                       'pdf:writer_pdf_Export', '--outdir', output_dir.resolve(), doc_file.resolve()],
                      stdout=PIPE, stderr=PIPE,
                      timeout=timeout, check=True)
        stdout = process.stdout.decode("utf-8")
        re_filename = re.search('-> (.*?) using filter', stdout)
        output = Path(re_filename[1]).resolve()
    except Exception as e:
        exception = e
    return (output, exception)


@st.cache_data(show_spinner=False)
def get_base64_encoded_bytes(file_bytes) -> str:
    base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
    return base64_encoded


@st.cache_data(show_spinner=False)
def show_pdf_base64(base64_pdf):
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_versions() -> str:
    result = run(["soffice", "--version"], capture_output=True, text=True)
    libreoffice_version = result.stdout.strip()
    versions = f'''
    - `Streamlit {st.__version__}`
    - `{libreoffice_version}`
    '''
    return versions


def get_all_files_in_tempdir(tempfiledir: Path) -> list:
    files = [x for x in tempfiledir.iterdir() if x.is_file()]
    files = sorted(files, key=lambda f: f.stat().st_mtime)
    return files


def get_pdf_files_in_tempdir(tempfiledir: Path) -> list:
    files = [x for x in tempfiledir.iterdir() if x.is_file()
             and x.suffix == '.pdf']
    files = sorted(files, key=lambda f: f.stat().st_mtime)
    return files


def get_zip_files_in_tempdir(tempfiledir: Path) -> list:
    files = [x for x in tempfiledir.iterdir() if x.is_file()
             and x.suffix == '.zip']
    files = sorted(files, key=lambda f: f.stat().st_mtime)
    return files


def make_zipfile_from_filelist(filelist: list, output_dir: Path = Path("."), zipname: str = "Converted.zip") -> Path:
    """Make zipfile from list of files
    params: filelist: list of files
            output_dir: Path to output dir
            zipname: name of zipfile
    returns: Path to zipfile
    """
    zip_path = output_dir.joinpath(zipname)
    # check if filelist is empty and don't create zipfile resp. delete it
    if not filelist:
        zip_path.unlink(missing_ok=True)
        return None
    else:
        with ZipFile(zip_path, 'w') as zipObj:
            for file in filelist:
                zipObj.write(file, file.name)
        return zip_path


def delete_all_files_in_tempdir(tempfiledir: Path):
    for file in get_all_files_in_tempdir(tempfiledir):
        file.unlink()


def delete_files_from_tempdir_with_same_stem(tempfiledir: Path, file_path: Path):
    file_stem = file_path.stem
    for file in get_all_files_in_tempdir(tempfiledir):
        if file.stem == file_stem:
            file.unlink()


def get_bytes_from_file(file_path: Path) -> bytes:
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return file_bytes


def check_if_file_with_same_name_and_hash_exists(tempfiledir: Path, file_name: str, hashval: int) -> bool:
    """Check if file with same name and hash already exists in tempdir
    params: tempfiledir: Path to file
            file_name: name of file
            hashval: hash of file
    returns: True if file with same name and hash already exists in tempdir
    """
    file_path = tempfiledir.joinpath(file_name)
    if file_path.exists():
        file_hash = hash((file_path.name, file_path.stat().st_size))
        if file_hash == hashval:
            return True
    return False


def get_related_topics_content(predicted_topic, topics_texts_dict, top_n=2):
    """
    Given a predicted topic and a dictionary of topics with their contents,
    this function returns the top N related topics and their content.

    :param predicted_topic: The predicted topic as a string
    :param topics_texts_dict: Dictionary of topics and their corresponding content
    :param top_n: Number of top related topics to return
    :return: Dictionary of top N related topics and their content
    """
    # Ensure the number of topics requested does not exceed the number of available topics
    top_n = min(top_n, len(topics_texts_dict))
    
    # Vectorize the topics and the predicted topic
    vectorizer = TfidfVectorizer()
    topic_vectors = vectorizer.fit_transform(topics_texts_dict.keys())
    predicted_vector = vectorizer.transform([predicted_topic])
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(predicted_vector, topic_vectors).flatten()
    
    # Rank the topics based on similarity
    top_topic_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Get the top N related topics and their content
    top_topics_content = {list(topics_texts_dict.keys())[index]: topics_texts_dict[list(topics_texts_dict.keys())[index]]
                          for index in top_topic_indices}
    
    return top_topics_content

topics = []
texts = []

def topic_content_extractor_for_docx(file, html_directory):
    os.makedirs(html_directory, exist_ok=True)
    docx_file = os.path.join(html_directory, file)
    html_file_name = os.path.splitext(file)[0] + ".html"
    html_file_path = os.path.join(html_directory, html_file_name)

    try:
        # Convert the DOCX file to HTML
        pypandoc.convert_file(docx_file, 'html', outputfile=html_file_path)
        print(f"Conversion complete: {file} -> {html_file_name}")

        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find and manipulate the specific tags
        topics = [tag.get_text(strip=True) for tag in soup.find_all('h1')]
        texts = []
        html_texts = []
        tag_list = {'p', 'td', 'h2'}

        for section in soup.find_all('h1'):
            section_text = ""
            section_html = ""

            # Iterate over sibling elements after each h1 tag until the next h1 tag
            for sibling in section.find_all_next():
                if sibling.name == 'h1':
                    break
                elif sibling.name in tag_list:
                    section_text += sibling.get_text(strip=True) + ' '

            for sibling in section.find_all_next():
                if sibling.name == 'h1':
                    break
                else:
                    section_html += str(sibling)

            texts.append(section_text)
            html_texts.append(section_html)

        topics_texts_dict = dict(zip(topics, texts))
        topics_html_dict = dict(zip(topics, html_texts))

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

    return topics_texts_dict,topics_html_dict

def topic_content_extractor_for_doc(file, html_directory):
    os.makedirs(html_directory, exist_ok=True)
    docx_file = os.path.join(html_directory, file)
    output_docx_file = os.path.join(html_directory, os.path.splitext(file)[0]+ ".docx")

    try:
        subprocess.run(['unoconv', '--format', 'docx', '--output', output_docx_file, docx_file])
        print(f"Conversion completed {file} -> {file}. Output saved to: {output_docx_file}")
        # Construct the output HTML file path with the same name (but ".html" extension)
        html_file_name = os.path.splitext(output_docx_file)[0] + ".html"
        html_file_path = os.path.join(html_directory, html_file_name)

        try:
            # Convert the DOCX file to HTML
            pypandoc.convert_file(output_docx_file, 'html', outputfile=html_file_path)
            print(f"HTML Conversion complete: {output_docx_file} -> {html_file_name}")

            # Read the HTML file
            with open(html_file_path, 'r', encoding='utf-8') as html_file:
                html_content = html_file.read()

            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find and manipulate the specific tags
            topics = [tag.get_text(strip=True) for tag in soup.find_all('h1')]
            texts = []
            tag_list = {'p', 'td', 'h2'}

            for i, section in enumerate(soup.find_all('h1')):
                section_text = ""

                # Iterate over sibling elements after each h1 tag until the next h1 tag
                for sibling in section.find_all_next():
                    if sibling.name == 'h1':
                        break
                    elif sibling.name in tag_list:
                        section_text += sibling.get_text(strip=True) + ' '

                texts.append(section_text)

            topics_texts_dict = dict(zip(topics, texts))

        except Exception as e:
            print(e)

    except Exception as e:
        print(f"Extraction failed: {e}")

    return topics_texts_dict

def parse_wealth_recommendations_type_1_4(html,titles_list):

    soup = BeautifulSoup(html, 'html.parser')

    table_dict = {}

    table_list = []

    prev_table_dict = {}

    tags = soup.find_all(['h2','strong'])

    tags_with_text = [tag for tag in tags if tag.get_text(strip=True)]

    tags_list = [i for n, i in enumerate(tags_with_text) if i not in tags_with_text[:n]]

    for section in tags_with_text:

        table_list = []
        temp_dict = {}
        final_list = []

        capture_li_items = True
        li_items_before_table = []

        for sibling in section.find_all_next():
            if sibling.name in (['h2','strong']):
                break
            elif sibling.name == 'table':
                table_list.append(sibling)

                for prev_sibling in sibling.find_all_previous():
                    
                    if capture_li_items and prev_sibling.name in ('p'):
                        # print(prev_sibling.get_text())
                        li_items_before_table.append(prev_sibling.get_text())

                    elif prev_sibling.name in (['h2', 'strong']):
                        break
                    # Reset the flag after the first table
                capture_li_items = False

                res = [i for n, i in enumerate(li_items_before_table) if i not in li_items_before_table[:n]]
                try:
                    temp_dict['recommendations'] = res
                except:
                    continue

        
        table_dict[section.text.strip()] = table_list
        prev_table_dict[section.text.strip()] = temp_dict

        final_dict = {}
        list_1 = []

        for k,v in table_dict.items():
            section_dict = {}
            list_1 = []
            k = k.replace('\n', ' ')
            

            for key, value in prev_table_dict.items():
                key = key.replace('\n', ' ')

                if key in final_dict:
                    final_dict[key]['recommendations'] = value.get('recommendations', [])


            for table in v:
                list_1 = []
                rows = table.find_all('tr')
                for row in rows:
                    title = row.find(['td','th']).text.strip()
                    li = row.find_all('li')

                    for list in li:
                        list_1.append(list.text.strip())
                section_dict[title] = list_1

            final_dict[k.strip()] =  section_dict 
                    
            filtered_dict = {key: value for key, value in final_dict.items() if key in titles_list}

    return filtered_dict


from bs4 import BeautifulSoup

def parse_wealth_recommendations_type_3(html):
    soup = BeautifulSoup(html, 'html.parser')

    table_dict = {}
    prev_table_dict = {}
    capture_li_items = True


    tags = soup.find_all(['h2', 'strong'])

    tags_with_text = [tag for tag in tags if tag.get_text(strip=True)]

    for section in tags_with_text:
        item_dict = {}
        list_items = []
        temp_dict = {}
        prev_table_dict = {}
        li_items_before_table = []
        capture_li_items = True

        for sibling in section.find_all_next():
            res = []
            if sibling.name in (['h2','strong']):
                break
            elif sibling.name == 'h3':
                for prev_sibling in sibling.find_all_previous():
                        
                    if capture_li_items and prev_sibling.name in ('li','p'):
                        # print(prev_sibling.get_text())
                        li_items_before_table.append(prev_sibling.get_text())

                    elif prev_sibling.name in (['h2','strong']):
                        break
                        # Reset the flag after the first table
                capture_li_items = False

                recommendations_res = [i for n, i in enumerate(li_items_before_table) if i not in li_items_before_table[:n]]
                try:
                    temp_dict['recommendations'] = recommendations_res
                except:
                    continue

                list_items = []
                for item in sibling.find_all_next():
                    
                    if item.name in ['h3','strong']:
                        break
                    else:   
                        list_items.append(item.get_text())
                        [res.append(x) for x in list_items if x not in res]
                item_dict[sibling.get_text()] = res
                
        table_dict[section.text.strip()] = item_dict
        prev_table_dict[section.text.strip()] = temp_dict
        
        for key, value in prev_table_dict.items():
            key = key.replace('\n', ' ')

            if key in table_dict:
                table_dict[key]['recommendations'] = value.get('recommendations', [])
                
    return table_dict



def parse_wealth_recommendations_type_2(html):

    soup = BeautifulSoup(html, 'html.parser')

    final_dict = {}

    section_dict = {}
    
    tables = soup.findAll("table")

    
    for table in tables:
            title = ''
            list_1 = []
            title_list = []
            title_list_string = ''
            rows = table.find_all('tr')
            for row in rows:
                try:
                    title = row.find('th').text.strip()
                except:
                    pass

                li = row.find_all('li')

                if(len(li)==0):
                     title_list = row.find_all('th')
                     for title in title_list:
                        title_list_string = ''.join(title.text.strip())
                     final_dict[title_list_string] = section_dict
                     section_dict = {}
                     break

                for list in li:
                    list_1.append(list.text.strip())
                section_dict[title] = list_1

    return final_dict

def get_document_type(file,html_directory):
    os.makedirs(html_directory, exist_ok=True)
    docx_file = os.path.join(html_directory, file)
    html_file_name = os.path.splitext(file)[0] + ".html"
    html_file_path = os.path.join(html_directory, html_file_name)

    try:
        # Convert the DOCX file to HTML
        pypandoc.convert_file(docx_file, 'html', outputfile=html_file_path)
        # print(f"Conversion complete: {file} -> {html_file_name}")

        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

            # Get the tag list for the file
        tag_list = [tag.name for tag in soup.find_all()]


        result = ','.join(tag_list)

        # To load the model from the file
        model_type = load('nb_model_type.joblib')

        topics = model_type.inference(document=result, iteration=100, times=10)
        
        if topics[0][0] == 'common_topic':
            topic_name =  topics[1][0]
        else:
            topic_name =  topics[0][0]

        return topic_name
    
    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    load_dotenv()
    cleanup_tempdir()  # cleanup temp dir from previous user sessions
    tmpdirname = make_tempdir()  # make temp dir for each user session
    headercol1, headercol2 = st.columns([8, 1], gap='large')
    with headercol1:
        st.title('PlanLogic - Info. Retrieval Tool 📄')
    # with headercol2:
    #     st.image('resources/pdf.png', width=60)
    st.markdown('''---''')
    # add streamlit 2 column layout
    col1, col2 = st.columns([6, 8], gap='large')
    pdf_file = None
    pdf_bytes = None
    
    with col1:
        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        st.subheader('Upload File')
        with st.form("convert", clear_on_submit=True):
            uploaded_file = st.file_uploader(label='Upload a file', type=['docx', 'doc', 'pdf'], accept_multiple_files=False, key='file_uploader')
            submitted = st.form_submit_button("Start Process")

        if submitted and uploaded_file is not None:

            file_extension = uploaded_file.name.split('.')[-1].lower()

            file_type = uploaded_file.name.split(' ')[2].lower()

            uploaded_file_hash = hash((uploaded_file.name, uploaded_file.size))
            if check_if_file_with_same_name_and_hash_exists(tempfiledir=tmpdirname, file_name=uploaded_file.name, hashval=uploaded_file_hash) is False:
                
                # store file in temp dir
                tmpfile = store_file_in_tempdir(
                    tmpdirname, uploaded_file)

                
                if file_extension == 'docx':
                    # convert file to pdf
                    with st.spinner('Talking to LLM..'):
                        topics_texts_dict,topics_html_dict = topic_content_extractor_for_docx(tmpfile, tmpdirname)

                        # file_type = get_document_type(tmpfile,tmpdirname)
                        chain = create_extraction_chain(llm, schema_1, encoder_or_encoder_class='json')

                        predicted_topic = "Wealth Strategy recommendations"
                        start_time = time.time()
                        top_related_topics_content = get_related_topics_content(predicted_topic, topics_texts_dict, top_n=1)
                        top_related_topics_html = get_related_topics_content(predicted_topic, topics_html_dict, top_n=1)
                        # Join the values into a single string
                        content_string = ' '.join(top_related_topics_content.values())
                        # print(content_string)
                        html_string = ' '.join(top_related_topics_html.values())
                        
                        if file_type == '1':
                            chain = create_extraction_chain(llm, schema_1, encoder_or_encoder_class='json')

                            my_dict = chain.run(content_string)['data']

                            # Extract titles into a separate list
                            titles_list = [value['title'] for values in my_dict.values() for value in values]

                            final_dict= parse_wealth_recommendations_type_1_4(html_string,titles_list)
                        
                            users = json.dumps(final_dict, indent=4)

                            print(users)

                            actual_prompt = f"{HUMAN_PROMPT}<json>{users}</json>,\n\n{summarization_query_1_4}\n\n{AI_PROMPT}"
                            stream = anthropic.completions.create(prompt=actual_prompt, max_tokens_to_sample=4000, model="claude-v1.3-100k", stream=True)

                            # json_string = re.search('<json>(.*?)</json>',stream.completion)
                            # print(str(stream.completion))
                            print("SUMMARY")
                            for completion in stream:
                                    print(completion.completion, end="", flush=True)
                            # if json_string:
                            #     json_string = json_string.group(1)
                            #     try:
                            #         json_data = json.loads(json_string)
                            #         st.write(json_data)
                            #     except json.JSONDecodeError:
                            #         st.write("Invalid JSON format")
                            # else:
                            #     st.write("No JSON found in the string")
                            

                        elif file_type == '2':

                            final_dict= parse_wealth_recommendations_type_2(html_string)
                        
                            users = json.dumps(final_dict, indent=4)

                            print(users)

                            actual_prompt = f"{HUMAN_PROMPT}<json>{users}</json>,\n\n{summarization_query_2}\n\n{AI_PROMPT}"
                            stream = anthropic.completions.create(prompt=actual_prompt, max_tokens_to_sample=4000, model="claude-v1.3-100k", stream=True)
                            
                            for completion in stream:
                                    print(completion.completion, end="", flush=True)

                        elif file_type == '3':

                            final_dict= parse_wealth_recommendations_type_3(html_string)
                        
                            users = json.dumps(final_dict, indent=4)

                            print(users)

                            actual_prompt = f"{HUMAN_PROMPT}<json>{users}</json>,\n\n{summarization_query_3}\n\n{AI_PROMPT}"
                            stream = anthropic.completions.create(prompt=actual_prompt, max_tokens_to_sample=4000, model="claude-v1.3-100k", stream=True)
                            
                            for completion in stream:
                                    print(completion.completion, end="", flush=True)

                        elif file_type == '4':

                            chain = create_extraction_chain(llm, schema_1, encoder_or_encoder_class='json')

                            my_dict = chain.run(content_string)['data']

                            # Extract titles into a separate list
                            titles_list = [value['title'] for values in my_dict.values() for value in values]

                            final_dict= parse_wealth_recommendations_type_1_4(html_string,titles_list)
                        
                            users = json.dumps(final_dict, indent=4)

                            print(users)

                            actual_prompt = f"{HUMAN_PROMPT}<json>{users}</json>,\n\n{summarization_query_1_4}\n\n{AI_PROMPT}"
                            stream = anthropic.completions.create(prompt=actual_prompt, max_tokens_to_sample=4000, model="claude-v1.3-100k", stream=True)
                            
                            for completion in stream:
                                    print(completion.completion, end="", flush=True)

                        
                        # actual_prompt = f"{HUMAN_PROMPT}<json>{users}</json>,\n\n{summarization_query}\n\n{AI_PROMPT}"
                        # stream = anthropic.completions.create(prompt=actual_prompt, max_tokens_to_sample=4000, model="claude-v1.3-100k", stream=True)
                        
                        # for completion in stream:
                        #         print(completion.completion, end="", flush=True)
                        end_time = time.time()
                        duration = end_time - start_time
                        st.success('Completed!')
                        st.write(f"The process took {duration:.2f} seconds.")
                            
                        # client = Anthropic()
                        # token_count = client.count_tokens(pdf_text)
                        # st.write(token_count)     

               

    with col2:
        st.subheader('Download/Delete File')
        pdf_files_in_temp = get_pdf_files_in_tempdir(tmpdirname)
        # make subcolums for download or deleting single pdf files
        if len(pdf_files_in_temp) > 0:
            file_bytes = list()
            for index, file in enumerate(pdf_files_in_temp):
                file_bytes.append(get_bytes_from_file(file))
                subcol1, subcol2, subcol3, subcol4 = st.columns([12, 3, 3, 3])
                with subcol1:
                    st.info(file.name)

                with subcol2:
                    st.download_button(label='Download',
                                       data=file_bytes[index],
                                       file_name=file.name,
                                       mime='application/octet-stream',
                                       key=f'download_button_{index}')
                with subcol3:
                    if st.button('Delete', key=f'delete_button_{index}'):
                        delete_files_from_tempdir_with_same_stem(
                            tmpdirname, file)
                        st.rerun()
            # download button for all pdf files as zip
            st.markdown('''<br>''', unsafe_allow_html=True)
            zip_path = make_zipfile_from_filelist(
                pdf_files_in_temp, tmpdirname)
            zip_bytes = get_bytes_from_file(zip_path)
            st.download_button(label='Download all PDF files as single ZIP file',
                               data=zip_bytes,
                               file_name=zip_path.name,
                               mime='application/octet-stream')
            if st.button('Delete all files from Temporary folder', key='delete_all_button'):
                delete_all_files_in_tempdir(tmpdirname)
                st.rerun()
        else:
            st.warning('No PDF files available for download.')