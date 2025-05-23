import streamlit as st
import subprocess
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Property Recommender",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create a dummy style.css if it doesn't exist, or use your actual one
css_file_path = "style.css"
if not os.path.exists(css_file_path):
    with open(css_file_path, "w") as f:
        f.write("""
        /* General Body Styles */
        body {
            font-family: 'Arial', sans-serif;
            color: #333;
            background-color: #f4f4f9; /* Light grey background */
        }

        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* Titles and Headers */
        h1, h2, h3 {
            color: #2c3e50; /* Dark blue-grey for headers */
            font-weight: bold;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 0.5em;
            border-bottom: 2px solid #3498db; /* Blue accent for main title */
            padding-bottom: 0.3em;
        }

        h2 {
            font-size: 1.8em;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }

        /* Buttons */
        .stButton>button {
            background-color: #3498db; /* Blue */
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 1.1em;
            font-weight: bold;
            transition: background-color 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: #2980b9; /* Darker Blue */
        }
        .stButton>button:active {
            background-color: #1f6a9c; /* Even Darker Blue */
        }


        /* Expander for script output */
        .stExpander {
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-top: 1rem;
        }
        .stExpander header {
            background-color: #ecf0f1; /* Light grey for expander header */
            padding: 0.5rem 1rem;
            font-weight: bold;
            color: #2c3e50;
        }

        /* Code blocks (like the script output) */
        pre {
            background-color: #2c3e50; /* Dark background for code */
            color: #f8f8f2; /* Light text for code */
            padding: 1rem;
            border-radius: 5px;
            overflow-x: auto; /* Allow horizontal scrolling for long lines */
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.95em;
            white-space: pre-wrap; /* Wrap long lines */
            word-wrap: break-word;
        }
        
        /* Sidebar */
        .css-1d391kg { /* Streamlit's default sidebar class might change, target more generally if needed */
            background-color: #34495e; /* Darker sidebar */
            padding: 1rem;
        }
        .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown li {
             color: #ecf0f1; /* Lighter text for sidebar */
        }


        /* Custom class for README display */
        .readme-content {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            line-height: 1.6;
        }
        .readme-content img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        """)

local_css(css_file_path)


# --- Sidebar Content ---
st.sidebar.title("About")
st.sidebar.info(
    "This application demonstrates the Property Recommendation System."
    "It runs a Python script to find similar properties based on a dataset."
)
st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown(
    "The README content is displayed on the main page. Click the **'Run Property Similarity Script'** button to execute the analysis."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Streamlit")


# --- Main Application ---
st.title("🏠 Property Recommendation System")

# Display README.md content
st.header("Project Overview (README)")
try:
    with open("README.md", "r", encoding="utf-8") as readme_file:
        readme_content = readme_file.read()
    st.markdown(f"<div class='readme-content'>{readme_content}</div>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("README.md not found. Please make sure it's in the same directory as app.py.")
except Exception as e:
    st.error(f"Error reading README.md: {e}")

st.markdown("---") # Visual separator

# Button to run the property_similarity.py script
st.header("Run Analysis")
st.write(
    "Click the button below to run the `property_similarity.py` script. "
    "The script will process the `appraisals_dataset.json` file (make sure it's present) "
    "and display its output."
)

if st.button("Run Property Similarity Script", key="run_script_button"):
    script_path = "property_similarity.py"
    dataset_path = "appraisals_dataset.json"

    if not os.path.exists(script_path):
        st.error(f"Error: The script '{script_path}' was not found.")
    elif not os.path.exists(dataset_path):
        st.error(f"Error: The dataset '{dataset_path}' was not found. The script requires this file to run.")
    else:
        with st.spinner("Running property similarity analysis... Please wait."):
            try:
                # Ensure the script can find pandas, sklearn, etc.
                # Running with `python -u` for unbuffered output
                process = subprocess.Popen(
                    ["python", "-u", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1, # Line buffered
                    encoding='utf-8'
                )
                
                output_placeholder = st.empty()
                full_output = ""
                
                # Stream output
                for line in process.stdout:
                    full_output += line
                    output_placeholder.code(full_output, language="text") # Use st.code for better formatting of script output

                # Wait for the process to complete and capture any remaining output/errors
                stdout, stderr = process.communicate()
                full_output += stdout # Append any remaining stdout
                
                if process.returncode == 0:
                    st.success("Script executed successfully!")
                    if not full_output.strip(): # If stdout was empty but successful
                        output_placeholder.code("Script ran successfully, but produced no output.", language="text")
                    else:
                         output_placeholder.code(full_output, language="text")
                else:
                    st.error(f"Script execution failed with return code {process.returncode}.")
                    error_message = full_output + stderr # Combine stdout (if any) and stderr
                    if not error_message.strip():
                         output_placeholder.code(f"Script failed. Error: No specific error message captured. Return code: {process.returncode}", language="text")
                    else:
                        output_placeholder.code(error_message, language="text")

            except FileNotFoundError:
                st.error(f"Error: Python interpreter not found or '{script_path}' not found. Make sure Python is installed and the script path is correct.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.code(str(e), language="text")

st.markdown("---")
st.info("End of page.") 